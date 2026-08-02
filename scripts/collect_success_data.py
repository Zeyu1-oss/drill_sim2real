

import argparse
import os
import sys
import pickle
import time
from pathlib import Path

import numpy as np
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (scripts/ -> ../..)
sys.path.insert(0, project_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_envs', type=int, default=2048)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--output', type=str, default='collected_data/success_data.pkl')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--drill_configs', type=str, default=None)
    parser.add_argument('--frame-scale', type=float, default=0.5)
    parser.add_argument('--history-len', type=int, default=0)
    parser.add_argument('--playback', type=str, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] checkpoint file not found: {args.checkpoint}")
        return

    # ============================================================
    # init Isaac Lab
    # ============================================================
    from isaaclab.app import AppLauncher
    headless_mode = args.headless
    print(f"[INFO] launching Isaac Sim (headless={headless_mode})...")
    app_launcher = AppLauncher(headless=headless_mode)
    simulation_app = app_launcher.app

    try:
        import math
        import yaml

        import omni.timeline
        import carb

        from rl_games.common import env_configurations, vecenv
        from rl_games.common.player import BasePlayer
        from rl_games.torch_runner import Runner

        from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
        from tasks.grasp_drill_env import GraspDrillEnv, create_grasp_drill_env_cfg

        # ============================================================
        # load RL Games config
        # ============================================================
        rl_config_path = os.path.join(project_root, "config/agents/rl_games_ppo_cfg.yaml")
        with open(rl_config_path, "r") as f:
            agent_cfg = yaml.safe_load(f)

        agent_cfg["params"]["config"]["num_actors"] = args.num_envs
        agent_cfg["params"]["config"]["device"] = args.device
        agent_cfg["params"]["config"]["device_name"] = args.device

        # ============================================================
        # create env config (identical to play_drill.py)
        # ============================================================
        cfg = create_grasp_drill_env_cfg(
            num_envs=args.num_envs,
            device=args.device,
            headless=args.headless,
            debug=args.debug,
            drill_config_path=args.drill_configs,
            enable_cameras=False,   # state-only collection, no camera needed (default True crashes / uses VRAM)
        )
        cfg.debug_frame_scale = args.frame_scale
        cfg.seed = args.seed

        log_dir = os.path.dirname(os.path.dirname(args.checkpoint))
        cfg.log_dir = log_dir

        print(f"[INFO] creating env (num_envs={args.num_envs})...")
        env = GraspDrillEnv(cfg=cfg, debug=args.debug)

        # wrap env
        clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
        clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
        rl_device = args.device

        env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

        # register env
        vecenv.register(
            "IsaacRlgWrapper",
            lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
        )
        env_configurations.register(
            "rlgpu",
            {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env}
        )

        # ============================================================
        # load model
        # ============================================================
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = args.checkpoint
        agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

        runner = Runner()
        runner.load(agent_cfg)
        agent: BasePlayer = runner.create_player()
        agent.restore(args.checkpoint)
        agent.reset()

        # ============================================================
        # sim timestep config
        # ============================================================
        dt = env.unwrapped.step_dt

        settings = carb.settings.get_settings()
        settings.set("/app/player/useFixedTimeStepping", True)
        settings.set("/app/player/targetFrameRate", int(1.0 / dt))
        settings.set("/app/runLoops/rendering_0/rateLimitEnabled", True)
        settings.set("/app/runLoops/rendering_0/rateLimit", 60)
        settings.set("/physics/updateToUsd", False)
        settings.set("/physics/updateParticlesToUsd", False)

        timeline = omni.timeline.get_timeline_interface()

        # ============================================================
        # start sim
        # ============================================================
        sys.stdout.flush()
        timeline.play()
        sys.stdout.flush()

        # ============================================================
        # reset env, get initial observation
        # ============================================================
        obs = env.reset()
        if isinstance(obs, dict):
            obs = obs["obs"]
        _ = agent.get_batch_size(obs, 1)

        # ============================================================
        # Playback mode: load PKL data to drive the initial state
        # ============================================================
        pkl_samples = None
        if args.playback:
            if not os.path.exists(args.playback):
                print(f"[ERROR] Playback PKL file not found: {args.playback}")
                return
            with open(args.playback, "rb") as f:
                pkl_data = pickle.load(f)
            pkl_samples = pkl_data["samples"]
            print(f"[INFO] Playback mode: loaded {len(pkl_samples)} PKL samples")
            print(f"  PKL metadata: {pkl_data['metadata']}")
            _pkl_idx = 0  # round-robin pointer

        # ============================================================
        # collection state tracking
        # ============================================================
        # episode_state: 0=never succeeded, 1=succeeded (waiting for episode end), 2=collected this episode
        episode_state = torch.zeros(args.num_envs, dtype=torch.long, device=args.device)

        _success_reward_cache = [0.0] * args.num_envs
        _success_step_cache = [0] * args.num_envs

        # previous step's obs and joint state (to save the \"step before success\" data on success)
        env_unwrapped = env.unwrapped

        # ring buffer: keep the last N steps of Franka/drill state (captured before step)
        # Franka and drill get reset inside env.step(), so they must be saved before step
        # fix 2: buffer length = history_len + 1, ensuring step N-5 is reachable
        HISTORY = args.history_len + 1
        _hist_obs = [None] * HISTORY
        _hist_joint_pos = [None] * HISTORY
        _hist_joint_vel = [None] * HISTORY
        _hist_drill_pos_w = [None] * HISTORY
        _hist_drill_quat_w = [None] * HISTORY
        _hist_drill_lin_vel = [None] * HISTORY
        _hist_drill_ang_vel = [None] * HISTORY
        _hist_env_origins = [None] * HISTORY
        _hist_head = 0  # current head pointer of the ring buffer

        # preallocate buffer tensors (avoid repeated clone in the loop)
        _obs_buf = obs.cpu().clone()
        _jp_buf = env_unwrapped.franka.data.joint_pos.cpu().clone()
        _jv_buf = env_unwrapped.franka.data.joint_vel.cpu().clone()
        _dp_buf = env_unwrapped.drill.data.root_pos_w.cpu().clone()
        _dq_buf = env_unwrapped.drill.data.root_quat_w.cpu().clone()
        _dlv_buf = env_unwrapped.drill.data.root_lin_vel_w.cpu().clone()
        _dav_buf = env_unwrapped.drill.data.root_ang_vel_w.cpu().clone()
        _eo_buf = env_unwrapped.scene.env_origins.cpu().clone()

        # batched success cache (vectorized, CPU): overwritten each step while success persists; at episode end
        # the cache holds the full state of \"the last time still in success\" -- that is the final grasp pose to collect.
        # (old logic cached only once at first success, storing an intermediate pose just past the threshold, even N steps back)
        _cache_valid = torch.zeros(args.num_envs, dtype=torch.bool)
        _cache_obs = torch.zeros_like(_obs_buf)
        _cache_jp = torch.zeros_like(_jp_buf)
        _cache_jv = torch.zeros_like(_jv_buf)
        _cache_dp = torch.zeros_like(_dp_buf)
        _cache_dq = torch.zeros_like(_dq_buf)
        _cache_dlv = torch.zeros_like(_dlv_buf)
        _cache_dav = torch.zeros_like(_dav_buf)
        _cache_eo = torch.zeros_like(_eo_buf)

        collected_samples = []
        collected_count = 0
        step_count = 0

        print_interval = 100
        print(f"\n[INFO] starting data collection... (target: {args.num_samples})")
        print(f"[INFO] print stats every {print_interval} steps")
        start_time = time.time()
        sys.stdout.flush()

        with torch.inference_mode():
            while collected_count < args.num_samples:
                # --- save Franka/drill state before step ---
                # Franka and drill get reset inside env.step(), must capture before step
                if isinstance(obs, dict):
                    _obs_buf.copy_(obs["obs"])
                else:
                    _obs_buf.copy_(obs)

                _jp_buf.copy_(env_unwrapped.franka.data.joint_pos)
                _jv_buf.copy_(env_unwrapped.franka.data.joint_vel)
                _dp_buf.copy_(env_unwrapped.drill.data.root_pos_w)
                _dq_buf.copy_(env_unwrapped.drill.data.root_quat_w)
                _dlv_buf.copy_(env_unwrapped.drill.data.root_lin_vel_w)
                _dav_buf.copy_(env_unwrapped.drill.data.root_ang_vel_w)
                _eo_buf.copy_(env_unwrapped.scene.env_origins)

                # write to ring buffer
                _hist_obs[_hist_head] = _obs_buf.clone()
                _hist_joint_pos[_hist_head] = _jp_buf.clone()
                _hist_joint_vel[_hist_head] = _jv_buf.clone()
                _hist_drill_pos_w[_hist_head] = _dp_buf.clone()
                _hist_drill_quat_w[_hist_head] = _dq_buf.clone()
                _hist_drill_lin_vel[_hist_head] = _dlv_buf.clone()
                _hist_drill_ang_vel[_hist_head] = _dav_buf.clone()
                _hist_env_origins[_hist_head] = _eo_buf.clone()

                # --- execute step (success triggers reset) ---
                obs_tensor = agent.obs_to_torch(obs)
                actions = agent.get_action(obs_tensor, is_deterministic=True)
                obs, rewards, dones, _ = env.step(actions)
                step_count += 1

                env_unwrapped = env.unwrapped

                # update the ring buffer head pointer (advance after each step)
                _hist_head = (_hist_head + 1) % HISTORY

                # --- update obs after step (obs returns post-reset) ---
                if isinstance(obs, dict):
                    obs_dict = obs["obs"]
                else:
                    obs_dict = obs
                _obs_buf.copy_(obs_dict)

                # ============================================================
                # success check (identical to play_drill.py)
                # ============================================================
                try:
                    lenient_success = env_unwrapped._cached_lenient_success
                except AttributeError:
                    lenient_success = torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)

                current_success = lenient_success.bool()
                is_done = dones.bool() if hasattr(dones, 'bool') else dones > 0

                # ============================================================
                # track each env's episode state
                # ============================================================
                just_became_success = (episode_state == 0) & current_success

                # -- cache refresh (vectorized): overwrite every step while success persists; at episode end
                #    the cache = the last pose still in success (history_len>0 can go further back).
                #    tail points to the state captured before this step (_hist_head already +1).
                tail = (_hist_head - args.history_len - 1) % HISTORY
                if _hist_obs[tail] is not None and current_success.any():
                    sel = current_success.detach().cpu()
                    _cache_valid |= sel
                    _cache_obs[sel] = _hist_obs[tail][sel]
                    _cache_jp[sel] = _hist_joint_pos[tail][sel]
                    _cache_jv[sel] = _hist_joint_vel[tail][sel]
                    _cache_dp[sel] = _hist_drill_pos_w[tail][sel]
                    _cache_dq[sel] = _hist_drill_quat_w[tail][sel]
                    _cache_dlv[sel] = _hist_drill_lin_vel[tail][sel]
                    _cache_dav[sel] = _hist_drill_ang_vel[tail][sel]
                    _cache_eo[sel] = _hist_env_origins[tail][sel]

                # first success: initialize episode stats
                for env_idx in torch.where(just_became_success)[0]:
                    idx = env_idx.item()
                    _success_reward_cache[idx] = 0.0
                    _success_step_cache[idx] = step_count
                    if args.debug and idx < 4:
                        print(f"  [DEBUG] env={idx} first success (step={step_count})")

                # accumulate episode reward (keep accumulating within the success window)
                success_mask = episode_state == 1
                for env_idx in torch.where(success_mask)[0]:
                    idx = env_idx.item()
                    _success_reward_cache[idx] += rewards[idx].item()

                # mark as succeeded (waiting for episode end)
                episode_state[just_became_success] = 1

                # ============================================================
                # can collect: has succeeded (episode_state==1) and episode ended
                # ============================================================
                can_collect = (episode_state == 1) & is_done
                if can_collect.any():
                    for env_idx in torch.where(can_collect)[0]:
                        idx = env_idx.item()

                        if _cache_valid[idx]:
                            # must clone: single-row indexing is a view, _build_sample's .numpy() shares memory,
                            # without clone the env's next episode cache refresh would corrupt the collected sample
                            cached_state = {
                                "joint_pos": _cache_jp[idx].clone(),
                                "joint_vel": _cache_jv[idx].clone(),
                                "drill_pos_world": _cache_dp[idx].clone(),
                                "drill_quat": _cache_dq[idx].clone(),
                                "drill_lin_vel": _cache_dlv[idx].clone(),
                                "drill_ang_vel": _cache_dav[idx].clone(),
                                "env_origin": _cache_eo[idx].clone(),
                            }
                            sample = _build_sample(env_unwrapped, idx, _cache_obs[idx].clone(),
                                                   _success_reward_cache[idx],
                                                   step_count - _success_step_cache[idx],
                                                   cached_state)
                            collected_samples.append(sample)
                            collected_count += 1

                            if collected_count >= args.num_samples:
                                break

                        # reset this env's cache
                        _cache_valid[idx] = False
                        _success_reward_cache[idx] = 0.0
                        _success_step_cache[idx] = 0
                        episode_state[env_idx] = 2

                # ============================================================
                # episode reset handling
                # ============================================================
                if is_done.any():
                    agent.reset()
                    reset_0 = (episode_state == 0)
                    reset_1 = (episode_state == 1)
                    reset_2 = (episode_state == 2)
                    # state==1: timed out before the success window ended, not collected
                    # state==2: already collected, waiting to reset back to active state
                    episode_state[reset_0 | reset_1 | reset_2] = 0
                    # clear the cache for state 0/1 (state 2 cache was cleared at collection time, no need)
                    _cache_valid[(reset_0 | reset_1).detach().cpu()] = False
                    for env_idx in torch.where(reset_0 | reset_1)[0]:
                        idx = env_idx.item()
                        _success_reward_cache[idx] = 0.0
                        _success_step_cache[idx] = 0

                # ============================================================
                # progress print (based on step count + ETA)
                # ============================================================
                if step_count % print_interval == 0:
                    n_waiting = (episode_state == 1).sum().item()
                    n_success_this_batch = just_became_success.sum().item()
                    n_collected = collected_count
                    elapsed = time.time() - start_time
                    steps_per_sample = step_count / max(collected_count, 1)
                    eta = steps_per_sample * (args.num_samples - collected_count)
                    m, s = divmod(int(eta), 60)
                    h, m = divmod(m, 60)
                    eta_str = f"{h}h{m}m{s}s" if h > 0 else f"{m}m{s}s"
                    bar_len = 30
                    filled = int(bar_len * collected_count / args.num_samples)
                    bar = "#" * filled + "-" * (bar_len - filled)
                    print(f"  [{bar}] step={step_count:,} | collected={collected_count}/{args.num_samples} "
                          f"| rate={step_count/elapsed:.0f}stp/s | waiting={n_waiting} "
                          f"| successes_detected={n_success_this_batch} | ETA={eta_str}")
                    sys.stdout.flush()

                    # ---- diagnostic: print the distribution of success components ----
                    if args.debug and hasattr(env_unwrapped, '_check_success'):
                        with torch.inference_mode():
                            try:
                                # temporarily force a single computation (bypass cache)
                                temp = env_unwrapped._cached_lenient_success.clone()
                                env_unwrapped._cached_lenient_success.zero_()
                                raw_success = env_unwrapped._check_success()
                                env_unwrapped._cached_lenient_success.copy_(temp)

                                # drill pos
                                drill_z = env_unwrapped.drill.data.root_pos_w[:, 2]
                                init_z = env_unwrapped.initial_drill_pos[:, 2]
                                lift_ok = (drill_z - init_z) > env_unwrapped.lift_z_threshold
                                window_sum = env_unwrapped._success_window.sum(dim=1)
                                print(f"  [DIAG] step={step_count} | "
                                      f"raw_success={raw_success.sum().item()}/{args.num_envs} | "
                                      f"lenient_success={current_success.sum().item()}/{args.num_envs} | "
                                      f"lift_ok={lift_ok.sum().item()}/{args.num_envs} | "
                                      f"window_sum>={window_sum[window_sum>0].min().item() if (window_sum>0).any() else 0} "
                                      f"(max={window_sum.max().item()})")
                                sys.stdout.flush()
                            except Exception as e:
                                pass

                if isinstance(obs, dict):
                    obs = obs["obs"]

        # ============================================================
        # save data
        # ============================================================
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        save_data(collected_samples, args.output, args.history_len)

        elapsed = time.time() - start_time
        m, s = divmod(int(elapsed), 60)
        h, m = divmod(m, 60)
        print(f"\n{'='*60}")
        print(f"data collection complete!")
        print(f"  target samples: {args.num_samples}")
        print(f"  actually collected:   {len(collected_samples)}")
        print(f"  total steps:     {step_count:,}")
        print(f"  total time:     {h}h{m}m{s}s")
        print(f"  avg rate:   {step_count/elapsed:.0f} steps/s")
        print(f"  output file:   {args.output}")
        print(f"{'='*60}")

        _print_variant_stats(collected_samples)

        env.close()

    finally:
        simulation_app.close()


def _build_sample(env, env_idx: int, cached_obs: torch.Tensor,
                  episode_reward: float, success_hold_steps: int,
                  cached_state: dict) -> dict:
    """Build one complete success sample from the cache.

    cached_state is read from env at the first success moment, then no longer depends on env state,
    avoiding reading the new episode's state after _reset_idx.
    """
    try:
        # ---- joint state (read from cache, extract controlled joints, consistent with _get_observations) ----
        joint_pos_23 = cached_state["joint_pos"].cpu().numpy()
        joint_vel_23 = cached_state["joint_vel"].cpu().numpy()
        # controlled_joint_indices: fr3_joint1~7 (7) + 6 proximal hand joints = 13
        controlled = env.controlled_joint_indices.cpu().numpy()  # shape=(13,)
        joint_pos = joint_pos_23[controlled]
        joint_vel = joint_vel_23[controlled]

        # ---- Drill state (world -> local coordinates) ----
        drill_pos_world = cached_state["drill_pos_world"].cpu().numpy()
        env_origin = cached_state["env_origin"].cpu().numpy()
        drill_pos = drill_pos_world - env_origin  # convert to local coordinates (relative to env_origin)
        drill_quat = cached_state["drill_quat"].cpu().numpy()
        drill_lin_vel = cached_state["drill_lin_vel"].cpu().numpy()
        drill_ang_vel = cached_state["drill_ang_vel"].cpu().numpy()

        # ---- Variant info ----
        variant_idx = env._drill_variant_indices[env_idx].item()
        variant_name = env._variant_attrs.get(variant_idx, {}).get(
            "name", f"variant_{variant_idx}"
        )

        # ---- Success window stats ----
        success_window_sum = env._success_window[env_idx].sum().item()

        # ---- Policy observation (raw tensor -> numpy) ----
        full_obs = cached_obs.cpu().numpy()

        return {
            # full policy observation vector (usable directly for relabel / BC / SAC etc.)
            "full_obs": full_obs,
            # all joint positions/velocities (13 dims, controlled joints only)
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            # Drill world-coordinate state
            "drill_pos": drill_pos,
            "drill_quat": drill_quat,
            "drill_lin_vel": drill_lin_vel,
            "drill_ang_vel": drill_ang_vel,
            # Variant info
            "variant_idx": variant_idx,
            "variant_name": variant_name,
            # Episode stats
            "episode_reward": episode_reward,
            "success_hold_steps": success_hold_steps,
            "success_window_sum": int(success_window_sum),
        }
    except Exception as e:
        print(f"[WARN] _build_sample failed for env_idx={env_idx}: {e}")
        return None


def _print_variant_stats(samples: list):
    """Print collection stats per variant."""
    if not samples:
        return
    from collections import Counter
    variant_counts = Counter(s["variant_name"] for s in samples)
    print("\nPer-variant collection stats:")
    for name, count in sorted(variant_counts.items()):
        print(f"  {name}: {count} ({count / len(samples) * 100:.1f}%)")


def save_data(samples: list, output_path: str, history_len: int = 5):
    """Save the collected data to a pickle file."""
    # filter out None (failed collections)
    valid_samples = [s for s in samples if s is not None]
    skipped = len(samples) - len(valid_samples)
    if skipped > 0:
        print(f"[WARN] skipped {skipped} invalid samples")

    # fix 3: define metadata whether or not skipped > 0, to avoid a crash when skipped==0
    metadata = {
        "num_samples": len(valid_samples),
        "obs_dim": valid_samples[0]["full_obs"].shape[-1] if valid_samples else 0,
        "joint_pos_dim": valid_samples[0]["joint_pos"].shape[-1] if valid_samples else 0,
        "joint_vel_dim": valid_samples[0]["joint_vel"].shape[-1] if valid_samples else 0,
        "version": "1.2",
        "history_len": history_len,
        "description": (
            f"Success episode data for grasp_drill task. "
            f"full_obs is the policy observation vector "
            f"({valid_samples[0]['full_obs'].shape[-1] if valid_samples else '?'}-dim, matches stage1). "
            f"Collected lenient_success=True. "
            f"Stores the LAST state at which success still held (cache refreshed every "
            f"success step; history_len={history_len} extra backtrack). "
            "drill_pos is stored in LOCAL coordinates (relative to env_origin). "
            "joint_pos/vel are 13-dim controlled joints only (matching _get_observations)."
        ),
    }

    output = {
        "metadata": metadata,
        "samples": valid_samples,
    }

    with open(output_path, "wb") as f:
        pickle.dump(output, f)

    print(f"[INFO] saved {len(valid_samples)} samples to: {output_path}")


if __name__ == "__main__":
    main()
