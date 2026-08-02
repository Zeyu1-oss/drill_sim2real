
import argparse
import os
import sys
from pathlib import Path

# add project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (scripts/ -> ../..)
sys.path.insert(0, project_root)


def _body_velocities(franka_data):
    """Return (lin, ang): each link's world-frame linear/angular velocity (num_envs, num_bodies, 3).
    Prefer IsaacLab's body_lin_vel_w/body_ang_vel_w, fall back to body_vel_w[...,:3]/[...,3:] if missing."""
    lin = getattr(franka_data, "body_lin_vel_w", None)
    ang = getattr(franka_data, "body_ang_vel_w", None)
    if lin is None or ang is None:
        bv = franka_data.body_vel_w          # (N, B, 6) = [lin(3), ang(3)]
        lin, ang = bv[..., :3], bv[..., 3:]
    return lin, ang


def print_link_velocities(franka, env_id, step):
    """Print each robot link's world-frame velocity for the given env (linear vel vector+magnitude, angular vel magnitude)."""
    import torch
    lin, ang = _body_velocities(franka.data)
    lin = lin[env_id]                        # (B, 3)
    ang = ang[env_id]                        # (B, 3)
    names = franka.body_names
    lin_sp = torch.linalg.norm(lin, dim=-1)  # (B,) linear speed m/s
    ang_sp = torch.linalg.norm(ang, dim=-1)  # (B,) angular speed rad/s
    print(f"\n[link-vel] step={step} env={env_id}  (world frame)")
    print(f"  {'link':<26} {'|v|(m/s)':>9} {'vx':>7} {'vy':>7} {'vz':>7} {'|w|(rad/s)':>11}")
    for i, nm in enumerate(names):
        v = lin[i]
        print(f"  {nm:<26} {lin_sp[i].item():9.3f} {v[0].item():7.3f} "
              f"{v[1].item():7.3f} {v[2].item():7.3f} {ang_sp[i].item():11.3f}")
    print(f"  -> max linear speed over links={lin_sp.max().item():.3f} m/s "
          f"(link={names[int(lin_sp.argmax())]}), max angular speed={ang_sp.max().item():.3f} rad/s",
          flush=True)


def _wrist_body_index(franka, wrist_name="R_hand_base_link"):
    """Index of the wrist (palm base) link in body_names; fall back to the 'hand_base' substring, else None."""
    names = list(franka.body_names)
    if wrist_name in names:
        return names.index(wrist_name)
    for i, nm in enumerate(names):
        if "hand_base" in nm:
            return i
    return None


def print_wrist_velocity(franka, env_id, step):
    """Print the wrist (R_hand_base_link) world-frame linear velocity for the given env (vector + magnitude, m/s)."""
    import torch
    idx = _wrist_body_index(franka)
    if idx is None:
        print(f"[wrist-vel] step={step} env={env_id}: wrist link not found (R_hand_base_link)", flush=True)
        return
    lin, _ = _body_velocities(franka.data)
    v = lin[env_id, idx]                          # (3,) world-frame linear velocity
    sp = torch.linalg.norm(v).item()
    # print(f"  [wrist-vel] env={env_id} link={franka.body_names[idx]} "
        #   f"|v|={sp:6.3f} m/s  v=({v[0].item():7.3f}, {v[1].item():7.3f}, {v[2].item():7.3f})",
        #   flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_envs', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--checkpoint', type=str, default='runs/inspire_hand_grasp_drill_26-06-13-21-10/inspire_hand_grasp_drill_26-06-13-21-10/nn/inspire_hand_grasp_drill_26-06-13-21-10.pth')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--frame-scale', type=float, default=0.5)
    parser.add_argument('--real_time', action='store_true')
    parser.add_argument('--drill_configs', type=str, default=None)
    parser.add_argument('--drill_variants', type=str, default=None)
    parser.add_argument('--render_width', type=int, default=2560)
    parser.add_argument('--render_height', type=int, default=1440)
    parser.add_argument('--link_vel_interval', type=int, default=0)
    parser.add_argument('--link_vel_env', type=int, default=0)
    parser.add_argument('--target_drive', action='store_true')

    args = parser.parse_args()

    # check the checkpoint path before launching Isaac Sim, so a typo does not waste minutes of startup
    if args.checkpoint and not os.path.exists(args.checkpoint):
        print(f"[ERROR] checkpoint not found: {args.checkpoint}")
        return

    # init Isaac Lab
    from isaaclab.app import AppLauncher
    
    # play script uses GUI mode (visualization) by default
    headless_mode = args.headless
    print(f"[INFO] launching Isaac Sim (headless={headless_mode}, render={args.render_width}x{args.render_height})...")
    app_launcher = AppLauncher(headless=headless_mode)
    simulation_app = app_launcher.app
    
    # enable real-time mode (if specified)
    if args.real_time:
        import carb
        settings = carb.settings.get_settings()
        settings.set_bool("/app/runLoops/main/enabled", True)
        settings.set_bool("/app/runLoops/main/syncTooFast", True)
    
    try:
        import math
        import random
        import time

        import numpy as np
        import gymnasium as gym
        import torch
        from rl_games.common import env_configurations, vecenv
        from rl_games.common.player import BasePlayer
        from rl_games.torch_runner import Runner
        
        from isaaclab.envs import ManagerBasedRLEnvCfg
        from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
        
        from tasks.grasp_drill_env import GraspDrillEnv, create_grasp_drill_env_cfg
        
        # load RL Games config
        import yaml
        rl_config_path = os.path.join(project_root, "config/agents/rl_games_ppo_cfg.yaml")
        with open(rl_config_path, "r") as f:
            agent_cfg = yaml.safe_load(f)
        
        # update config
        agent_cfg["params"]["config"]["num_actors"] = args.num_envs
        agent_cfg["params"]["config"]["device"] = args.device
        agent_cfg["params"]["config"]["device_name"] = args.device
        
        # set checkpoint
        if args.checkpoint:
            resume_path = args.checkpoint
        else:
            # by default find the latest checkpoint
            run_name = "inspire_hand_grasp_drill"
            runs_dir = Path(project_root) / "runs"
            if runs_dir.exists():
                for run_dir in sorted(runs_dir.iterdir(), reverse=True):
                    if run_name in run_dir.name:
                        # find checkpoints in nested directories
                        for subdir in run_dir.iterdir():
                            if subdir.is_dir():
                                nn_dir = subdir / "nn"
                                if nn_dir.exists():
                                    pth_files = list(nn_dir.glob("*.pth"))
                                    if pth_files:
                                        resume_path = str(max(pth_files, key=lambda x: x.stat().st_mtime))
                                        break
                        else:
                            continue
                        break
            else:
                resume_path = None
        
        if not resume_path or not os.path.exists(resume_path):
            print(f"[ERROR] no usable checkpoint found (resume_path={resume_path})")
            return
        
        
        # create env config
        cfg = create_grasp_drill_env_cfg(
            num_envs=args.num_envs,
            device=args.device,
            headless=args.headless,
            debug=args.debug,
            drill_config_path=args.drill_configs,
            drill_variants_path=args.drill_variants,
            enable_cameras=False,
        )
        # set frame size
        cfg.debug_frame_scale = args.frame_scale
        cfg.seed = args.seed
        
        # set log_dir
        log_dir = os.path.dirname(os.path.dirname(resume_path))
        cfg.log_dir = log_dir
        
        # create env (instantiate the env class directly)
        print(f"\ncreating env (num_envs={args.num_envs})...")
        env = GraspDrillEnv(cfg=cfg, debug=args.debug)

        # delete the template prototype left by MultiAssetSpawner, to avoid extra objects in the scene
        from isaaclab.sim.utils import delete_prim
        try:
            delete_prim("/World/Template")
            print("[INFO] deleted /World/Template")
        except Exception as e:
            print(f"[WARN] failed to delete /World/Template: {e}")
        

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
        env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})
        
        # load model
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
        
        runner = Runner()
        runner.load(agent_cfg)
        agent: BasePlayer = runner.create_player()
        agent.restore(resume_path)
        agent.reset()
        

        # init
        dt = env.unwrapped.step_dt
        
        # reset env, get initial observation
        obs = env.reset()
        if isinstance(obs, dict):
            obs = obs["obs"]
        _ = agent.get_batch_size(obs, 1)

        # use omni.timeline to control the sim
        import omni.timeline

        import carb
        from perception.camera_setup import apply_render_settings
        settings = apply_render_settings(dt)
        settings.set("/app/renderer/resolution/width", args.render_width)
        settings.set("/app/renderer/resolution/height", args.render_height)
        settings.set("/app/renderer/resolution/multiplier", 1.0)
        settings.set("/rtx/rendermode", "RaytracedLighting")
        timeline = omni.timeline.get_timeline_interface()
        
        # start sim
        sys.stdout.flush()
        timeline.play()
        sys.stdout.flush()

        total_episodes = 0
        successful_episodes = 0
        episode_rewards = []
        current_episode_reward = torch.zeros(args.num_envs, device=args.device)
        success_log = []
        ep_len_buf = torch.zeros(args.num_envs, device=args.device)   # steps taken this episode per env
        _done_dbg = 0   # number of reset diagnostics printed (capped to avoid spam)

        # ===== direct target drive (--target_drive) =====
        # MLP outputs raw -> raw_to_target converts to joint target angles q* -> commanded directly to PD (skipping the raw->EMA inverse mapping).
        # shares perception/target_drive.py (single source) with collect/deploy.
        _td = None
        if args.target_drive:
            from perception.target_drive import install_direct_drive, raw_to_target
            _drive_p = install_direct_drive(env.unwrapped)
            print(f"[TARGET_DRIVE] play direct target drive: {_drive_p}", flush=True)
            _td = (raw_to_target, _drive_p)

        # ============================================================
        # PhysX distance validation (if debug mode enabled)
        # ============================================================
        if args.debug:
            print("\n" + "=" * 60)
            print("running PhysX distance validation...")
            print("=" * 60)
            try:
                from omni.physx import get_physx_scene_query_interface
                query_interface = get_physx_scene_query_interface()
                print("[OK] PhysX query interface acquired")
            except Exception as e:
                print(f"[X] PhysX API unavailable: {e}")

        # main loop
        step_count = 0
        print_interval = 1  # print every 100 steps
        try:
            sys.stdout.flush()
            sys.stdout.flush()

            while simulation_app.is_running():
                import time

                if step_count == 0:
                    wall_start_time = time.time()                
                with torch.inference_mode():
                    obs = agent.obs_to_torch(obs)
                    actions = agent.get_action(obs, is_deterministic=True)
                    if _td is not None:
                        # raw -> target (q*); stored to env, patched _apply_action commands it directly, no inverse mapping
                        cur0 = env.unwrapped.cur_targets.clone()
                        env.unwrapped._direct_target = _td[0](actions, cur0, _td[1])
                    obs, rewards, dones, _ = env.step(actions)

                    step_count += 1
                    ep_len_buf += 1

                    # print env link velocity (every link_vel_interval steps, 0=off)
                    if args.link_vel_interval > 0 and step_count % args.link_vel_interval == 0:
                        print_link_velocities(env.unwrapped.franka, args.link_vel_env, step_count)

                    # if step_count % 200 == 0:
                    #     wall_time = time.time() - wall_start_time
                    #     sim_time = step_count * dt
                    #     print(f"[speed check] sim_time={sim_time:.2f}s | wall_time={wall_time:.2f}s | ratio={sim_time/wall_time:.2f}x")
                    #                     # accumulate episode reward
                    current_episode_reward += rewards

                    # detect finished envs
                    if len(dones) > 0:
                        done_indices = torch.where(dones)[0]
                        for env_idx in done_indices:
                            total_episodes += 1
                            ep_rew = current_episode_reward[env_idx].item()
                            episode_rewards.append(ep_rew)

                            # consistent with training: use _cached_lenient_success (filled by grasp_drill_env._get_dones)
                            env_unwrapped = env.unwrapped
                            try:
                                is_success = env_unwrapped._cached_lenient_success[env_idx].item()
                            except Exception:
                                is_success = False

                            if is_success:
                                successful_episodes += 1
                                success_log.append(1)
                            else:
                                success_log.append(0)

                            # === reset diagnostic: print this episode's steps + likely reason (first 40) ===
                            if _done_dbg < 40 and not is_success:
                                _ei = int(env_idx)
                                _len = int(ep_len_buf[_ei].item())
                                try:
                                    _nan = bool(env_unwrapped._pending_nan_mask[_ei].item())
                                except Exception:
                                    _nan = False
                                # whether finger joints exceed limits (eps=0.2, same as _get_dones); post-reset reads the new episode's values, reference only
                                _fj = env_unwrapped.franka.data.joint_pos[_ei, env_unwrapped.finger_indices]
                                _fl = env_unwrapped.joint_lower_limits[env_unwrapped.num_arm_joints:]
                                _fu = env_unwrapped.joint_upper_limits[env_unwrapped.num_arm_joints:]
                                _viol = float(((_fj - _fl).clamp(max=0).abs() + (_fj - _fu).clamp(min=0)).max().item())
                                print(f"[RESET] env={_ei} len={_len} steps success={is_success} "
                                      f"nan={_nan} finger_viol(post)={_viol:.3f} reward={ep_rew:.2f}", flush=True)
                                _done_dbg += 1

                            ep_len_buf[env_idx] = 0.0
                            current_episode_reward[env_idx] = 0.0

                    # print progress every print_interval steps (debug mode also prints distance info)
                    if step_count % print_interval == 0:
                        running_success_rate = (successful_episodes / total_episodes * 100) if total_episodes > 0 else 0.0
                        print(f"  Step: {step_count} | dones: {dones.sum().item():.0f} | episodes: {total_episodes} | success: {successful_episodes} ({running_success_rate:.1f}%)")
                        # wrist (R_hand_base_link) linear velocity, to observe wrist motion during grasp/move
                        print_wrist_velocity(env.unwrapped.franka, args.link_vel_env, step_count)

                        # print policy.mu stats
                        if hasattr(agent, 'a2c') and hasattr(agent.a2c, 'model'):
                            policy = agent.a2c.model
                            if hasattr(policy, 'mu'):
                                mu = policy.mu(obs)
                                print(f"  [Policy mu] mean={mu.mean().item():.4f}, std={mu.std().item():.4f}, max={mu.max().item():.4f}, min={mu.min().item():.4f}")



                        sys.stdout.flush()

                if isinstance(obs, dict):
                    obs = obs["obs"]

                if len(dones) > 0:
                    agent.reset()

            print(f"[DEBUG] exited while loop, is_running = {simulation_app.is_running()}")
            sys.stdout.flush()

        except KeyboardInterrupt:
            print("\n\nplayback stopped")

        print(f"\ntotal steps: {step_count}")
        print(f"total episodes: {total_episodes}")
        print(f"successful episodes: {successful_episodes}")
        if total_episodes > 0:
            success_rate = successful_episodes / total_episodes * 100
            print(f"success rate: {success_rate:.2f}%")
        if _td is not None:
            print(f"[TARGET_DRIVE] direct target drive finished. Compare this success rate with the baseline (without --target_drive),"
                  f"~baseline = target drive is lossless, so DP3 deploy can skip the target->raw inverse mapping.")
        else:
            print("success rate: N/A (no episodes)")

        if episode_rewards:
            ep_rewards_np = np.array(episode_rewards)
            print(f"Episode reward stats:")
            print(f"  mean: {ep_rewards_np.mean():.2f}")
            print(f"  max: {ep_rewards_np.max():.2f}")
            print(f"  min: {ep_rewards_np.min():.2f}")
            print(f"  std: {ep_rewards_np.std():.2f}")
        else:
            print("Episode reward: N/A (no episodes)")

        if success_log:
            print(f"success sequence: {success_log}")

        env.close()

    except Exception:
        # simulation_app.close() does a kit fast-exit (_exit); if not printed first, exceptions get swallowed
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()