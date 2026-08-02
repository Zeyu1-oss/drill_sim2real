
import argparse
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def _ckpt_obs_dim(path):
    """Read observation dim from the checkpoint input-layer weight (same as train2.py)."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    for k, v in ck.get("model", {}).items():
        if k.endswith("actor_mlp.0.weight") and torch.is_tensor(v) and v.dim() == 2:
            return int(v.shape[1])
    raise RuntimeError(f"cannot read observation dim from {path} (no actor_mlp.0.weight)")


def _build_player(agent_cfg_template, ckpt, obs_dim, act_dim, num_envs, device):
    """Build an rl_games player that does not create its own env: give obs/action space
    explicitly via env_info, so two policies with different input dims share one env."""
    import copy
    import gym
    import numpy as np
    from rl_games.torch_runner import Runner

    cfg = copy.deepcopy(agent_cfg_template)
    cfg["params"]["config"]["num_actors"] = num_envs
    cfg["params"]["config"]["device"] = device
    cfg["params"]["config"]["device_name"] = device
    cfg["params"]["load_checkpoint"] = True
    cfg["params"]["load_path"] = ckpt
    cfg["params"]["config"]["env_info"] = {
        "observation_space": gym.spaces.Box(-np.inf, np.inf, (obs_dim,)),
        "action_space": gym.spaces.Box(-1.0, 1.0, (act_dim,)),
        "agents": 1,
    }

    runner = Runner()
    runner.load(cfg)
    player = runner.create_player()
    player.restore(ckpt)
    player.reset()
    return player


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--stage1_checkpoint", type=str,
                        default="runs/inspire_hand_grasp_drill_26-06-13-21-10/inspire_hand_grasp_drill_26-06-13-21-10/nn/inspire_hand_grasp_drill_26-06-13-21-10.pth")
    parser.add_argument("--stage2_checkpoint", type=str,
                        default="runs/stage2_26-07-11-18-36/stage2_26-07-11-18-36/nn/stage2_26-07-11-18-36.pth")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--real_time", action="store_true")
    parser.add_argument("--episode_length_s", type=float, default=20.0)
    parser.add_argument("--success_hold_stop", type=int, default=20)
    parser.add_argument("--target_pos", type=float, nargs=3, default=[0.0, 0.0, 0.85])
    parser.add_argument("--target_quat", type=float, nargs=4, default=[1.0, 0.0, 0.0, 0.0])
    parser.add_argument("--drill_variants", type=str, default=None)
    parser.add_argument("--zero_action_idx", type=int, nargs="+", default=None)

    args = parser.parse_args()

    for p in (args.stage1_checkpoint, args.stage2_checkpoint):
        if not os.path.exists(p):
            print(f"[ERROR] checkpoint not found: {p}")
            return

    from isaaclab.app import AppLauncher
    print(f"[INFO] launching Isaac Sim (headless={args.headless})...")
    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app

    if args.real_time:
        import carb
        settings = carb.settings.get_settings()
        settings.set_bool("/app/runLoops/main/enabled", True)
        settings.set_bool("/app/runLoops/main/syncTooFast", True)

    try:
        import math

        import numpy as np
        import torch
        import yaml
        from rl_games.common import env_configurations, vecenv
        from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

        from tasks.stage2_env import create_stage2_env_cfg
        from tasks.chained_env import get_chained_env_class

        rl_config_path = os.path.join(project_root, "config/agents/rl_games_ppo_cfg.yaml")
        with open(rl_config_path, "r") as f:
            agent_cfg = yaml.safe_load(f)

        # -- read both policies' observation dims from the checkpoints --
        d1 = _ckpt_obs_dim(args.stage1_checkpoint)   # expected 71
        d2 = _ckpt_obs_dim(args.stage2_checkpoint)   # expected 71 + 7 = 78
        print(f"[INFO] stage1 obs dim = {d1}, stage2 obs dim = {d2}")
        if d2 != d1 + 7:
            print(f"[WARN] stage2 dim ({d2}) != stage1 ({d1})+7, check the two checkpoints match")

        # -- env: stage2 scene (with plate), ChainedEnv handles stage1-style reset and switching --
        ChainedEnv = get_chained_env_class()
        cfg = create_stage2_env_cfg(
            num_envs=args.num_envs,
            device=args.device,
            headless=args.headless,
            debug=args.debug,
            target_pos=args.target_pos,
            target_quat=args.target_quat,
            drill_variants_path=args.drill_variants,
        )
        cfg.seed = args.seed
        cfg.episode_length_s = args.episode_length_s
        cfg.log_dir = os.path.dirname(os.path.dirname(args.stage2_checkpoint))

        print(f"\ncreating Chained env (num_envs={args.num_envs}, episode={args.episode_length_s}s)...")
        env = ChainedEnv(
            cfg=cfg,
            debug=args.debug,
            success_hold_stop=args.success_hold_stop,
            target_pos=tuple(args.target_pos),
            target_quat=tuple(args.target_quat),
        )

        from isaaclab.sim.utils import delete_prim
        try:
            delete_prim("/World/Template")
            print("[INFO] deleted /World/Template")
        except Exception as e:
            print(f"[WARN] failed to delete /World/Template: {e}")

        clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
        clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
        env = RlGamesVecEnvWrapper(env, args.device, clip_obs, clip_actions)

        # env still registered once (other tools may use it); player does not build env through it
        vecenv.register(
            "IsaacRlgWrapper",
            lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
        )
        env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

        act_dim = env.unwrapped.cfg.action_space
        agent1 = _build_player(agent_cfg, args.stage1_checkpoint, d1, act_dim,
                               args.num_envs, args.device)
        agent2 = _build_player(agent_cfg, args.stage2_checkpoint, d2, act_dim,
                               args.num_envs, args.device)
        print("[INFO] both policies loaded")

        dt = env.unwrapped.step_dt
        obs = env.reset()
        if isinstance(obs, dict):
            obs = obs["obs"]
        _ = agent1.get_batch_size(obs[:, :d1], 1)
        _ = agent2.get_batch_size(obs, 1)

        import omni.timeline
        from perception.camera_setup import apply_render_settings
        apply_render_settings(dt)
        timeline = omni.timeline.get_timeline_interface()

        sys.stdout.flush()
        timeline.play()
        sys.stdout.flush()

        env_unwrapped = env.unwrapped

        # ---- force-zero action dims (e.g. --zero_action_idx 7 freezes the index finger) ----
        _zero_idx = args.zero_action_idx
        if _zero_idx is not None:
            _names = list(env_unwrapped.controlled_joint_names)
            for _i in _zero_idx:
                if not (0 <= _i < int(act_dim)):
                    raise ValueError(f"--zero_action_idx {_i} out of range (action has {act_dim} dims)")
            print(f"[ZERO-ACTION] force-zero action dims {_zero_idx}: "
                  f"{[_names[_i] for _i in _zero_idx]} "
                  f"(fingers are incremental control, zero = hold initial angle)")

        total_episodes = 0
        reached_stage2 = 0      # episodes that grasped and switched to stage2
        chained_success = 0     # episodes fully successful (grasp + align)
        success_log = []

        step_count = 0
        print_interval = 100
        try:
            print("\nstarting main loop...")
            sys.stdout.flush()

            while simulation_app.is_running():
                with torch.inference_mode():
                    phase = env_unwrapped.phase.clone()          # phase before step
                    a1 = agent1.get_action(agent1.obs_to_torch(obs[:, :d1]), is_deterministic=True)
                    a2 = agent2.get_action(agent2.obs_to_torch(obs), is_deterministic=True)
                    actions = torch.where((phase == 1).unsqueeze(1), a2, a1)
                    if _zero_idx is not None:      # force-zero given dims (both stages)
                        actions[:, _zero_idx] = 0.0

                    obs, rewards, dones, _ = env.step(actions)
                    step_count += 1

                    switched_now = (phase == 0) & (env_unwrapped.phase == 1)
                    for env_idx in switched_now.nonzero(as_tuple=True)[0]:
                        s = env_unwrapped._switch_step_buf[env_idx].item()

                    if len(dones) > 0:
                        done_indices = torch.where(dones)[0]
                        for env_idx in done_indices:
                            total_episodes += 1
                            # switch step before reset is stored by ChainedEnv in _last_episode_switch_step
                            sw = env_unwrapped._last_episode_switch_step[env_idx].item()
                            if sw >= 0:
                                reached_stage2 += 1
                            try:
                                ok = env_unwrapped._cached_lenient_success[env_idx].item()
                            except Exception:
                                ok = False
                            if ok:
                                chained_success += 1
                                success_log.append(1)
                            else:
                                success_log.append(0)

                    if step_count % print_interval == 0:
                        n_s2 = (env_unwrapped.phase == 1).sum().item()
                        r1 = (reached_stage2 / total_episodes * 100) if total_episodes else 0.0
                        r2 = (chained_success / total_episodes * 100) if total_episodes else 0.0
                        print(f"  Step: {step_count} | currently in stage2: {n_s2}/{args.num_envs} | "
                              f"episodes: {total_episodes} | grasp rate: {r1:.1f}% | chained success: {r2:.1f}%")
                        sys.stdout.flush()

                if isinstance(obs, dict):
                    obs = obs["obs"]

                if len(dones) > 0:
                    agent1.reset()
                    agent2.reset()

            sys.stdout.flush()

        except KeyboardInterrupt:
            print("\n\nplayback stopped")

        print(f"\ntotal steps: {step_count}")
        print(f"total episodes: {total_episodes}")
        if total_episodes > 0:
            print(f"grasped and switched to stage2: {reached_stage2} ({reached_stage2/total_episodes*100:.2f}%)")
            print(f"chained full success: {chained_success} ({chained_success/total_episodes*100:.2f}%)")
            if reached_stage2 > 0:
                print(f"stage2 conditional success (after switch): {chained_success/reached_stage2*100:.2f}%")
        if success_log:
            print(f"success sequence: {success_log}")

        env.close()

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
