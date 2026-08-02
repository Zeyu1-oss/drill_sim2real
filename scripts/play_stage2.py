#!/usr/bin/env python3
"""
Play script for Stage2 environment — visualize trained checkpoint.
Usage:
    python play_stage2.py --checkpoint runs/stage2_26-04-28-21-36/nn/last_xxx.pth
    python play_stage2.py --num_envs 1 --headless  # headless mode
"""

import argparse
import os
import sys
from pathlib import Path

# add project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (scripts/ -> ../..)
sys.path.insert(0, project_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_envs', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--frame-scale', type=float, default=0.5)
    parser.add_argument('--real_time', action='store_true')
    parser.add_argument('--target_pos', type=float, nargs=3, default=[0.0, 0.0, 0.85])
    parser.add_argument('--target_quat', type=float, nargs=4, default=[1.0, 0.0, 0.0, 0.0])
    parser.add_argument('--drill_variants', type=str, default=None)
    parser.add_argument('--success_dataset', type=str, default='collected_data/success_data_dp3.pkl')

    args = parser.parse_args()

    # init Isaac Lab
    from isaaclab.app import AppLauncher

    headless_mode = args.headless
    print(f"[INFO] launching Isaac Sim (headless={headless_mode})...")
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
        import time

        import numpy as np
        import gymnasium as gym
        import torch
        from rl_games.common import env_configurations, vecenv
        from rl_games.common.player import BasePlayer
        from rl_games.torch_runner import Runner

        from isaaclab.envs import ManagerBasedRLEnvCfg
        from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

        from tasks.stage2_env import get_stage2_env_class, create_stage2_env_cfg

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
            # by default find the latest stage2 checkpoint
            runs_dir = Path(project_root) / "runs"
            if runs_dir.exists():
                for run_dir in sorted(runs_dir.iterdir(), reverse=True):
                    if run_dir.name.startswith("stage2_"):
                        for subdir in run_dir.iterdir():
                            if subdir.is_dir():
                                nn_dir = subdir / "nn"
                                if nn_dir.exists():
                                    pth_files = list(nn_dir.glob("*.pth"))
                                    if pth_files:
                                        resume_path = str(max(pth_files, key=lambda x: x.stat().st_mtime))
                                        print(f"[INFO] auto-found checkpoint: {resume_path}")
                                        break
                        else:
                            continue
                        break
            if not resume_path:
                print("[ERROR] no checkpoint specified and no default checkpoint found")
                return

        if not os.path.exists(resume_path):
            print(f"[ERROR] checkpoint not found: {resume_path}")
            return

        # load dataset (after AppLauncher)
        success_dataset = None
        if args.success_dataset:
            from tasks.stage2_env import SuccessDataDataset
            success_dataset = SuccessDataDataset(args.success_dataset)
            print(f"[INFO] loading dataset: {args.success_dataset} ({len(success_dataset)} samples)")

        # get the Stage2Env class (importable only after AppLauncher)
        from tasks.stage2_env import get_stage2_env_class
        Stage2Env = get_stage2_env_class()

        # create env config
        cfg = create_stage2_env_cfg(
            num_envs=args.num_envs,
            device=args.device,
            headless=args.headless,
            debug=args.debug,
            target_pos=args.target_pos,
            target_quat=args.target_quat,
            drill_variants_path=args.drill_variants,
        )
        cfg.debug_frame_scale = args.frame_scale
        cfg.seed = args.seed

        # set log_dir
        log_dir = os.path.dirname(os.path.dirname(resume_path))
        cfg.log_dir = log_dir

        # create env
        print(f"\ncreating Stage2 env (num_envs={args.num_envs})...")
        env = Stage2Env(
            cfg=cfg,
            debug=args.debug,
            success_dataset=success_dataset,
            target_pos=tuple(args.target_pos),
            target_quat=tuple(args.target_quat),
        )

        # delete the template prototype left by MultiAssetSpawner
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
        apply_render_settings(dt)
        timeline = omni.timeline.get_timeline_interface()

        # start sim
        sys.stdout.flush()
        timeline.play()
        sys.stdout.flush()

        # stats variables
        total_episodes = 0
        successful_episodes = 0
        episode_rewards = []
        current_episode_reward = torch.zeros(args.num_envs, device=args.device)
        success_log = []

        # main loop
        step_count = 0
        print_interval = 100
        try:
            print("\nstarting main loop...")
            sys.stdout.flush()

            while simulation_app.is_running():
                with torch.inference_mode():
                    obs = agent.obs_to_torch(obs)
                    actions = agent.get_action(obs, is_deterministic=True)
                    obs, rewards, dones, _ = env.step(actions)

                    step_count += 1
                    current_episode_reward += rewards

                    # detect finished envs
                    if len(dones) > 0:
                        done_indices = torch.where(dones)[0]
                        for env_idx in done_indices:
                            total_episodes += 1
                            ep_rew = current_episode_reward[env_idx].item()
                            episode_rewards.append(ep_rew)

                            # get success status
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

                            current_episode_reward[env_idx] = 0.0

                    # print progress every print_interval steps
                    if step_count % print_interval == 0:
                        running_success_rate = (successful_episodes / total_episodes * 100) if total_episodes > 0 else 0.0
                        print(f"  Step: {step_count} | dones: {dones.sum().item():.0f} | episodes: {total_episodes} | success: {successful_episodes} ({running_success_rate:.1f}%)")

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

            sys.stdout.flush()

        except KeyboardInterrupt:
            print("\n\nplayback stopped")

        # print result stats
        print(f"\ntotal steps: {step_count}")
        print(f"total episodes: {total_episodes}")
        print(f"successful episodes: {successful_episodes}")
        if total_episodes > 0:
            success_rate = successful_episodes / total_episodes * 100
            print(f"success rate: {success_rate:.2f}%")
        else:
            print("success rate: N/A (no episodes)")

        if episode_rewards:
            ep_rewards_np = np.array(episode_rewards)
            print(f"Episode reward stats:")
            print(f"  mean: {ep_rewards_np.mean():.2f}")
            print(f"  max: {ep_rewards_np.max():.2f}")
            print(f"  min: {ep_rewards_np.min():.2f}")
            print(f"  std: {ep_rewards_np.std():.2f}")

        if success_log:
            print(f"success sequence: {success_log}")

        env.close()

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
