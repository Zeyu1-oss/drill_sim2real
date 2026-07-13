#!/usr/bin/env python3
"""
Camera Visualization Script - 输出几张 RGB 图像用于调整相机位置。

Usage:
    python debug_camera_viz.py --checkpoint runs/xxx/nn/xxx.pth --save_images
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (scripts/ -> ../..)
sys.path.insert(0, project_root)


def parse_args():
    parser = argparse.ArgumentParser(description="相机可视化 - 持续播放 checkpoint 并显示相机画面")
    parser.add_argument("--checkpoint", type=str, required=True, help="RL checkpoint path")
    parser.add_argument("--num_envs", type=int, default=1, help="并行环境数量")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--headless", action="store_true", help="无头模式")
    parser.add_argument("--save_images", action="store_true", help="周期性保存 RGB 图像到 output/ 目录")
    parser.add_argument("--img_height", type=int, default=384, help="图像高度")
    parser.add_argument("--img_width", type=int, default=384, help="图像宽度")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--debug", action="store_true", help="启用环境调试模式")
    parser.add_argument("--drill_configs", type=str, default=None, help="钻头配置 YAML 路径")
    parser.add_argument("--log_interval", type=int, default=30, help="每隔多少步打印一次相机统计")
    parser.add_argument("--save_interval", type=int, default=10, help="每隔多少步保存一次图像")
    parser.add_argument("--max_steps", type=int, default=1, help="最大运行步数，0 表示一直运行到手动关闭")
    return parser.parse_args()


def describe_depth(depth_tensor):
    depth_np = depth_tensor.detach().cpu().numpy()
    positive = depth_np[depth_np > 0]
    if positive.size == 0:
        return f"shape={tuple(depth_np.shape)}, all_zeros_or_invalid"
    return (
        f"shape={tuple(depth_np.shape)}, "
        f"min={positive.min():.3f}m, max={positive.max():.3f}m, mean={positive.mean():.3f}m"
    )


def print_camera_pose(cam, name):
    pos = cam.data.pos_w[0].detach().cpu().numpy()
    quat = cam.data.quat_w[0].detach().cpu().numpy()
    print(f"{name} pose:")
    print(f"  position = ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}) m")
    print(f"  quaternion = ({quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f})")
    if hasattr(cam.data, "intrinsic_matrices"):
        k = cam.data.intrinsic_matrices[0].detach().cpu().numpy()
        print(f"  intrinsics: fx={k[0,0]:.2f}, fy={k[1,1]:.2f}, cx={k[0,2]:.2f}, cy={k[1,2]:.2f}")


def save_rgb_image(path, rgb):
    import cv2

    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def main():
    args = parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        return

    output_dir = os.path.join(project_root, "output", "camera_debug")
    if args.save_images:
        os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  Camera Debug Visualization")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Resolution: {args.img_width}x{args.img_height}")
    print(f"  Save images: {args.save_images}")
    print(f"  Log interval: {args.log_interval} steps")
    print(f"  Save interval: {args.save_interval} steps")
    print(f"  Max steps: {'infinite' if args.max_steps == 0 else args.max_steps}")
    print("=" * 60)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=args.headless, enable_cameras=True)
    simulation_app = app_launcher.app

    env = None
    try:
        import carb
        import omni.timeline
        import omni.usd
        import yaml
        from rl_games.common import env_configurations, vecenv
        from rl_games.torch_runner import Runner

        from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
        from tasks.grasp_drill_env import GraspDrillEnv, create_grasp_drill_env_cfg

        omni.usd.get_context().new_stage()

        rl_config_path = os.path.join(project_root, "config/agents/rl_games_ppo_cfg.yaml")
        with open(rl_config_path, "r") as f:
            agent_cfg = yaml.safe_load(f)

        agent_cfg["params"]["config"]["num_actors"] = args.num_envs
        agent_cfg["params"]["config"]["device"] = args.device
        agent_cfg["params"]["config"]["device_name"] = args.device

        cfg = create_grasp_drill_env_cfg(
            num_envs=args.num_envs,
            device=args.device,
            headless=args.headless,
            debug=args.debug,
            drill_config_path=args.drill_configs,
            img_height=args.img_height,
            img_width=args.img_width,
            enable_cameras=True,
        )
        # Add rgb data type for visualization (doesn't affect trained policy)
        for _cn in ("cam1", "cam2"):   # cam1=30cm跟随 cam2=固定工作区,两个都要 RGB
            _c = getattr(cfg.scene, _cn, None)
            if _c is not None and "rgb" not in _c.data_types:
                _c.data_types = ["rgb"] + list(_c.data_types)
        cfg.seed = args.seed
        cfg.log_dir = os.path.dirname(os.path.dirname(args.checkpoint))

        print("\n[INFO] Creating environment with cameras...")
        base_env = GraspDrillEnv(cfg=cfg, debug=args.debug)
        env_unwrapped = base_env.unwrapped

        # Follow play_drill.py order: reset BEFORE timeline.play()
        print("[INFO] Resetting environment...")
        obs = base_env.reset()
        if isinstance(obs, dict):
            obs = obs.get("obs", obs)

        clip_obs = agent_cfg["params"]["env"].get("clip_observations", float("inf"))
        clip_actions = agent_cfg["params"]["env"].get("clip_actions", float("inf"))
        env = RlGamesVecEnvWrapper(base_env, args.device, clip_obs, clip_actions)

        vecenv.register(
            "IsaacRlgWrapper",
            lambda config_name, num_actors, **kw: RlGamesGpuEnv(config_name, num_actors, **kw),
        )
        env_configurations.register(
            "rlgpu",
            {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env},
        )

        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = args.checkpoint
        agent_cfg["params"]["config"]["num_actors"] = env_unwrapped.num_envs

        runner = Runner()
        runner.load(agent_cfg)
        agent = runner.create_player()
        agent.restore(args.checkpoint)
        agent.reset()

        scene_keys = list(env_unwrapped.scene.keys())
        print(f"[INFO] Available scene keys: {scene_keys}")
        for _cn in ("cam1", "cam2"):
            if _cn not in scene_keys:
                print(f"[ERROR] Camera '{_cn}' not found in scene. Available: {scene_keys}")
                return

        cam1 = env_unwrapped.scene["cam1"]   # 30cm 跟随电钻
        cam2 = env_unwrapped.scene["cam2"]   # 固定工作区
        print("[INFO] Cameras cam1 + cam2 found successfully.")

        dt = env_unwrapped.step_dt
        settings = carb.settings.get_settings()
        settings.set("/app/player/useFixedTimeStepping", True)
        settings.set("/app/player/targetFrameRate", int(1.0 / dt))
        settings.set("/app/runLoops/rendering_0/rateLimitEnabled", True)
        settings.set("/app/runLoops/rendering_0/rateLimit", 60)
        settings.set("/physics/updateToUsd", False)
        settings.set("/physics/updateParticlesToUsd", False)
        settings.set("/app/runLoops/main/enabled", True)
        settings.set("/app/runLoops/main/syncTooFast", True)

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        sys.stdout.flush()

        print(f"[DEBUG] simulation_app.is_running() before loop = {simulation_app.is_running()}")

        if not simulation_app.is_running():
            print("[ERROR] simulation_app.is_running() is False before entering loop!")
            return

        obs = env.reset()
        if isinstance(obs, dict):
            obs = obs["obs"]
        _ = agent.get_batch_size(obs, 1)
        _ = agent.obs_to_torch(obs)

        print("\n[INFO] Entering play loop - camera frames will be captured...")

        step_count = 0
        last_log_time = time.time()

        while simulation_app.is_running() and step_count < args.max_steps:
            obs_tensor = agent.obs_to_torch(obs)
            actions = agent.get_action(obs_tensor, is_deterministic=True)
            if not isinstance(actions, torch.Tensor):
                actions = torch.from_numpy(np.array(actions)).float().to(args.device)

            obs, rewards, dones, _ = env.step(actions)
            if isinstance(obs, dict):
                obs = obs["obs"]

            cam1.update(dt)
            cam2.update(dt)

            rgb1 = cam1.data.output["rgb"][0].detach().cpu().numpy().astype(np.uint8)
            depth1 = cam1.data.output["distance_to_image_plane"][0]
            rgb2 = cam2.data.output["rgb"][0].detach().cpu().numpy().astype(np.uint8)
            depth2 = cam2.data.output["distance_to_image_plane"][0]

            if step_count % args.log_interval == 0:
                now = time.time()
                elapsed = max(now - last_log_time, 1e-6)
                steps_per_sec = args.log_interval / elapsed if step_count > 0 else 0.0
                print(f"\n[Step {step_count}] approx_fps={steps_per_sec:.1f}")
                print(
                    f"  Cam1 RGB: shape={rgb1.shape}, range=[{rgb1.min()}, {rgb1.max()}], mean={rgb1.mean():.1f}"
                )
                print(f"  Cam1 Depth: {describe_depth(depth1)}")
                print(
                    f"  Cam2 RGB: shape={rgb2.shape}, range=[{rgb2.min()}, {rgb2.max()}], mean={rgb2.mean():.1f}"
                )
                print(f"  Cam2 Depth: {describe_depth(depth2)}")
                last_log_time = now

            if args.save_images and step_count % args.save_interval == 0:
                save_rgb_image(os.path.join(output_dir, f"cam1_step_{step_count:06d}.png"), rgb1)
                save_rgb_image(os.path.join(output_dir, f"cam2_step_{step_count:06d}.png"), rgb2)
                print(f"  [INFO] Saved cam1 + cam2 images for step {step_count} to {output_dir}")

            obs, rewards, dones, _ = env.step(actions)
            step_count += 1

            if isinstance(obs, dict):
                obs = obs["obs"]

            if len(dones) > 0 and dones.any():
                agent.reset()

        print(f"\n[INFO] Finished after {step_count} steps.")
        if args.save_images:
            print(f"[INFO] Images saved in: {output_dir}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] Exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if env is not None:
            env.close()
        simulation_app.close()

    print("[INFO] Done!")


if __name__ == "__main__":
    main()
