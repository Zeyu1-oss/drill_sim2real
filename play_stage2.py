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

# 添加项目根目录
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def main():
    parser = argparse.ArgumentParser(description="播放 Stage2 训练好的策略")
    parser.add_argument("--num_envs", type=int, default=4, help="并行环境数量")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--checkpoint", type=str, default=None, help="检查点路径")
    parser.add_argument("--headless", action="store_true", help="无头模式")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--debug", action="store_true", help="调试模式：启用坐标系可视化")
    parser.add_argument("--frame-scale", type=float, default=0.5, help="调试坐标系大小 (默认 0.5)")
    parser.add_argument("--real_time", action="store_true", help="实时模式：仿真速度与物理世界同步")
    parser.add_argument("--target_pos", type=float, nargs=3, default=[0.0, 0.0, 0.85],
                        help="目标位置 (默认 [0.0, 0.0, 0.85])")
    parser.add_argument("--target_quat", type=float, nargs=4, default=[1.0, 0.0, 0.0, 0.0],
                        help="目标旋转四元数 (默认 [1.0, 0.0, 0.0, 0.0])")
    parser.add_argument("--drill_variants", type=str, default=None,
                        help="电钻变体配置文件路径（YAML）")
    parser.add_argument("--success_dataset", type=str, default="collected_data/success_data.pkl",
                        help="成功数据数据集路径（pkl）")

    args = parser.parse_args()

    # 初始化 Isaac Lab
    from isaaclab.app import AppLauncher

    headless_mode = args.headless
    print(f"[INFO] 启动 Isaac Sim (headless={headless_mode})...")
    app_launcher = AppLauncher(headless=headless_mode)
    simulation_app = app_launcher.app

    # 启用实时模式（如果指定）
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

        # 加载 RL Games 配置
        import yaml
        rl_config_path = os.path.join(project_root, "config/agents/rl_games_ppo_cfg.yaml")
        with open(rl_config_path, "r") as f:
            agent_cfg = yaml.safe_load(f)

        # 更新配置
        agent_cfg["params"]["config"]["num_actors"] = args.num_envs
        agent_cfg["params"]["config"]["device"] = args.device
        agent_cfg["params"]["config"]["device_name"] = args.device

        # 设置检查点
        if args.checkpoint:
            resume_path = args.checkpoint
        else:
            # 默认查找最新的 stage2 检查点
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
                                        print(f"[INFO] 自动找到检查点: {resume_path}")
                                        break
                        else:
                            continue
                        break
            if not resume_path:
                print("[ERROR] 未指定检查点且未找到默认检查点")
                return

        if not os.path.exists(resume_path):
            print(f"[ERROR] 检查点不存在: {resume_path}")
            return

        # 加载数据集（在 AppLauncher 之后）
        success_dataset = None
        if args.success_dataset:
            from tasks.stage2_env import SuccessDataDataset
            success_dataset = SuccessDataDataset(args.success_dataset)
            print(f"[INFO] 加载数据集: {args.success_dataset} ({len(success_dataset)} 样本)")

        # 获取 Stage2Env 类（在 AppLauncher 之后才可导入）
        from tasks.stage2_env import get_stage2_env_class
        Stage2Env = get_stage2_env_class()

        # 创建环境配置
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

        # 设置 log_dir
        log_dir = os.path.dirname(os.path.dirname(resume_path))
        cfg.log_dir = log_dir

        # 创建环境
        print(f"\n创建 Stage2 环境 (num_envs={args.num_envs})...")
        env = Stage2Env(
            cfg=cfg,
            debug=args.debug,
            success_dataset=success_dataset,
            target_pos=tuple(args.target_pos),
            target_quat=tuple(args.target_quat),
        )

        # 删除 MultiAssetSpawner 留下的模板原型
        from isaaclab.sim.utils import delete_prim
        try:
            delete_prim("/World/Template")
            print("[INFO] 已删除 /World/Template")
        except Exception as e:
            print(f"[WARN] 删除 /World/Template 失败: {e}")

        # 包装环境
        clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
        clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
        rl_device = args.device

        env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

        # 注册环境
        vecenv.register(
            "IsaacRlgWrapper",
            lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
        )
        env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

        # 加载模型
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

        runner = Runner()
        runner.load(agent_cfg)
        agent: BasePlayer = runner.create_player()
        agent.restore(resume_path)
        agent.reset()

        # 初始化
        dt = env.unwrapped.step_dt

        # 重置环境，获取初始观察
        obs = env.reset()
        if isinstance(obs, dict):
            obs = obs["obs"]
        _ = agent.get_batch_size(obs, 1)

        # 使用 omni.timeline 来控制仿真
        import omni.timeline
        import carb
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

        # 启动仿真
        sys.stdout.flush()
        timeline.play()
        sys.stdout.flush()

        # 统计变量
        total_episodes = 0
        successful_episodes = 0
        episode_rewards = []
        current_episode_reward = torch.zeros(args.num_envs, device=args.device)
        success_log = []

        # 主循环
        step_count = 0
        print_interval = 100
        try:
            print("\n开始执行主循环...")
            sys.stdout.flush()

            while simulation_app.is_running():
                with torch.inference_mode():
                    obs = agent.obs_to_torch(obs)
                    actions = agent.get_action(obs, is_deterministic=True)
                    obs, rewards, dones, _ = env.step(actions)

                    step_count += 1
                    current_episode_reward += rewards

                    # 检测完成的 env
                    if len(dones) > 0:
                        done_indices = torch.where(dones)[0]
                        for env_idx in done_indices:
                            total_episodes += 1
                            ep_rew = current_episode_reward[env_idx].item()
                            episode_rewards.append(ep_rew)

                            # 获取 success 状态
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

                    # 每 print_interval 步打印一次进度
                    if step_count % print_interval == 0:
                        running_success_rate = (successful_episodes / total_episodes * 100) if total_episodes > 0 else 0.0
                        print(f"  Step: {step_count} | dones: {dones.sum().item():.0f} | episodes: {total_episodes} | success: {successful_episodes} ({running_success_rate:.1f}%)")

                        # 打印 policy.mu 的统计信息
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
            print("\n\n播放停止")

        # 打印统计结果
        print(f"\n总执行步数: {step_count}")
        print(f"总 Episode 数: {total_episodes}")
        print(f"成功 Episode 数: {successful_episodes}")
        if total_episodes > 0:
            success_rate = successful_episodes / total_episodes * 100
            print(f"成功率: {success_rate:.2f}%")
        else:
            print("成功率: N/A (无 Episode)")

        if episode_rewards:
            ep_rewards_np = np.array(episode_rewards)
            print(f"Episode 奖励统计:")
            print(f"  平均: {ep_rewards_np.mean():.2f}")
            print(f"  最大: {ep_rewards_np.max():.2f}")
            print(f"  最小: {ep_rewards_np.min():.2f}")
            print(f"  标准差: {ep_rewards_np.std():.2f}")

        if success_log:
            print(f"成功序列: {success_log}")

        env.close()

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
