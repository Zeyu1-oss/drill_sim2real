

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
    parser = argparse.ArgumentParser(description="收集成功 episode 的完整观测数据")
    parser.add_argument("--num_envs", type=int, default=2048, help="并行环境数量")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--checkpoint", type=str, required=True, help="检查点路径")
    parser.add_argument("--num_samples", type=int, default=1000, help="收集的样本数量")
    parser.add_argument("--output", type=str, default="collected_data/success_data.pkl",
                       help="输出文件路径")
    parser.add_argument("--headless", action="store_true", help="无头模式")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--debug", action="store_true", help="调试模式：打印更多信息")
    parser.add_argument("--drill_configs", type=str, default=None,
                       help="多电钻配置文件路径（YAML）")
    parser.add_argument("--frame-scale", type=float, default=0.5, help="调试坐标系大小")
    parser.add_argument("--history-len", type=int, default=0,
                       help="相对成功时刻回溯的步数;0=成功当步(step 前)的状态。"
                            "缓存在成功持续期间每步刷新,episode 结束时保存的是"
                            "最后一次仍处于成功状态的抓取姿态")
    parser.add_argument("--playback", type=str, default=None,
                       help="PKL文件路径：从PKL读取joint+drill数据来驱动每回合初始状态（joint和drill共用同一条数据）")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] 检查点文件不存在: {args.checkpoint}")
        return

    # ============================================================
    # 初始化 Isaac Lab
    # ============================================================
    from isaaclab.app import AppLauncher
    headless_mode = args.headless
    print(f"[INFO] 启动 Isaac Sim (headless={headless_mode})...")
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
        # 加载 RL Games 配置
        # ============================================================
        rl_config_path = os.path.join(project_root, "config/agents/rl_games_ppo_cfg.yaml")
        with open(rl_config_path, "r") as f:
            agent_cfg = yaml.safe_load(f)

        agent_cfg["params"]["config"]["num_actors"] = args.num_envs
        agent_cfg["params"]["config"]["device"] = args.device
        agent_cfg["params"]["config"]["device_name"] = args.device

        # ============================================================
        # 创建环境配置（与 play_drill.py 完全一致）
        # ============================================================
        cfg = create_grasp_drill_env_cfg(
            num_envs=args.num_envs,
            device=args.device,
            headless=args.headless,
            debug=args.debug,
            drill_config_path=args.drill_configs,
            enable_cameras=False,   # 纯状态收集,不需要相机(默认 True 会崩/耗显存)
        )
        cfg.debug_frame_scale = args.frame_scale
        cfg.seed = args.seed

        log_dir = os.path.dirname(os.path.dirname(args.checkpoint))
        cfg.log_dir = log_dir

        print(f"[INFO] 创建环境 (num_envs={args.num_envs})...")
        env = GraspDrillEnv(cfg=cfg, debug=args.debug)

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
        env_configurations.register(
            "rlgpu",
            {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env}
        )

        # ============================================================
        # 加载模型
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
        # 仿真时间步配置
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
        # 启动仿真
        # ============================================================
        sys.stdout.flush()
        timeline.play()
        sys.stdout.flush()

        # ============================================================
        # 重置环境，获取初始观察
        # ============================================================
        obs = env.reset()
        if isinstance(obs, dict):
            obs = obs["obs"]
        _ = agent.get_batch_size(obs, 1)

        # ============================================================
        # Playback 模式：加载 PKL 数据用于驱动初始状态
        # ============================================================
        pkl_samples = None
        if args.playback:
            if not os.path.exists(args.playback):
                print(f"[ERROR] Playback PKL 文件不存在: {args.playback}")
                return
            with open(args.playback, "rb") as f:
                pkl_data = pickle.load(f)
            pkl_samples = pkl_data["samples"]
            print(f"[INFO] Playback 模式：加载 {len(pkl_samples)} 条 PKL 样本")
            print(f"  PKL metadata: {pkl_data['metadata']}")
            _pkl_idx = 0  # 轮询指针

        # ============================================================
        # 收集状态追踪
        # ============================================================
        # episode_state: 0=未成功过, 1=已成功(等待episode结束), 2=本episode已收集
        episode_state = torch.zeros(args.num_envs, dtype=torch.long, device=args.device)

        _success_reward_cache = [0.0] * args.num_envs
        _success_step_cache = [0] * args.num_envs

        # 上一步的 obs 和关节状态（用于在成功时保存"成功前一 step"的数据）
        env_unwrapped = env.unwrapped

        # 循环缓冲区：保存最近 N 步的 Franka/drill 状态（step 前捕获）
        # Franka 和 drill 在 env.step() 内部会被 reset，所以必须在 step 前保存
        # 问题 2 fix: buffer 长度 = history_len + 1，确保能取到 step N-5
        HISTORY = args.history_len + 1
        _hist_obs = [None] * HISTORY
        _hist_joint_pos = [None] * HISTORY
        _hist_joint_vel = [None] * HISTORY
        _hist_drill_pos_w = [None] * HISTORY
        _hist_drill_quat_w = [None] * HISTORY
        _hist_drill_lin_vel = [None] * HISTORY
        _hist_drill_ang_vel = [None] * HISTORY
        _hist_env_origins = [None] * HISTORY
        _hist_head = 0  # 环形缓冲区当前头指针

        # 预分配缓冲区 tensor（避免循环中反复 clone）
        _obs_buf = obs.cpu().clone()
        _jp_buf = env_unwrapped.franka.data.joint_pos.cpu().clone()
        _jv_buf = env_unwrapped.franka.data.joint_vel.cpu().clone()
        _dp_buf = env_unwrapped.drill.data.root_pos_w.cpu().clone()
        _dq_buf = env_unwrapped.drill.data.root_quat_w.cpu().clone()
        _dlv_buf = env_unwrapped.drill.data.root_lin_vel_w.cpu().clone()
        _dav_buf = env_unwrapped.drill.data.root_ang_vel_w.cpu().clone()
        _eo_buf = env_unwrapped.scene.env_origins.cpu().clone()

        # 批量成功缓存(向量化,CPU):成功持续期间每步覆盖刷新,episode 结束时
        # 缓存内容即"最后一次仍处于成功状态"的完整状态——这才是要收集的最终抓取姿态。
        # (旧逻辑只在首次成功时缓存一次,存的是刚过成功阈值、甚至回退 N 步的中间姿态)
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
        print(f"\n[INFO] 开始数据收集... (目标: {args.num_samples} 条)")
        print(f"[INFO] 每 {print_interval} 步打印一次统计")
        start_time = time.time()
        sys.stdout.flush()

        with torch.inference_mode():
            while collected_count < args.num_samples:
                # --- 在 step 之前保存 Franka/drill 状态 ---
                # Franka 和 drill 会在 env.step() 内部被 reset，必须在 step 前捕获
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

                # 写入环形缓冲区
                _hist_obs[_hist_head] = _obs_buf.clone()
                _hist_joint_pos[_hist_head] = _jp_buf.clone()
                _hist_joint_vel[_hist_head] = _jv_buf.clone()
                _hist_drill_pos_w[_hist_head] = _dp_buf.clone()
                _hist_drill_quat_w[_hist_head] = _dq_buf.clone()
                _hist_drill_lin_vel[_hist_head] = _dlv_buf.clone()
                _hist_drill_ang_vel[_hist_head] = _dav_buf.clone()
                _hist_env_origins[_hist_head] = _eo_buf.clone()

                # --- 执行 step（成功时会触发 reset）---
                obs_tensor = agent.obs_to_torch(obs)
                actions = agent.get_action(obs_tensor, is_deterministic=True)
                obs, rewards, dones, _ = env.step(actions)
                step_count += 1

                env_unwrapped = env.unwrapped

                # 更新环形缓冲区头指针（每次 step 后移动）
                _hist_head = (_hist_head + 1) % HISTORY

                # --- 在 step 之后更新 obs（obs 返回时已经 reset 后了）---
                if isinstance(obs, dict):
                    obs_dict = obs["obs"]
                else:
                    obs_dict = obs
                _obs_buf.copy_(obs_dict)

                # ============================================================
                # 成功判定（与 play_drill.py 完全一致）
                # ============================================================
                try:
                    lenient_success = env_unwrapped._cached_lenient_success
                except AttributeError:
                    lenient_success = torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)

                current_success = lenient_success.bool()
                is_done = dones.bool() if hasattr(dones, 'bool') else dones > 0

                # ============================================================
                # 跟踪每个 env 的 episode 状态
                # ============================================================
                just_became_success = (episode_state == 0) & current_success

                # —— 缓存刷新(向量化):成功持续期间每一步都覆盖写,episode 结束时
                #    缓存内容 = 最后一次仍处于成功状态的姿态(history_len>0 可再回退)。
                #    tail 指向本步 step 前捕获的状态(_hist_head 已 +1)。
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

                # 首次成功:初始化 episode 统计
                for env_idx in torch.where(just_became_success)[0]:
                    idx = env_idx.item()
                    _success_reward_cache[idx] = 0.0
                    _success_step_cache[idx] = step_count
                    if args.debug and idx < 4:
                        print(f"  [DEBUG] env={idx} 首次成功 (step={step_count})")

                # 累加 episode 奖励（在成功窗口内持续累加）
                success_mask = episode_state == 1
                for env_idx in torch.where(success_mask)[0]:
                    idx = env_idx.item()
                    _success_reward_cache[idx] += rewards[idx].item()

                # 标记为已成功（等待 episode 结束）
                episode_state[just_became_success] = 1

                # ============================================================
                # 可以收集：已成功过(episode_state==1) 且 episode 结束
                # ============================================================
                can_collect = (episode_state == 1) & is_done
                if can_collect.any():
                    for env_idx in torch.where(can_collect)[0]:
                        idx = env_idx.item()

                        if _cache_valid[idx]:
                            # 必须 clone:单行索引是视图,_build_sample 的 .numpy() 共享内存,
                            # 不 clone 的话该 env 下一局刷新缓存会篡改已收集样本
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

                        # 重置该 env 的缓存
                        _cache_valid[idx] = False
                        _success_reward_cache[idx] = 0.0
                        _success_step_cache[idx] = 0
                        episode_state[env_idx] = 2

                # ============================================================
                # Episode 重置处理
                # ============================================================
                if is_done.any():
                    agent.reset()
                    reset_0 = (episode_state == 0)
                    reset_1 = (episode_state == 1)
                    reset_2 = (episode_state == 2)
                    # state==1: 成功窗口未结束就超时了，未能收集
                    # state==2: 已被收集，等待重置回参与状态
                    episode_state[reset_0 | reset_1 | reset_2] = 0
                    # 清理 state 0/1 的缓存（state 2 缓存已在收集时清空，无需再清）
                    _cache_valid[(reset_0 | reset_1).detach().cpu()] = False
                    for env_idx in torch.where(reset_0 | reset_1)[0]:
                        idx = env_idx.item()
                        _success_reward_cache[idx] = 0.0
                        _success_step_cache[idx] = 0

                # ============================================================
                # 进度打印（基于步数 + ETA）
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

                    # ---- 诊断：打印 success 各组件分布 ----
                    if args.debug and hasattr(env_unwrapped, '_check_success'):
                        with torch.inference_mode():
                            try:
                                # 临时强制计算一次（绕缓存）
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
        # 保存数据
        # ============================================================
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        save_data(collected_samples, args.output, args.history_len)

        elapsed = time.time() - start_time
        m, s = divmod(int(elapsed), 60)
        h, m = divmod(m, 60)
        print(f"\n{'='*60}")
        print(f"数据收集完成!")
        print(f"  目标样本数: {args.num_samples}")
        print(f"  实际收集:   {len(collected_samples)}")
        print(f"  总步数:     {step_count:,}")
        print(f"  总耗时:     {h}h{m}m{s}s")
        print(f"  平均速率:   {step_count/elapsed:.0f} 步/秒")
        print(f"  输出文件:   {args.output}")
        print(f"{'='*60}")

        _print_variant_stats(collected_samples)

        env.close()

    finally:
        simulation_app.close()


def _build_sample(env, env_idx: int, cached_obs: torch.Tensor,
                  episode_reward: float, success_hold_steps: int,
                  cached_state: dict) -> dict:
    """从缓存中构建一条完整的成功样本。

    cached_state 在首次成功时刻从 env 读取，之后不再依赖 env 状态，
    避免 _reset_idx 后读取到新 episode 状态。
    """
    try:
        # ---- 关节状态 (从缓存读取，提取 controlled 关节，与 _get_observations 一致) ----
        joint_pos_23 = cached_state["joint_pos"].cpu().numpy()
        joint_vel_23 = cached_state["joint_vel"].cpu().numpy()
        # controlled_joint_indices: fr3_joint1~7 (7) + 6 proximal hand joints = 13
        controlled = env.controlled_joint_indices.cpu().numpy()  # shape=(13,)
        joint_pos = joint_pos_23[controlled]
        joint_vel = joint_vel_23[controlled]

        # ---- Drill 状态 (世界坐标 -> 局部坐标) ----
        drill_pos_world = cached_state["drill_pos_world"].cpu().numpy()
        env_origin = cached_state["env_origin"].cpu().numpy()
        drill_pos = drill_pos_world - env_origin  # 转为局部坐标（相对于 env_origin）
        drill_quat = cached_state["drill_quat"].cpu().numpy()
        drill_lin_vel = cached_state["drill_lin_vel"].cpu().numpy()
        drill_ang_vel = cached_state["drill_ang_vel"].cpu().numpy()

        # ---- Variant 信息 ----
        variant_idx = env._drill_variant_indices[env_idx].item()
        variant_name = env._variant_attrs.get(variant_idx, {}).get(
            "name", f"variant_{variant_idx}"
        )

        # ---- Success 窗口统计 ----
        success_window_sum = env._success_window[env_idx].sum().item()

        # ---- Policy 观测 (原始 tensor → numpy) ----
        full_obs = cached_obs.cpu().numpy()

        return {
            # 完整 policy 观测向量（可直接用于 relabel / BC / SAC 等）
            "full_obs": full_obs,
            # 所有关节位置/速度 (13维, controlled joints only)
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            # Drill 世界坐标状态
            "drill_pos": drill_pos,
            "drill_quat": drill_quat,
            "drill_lin_vel": drill_lin_vel,
            "drill_ang_vel": drill_ang_vel,
            # Variant 信息
            "variant_idx": variant_idx,
            "variant_name": variant_name,
            # Episode 统计
            "episode_reward": episode_reward,
            "success_hold_steps": success_hold_steps,
            "success_window_sum": int(success_window_sum),
        }
    except Exception as e:
        print(f"[WARN] _build_sample failed for env_idx={env_idx}: {e}")
        return None


def _print_variant_stats(samples: list):
    """打印各 variant 的收集统计。"""
    if not samples:
        return
    from collections import Counter
    variant_counts = Counter(s["variant_name"] for s in samples)
    print("\n各 Variant 收集统计:")
    for name, count in sorted(variant_counts.items()):
        print(f"  {name}: {count} ({count / len(samples) * 100:.1f}%)")


def save_data(samples: list, output_path: str, history_len: int = 5):
    """保存收集的数据到 pickle 文件。"""
    # 过滤掉 None（收集失败的）
    valid_samples = [s for s in samples if s is not None]
    skipped = len(samples) - len(valid_samples)
    if skipped > 0:
        print(f"[WARN] 跳过 {skipped} 条无效样本")

    # 问题 3 fix: metadata 无论 skipped 是否 > 0 都要定义，避免 skipped==0 时 crash
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

    print(f"[INFO] 保存 {len(valid_samples)} 条样本到: {output_path}")


if __name__ == "__main__":
    main()
