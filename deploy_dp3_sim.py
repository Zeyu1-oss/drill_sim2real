#!/usr/bin/env python3
"""
DP3 checkpoint evaluation in IsaacSim simulation.

Loads the DP3 checkpoint (trained via scripts/train_policy_inspire_drill.sh) and
runs closed-loop rollout in the GraspDrillEnv, evaluating success rate.

Initialization order mirrors collect_dp3_data.py (which works):
  new_stage -> create env -> reset (BEFORE play) -> carb settings -> timeline.play()
  -> main loop

Loop order mirrors collect_dp3_data.py:
  policy(action) -> env_unwrapped.step(action) -> cam.update()
  (NOT: cam.update -> policy -> step)

Usage:
    python deploy_dp3_sim.py \
        --dp3_ckpt /path/to/checkpoint.ckpt \
        --data_path /path/to/data.zarr \
        --num_episodes 100 \
        --num_envs 16 \
        --device cuda:0 \
        --headless
"""

import argparse
import os
import sys
import time

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import numpy as np
import torch

DP3_ROOT = "/home/zeyu/3D-Diffusion-Policy/3D-Diffusion-Policy"
sys.path.insert(0, DP3_ROOT)


def _debug_step(label):
    """Print a timestamped debug label with flush."""
    print(f"[DEBUG-{time.strftime('%H:%M:%S')}] {label}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="DP3 仿真部署与性能验证")
    parser.add_argument("--dp3_ckpt", type=str,
                        default="/home/zeyu/3D-Diffusion-Policy/3D-Diffusion-Policy/data/outputs/inspire_drill-simple_dp3-simple_dp3_seed0/checkpoints/latest.ckpt",
                        help="DP3 checkpoint (.ckpt)")
    parser.add_argument("--data_path", type=str,
                        default="/home/zeyu/inspire_drill/data/inspire_drill_dp3.zarr",
                        help="训练数据集路径（用于计算 normalizer）")
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--num_episodes", type=int, default=100,
                        help="评估的总 episode 数（达到即停止）")
    parser.add_argument("--num_inference_steps", type=int, default=None,
                        help="DDIM inference steps（默认从 checkpoint 读取）")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--drill_configs", type=str, default=None)
    parser.add_argument("--img_height", type=int, default=128)
    parser.add_argument("--img_width", type=int, default=128)
    parser.add_argument("--pc_num_points", type=int, default=2048,
                        help="仿真中每个相机的深度点数量（从depth生成点云用）")
    parser.add_argument("--policy_pc_points", type=int, default=500,
                        help="DP3 policy 期望的每帧点云数量（必须与训练数据一致）")
    parser.add_argument("--workspace", nargs=6, type=float,
                        default=[0, 1.75, -0.6, 1, 0.03, 2],
                        help="[x_min, x_max, y_min, y_max, z_min, z_max] env-local (m)")
    parser.add_argument("--save_traj", type=str, default=None,
                        help="保存每条轨迹的 state/action/reward 到 npz 文件")
    return parser.parse_args()


# =========================================================================
#  点云工具：与 collect_dp3_data.py 完全一致的 FPS + 深度转点云实现
# =========================================================================
def farthest_point_sample_torch(points: torch.Tensor, num_points: int) -> torch.Tensor:
    """GPU-batched Farthest Point Sample. points: (B, N, 3) -> indices: (B, num_points)."""
    B, N, _ = points.shape
    if N <= num_points:
        return torch.arange(N, device=points.device).unsqueeze(0).expand(B, -1)

    device = points.device
    result = torch.zeros(B, num_points, dtype=torch.long, device=device)
    batch_idx = torch.arange(B, device=device)

    min_dists = torch.full((B, N), float("inf"), device=device)
    farthest = torch.zeros(B, dtype=torch.long, device=device)
    result[:, 0] = farthest

    for i in range(1, num_points):
        last_farthest_expanded = points[batch_idx, farthest]
        dists = torch.sum((points - last_farthest_expanded.unsqueeze(1)) ** 2, dim=-1)
        min_dists = torch.minimum(min_dists, dists)
        farthest = torch.argmax(min_dists, dim=1)
        result[:, i] = farthest

    return result


def depth_to_pointcloud_batch(depth, intrinsic, pos_w, quat_w, workspace, num_points,
                              env_origins=None, pool_size=2048):
    """Env-local point clouds. Vectorized: single batched FPS over all envs."""
    B, H, W, _ = depth.shape
    device = depth.device
    HW = H * W

    j_coords = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, W)
    i_coords = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1)
    cx = intrinsic[:, 0, 2].view(B, 1, 1)
    cy = intrinsic[:, 1, 2].view(B, 1, 1)
    fx = intrinsic[:, 0, 0].view(B, 1, 1)
    fy = intrinsic[:, 1, 1].view(B, 1, 1)

    z = depth[..., 0].float()
    z = torch.where(torch.isfinite(z), z, torch.zeros_like(z))
    valid_depth = z > 0

    x = (j_coords - cx) * z / fx
    y = (i_coords - cy) * z / fy
    points_cam_flat = torch.stack([x, y, z], dim=-1).view(B, HW, 3)

    q = quat_w.float()
    q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)
    qw, qx, qy, qz = q.unbind(dim=-1)
    rot = torch.empty(B, 3, 3, device=device, dtype=torch.float32)
    rot[:, 0, 0] = 1 - 2*(qy*qy + qz*qz); rot[:, 0, 1] = 2*(qx*qy - qw*qz); rot[:, 0, 2] = 2*(qx*qz + qw*qy)
    rot[:, 1, 0] = 2*(qx*qy + qw*qz); rot[:, 1, 1] = 1 - 2*(qx*qx + qz*qz); rot[:, 1, 2] = 2*(qy*qz - qw*qx)
    rot[:, 2, 0] = 2*(qx*qz - qw*qy); rot[:, 2, 1] = 2*(qy*qz + qw*qx); rot[:, 2, 2] = 1 - 2*(qx*qx + qy*qy)

    points_world = torch.bmm(points_cam_flat, rot.transpose(1, 2)) + pos_w.float().unsqueeze(1)
    if env_origins is None:
        env_origins = torch.zeros(B, 3, device=device, dtype=torch.float32)
    points_local = points_world - env_origins.float().unsqueeze(1)

    x_min, x_max, y_min, y_max, z_min, z_max = workspace
    mask = valid_depth.view(B, HW) & (
        (points_local[..., 0] > x_min) & (points_local[..., 0] < x_max) &
        (points_local[..., 1] > y_min) & (points_local[..., 1] < y_max) &
        (points_local[..., 2] > z_min) & (points_local[..., 2] < z_max)
    )
    zero_pc_mask = mask.sum(dim=1) == 0

    # Stage 1: 每个 env 随机取至多 pool_size 个有效点
    rand = torch.rand(B, HW, device=device).masked_fill(~mask, float("inf"))
    k = min(pool_size, HW)
    sel_vals, sel_idx = torch.topk(rand, k, dim=1, largest=False)
    cand = torch.gather(points_local, 1, sel_idx.unsqueeze(-1).expand(-1, -1, 3))
    pool_valid = torch.isfinite(sel_vals)
    first_valid = cand[:, :1, :]
    cand = torch.where(pool_valid.unsqueeze(-1), cand, first_valid)

    # Stage 2: 对所有 env 做一次批量 FPS
    idx = farthest_point_sample_torch(cand, num_points)
    result = torch.gather(cand, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
    result[zero_pc_mask] = 0.0
    return result, zero_pc_mask


def compute_normalizer_from_zarr(data_path: str, device: str):
    """从 zarr 数据集计算 normalizer（训练时 save 的 pickle 未必在 checkpoint 中）。"""
    import zarr
    from diffusion_policy_3d.model.common.normalizer import LinearNormalizer

    z = zarr.open_group(data_path)
    state_data = z['data']['state'][:]
    action_data = z['data']['action'][:]
    pc_data = z['data']['point_cloud'][:]

    data = {
        'action': action_data,
        'agent_pos': state_data,
        'point_cloud': pc_data,
    }

    normalizer = LinearNormalizer()
    normalizer.fit(data=data, last_n_dims=1, mode='limits')
    print(f"[Normalizer] Computed from zarr: {data_path}")
    print(f"  action  range: {action_data.min():.3f} ~ {action_data.max():.3f}")
    print(f"  agent_pos range: {state_data.min():.3f} ~ {state_data.max():.3f}")
    print(f"  point_cloud range: {pc_data.min():.3f} ~ {pc_data.max():.3f}")
    return normalizer


def main():
    args = parse_args()

    if not os.path.exists(args.dp3_ckpt):
        print(f"[ERROR] DP3 checkpoint not found: {args.dp3_ckpt}")
        return

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=args.headless, enable_cameras=True)
    simulation_app = app_launcher.app
    _debug_step("AppLauncher done")

    try:
        import carb
        import omni.timeline
        import omni.usd
        from tasks.grasp_drill_env import GraspDrillEnv, create_grasp_drill_env_cfg

        # 从干净 stage 开始（与采集脚本一致）
        omni.usd.get_context().new_stage()
        _debug_step("new_stage done")

        # ---------- Env cfg ----------
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
        cfg.seed = args.seed
        cfg.log_dir = os.path.dirname(args.dp3_ckpt)

        # 给相机加上 rgb AOV（depth-only 的 TiledCamera 首帧可能死锁）
        for _cn in ("cam1", "cam2"):
            _c = getattr(cfg.scene, _cn, None)
            if _c is not None and "rgb" not in _c.data_types:
                _c.data_types = ["rgb"] + list(_c.data_types)

        print("[TRACE] before env creation", flush=True)
        base_env = GraspDrillEnv(cfg=cfg, debug=args.debug)
        env_unwrapped = base_env.unwrapped
        print("[TRACE] after env creation", flush=True)
        _debug_step("env creation done")

        # ---------- Carb 设置（固定时间步，与采集脚本一致）----------
        settings = carb.settings.get_settings()
        settings.set("/app/player/useFixedTimeStepping", True)
        settings.set("/app/player/targetFrameRate", int(1.0 / env_unwrapped.step_dt))
        settings.set("/app/runLoops/rendering_0/rateLimitEnabled", True)
        settings.set("/app/runLoops/rendering_0/rateLimit", 60)
        settings.set("/physics/updateToUsd", False)
        settings.set("/physics/updateParticlesToUsd", False)
        settings.set("/app/runLoops/main/enabled", True)
        settings.set("/app/runLoops/main/syncTooFast", True)
        _debug_step("carb settings done")

        # reset BEFORE timeline.play()
        print("[INFO] Resetting environment (before play)...", flush=True)
        env_unwrapped.reset()
        _debug_step("reset(before play) done")

        dt = env_unwrapped.step_dt
        cam1 = env_unwrapped.scene.sensors.get("cam1")
        cam2 = env_unwrapped.scene.sensors.get("cam2")
        if cam1 is None or cam2 is None:
            raise RuntimeError(
                f"Expected env-managed cameras 'cam1'/'cam2', got "
                f"sensors={list(env_unwrapped.scene.sensors.keys())}"
            )
        _debug_step(f"cameras acquired: cam1={cam1.is_initialized}, cam2={cam2.is_initialized}")

        timeline = omni.timeline.get_timeline_interface()
        _debug_step("timeline acquired, calling play()...")
        timeline.play()
        _debug_step("timeline.play() done")
        print(f"  simulation_app.is_running() = {simulation_app.is_running()}", flush=True)

        if not simulation_app.is_running():
            print("[ERROR] simulation_app.is_running() is False before loop!")
            return
        _debug_step("app running check passed")

        # 注意：不要在这里再调用 env_unwrapped.reset()！
        # collect_dp3_data.py 也没有第二次 reset：reset(BEFORE play) -> timeline.play() -> main loop
        # 第二次 reset(AFTER play) 会触发 TiledCamera 首帧 GPU render，此时没有 render run loop 驱动，
        # TiledCamera.wait_for_next_frame() 永远等待，导致死锁。
        # 第一帧的观测会在主循环第一次 step() 时自动计算（reset 后初始状态）。
        _debug_step("starting main loop...")

        # ---------- Load DP3 ----------
        from diffusion_policy_3d.policy.dp3 import DP3
        import dill
        from hydra.utils import instantiate as hydra_instantiate
        from omegaconf import OmegaConf
        OmegaConf.register_new_resolver("eval", eval, replace=True)

        print(f"\nLoading DP3 checkpoint: {args.dp3_ckpt}", flush=True)
        _debug_step("loading checkpoint...")
        payload = torch.load(args.dp3_ckpt, pickle_module=dill, map_location="cpu")
        _debug_step("checkpoint loaded")
        dp3_cfg = payload["cfg"]

        _debug_step("instantiating policy...")
        dp3_policy: DP3 = hydra_instantiate(dp3_cfg.policy)
        _debug_step("policy instantiated, moving to device...")
        dp3_policy.to(args.device)
        dp3_policy.eval()
        dp3_policy.load_state_dict(payload["state_dicts"]["model"], strict=False)
        _debug_step("model state dict loaded")

        if args.num_inference_steps is not None:
            dp3_policy.num_inference_steps = args.num_inference_steps
            print(f"  Override num_inference_steps -> {args.num_inference_steps}")

        # 优先使用 EMA 权重
        if "ema_model" in payload["state_dicts"]:
            dp3_policy_ema: DP3 = hydra_instantiate(dp3_cfg.policy)
            dp3_policy_ema.load_state_dict(payload["state_dicts"]["ema_model"], strict=False)
            if args.num_inference_steps is not None:
                dp3_policy_ema.num_inference_steps = args.num_inference_steps
            dp3_policy_ema.to(args.device)
            dp3_policy_ema.eval()
            print("  Using EMA model.")
        else:
            dp3_policy_ema = dp3_policy
        _debug_step("policy ready")

        # 设置 normalizer
        if "pickles" in payload and "normalizer" in payload["pickles"]:
            normalizer = dill.loads(payload["pickles"]["normalizer"])
            dp3_policy.set_normalizer(normalizer)
            if "ema_model" in payload["state_dicts"]:
                dp3_policy_ema.set_normalizer(dill.loads(payload["pickles"]["normalizer"]))
            print("  Loaded normalizer from checkpoint pickles.")
        else:
            normalizer = compute_normalizer_from_zarr(args.data_path, args.device)
            dp3_policy.set_normalizer(normalizer)
            if "ema_model" in payload["state_dicts"]:
                dp3_policy_ema.set_normalizer(normalizer)
            print("  Computed normalizer from dataset.")
        _debug_step("normalizer ready")

        n_obs = dp3_policy.n_obs_steps          # 2
        n_act = dp3_policy.n_action_steps       # 8
        horizon = dp3_policy.horizon           # 16
        num_inf_steps = args.num_inference_steps or dp3_policy.num_inference_steps
        policy_pc_points = args.policy_pc_points  # 500（训练时用的点云数量）
        sim_pc_per_cam = args.pc_num_points // 2  # 2048 // 2 = 1024 per camera

        print(f"\n  DP3 config: horizon={horizon}, n_obs={n_obs}, n_act={n_act}")
        print(f"  inference steps={num_inf_steps}")
        print(f"  sim depth -> {args.pc_num_points} pts (from {args.pc_num_points//2} per cam)")
        print(f"  policy expects {policy_pc_points} pts/frame")

        # ---------- State helpers ----------
        controlled_indices = env_unwrapped.controlled_joint_indices.cpu()
        workspace = tuple(args.workspace)
        env_origins = env_unwrapped.scene.env_origins
        initial_pc = np.zeros((args.num_envs, policy_pc_points, 3), dtype=np.float32)

        from collections import deque
        env_obs_histories = [deque(maxlen=n_obs + n_act) for _ in range(args.num_envs)]
        env_episode_rewards = torch.zeros(args.num_envs, device=args.device)
        total_episodes = 0
        successful_episodes = 0
        success_log = []
        episode_rewards = []
        total_zero_pc = 0
        total_steps = 0

        # ---------- 轨迹保存（可选）----------
        traj_buffers = [[] for _ in range(args.num_envs)] if args.save_traj else None
        env_started = np.zeros(args.num_envs, dtype=bool)

        # ---------- 预填充每个 env 的观测历史 ----------
        initial_jp = env_unwrapped.franka.data.joint_pos[:, controlled_indices]
        initial_jv = env_unwrapped.franka.data.joint_vel[:, controlled_indices]
        initial_state = torch.cat([initial_jp, initial_jv], dim=-1).cpu().numpy()
        for env_id in range(args.num_envs):
            for _ in range(n_obs):
                env_obs_histories[env_id].append({
                    "agent_pos": initial_state[env_id],
                    "point_cloud": initial_pc[env_id],
                })

        pending_actions = [None] * args.num_envs
        pending_idx = [0] * args.num_envs

        _debug_step("initial state collected")

        start_time = time.time()
        print(f"\nStarting DP3 rollout (target {args.num_episodes} episodes, "
              f"{args.num_envs} envs)...")
        print(f"  workspace(env-local)={workspace}")
        print(f"  obs: agent_pos[26] + point_cloud[{policy_pc_points},3]")
        sys.stdout.flush()

        _debug_step("main loop entering...")
        try:
            with torch.inference_mode():
                while total_episodes < args.num_episodes and simulation_app.is_running():
                    try:
                        # ---- 0) 进度追踪 ----
                        if total_steps == 0:
                            _debug_step("main loop first iteration")

                        # ---- 1) 哪些 env 需要新的 action chunk ----
                        need_policy_call = [
                            env_id for env_id in range(args.num_envs)
                            if pending_actions[env_id] is None or pending_idx[env_id] >= n_act
                        ]

                        # ---- 2) DP3 推理（批处理）----
                        if need_policy_call:
                            obs_batch_list, pc_batch_list = [], []
                            for env_id in need_policy_call:
                                hist = list(env_obs_histories[env_id])
                                if len(hist) < n_obs:
                                    hist = [hist[0]] * (n_obs - len(hist)) + hist
                                obs_timestep = hist[-n_obs:]
                                obs_batch_list.append(np.stack([s["agent_pos"] for s in obs_timestep]))
                                pc_batch_list.append(np.stack([s["point_cloud"] for s in obs_timestep]))

                            obs_dict = {
                                "agent_pos": torch.from_numpy(np.stack(obs_batch_list)).to(args.device),
                                "point_cloud": torch.from_numpy(np.stack(pc_batch_list)).to(args.device),
                            }
                            result = dp3_policy_ema.predict_action(obs_dict)
                            action_chunks = result["action"].cpu().numpy()

                            for i, env_id in enumerate(need_policy_call):
                                pending_actions[env_id] = action_chunks[i]
                                pending_idx[env_id] = 0

                        # ---- 3) 执行一帧动作 ----
                        actions_np = np.zeros((args.num_envs, 13), dtype=np.float32)
                        for env_id in range(args.num_envs):
                            if pending_actions[env_id] is not None:
                                actions_np[env_id] = pending_actions[env_id][pending_idx[env_id]]
                                pending_idx[env_id] += 1

                        actions = torch.from_numpy(actions_np).to(args.device)
                        obs_dict_step, rewards, terminated, truncated, _ = env_unwrapped.step(actions)
                        total_steps += 1

                        # ---- 4) 更新相机：在 step() 之后（与 collect_dp3_data.py 一致）----
                        cam1.update(dt)
                        cam2.update(dt)

                        # ---- 5) 在 step() 之后读取状态（与 collect_dp3_data.py 一致）----
                        joint_pos = env_unwrapped.franka.data.joint_pos
                        joint_vel = env_unwrapped.franka.data.joint_vel
                        state_26 = torch.cat(
                            [joint_pos[:, controlled_indices], joint_vel[:, controlled_indices]],
                            dim=-1,
                        ).cpu().numpy()

                        # ---- 6) 深度图 -> 点云 ----
                        depth1 = torch.nan_to_num(cam1.data.output["distance_to_image_plane"],
                                                  nan=0.0, posinf=0.0, neginf=0.0)
                        depth2 = torch.nan_to_num(cam2.data.output["distance_to_image_plane"],
                                                  nan=0.0, posinf=0.0, neginf=0.0)

                        pc1_t, zmask1 = depth_to_pointcloud_batch(
                            depth1, cam1.data.intrinsic_matrices, cam1.data.pos_w,
                            cam1.data.quat_w_ros, workspace, sim_pc_per_cam, env_origins)
                        pc2_t, zmask2 = depth_to_pointcloud_batch(
                            depth2, cam2.data.intrinsic_matrices, cam2.data.pos_w,
                            cam2.data.quat_w_ros, workspace, sim_pc_per_cam, env_origins)
                        total_zero_pc += int(zmask1.sum().item() + zmask2.sum().item())

                        pc_fused_2048 = torch.zeros(args.num_envs, args.pc_num_points, 3,
                                                   device=args.device, dtype=torch.float32)
                        pc_fused_2048[:, :sim_pc_per_cam] = pc1_t
                        pc_fused_2048[:, sim_pc_per_cam:] = pc2_t

                        if policy_pc_points < args.pc_num_points:
                            down_idx = farthest_point_sample_torch(pc_fused_2048, policy_pc_points)
                            pc_for_policy = torch.gather(
                                pc_fused_2048, 1,
                                down_idx.unsqueeze(-1).expand(-1, -1, 3)
                            ).cpu().numpy()
                        else:
                            pc_for_policy = pc_fused_2048.cpu().numpy()

                        env_episode_rewards += rewards
                        is_done = terminated.bool() | truncated.bool()

                        try:
                            lenient_success = env_unwrapped._cached_lenient_success
                        except AttributeError:
                            lenient_success = torch.zeros(args.num_envs, dtype=torch.bool,
                                                          device=args.device)

                        # ---- 7) Episode 完成处理 ----
                        finished = torch.where(is_done)[0]
                        if finished.numel() > 0:
                            reset_jp = env_unwrapped.franka.data.joint_pos[:, controlled_indices]
                            reset_jv = env_unwrapped.franka.data.joint_vel[:, controlled_indices]
                            reset_state = torch.cat([reset_jp, reset_jv], dim=-1).cpu().numpy()

                            for env_idx in finished:
                                env_id = env_idx.item()
                                total_episodes += 1
                                episode_rewards.append(env_episode_rewards[env_id].item())

                                if lenient_success[env_id]:
                                    successful_episodes += 1
                                    success_log.append(1)
                                else:
                                    success_log.append(0)

                                if traj_buffers is not None and len(traj_buffers[env_id]) > 0:
                                    traj_buffers[env_id].append({
                                        "state": state_26[env_id],
                                        "action": actions_np[env_id],
                                        "reward": rewards[env_id].item(),
                                        "done": True,
                                    })
                                    save_traj_as_npz(traj_buffers[env_id], args.save_traj, env_id)

                                env_episode_rewards[env_id] = 0.0
                                pending_actions[env_id] = None
                                pending_idx[env_id] = 0
                                env_obs_histories[env_id].clear()
                                for _ in range(n_obs):
                                    env_obs_histories[env_id].append({
                                        "agent_pos": reset_state[env_id],
                                        "point_cloud": initial_pc[env_id],
                                    })
                                env_started[env_id] = True
                                traj_buffers[env_id] = []

                        # ---- 8) 更新未结束 env 的观测历史（用 step 后读取的新 pc）----
                        for env_id in range(args.num_envs):
                            if not is_done[env_id]:
                                if not env_started[env_id]:
                                    env_started[env_id] = True
                                env_obs_histories[env_id].append({
                                    "agent_pos": state_26[env_id],
                                    "point_cloud": pc_for_policy[env_id],
                                })
                                if traj_buffers is not None:
                                    traj_buffers[env_id].append({
                                        "state": state_26[env_id],
                                        "action": actions_np[env_id],
                                        "reward": rewards[env_id].item(),
                                        "done": False,
                                    })

                        # ---- 9) 进度打印 ----
                        if total_steps % 100 == 0:
                            _debug_step(f"step {total_steps}")
                            elapsed = time.time() - start_time
                            sr = successful_episodes / max(total_episodes, 1) * 100
                            print(f"  step={total_steps} | eps={total_episodes}/{args.num_episodes} "
                                  f"| succ={successful_episodes} ({sr:.1f}%) "
                                  f"| zero_pc={total_zero_pc} | {total_steps/elapsed:.0f} step/s")
                            sys.stdout.flush()

                    except Exception as loop_err:
                        import traceback
                        _debug_step(f"[ERROR] loop body failed at step {total_steps}: {loop_err}")
                        traceback.print_exc()
                        raise

        except Exception as main_err:
            import traceback
            _debug_step(f"[FATAL] main loop error: {main_err}")
            traceback.print_exc()
            raise

        # ---- 最终统计 ----
        elapsed = time.time() - start_time
        m, s = divmod(int(elapsed), 60)
        h, m = divmod(m, 60)

        print(f"\n{'='*60}")
        print(f"DP3 Evaluation Results")
        print(f"{'='*60}")
        print(f"Checkpoint: {args.dp3_ckpt}")
        print(f"Total episodes: {total_episodes}")
        print(f"Successful episodes: {successful_episodes}")
        if total_episodes > 0:
            print(f"Success rate: {successful_episodes/total_episodes*100:.1f}%")
        if episode_rewards:
            ep_np = np.array(episode_rewards)
            print(f"Reward: mean={ep_np.mean():.1f}, max={ep_np.max():.1f}, min={ep_np.min():.1f}, std={ep_np.std():.1f}")
        print(f"Total steps: {total_steps}")
        print(f"Total zero-pc frames: {total_zero_pc}")
        print(f"Time: {h}h{m}m{s}s, {total_steps/elapsed:.0f} steps/s")
        if success_log:
            print(f"Success log: {success_log}")
        print(f"{'='*60}")

    finally:
        simulation_app.close()


def save_traj_as_npz(buffer, output_dir, env_id):
    """将轨迹数据保存为 npz 文件。"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    states = np.stack([f["state"] for f in buffer])
    actions = np.stack([f["action"] for f in buffer])
    rewards = np.array([f["reward"] for f in buffer])
    path = os.path.join(output_dir, f"traj_env{env_id}.npz")
    np.savez(path, states=states, actions=actions, rewards=rewards)
    print(f"  Saved trajectory: {path} ({len(buffer)} steps)")


if __name__ == "__main__":
    main()
