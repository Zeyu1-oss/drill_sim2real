#!/usr/bin/env python3
"""
DP3 visual data collection script using RL agent rollouts.
Saves successful episodes as DP3-compatible Zarr format.

Initialization sequence is copied verbatim from the working debug_camera_viz.py:
  new_stage -> create env -> reset (BEFORE play) -> carb settings -> timeline.play()
  -> reset (AFTER play) -> loop(env.step + cam.update, NO explicit sim.render).

DEBUG INSTRUMENTATION (this version):
  * zarr / numcodecs / psutil are imported AFTER rl_games (which pulls in triton).
    Importing numcodecs/blosc BEFORE triton loads a conflicting native/OpenMP
    runtime that deadlocks triton's C-extension dlopen. debug_camera_viz.py never
    imports those, which is why it doesn't hang.
  * Every import in main() is bracketed by [IMP] trace prints -> the LAST printed
    [IMP] line is the one that hung.
  * faulthandler.dump_traceback_later(60, repeat=True) auto-dumps the full stack of
    every thread to stderr if anything blocks >60s, so you don't need py-spy.
    It's cancelled right before the main collection loop (so it won't spam during
    the long rollout) and again in finally.
"""

import argparse
import os
import sys
import time
import shutil

import faulthandler

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Only the lightweight / always-safe native libs at top.
# zarr / numcodecs / psutil are intentionally NOT imported here (see module docstring).
import numpy as np
import torch

# ---------------------------------------------------------------------------
# REAL FIX: neutralize torch.compile BEFORE rl_games is imported.
# rl_games/envs/test_network.py calls torch.compile() at module-import time,
# which lazily imports torch._dynamo -> triton, and triton's C-extension dlopen
# deadlocks inside the omni/Isaac process. We never use torch.compile anywhere,
# so making it a passthrough removes the triton dependency completely and the
# hang can no longer happen, regardless of the underlying dlopen race.
# ---------------------------------------------------------------------------
def _safe_compile(model=None, *args, **kwargs):
    if callable(model):
        return model                 # torch.compile(net) / @torch.compile
    return lambda fn: fn             # torch.compile(mode=...) used as decorator
torch.compile = _safe_compile


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="data/inspire_drill_dp3.zarr")
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--drill_configs", type=str, default=None)
    parser.add_argument("--img_height", type=int, default=128)
    parser.add_argument("--img_width", type=int, default=128)
    parser.add_argument("--pc_num_points", type=int, default=2048)
    parser.add_argument("--workspace", nargs=6, type=float,
                        default=[0, 1.75, -0.6, 1, 0.02, 2],
                        help="[x_min, x_max, y_min, y_max, z_min, z_max] in env-local coords (m). "
                             "Default covers table (y~0, z~-0.23) + drill (x~0.35, z~-0.18) with margin.")
    # Optional: drop the first frame(s) of every fresh episode in case the very
    # first post-reset render is stale. 0 keeps everything.
    parser.add_argument("--skip_first_frames", type=int, default=0)
    return parser.parse_args()


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
    """Env-local point clouds. Vectorized: single batched FPS over all envs (no per-env python loop)."""
    B, H, W, _ = depth.shape
    device = depth.device
    HW = H * W

    j_coords = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, W)
    i_coords = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1)
    cx = intrinsic[:, 0, 2].view(B, 1, 1); cy = intrinsic[:, 1, 2].view(B, 1, 1)
    fx = intrinsic[:, 0, 0].view(B, 1, 1); fy = intrinsic[:, 1, 1].view(B, 1, 1)

    z = depth[..., 0].float()
    z = torch.where(torch.isfinite(z), z, torch.zeros_like(z))
    valid_depth = z > 0

    x = (j_coords - cx) * z / fx
    y = (i_coords - cy) * z / fy
    points_cam_flat = torch.stack([x, y, z], dim=-1).view(B, HW, 3)

    q = quat_w.float(); q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)
    qw, qx, qy, qz = q.unbind(dim=-1)
    rot = torch.empty(B, 3, 3, device=device, dtype=torch.float32)
    rot[:, 0, 0] = 1 - 2*(qy*qy + qz*qz); rot[:, 0, 1] = 2*(qx*qy - qw*qz); rot[:, 0, 2] = 2*(qx*qz + qw*qy)
    rot[:, 1, 0] = 2*(qx*qy + qw*qz); rot[:, 1, 1] = 1 - 2*(qx*qx + qz*qz); rot[:, 1, 2] = 2*(qy*qz - qw*qx)
    rot[:, 2, 0] = 2*(qx*qz - qw*qy); rot[:, 2, 1] = 2*(qy*qz + qw*qx); rot[:, 2, 2] = 1 - 2*(qx*qx + qy*qy)

    points_world = torch.bmm(points_cam_flat, rot.transpose(1, 2)) + pos_w.float().unsqueeze(1)
    if env_origins is None:
        env_origins = torch.zeros(B, 3, device=device, dtype=torch.float32)
    points_local = points_world - env_origins.unsqueeze(1)          # (B, HW, 3)

    x_min, x_max, y_min, y_max, z_min, z_max = workspace
    mask = valid_depth.view(B, HW) & (
        (points_local[..., 0] > x_min) & (points_local[..., 0] < x_max) &
        (points_local[..., 1] > y_min) & (points_local[..., 1] < y_max) &
        (points_local[..., 2] > z_min) & (points_local[..., 2] < z_max)
    )                                                               # (B, HW)
    zero_pc_mask = mask.sum(dim=1) == 0                             # (B,)

    # --- Stage 1: vectorized pick up to pool_size valid points per env (no loop, no sync) ---
    rand = torch.rand(B, HW, device=device).masked_fill(~mask, float("inf"))
    k = min(pool_size, HW)
    sel_vals, sel_idx = torch.topk(rand, k, dim=1, largest=False)   # (B, k)
    cand = torch.gather(points_local, 1, sel_idx.unsqueeze(-1).expand(-1, -1, 3))  # (B, k, 3)
    pool_valid = torch.isfinite(sel_vals)                          # (B, k): real valid points
    # replace junk slots (envs with <k valid) by that env's first valid point so FPS won't pick garbage
    first_valid = cand[:, :1, :]
    cand = torch.where(pool_valid.unsqueeze(-1), cand, first_valid)

    # --- Stage 2: ONE batched FPS over all B envs (250 iters total, not B*250) ---
    idx = farthest_point_sample_torch(cand, num_points)            # (B, num_points)
    result = torch.gather(cand, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
    result[zero_pc_mask] = 0.0
    return result, zero_pc_mask


def main():
    args = parse_args()
    simulation_app = None

    # --- DEBUG: enable() only installs signal handlers (no extra thread), safe to
    #     do early. The dump_traceback_later watchdog THREAD is armed later, AFTER
    #     all imports, so it can't interfere with any C-extension dlopen. ---
    faulthandler.enable()  # dump on hard crash / segfault (native lib issues)
    print("[TRACE] faulthandler.enable() done", flush=True)

    if args.headless:
        print("[INFO] Running headless collection with IsaacLab-managed TiledCamera sensors.", flush=True)

    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        return

    if os.path.exists(args.output):
        shutil.rmtree(args.output)

    print("[TRACE] before AppLauncher", flush=True)
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=args.headless, enable_cameras=True)
    simulation_app = app_launcher.app
    print("[TRACE] after AppLauncher", flush=True)

    try:
        # ---------------- instrumented imports ----------------
        # The LAST [IMP] line printed before a hang = the import that deadlocked.
        print("[IMP] math/yaml ...", flush=True)
        import math
        import yaml
        print("[IMP] carb/omni ...", flush=True)
        import carb
        import omni.timeline
        import omni.usd
        print("[IMP] rl_games.common (this triggers torch.compile -> torch._dynamo -> triton dlopen) ...", flush=True)
        from rl_games.common import env_configurations, vecenv
        print("[IMP] rl_games.torch_runner ...", flush=True)
        from rl_games.torch_runner import Runner
        print("[IMP] isaaclab_rl.rl_games ...", flush=True)
        from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
        print("[IMP] tasks.grasp_drill_env ...", flush=True)
        from tasks.grasp_drill_env import GraspDrillEnv, create_grasp_drill_env_cfg
        print("[IMP] triton-dependent imports done", flush=True)

        # zarr / numcodecs / psutil imported HERE, AFTER triton is fully loaded.
        # (Importing numcodecs before triton deadlocks triton's C-extension dlopen.)
        print("[IMP] zarr ...", flush=True)
        import zarr
        print("[IMP] numcodecs ...", flush=True)
        import numcodecs
        print("[IMP] psutil ...", flush=True)
        import psutil
        print("[IMP] ALL imports done", flush=True)

        # Now that all C-extension dlopens are done, it's safe to start the
        # watchdog thread. Covers env creation / timeline.play() / first render.
        faulthandler.dump_traceback_later(60, repeat=True)
        print("[TRACE] faulthandler auto-dump armed (60s)", flush=True)

        # Match debug_camera_viz.py: start from a clean stage.
        print("[TRACE] new_stage ...", flush=True)
        omni.usd.get_context().new_stage()
        print("[TRACE] new_stage done", flush=True)

        rl_config_path = os.path.join(project_root, "config/agents/rl_games_ppo_cfg.yaml")
        with open(rl_config_path, "r") as f:
            agent_cfg = yaml.safe_load(f)

        agent_cfg["params"]["config"]["num_actors"] = args.num_envs
        agent_cfg["params"]["config"]["device"] = args.device
        agent_cfg["params"]["config"]["device_name"] = args.device

        print("[TRACE] create_grasp_drill_env_cfg ...", flush=True)
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
        cfg.log_dir = os.path.dirname(os.path.dirname(args.checkpoint))
        print("[TRACE] cfg built", flush=True)

        # Fix: add rgb to camera data_types so TiledCamera's render product has a color AOV.
        # Without rgb, a depth-only TiledCamera can deadlock on its first frame.
        for _cn in ("cam1", "cam2"):
            _c = getattr(cfg.scene, _cn, None)
            if _c is not None and "rgb" not in _c.data_types:
                _c.data_types = ["rgb"] + list(_c.data_types)

        print("[TRACE] before env creation", flush=True)
        base_env = GraspDrillEnv(cfg=cfg, debug=args.debug)
        env_unwrapped = base_env.unwrapped
        print("[TRACE] after env creation", flush=True)

        # --- reset BEFORE timeline.play() (this is the order that works) ---
        print("[INFO] Resetting environment (before play)...", flush=True)
        _ = env_unwrapped.reset()
        print("[TRACE] reset (before play) done", flush=True)

        clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
        clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
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

        print("[TRACE] building rl_games runner/agent ...", flush=True)
        runner = Runner()
        runner.load(agent_cfg)
        agent = runner.create_player()
        agent.restore(args.checkpoint)
        agent.reset()
        print("[TRACE] after agent restore/reset", flush=True)

        dt = env_unwrapped.step_dt
        cam1 = env_unwrapped.scene.sensors.get("cam1")
        cam2 = env_unwrapped.scene.sensors.get("cam2")
        if cam1 is None or cam2 is None:
            raise RuntimeError(
                f"Expected env-managed cameras 'cam1' and 'cam2', got sensors={list(env_unwrapped.scene.sensors.keys())}"
            )
        print(
            f"[TRACE] cameras ready cam1={type(cam1).__name__} init={cam1.is_initialized} "
            f"cam2={type(cam2).__name__} init={cam2.is_initialized}",
            flush=True,
        )

        # --- carb settings copied from debug_camera_viz.py: these keep the render
        #     run loop alive so env.step()'s internal TiledCamera render completes.
        #     Without them step() blocks forever waiting on GPU frames (the deadlock). ---
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
        print("[TRACE] timeline.play() ...", flush=True)
        timeline.play()
        sys.stdout.flush()
        print(f"[DEBUG] timeline.play() done, is_running={simulation_app.is_running()}", flush=True)
        if not simulation_app.is_running():
            print("[ERROR] simulation_app.is_running() is False before loop!")
            return

        # --- reset AFTER play: use wrapped env (env), NOT env_unwrapped.
        # env_unwrapped.reset() returns (obs_dict, extras) tuple which rl_games can't parse.
        # RlGamesVecEnvWrapper.reset() returns the format rl_games expects.
        print("[TRACE] env.reset() (after play) ...", flush=True)
        obs = env.reset()
        if isinstance(obs, dict):
            obs = obs.get("obs", obs)
        agent.get_batch_size(obs, 1)
        agent.reset()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(args.device)
        print("[TRACE] reset (after play) done", flush=True)

        controlled_indices = env_unwrapped.controlled_joint_indices.cpu()
        workspace = tuple(args.workspace)

        # ---------------- Incremental Zarr setup ----------------
        print("[TRACE] zarr setup ...", flush=True)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        # zarr v3 requires zarr_format=2 to accept numcodecs.Blosc as compressor.
        # v3 native codecs don't support incremental resize well, so we use v2 format.
        blosc_compressor = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=1)
        zarr_root = zarr.group(args.output, zarr_format=2)
        zarr_data = zarr_root.create_group("data")
        zarr_meta = zarr_root.create_group("meta")

        half_points = args.pc_num_points // 2
        zarr_data.create_dataset("state", shape=(0, 26), dtype="float32",
                                 compressor=blosc_compressor, chunks=(100, 26), overwrite=True)
        zarr_data.create_dataset("action", shape=(0, 13), dtype="float32",
                                 compressor=blosc_compressor, chunks=(100, 13), overwrite=True)
        zarr_data.create_dataset("point_cloud", shape=(0, args.pc_num_points, 3), dtype="float32",
                                 compressor=blosc_compressor, chunks=(100, args.pc_num_points, 3), overwrite=True)
        zarr_meta.create_dataset("episode_ends", shape=(0,), dtype="int64",
                                 compressor=blosc_compressor, chunks=(100,), overwrite=True)
        print("[TRACE] zarr setup done", flush=True)

        episode_ends = []

        # ---------------- collection state ----------------
        buffers = [[] for _ in range(args.num_envs)]      # list of (state, action, pc) per env
        env_started = np.zeros(args.num_envs, dtype=bool)  # whether env is currently recording
        total_collected = 0
        total_steps = 0
        total_zero_pc = 0
        print_interval = 100
        start_time = time.time()

        def flush_episode(states, actions, pcs):
            nonlocal total_collected
            n = len(states)
            if n == 0:
                return
            states_arr = np.stack(states).astype(np.float32)
            actions_arr = np.stack(actions).astype(np.float32)
            pcs_arr = np.stack(pcs).astype(np.float32)
            idx = zarr_data["state"].shape[0]
            zarr_data["state"].resize((idx + n, 26))
            zarr_data["action"].resize((idx + n, 13))
            zarr_data["point_cloud"].resize((idx + n, args.pc_num_points, 3))
            zarr_data["state"][idx:idx + n] = states_arr
            zarr_data["action"][idx:idx + n] = actions_arr
            zarr_data["point_cloud"][idx:idx + n] = pcs_arr
            episode_ends.append(idx + n)
            ends = np.array(episode_ends, dtype=np.int64)
            zarr_meta["episode_ends"].resize((len(ends),))
            zarr_meta["episode_ends"][:] = ends
            total_collected += 1

        # init done -> stop the auto-dump so it doesn't spam during the long rollout.
        faulthandler.cancel_dump_traceback_later()
        print("[TRACE] faulthandler auto-dump cancelled; entering main loop", flush=True)
        print(f"Collecting {args.num_episodes} episodes (num_envs={args.num_envs})...", flush=True)

        try:
            with torch.inference_mode():
                while total_collected < args.num_episodes and simulation_app.is_running():
                    # 1) policy action
                    obs_tensor = agent.obs_to_torch(obs)
                    actions = agent.get_action(obs_tensor, is_deterministic=True)
                    if not isinstance(actions, torch.Tensor):
                        actions = torch.from_numpy(np.array(actions)).float().to(args.device)

                    # 2) step: use env_unwrapped.step() to avoid deadlock from sim.render().
                    # env.step() internally calls sim.render() which deadlocks with the carb run loop.
                    # env_unwrapped.step() returns (obs_dict, rewards, terminated, truncated, extras).
                    obs_dict, rewards, terminated, truncated, extras = env_unwrapped.step(actions)
                    total_steps += 1
                    cam1.update(dt)
                    cam2.update(dt)

                    # Convert obs to rl_games format for agent (mirrors env._process_obs)
                    obs = env._process_obs(obs_dict)

                    # 3) state (13 pos + 13 vel = 26)
                    joint_pos = env_unwrapped.franka.data.joint_pos
                    joint_vel = env_unwrapped.franka.data.joint_vel
                    state_26 = torch.cat(
                        [joint_pos[:, controlled_indices], joint_vel[:, controlled_indices]], dim=-1
                    )

                    # 4) depth -> fused point cloud
                    # Read depth directly from camera output buffers (bypasses _get_camera_observations
                    # which now returns rgb after we added it to data_types above).
                    depth1 = cam1.data.output["distance_to_image_plane"]
                    depth2 = cam2.data.output["distance_to_image_plane"]
                    depth1 = torch.nan_to_num(depth1, nan=0.0, posinf=0.0, neginf=0.0)
                    depth2 = torch.nan_to_num(depth2, nan=0.0, posinf=0.0, neginf=0.0)

                    pc1, zmask1 = depth_to_pointcloud_batch(
                        depth1, cam1.data.intrinsic_matrices, cam1.data.pos_w,
                        cam1.data.quat_w_ros, workspace, half_points,
                        env_unwrapped.scene.env_origins,
                    )
                    pc2, zmask2 = depth_to_pointcloud_batch(
                        depth2, cam2.data.intrinsic_matrices, cam2.data.pos_w,
                        cam2.data.quat_w_ros, workspace, half_points,
                        env_unwrapped.scene.env_origins,
                    )
                    total_zero_pc += int(zmask1.sum().item() + zmask2.sum().item())

                    pc_fused = torch.zeros(args.num_envs, args.pc_num_points, 3,
                                           device=depth1.device, dtype=torch.float32)
                    pc_fused[:, :half_points] = pc1
                    pc_fused[:, half_points:] = pc2

                    # 5) move everything to CPU once
                    actions_np = actions.detach().cpu().numpy()
                    state_np = state_26.detach().cpu().numpy()
                    pc_np = pc_fused.detach().cpu().numpy()

                    # is_done = terminated | truncated (both available directly from env_unwrapped.step)
                    is_done = terminated.bool() | truncated.bool()
                    try:
                        lenient_success = env_unwrapped._cached_lenient_success
                    except AttributeError:
                        lenient_success = torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)
                    is_done_np = is_done.detach().cpu().numpy()
                    success_np = lenient_success.detach().cpu().numpy().astype(bool)
                    # per-env "is this frame's pc empty"
                    bad_pc_np = (zmask1 | zmask2).detach().cpu().numpy()

                    # 6) buffer current frame for active, non-terminating envs.
                    #    Skip the terminating step: env auto-resets inside step(), so the
                    #    state/pc just read already belong to the NEXT episode.
                    for env_id in range(args.num_envs):
                        if not env_started[env_id] or is_done_np[env_id]:
                            continue
                        if args.skip_first_frames and len(buffers[env_id]) < args.skip_first_frames:
                            # still allow growth so the counter advances, but don't keep junk frames
                            buffers[env_id].append((state_np[env_id], actions_np[env_id], pc_np[env_id]))
                            continue
                        buffers[env_id].append((state_np[env_id], actions_np[env_id], pc_np[env_id]))

                    # 7) episode boundaries
                    any_done = False
                    for env_id in range(args.num_envs):
                        if is_done_np[env_id]:
                            any_done = True
                            if env_started[env_id] and buffers[env_id] and success_np[env_id]:
                                states, acts, pcs = zip(*buffers[env_id])
                                flush_episode(list(states), list(acts), list(pcs))
                            buffers[env_id] = []
                            env_started[env_id] = True  # fresh episode begins now
                        elif not env_started[env_id]:
                            buffers[env_id] = []
                            env_started[env_id] = True

                    if any_done:
                        agent.reset()

                    # 8) progress
                    if total_steps % print_interval == 0:
                        elapsed = max(time.time() - start_time, 1e-6)
                        eta_sec = (elapsed / total_collected * (args.num_episodes - total_collected)
                                   if total_collected > 0 else 0)
                        m, s = divmod(int(eta_sec), 60)
                        h, m = divmod(m, 60)
                        filled = int(30 * total_collected / args.num_episodes)
                        bar = "#" * filled + "-" * (30 - filled)
                        proc = psutil.Process()
                        cpu_mem_gb = proc.memory_info().rss / 1e9
                        gpu_mem_gb = (torch.cuda.max_memory_allocated(args.device) / 1e9
                                      if torch.cuda.is_available() else 0)
                        num_pending = sum(len(b) for b in buffers)
                        print(f"  [{bar}] {total_collected}/{args.num_episodes} | {total_steps}stp | "
                              f"{total_steps / elapsed:.0f}/s | ETA={h}h{m}m{s}s | "
                              f"RAM={cpu_mem_gb:.1f}GB GPU={gpu_mem_gb:.1f}GB | "
                              f"pending={num_pending} | zero_pc={total_zero_pc}", flush=True)

                    if isinstance(obs, dict):
                        obs = obs["obs"]

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user, saving collected data...", flush=True)
        except Exception as e:
            import traceback
            print(f"\n[ERROR] Exception in main loop: {e}")
            traceback.print_exc()
            print(f"[DEBUG] Last step count: {total_steps}, collected: {total_collected}")

        elapsed = max(time.time() - start_time, 1e-6)
        m, s = divmod(int(elapsed), 60)
        h, m = divmod(m, 60)
        gpu_mem_gb = (torch.cuda.max_memory_allocated(args.device) / 1e9
                      if torch.cuda.is_available() else 0)
        print(f"\nDone: collected={total_collected} episodes, steps={total_steps}, "
              f"zero_pc={total_zero_pc}, peak_gpu={gpu_mem_gb:.1f}GB, time={h}h{m}m{s}s", flush=True)
        print(f"Output: {args.output}", flush=True)

        env.close()
    except Exception:                          # 捕获 try 主体里(loop 之前)的异常
        import traceback
        print("\n[FATAL] 真正的异常在这里:", flush=True)
        traceback.print_exc()
    finally:
        faulthandler.cancel_dump_traceback_later()
        # py-spy 确认 simulation_app.close() 会卡在收尾渲染里;数据已增量写盘,直接硬退出
        os._exit(0)


if __name__ == "__main__":
    main()