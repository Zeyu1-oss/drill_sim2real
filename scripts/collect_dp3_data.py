import argparse
import os
import sys
import time
import shutil

import faulthandler

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (scripts/ -> ../..)
sys.path.insert(0, project_root)

import numpy as np
import torch


def _safe_compile(model=None, *args, **kwargs):
    if callable(model):
        return model                 # torch.compile(net) / @torch.compile
    return lambda fn: fn             # torch.compile(mode=...) used as decorator
torch.compile = _safe_compile


from perception.dp3_pointcloud import build_fused_camera_pc, init_fps_kernel


_PERCEPTION_HP = None
def _load_perception_hp():
    """按文件路径加载 hyperparameters.py(纯 stdlib)取 PerceptionHyperparameters,
    绕过 tasks 包 __init__(它 import grasp_drill_env 需要 IsaacLab,而 parse_args
    阶段 app 还没起)。collect/deploy 共用同一真源。"""
    global _PERCEPTION_HP
    if _PERCEPTION_HP is None:
        import importlib.util
        _p = os.path.join(project_root, "tasks", "config", "hyperparameters.py")
        _spec = importlib.util.spec_from_file_location("_hp_standalone", _p)
        _m = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_m)
        _PERCEPTION_HP = _m.DEFAULT_HYPERPARAMETERS.perception
    return _PERCEPTION_HP


def parse_args():
    _PERC = _load_perception_hp()   # PerceptionHyperparameters(单一真源)
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="data/inspire_drill_dp3_target_forces_1200.zarr")
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--episodes_per_variant", type=int, default=10000,
                        help="每种电钻收集的 episode 数；设置后总数 = 该值 × 电钻种类数，"
                             "并覆盖 --num_episodes")
    # 动作恒为 target 直接驱动:teacher raw→q*→直接下发,标签 = cur_targets(rad)
    # 经 --label_smooth_k 步前向滑动平均(见 target_drive.smooth_labels_forward:
    # 消掉 teacher 的 PWM 抖动噪声,保留握力/抗重力直流偏置)。
    # 与 deploy 用同一动力学路径,采集↔部署完全一致(已去掉 raw 模式)。
    parser.add_argument("--label_smooth_k", type=int, default=1,
                        help="动作标签前向滑动平均窗口(步);1=关闭(默认,直接用原始 target 标签)。"
                             ">1 则对 teacher 的 PWM 抖动做前向滑动平均。")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--drill_configs", type=str, default=None)
    parser.add_argument("--img_height", type=int, default=_PERC.img_height,
                        help="相机分辨率,默认来自 PerceptionHyperparameters(单一真源,与 deploy 一致)")
    parser.add_argument("--img_width", type=int, default=_PERC.img_width)
    parser.add_argument("--pc_num_points", type=int, default=1024,
                        help="相机段点数=两相机合计(cam1/cam2 各取一半,即各 512)。"
                             "默认 1024,配合 robot 160 + ground 96 = 总 1280")
    parser.add_argument("--robot_pc_points", type=int, default=160,
                        help="机器人 FK 点云降采样点数(canonical npz 1280 点分层抽样;"
                             "None/负数=不降。须与 deploy 一致)")
    parser.add_argument("--pc_dtype", type=str, default="float16",
                        choices=["float16", "float32"],
                        help="point_cloud 在 zarr 中的存储精度。fp16 在 ≤2m 坐标下分辨率≈1mm,"
                             "省一半盘;训练端读取时统一转 float32")
    parser.add_argument("--workspace", nargs=6, type=float,
                        default=list(_PERC.workspace),
                        help="[x_min, x_max, y_min, y_max, z_min, z_max] env-local (m);"
                             "默认来自 hyperparameters.PerceptionHyperparameters.workspace(单一真源)")
    parser.add_argument("--state_abs_limit", type=float, default=20.0)
    parser.add_argument("--force_state", action="store_true",
                        help="force 模式:state 存 26 维 [关节位置13|接触力13(log1p归一)](速度已移除)")
    parser.add_argument("--save_privileged", action="store_true",
                        help="额外保存特权信息到 zarr 的 data/privileged(用于特权重建蒸馏)")

    parser.add_argument("--pc_mode", type=str, default="robot",
                        choices=["camera", "robot", "robot_drill"],
                        help="点云组成:camera=仅相机;robot=+FK机器人;robot_drill=+机器人+电钻mesh完整云")
    parser.add_argument("--robot_pc_npz", type=str,
                        default="assets/inspire_tac/robot_canonical_points.npz",
                        help="tools/build_robot_pointcloud.py 产出的 canonical 点(link 局部系)")
    # 电钻完整云(mesh + 真值位姿)的点数;仅 robot_drill 模式用。仅 sim(用真值位姿)。
    parser.add_argument("--drill_pc_points", type=int, default=512,
                        help="完整电钻点云的点数(仅 pc_mode=robot_drill 用)")
    parser.add_argument("--save_init_poses", action="store_true",
                        help="overfit 测试用:额外保存每条 episode 的电钻初始位姿(env-local pos+quat+goal)"
                             "到 <output>_init_poses/init_poses.npz,与 zarr 的 episode 顺序 1:1。"
                             "deploy 可用 --init_pose_file 回放,验证 DP3 能否在相同场景下复刻。")
    # ---- 网格覆盖采集:用确定性网格 round-robin 替代随机采样,保证均匀覆盖位姿空间 ----
    parser.add_argument("--cover_grid", action="store_true",
                        help="开启确定性网格 round-robin 覆盖(替代 env 随机采样),均匀覆盖 x/y/yaw。")
    parser.add_argument("--grid_nx", type=int, default=4, help="x 方向网格点数")
    parser.add_argument("--grid_ny", type=int, default=4, help="y 方向网格点数")
    parser.add_argument("--grid_nyaw", type=int, default=6, help="yaw 方向网格点数")
    parser.add_argument("--grid_pos_range", nargs=2, type=float, default=None,
                        help="x,y 半幅(米);默认用 env 的 drill_pos_random_range")
    parser.add_argument("--grid_yaw_range", type=float, default=None,
                        help="yaw 半幅(rad);默认用 env 的 drill_rot_random_range[2]")
    parser.add_argument("--cover_sobol", action="store_true",
                        help="用 Sobol 低差异序列连续均匀采样(x,y,yaw);密度只由 episodes_per_variant 决定,"
                             "无需手动定义网格分辨率。优先于 --cover_grid。")
    args = parser.parse_args()

    args.save_robot_pc = args.pc_mode in ("robot", "robot_drill")
    args.save_drill_pc = args.pc_mode == "robot_drill"
    return args


PRIV_DIM = 26

def compute_privileged(env_unwrapped):
    """返回 (num_envs, PRIV_DIM) 的特权向量,逐 env 一行。"""
    d = env_unwrapped.drill.data
    drill_pos = d.root_pos_w - env_unwrapped.scene.env_origins   # (N,3) env-local
    drill_quat = d.root_quat_w                                    # (N,4) wxyz
    drill_lin_vel = d.root_lin_vel_w                              # (N,3)
    drill_ang_vel = d.root_ang_vel_w                              # (N,3)
    contact_forces = env_unwrapped._get_contact_forces_obs()      # (N,13)
    return torch.cat([drill_pos, drill_quat, drill_lin_vel, drill_ang_vel, contact_forces], dim=-1)


def build_grid_override(env_unwrapped, nx, ny, nyaw, pos_range, yaw_range, device):
    """分层网格 + 格内抖动 的电钻初始位姿生成器(= 分层随机 / 连续分辨率)。

    把 ±pos_range / ±yaw_range 切成 nx×ny×nyaw 个等宽格子(× 基础位姿 × goal);每次 reset
    按该 variant 游标轮流取下一格,并在格内做均匀抖动(±半格)。于是:
      - round-robin 保证"分层均匀"(每格都被访问、不留空洞);
      - 格内抖动让落点连续覆盖整段,而不是量化到格点 → 分辨率实际是"连续的",
        格子粗细只决定分层粒度,不决定最终覆盖精度。
    返回的 fn 直接赋给 env._drill_pose_override_fn。
    """
    from isaaclab.utils.math import quat_mul, quat_from_euler_xyz
    VA = env_unwrapped._variant_attrs

    def _centers(half, k):
        # 把 [-half, half] 切成 k 个等宽格,返回 (格心[k], 半格宽);抖动 ±半格 恰好填满该段。
        if k <= 1:
            return torch.tensor([0.0], device=device), float(half)
        step = 2.0 * half / k
        c = -half + (torch.arange(k, device=device) + 0.5) * step
        return c, step / 2.0

    xs, hx = _centers(pos_range[0], nx)
    ys, hy = _centers(pos_range[1], ny)
    yaws, hyaw = _centers(yaw_range, nyaw)
    z = torch.zeros(1, device=device)

    grids = {}   # vid -> list of (base_pos[3], base_rot[4], goal[4], center[3]=(dx,dy,dyaw))
    for vid, vdata in VA.items():
        pos_list = vdata["initial_pos_list"].to(device).float()
        rot_list = vdata["initial_rot_list"].to(device).float()
        goal_list = vdata["goal_rot_list"].to(device).float()
        cells = []
        for bi in range(pos_list.shape[0]):
            for gi in range(goal_list.shape[0]):
                for dx in xs:
                    for dy in ys:
                        for dyaw in yaws:
                            cells.append((pos_list[bi], rot_list[bi], goal_list[gi],
                                          torch.stack([dx, dy, dyaw])))
        grids[vid] = cells
        print(f"[GRID] variant {vid}: {len(cells)} cells "
              f"(bases={pos_list.shape[0]} goals={goal_list.shape[0]} "
              f"x={len(xs)} y={len(ys)} yaw={len(yaws)}); jitter=±({hx:.3f}m,{hy:.3f}m,{hyaw:.3f}rad)",
              flush=True)

    cursors = {vid: 0 for vid in grids}
    halfs = torch.tensor([hx, hy, hyaw], device=device)

    def override(env_ids, variant_ids):
        n = len(env_ids)
        pos = torch.zeros(n, 3); quat = torch.zeros(n, 4); goal = torch.zeros(n, 4)
        for i in range(n):
            vid = int(variant_ids[i].item())
            cells = grids[vid]
            base_pos, base_rot, gl, ctr = cells[cursors[vid] % len(cells)]
            cursors[vid] += 1
            off = ctr + (torch.rand(3, device=device) * 2.0 - 1.0) * halfs   # 格内均匀抖动
            p = base_pos.clone(); p[0] += off[0]; p[1] += off[1]
            yq = quat_from_euler_xyz(z, z, off[2].view(1))[0]                 # 仅 yaw 偏置
            q = quat_mul(yq.unsqueeze(0), base_rot.unsqueeze(0))[0]
            pos[i], quat[i], goal[i] = p.cpu(), q.cpu(), gl.cpu()
        return pos, quat, goal

    return override


def build_sobol_override(env_unwrapped, pos_range, yaw_range, device, seed=0):
    """Sobol 低差异连续采样:密度只由 episodes_per_variant 控制,无需手动分辨率。

    对每个 variant,在 (x,y,yaw) 的 ±范围内用 Sobol 序列均匀铺点(任意 N 的前缀都均匀,
    且 3 个维度联合覆盖——不用人为拆"位置 vs 旋转")。基础位姿 / goal 用 round-robin 轮流,
    保证每个 base 都被均匀覆盖。要采多少由 --episodes_per_variant 决定,采得越多越密。
    """
    from torch.quasirandom import SobolEngine
    from isaaclab.utils.math import quat_mul, quat_from_euler_xyz
    VA = env_unwrapped._variant_attrs
    half = torch.tensor([pos_range[0], pos_range[1], yaw_range], device=device)
    z = torch.zeros(1, device=device)

    state = {}
    for vid, vdata in VA.items():
        state[vid] = dict(
            engine=SobolEngine(dimension=3, scramble=True, seed=seed + int(vid)),
            pos=vdata["initial_pos_list"].to(device).float(),
            rot=vdata["initial_rot_list"].to(device).float(),
            goal=vdata["goal_rot_list"].to(device).float(),
            cur=0,
        )
        print(f"[SOBOL] variant {vid}: bases={state[vid]['pos'].shape[0]} "
              f"goals={state[vid]['goal'].shape[0]} "
              f"range=±({pos_range[0]}m,{pos_range[1]}m,{yaw_range}rad)", flush=True)

    def override(env_ids, variant_ids):
        n = len(env_ids)
        pos = torch.zeros(n, 3); quat = torch.zeros(n, 4); goal = torch.zeros(n, 4)
        for i in range(n):
            vid = int(variant_ids[i].item())
            st = state[vid]
            nb, ng = st["pos"].shape[0], st["goal"].shape[0]
            bi = st["cur"] % nb                 # base 轮流(覆盖均匀)
            gi = (st["cur"] // nb) % ng          # goal 轮流
            st["cur"] += 1
            u = st["engine"].draw(1).to(device)[0]       # (3,) ∈ [0,1)，Sobol 均匀
            off = (u * 2.0 - 1.0) * half                 # 映射到 ±range
            p = st["pos"][bi].clone(); p[0] += off[0]; p[1] += off[1]
            yq = quat_from_euler_xyz(z, z, off[2].view(1))[0]
            q = quat_mul(yq.unsqueeze(0), st["rot"][bi].unsqueeze(0))[0]
            pos[i], quat[i], goal[i] = p.cpu(), q.cpu(), st["goal"][gi].cpu()
        return pos, quat, goal

    return override


def main():
    args = parse_args()
    simulation_app = None

    # --- DEBUG: enable() only installs signal handlers (no extra thread), safe to
    #     do early. The dump_traceback_later watchdog THREAD is armed later, AFTER
    #     all imports, so it can't interfere with any C-extension dlopen. ---
    faulthandler.enable()  # dump on hard crash / segfault (native lib issues)

    if args.headless:
        print("[INFO] Running headless collection with IsaacLab-managed TiledCamera sensors.", flush=True)

    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        return

    if os.path.exists(args.output):
        shutil.rmtree(args.output)

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=args.headless, enable_cameras=True)
    simulation_app = app_launcher.app

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
        # pytorch3d 融合 FPS 内核:在 triton/zarr 之后再加载(C-extension dlopen 顺序安全)。
        # 通过 dp3_pointcloud.init_fps_kernel 设置共享模块内的全局,collect/deploy 选点一致。
        print("[IMP] pytorch3d FPS kernel ...", flush=True)
        if init_fps_kernel("/home/zeyu/3D-Diffusion-Policy/third_party/pytorch3d_simplified"):
            print("[IMP] pytorch3d FPS available (fused CUDA kernel)", flush=True)
        else:
            print("[IMP] pytorch3d FPS NOT available -> python 循环回退", flush=True)
        print("[IMP] ALL imports done", flush=True)

        # Now that all C-extension dlopens are done, it's safe to start the
        # watchdog thread. Covers env creation / timeline.play() / first render.
        faulthandler.dump_traceback_later(60, repeat=True)

        # Match debug_camera_viz.py: start from a clean stage.
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
        cfg.seed = args.seed
        cfg.log_dir = os.path.dirname(os.path.dirname(args.checkpoint))

        # Fix: add rgb to camera data_types so TiledCamera's render product has a color AOV.
        # Without rgb, a depth-only TiledCamera can deadlock on its first frame.
        for _cn in ("cam1", "cam2"):   # cam1=30cm跟随 cam2=固定工作区
            _c = getattr(cfg.scene, _cn, None)
            if _c is not None and "rgb" not in _c.data_types:
                _c.data_types = ["rgb"] + list(_c.data_types)

        base_env = GraspDrillEnv(cfg=cfg, debug=args.debug)
        env_unwrapped = base_env.unwrapped

        from isaaclab.sim.utils import delete_prim
        try:
            delete_prim("/World/Template")
        except Exception as e:
            print(f"[WARN] 删除 /World/Template 失败: {e}", flush=True)

        # --- reset BEFORE timeline.play() (this is the order that works) ---
        print("[INFO] Resetting environment (before play)...", flush=True)
        _ = env_unwrapped.reset()

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

        runner = Runner()
        runner.load(agent_cfg)
        agent = runner.create_player()
        agent.restore(args.checkpoint)
        agent.reset()

        dt = env_unwrapped.step_dt
        cam1 = env_unwrapped.scene.sensors.get("cam1")
        cam2 = env_unwrapped.scene.sensors.get("cam2")
        if cam1 is None or cam2 is None:
            raise RuntimeError(
                f"Expected env-managed cameras 'cam1'/'cam2', got sensors={list(env_unwrapped.scene.sensors.keys())}"
            )
        # cam1=30cm跟随框, cam2=固定工作区(不跟随)

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
        timeline.play()
        sys.stdout.flush()
        if not simulation_app.is_running():
            print("[ERROR] simulation_app.is_running() is False before loop!")
            return

        # --- reset AFTER play: use wrapped env (env), NOT env_unwrapped.
        # env_unwrapped.reset() returns (obs_dict, extras) tuple which rl_games can't parse.
        # RlGamesVecEnvWrapper.reset() returns the format rl_games expects.
        obs = env.reset()
        if isinstance(obs, dict):
            obs = obs.get("obs", obs)
        agent.get_batch_size(obs, 1)
        agent.reset()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(args.device)

        controlled_indices = env_unwrapped.controlled_joint_indices.cpu()
        workspace = tuple(args.workspace)

        # ---------------- target 直接驱动(默认,与 deploy/play 同一真源)----------------
        # 见 perception/target_drive.py。teacher raw → q* → 直接下发,标签恒为 cur_targets(rad)。
        from perception.target_drive import install_direct_drive, raw_to_target, smooth_labels_forward
        _drive_p = install_direct_drive(env_unwrapped)
        print(f"[TARGET_DRIVE] collect 直接 target 驱动:{_drive_p}", flush=True)
        print(f"[LABEL] label_smooth_k={args.label_smooth_k} "
              f"({'前向滑动平均,消 PWM 抖动噪声' if args.label_smooth_k > 1 else '关闭(原始抖动标签)'})",
              flush=True)

        # ---------------- 完整机器人点云 FK 模块(可选)----------------
        robot_fk = None
        robot_pc_M = 0
        if args.save_robot_pc:
            from perception.robot_pointcloud import RobotPointCloudFK
            _npz = args.robot_pc_npz
            if not os.path.isabs(_npz):
                _npz = os.path.join(project_root, _npz)
            _rmax = args.robot_pc_points if (args.robot_pc_points and args.robot_pc_points > 0) else None
            robot_fk = RobotPointCloudFK(_npz, list(env_unwrapped.franka.body_names), args.device,
                                         max_points=_rmax)
            robot_pc_M = robot_fk.num_points
            print(f"[INFO] save_robot_pc=ON -> 机器人 {robot_pc_M} 点合并进 point_cloud, npz={_npz}", flush=True)

        # ---------------- 完整电钻点云(mesh + 真值位姿,可选验证模式)----------------
        # 每个 env 按其 variant 预采 drill_M 个 mesh 点(电钻局部系,已乘 scale),每帧用
        # 电钻真值位姿刚体变换到 env-local —— 等价于"开局完整电钻点云附着在物体上、全程不遮挡"。
        drill_M = 0
        drill_canonical = None
        if args.save_drill_pc:
            from perception.robot_pointcloud import _quat_to_mat   # 复用 wxyz 四元数->旋转矩阵
            drill_M = args.drill_pc_points
            _va = env_unwrapped._variant_attrs
            _vids = env_unwrapped._drill_variant_indices  # (num_envs,)
            _var_canon = {}
            for _vid, _vdata in _va.items():
                _mp = _vdata.get("mesh_points")           # (N,3) 电钻局部系 or None
                _scale = _vdata["scale"]                  # (3,)
                if _mp is None or len(_mp) == 0:
                    _pts = torch.zeros(drill_M, 3, device=args.device)
                else:
                    _N = _mp.shape[0]
                    if _N >= drill_M:
                        _idx = torch.randperm(_N, device=_mp.device)[:drill_M]
                    else:
                        _idx = torch.randint(0, _N, (drill_M,), device=_mp.device)
                    _pts = (_mp[_idx] * _scale).to(args.device)
                _var_canon[int(_vid)] = _pts
            drill_canonical = torch.stack(
                [_var_canon[int(v.item())] for v in _vids], dim=0)  # (num_envs, drill_M, 3)
            print(f"[INFO] save_drill_pc=ON -> 电钻 {drill_M} 点(mesh+真值位姿)合并进 point_cloud", flush=True)

        # ---------------- Incremental Zarr setup ----------------
        # ---------------- 合成地面/桌面点云(默认开,补深度空洞)----------------
        from perception.robot_pointcloud import make_ground_points, jitter_ground_xy
        _PERC = _load_perception_hp()
        ground_M = _PERC.ground_points
        _ground_pts = make_ground_points(workspace, ground_M, _PERC.ground_z, args.device)
        ground_batch = _ground_pts.unsqueeze(0).expand(args.num_envs, -1, -1)  # (B,G,3) 网格基准
        _ground_xy_std = float(getattr(_PERC, "ground_xy_noise_std", 0.0))     # xy 每帧高斯抖动 σ
        print(f"[INFO] 合成地面:{ground_M} 点 @ z={_PERC.ground_z}(env-local), "
              f"xy 每帧高斯抖动 σ={_ground_xy_std}m(clamp 回区域),默认合并进 point_cloud", flush=True)
        _cam_follow = _PERC.camera_follow_drill          # 相机裁剪跟随电钻(oracle:真值位姿)
        _drill_crop_half = _PERC.drill_crop_half
        if _cam_follow:
            print(f"[INFO] 相机裁剪跟随电钻:中心=真值位姿,{2*_drill_crop_half*100:.0f}cm 立方体,"
                  f"框底 z>={workspace[4]}(不采桌面,地面交给合成点云),oracle 仅sim", flush=True)

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        # zarr v3 requires zarr_format=2 to accept numcodecs.Blosc as compressor.
        # v3 native codecs don't support incremental resize well, so we use v2 format.
        blosc_compressor = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=1)
        zarr_root = zarr.group(args.output, zarr_format=2)
        zarr_data = zarr_root.create_group("data")
        zarr_meta = zarr_root.create_group("meta")

        # point_cloud = 相机(pc_num_points = cam1 30cm跟随 1024 + cam2 工作区 1024) [+机器人 robot_pc_M] [+电钻 drill_M]。
        total_pc = (args.pc_num_points
                    + (robot_pc_M if args.save_robot_pc else 0)
                    + (drill_M if args.save_drill_pc else 0)
                    + ground_M)
        print(f"[INFO] pc_mode={args.pc_mode} -> total_pc={total_pc} "
              f"(camera={args.pc_num_points} + robot={robot_pc_M if args.save_robot_pc else 0} "
              f"+ drill={drill_M if args.save_drill_pc else 0} + ground={ground_M}) "
              f"| 训练 config point_cloud.shape 必须设成 [{total_pc}, 3]", flush=True)
        # chunk 帧维 = DP3 horizon(16):随机读一个 horizon 窗口最多跨 2 个 chunk,
        # 把随机读放大从 ~6-12x(原 100 帧/chunk)降到 ~2x。point_cloud@16帧≈732KB/chunk(zarr 理想区间)。
        _CHUNK_FRAMES = 16
        from perception.student_obs import build_agent_pos, agent_pos_dim, POS_DIM
        _state_dim = agent_pos_dim(args.force_state)   # 13 或 26
        print(f"[STATE] force_state={args.force_state} -> state dim={_state_dim} "
              f"({'pos13+contact13' if args.force_state else 'pos13'})", flush=True)
        zarr_data.create_dataset("state", shape=(0, _state_dim), dtype="float32",
                                 compressor=blosc_compressor, chunks=(_CHUNK_FRAMES, _state_dim), overwrite=True)
        zarr_data.create_dataset("action", shape=(0, 13), dtype="float32",
                                 compressor=blosc_compressor, chunks=(_CHUNK_FRAMES, 13), overwrite=True)
        zarr_data.create_dataset("point_cloud", shape=(0, total_pc, 3), dtype=args.pc_dtype,
                                 compressor=blosc_compressor, chunks=(_CHUNK_FRAMES, total_pc, 3), overwrite=True)
        print(f"[INFO] point_cloud 存储精度: {args.pc_dtype}"
              f"{'(fp16, 训练端 astype(float32) 自动转回)' if args.pc_dtype == 'float16' else ''}", flush=True)
        if args.save_privileged:
            zarr_data.create_dataset("privileged", shape=(0, PRIV_DIM), dtype="float32",
                                     compressor=blosc_compressor, chunks=(_CHUNK_FRAMES, PRIV_DIM), overwrite=True)
            print(f"[INFO] save_privileged=ON -> data/privileged dim={PRIV_DIM}", flush=True)
        zarr_meta.create_dataset("episode_ends", shape=(0,), dtype="int64",
                                 compressor=blosc_compressor, chunks=(100,), overwrite=True)

        episode_ends = []

        # 每个 env 的电钻类型由 _reset_idx 里 env_ids % num_drill_variants 决定，固定不变
        active_vids = sorted(v.variant_index for v in env_unwrapped.drill_variants)
        collected_per_variant = {vid: 0 for vid in active_vids}
        if args.episodes_per_variant is not None:
            args.num_episodes = args.episodes_per_variant * len(active_vids)
            print(f"[INFO] Per-variant quota: {args.episodes_per_variant} x "
                  f"{len(active_vids)} variants {active_vids} = {args.num_episodes} episodes",
                  flush=True)

        # ---- 位姿覆盖采集:Sobol 连续采样(只看 episodes_per_variant)或确定性网格 ----
        if args.cover_sobol or args.cover_grid:
            _rcfg = getattr(env_unwrapped.cfg, "randomization", None)
            _pr = (tuple(args.grid_pos_range) if args.grid_pos_range is not None
                   else getattr(_rcfg, "drill_pos_random_range", (0.1, 0.1, 0.0))[:2])
            _yr = (args.grid_yaw_range if args.grid_yaw_range is not None
                   else getattr(_rcfg, "drill_rot_random_range", (0.0, 0.0, 2.0))[2])
            if args.cover_sobol:
                env_unwrapped._drill_pose_override_fn = build_sobol_override(
                    env_unwrapped, _pr, _yr, args.device, seed=args.seed)
                print(f"[SOBOL] cover_sobol ON: pos_range=±{_pr} yaw_range=±{_yr} "
                      f"(密度仅由 episodes_per_variant={args.episodes_per_variant} 决定,无需手动分辨率;"
                      f"仅成功局入库)", flush=True)
            else:
                env_unwrapped._drill_pose_override_fn = build_grid_override(
                    env_unwrapped, args.grid_nx, args.grid_ny, args.grid_nyaw, _pr, _yr, args.device)
                print(f"[GRID] cover_grid ON: pos_range=±{_pr} yaw_range=±{_yr} "
                      f"nx={args.grid_nx} ny={args.grid_ny} nyaw={args.grid_nyaw} "
                      f"(round-robin 均匀覆盖,仅成功局入库)", flush=True)

        # ---------------- collection state ----------------
        buffers = [[] for _ in range(args.num_envs)]      # list of (state, action, pc) per env
        env_started = np.zeros(args.num_envs, dtype=bool)  # whether env is currently recording
        total_collected = 0
        total_steps = 0
        total_zero_pc = 0
        n_dropped_abnormal = 0   # 因 state 超过 --state_abs_limit 被丢弃的成功 episode 数
        print_interval = 100
        start_time = time.time()

        # ---- overfit 测试:逐 episode 记录电钻初始位姿(env-local) ----
        # read_init_pose 在每局开始(env 刚 reset)时读 env 写好的初始位姿;buf_init_pose 保存
        # "当前正在录制的这一局"的位姿,episode 真正写入 zarr 时(flush_episode 内)才追加到
        # flushed_init_poses,保证与 zarr episode 顺序严格 1:1。
        flushed_init_poses = []   # list of (vid, pos_local[3], quat[4], goal[4]),仅 --save_init_poses
        def read_init_pose(e):
            eo = env_unwrapped.scene.env_origins[e]
            pos_local = (env_unwrapped.initial_drill_pos[e] - eo).detach().cpu().numpy().astype(np.float32)
            quat = env_unwrapped.drill_initial_rot_tensor[e].detach().cpu().numpy().astype(np.float32)
            goal = env_unwrapped._drill_goal_quat_per_env[e].detach().cpu().numpy().astype(np.float32)
            vid = int(env_unwrapped._drill_variant_indices[e].item())
            return (vid, pos_local, quat, goal)
        buf_init_pose = [read_init_pose(e) for e in range(args.num_envs)] if args.save_init_poses else None

        def flush_episode(states, actions, pcs, privs, vid, init_pose=None):
            nonlocal total_collected, n_dropped_abnormal
            n = len(states)
            if n == 0:
                return
            if (args.episodes_per_variant is not None
                    and collected_per_variant[vid] >= args.episodes_per_variant):
                return  # 该种电钻已收满，丢弃多余 episode
            states_arr = np.stack(states).astype(np.float32)
            # 丢弃异常 episode:只检查位置维(前 POS_DIM),速度/接触力量纲不同不参与判定
            ep_max_abs = float(np.abs(states_arr[:, :POS_DIM]).max())
            if ep_max_abs > args.state_abs_limit:
                n_dropped_abnormal += 1
                print(f"  [DROP] episode (len={n}): state max|val|={ep_max_abs:.1f} "
                      f"> {args.state_abs_limit}, discarded", flush=True)
                return
            actions_arr = np.stack(actions).astype(np.float32)
            if args.label_smooth_k > 1:   # 标签平滑(单一真源,见 target_drive)
                actions_arr = smooth_labels_forward(actions_arr, args.label_smooth_k)
            pcs_arr = np.stack(pcs).astype(np.float32)   # 已是 camera+robot 合并后的 total_pc 点
            idx = zarr_data["state"].shape[0]
            zarr_data["state"].resize((idx + n, _state_dim))
            zarr_data["action"].resize((idx + n, 13))
            zarr_data["point_cloud"].resize((idx + n, total_pc, 3))
            zarr_data["state"][idx:idx + n] = states_arr
            zarr_data["action"][idx:idx + n] = actions_arr
            zarr_data["point_cloud"][idx:idx + n] = pcs_arr
            if args.save_privileged:
                privs_arr = np.stack(privs).astype(np.float32)
                zarr_data["privileged"].resize((idx + n, PRIV_DIM))
                zarr_data["privileged"][idx:idx + n] = privs_arr
            episode_ends.append(idx + n)
            ends = np.array(episode_ends, dtype=np.int64)
            zarr_meta["episode_ends"].resize((len(ends),))
            zarr_meta["episode_ends"][:] = ends
            collected_per_variant[vid] += 1
            total_collected += 1
            if args.save_init_poses and init_pose is not None:
                flushed_init_poses.append(init_pose)   # 与 zarr episode 顺序 1:1

        # init done -> stop the auto-dump so it doesn't spam during the long rollout.
        faulthandler.cancel_dump_traceback_later()
        print(f"Collecting {args.num_episodes} episodes (num_envs={args.num_envs}, "
              f"drive=direct-target)...", flush=True)

        try:
            with torch.inference_mode():
                while total_collected < args.num_episodes and simulation_app.is_running():
                    # 1) policy action
                    obs_tensor = agent.obs_to_torch(obs)
                    teacher_raw = agent.get_action(obs_tensor, is_deterministic=True)
                    if not isinstance(teacher_raw, torch.Tensor):
                        teacher_raw = torch.from_numpy(np.array(teacher_raw)).float().to(args.device)
                    actions = teacher_raw   # teacher 开车

                    # 1b) target 直接驱动:teacher raw → q*,存到 env,patched _apply_action 直接下发
                    cur0 = env_unwrapped.cur_targets.clone()
                    env_unwrapped._direct_target = raw_to_target(actions, cur0, _drive_p)

                    # 2) step: use env_unwrapped.step() to avoid deadlock from sim.render().
                    # env.step() internally calls sim.render() which deadlocks with the carb run loop.
                    # env_unwrapped.step() returns (obs_dict, rewards, terminated, truncated, extras).
                    obs_dict, rewards, terminated, truncated, extras = env_unwrapped.step(actions)
                    total_steps += 1
                    cam1.update(dt)
                    cam2.update(dt)

                    # Convert obs to rl_games format for agent (mirrors env._process_obs)
                    obs = env._process_obs(obs_dict)

                    # 3) state = agent_pos:13(仅位置)或 39(位置+速度+接触力),单一真源 student_obs
                    state_vec = build_agent_pos(env_unwrapped, args.force_state)

                    # 4) depth -> fused camera point cloud(单一真源 build_fused_camera_pc):
                    #    裁剪框由 _PERC.camera_follow_drill 决定(当前=False -> 两相机都用固定
                    #    workspace,不跟随电钻),各采 pc_num_points//2 点。逻辑全在 dp3_pointcloud.py。
                    pc_fused, zmask1, zmask2, _, _ = build_fused_camera_pc(
                        cam1, cam2, env_unwrapped.drill.data.root_pos_w,
                        env_unwrapped.scene.env_origins, _PERC, args.pc_num_points, workspace)
                    total_zero_pc += int(zmask1.sum().item() + zmask2.sum().item())

                    # 5b) 完整机器人点云(仅 --save_robot_pc):FK 从 body 世界位姿算 → 直接合并进
                    # point_cloud。body_pos_w 是含基座的绝对世界位姿,基座位置自动正确;减 env_origin
                    # 后与相机点云同系。两个点云合成一个:(camera 前段 | robot 后段)。
                    if args.save_robot_pc:
                        robot_pc = robot_fk(env_unwrapped.franka.data.body_pos_w,
                                            env_unwrapped.franka.data.body_quat_w,
                                            env_unwrapped.scene.env_origins)        # (B,M,3)
                        pc_fused = torch.cat([pc_fused, robot_pc], dim=1)           # (B, +robot)

                    # 5c) 完整电钻点云(仅 --save_drill_pc):mesh canonical 用电钻真值位姿刚体
                    # 变换到 env-local（= 开局完整电钻云"附着"在物体上,抓取全程不受遮挡）。
                    if args.save_drill_pc:
                        _dq = env_unwrapped.drill.data.root_quat_w           # (B,4) wxyz
                        _dp = env_unwrapped.drill.data.root_pos_w            # (B,3)
                        _Rd = _quat_to_mat(_dq)                             # (B,3,3)
                        _drill_world = torch.einsum('bij,bnj->bni', _Rd, drill_canonical) + _dp.unsqueeze(1)
                        _drill_pc = _drill_world - env_unwrapped.scene.env_origins.unsqueeze(1)
                        pc_fused = torch.cat([pc_fused, _drill_pc], dim=1)  # (B, +drill)

                    # 5d) 合成地面点云(默认开):接最后 -> [camera | robot | drill | ground]
                    # 地面 xy 每帧高斯抖动(z 不变,clamp 回区域),打破网格规整;deploy 同分布。
                    _ground_jit = jitter_ground_xy(ground_batch, _ground_xy_std,
                                                   workspace[0], workspace[1], workspace[2], workspace[3])
                    pc_fused = torch.cat([pc_fused, _ground_jit], dim=1)   # (B, +ground)

                    # 5) move everything to CPU once
                    # 动作标签 = step() 后的 cur_targets(= 本步直接下发的 q*,rad)。
                    actions_np = env_unwrapped.cur_targets.detach().cpu().numpy()
                    state_np = state_vec.detach().cpu().numpy()
                    pc_np = pc_fused.detach().cpu().numpy()    # camera + robot 合并后的 total_pc 点
                    # 特权信息(仅 --save_privileged):和 state 一样 step 后读,逐帧对齐
                    priv_np = (compute_privileged(env_unwrapped).detach().cpu().numpy()
                               if args.save_privileged else None)

                    # 首帧 FK 自检:机器人段(pc 后段)到最近相机段(pc 前段)点的距离。可见部分
                    # (p10)应在 cm 级;若基座/坐标系错了会普遍很大(几十 cm)。
                    if args.save_robot_pc and total_steps == 1:
                        _base = env_unwrapped.franka.data.body_pos_w[0, 0].detach().cpu().numpy()
                        _r = pc_np[0, args.pc_num_points:args.pc_num_points + robot_pc_M]  # 机器人段
                        _c = pc_np[0, :args.pc_num_points]        # 相机段
                        _cc = _c[(np.abs(_c).sum(1) > 1e-6)]      # 去掉占位 0 点
                        if len(_cc) > 0:
                            _d = np.linalg.norm(_r[:, None, :] - _cc[None, :, :], axis=-1).min(1)
                            print(f"[FK-CHECK] base(world)={np.round(_base,3)} | "
                                  f"robot 段 env-local bbox "
                                  f"x[{_r[:,0].min():.2f},{_r[:,0].max():.2f}] "
                                  f"y[{_r[:,1].min():.2f},{_r[:,1].max():.2f}] "
                                  f"z[{_r[:,2].min():.2f},{_r[:,2].max():.2f}] | "
                                  f"NN->camera p10={np.percentile(_d,10)*100:.1f}cm "
                                  f"median={np.median(_d)*100:.1f}cm "
                                  f"(p10 应为 cm 级,否则基座/坐标系有误)", flush=True)

                    # 首帧电钻自检:开局电钻可见,mesh 完整电钻云应贴合相机看到的电钻点。
                    if args.save_drill_pc and total_steps == 1:
                        _ds = pc_np[0, args.pc_num_points + robot_pc_M:args.pc_num_points + robot_pc_M + drill_M]   # 电钻段(末尾是 ground,要切到 drill_M 为止)
                        _c = pc_np[0, :args.pc_num_points]
                        _cc = _c[(np.abs(_c).sum(1) > 1e-6)]
                        if len(_cc) > 0:
                            _dd = np.linalg.norm(_ds[:, None, :] - _cc[None, :, :], axis=-1).min(1)
                            print(f"[DRILL-CHECK] 电钻段 NN->camera p50={np.percentile(_dd,50)*100:.1f}cm "
                                  f"p90={np.percentile(_dd,90)*100:.1f}cm "
                                  f"(开局电钻可见,应大多在 cm 级;过大说明 mesh/scale/位姿不一致)", flush=True)

                    # is_done = terminated | truncated (both available directly from env_unwrapped.step)
                    is_done = terminated.bool() | truncated.bool()
                    try:
                        lenient_success = env_unwrapped._cached_lenient_success
                    except AttributeError:
                        lenient_success = torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)
                    is_done_np = is_done.detach().cpu().numpy()
                    success_np = lenient_success.detach().cpu().numpy().astype(bool)
                    # per-env "is this frame's pc empty"(cam1 跟随框 或 cam2 工作区 任一空)
                    bad_pc_np = (zmask1 | zmask2).detach().cpu().numpy()

                    # 6) buffer current frame for active, non-terminating envs.
                    #    Skip the terminating step: env auto-resets inside step(), so the
                    #    state/pc just read already belong to the NEXT episode.
                    #    并跳过相机空帧(reset 后渲染滞后,跟随裁剪框里无有效点)——只在每集
                    #    开头出现,跳过=这集从第 2 帧起,时序连续,无副作用。
                    for env_id in range(args.num_envs):
                        if not env_started[env_id] or is_done_np[env_id]:
                            continue
                        if bad_pc_np[env_id]:        # 相机空(reset 渲染滞后)的帧不存
                            continue
                        buffers[env_id].append((state_np[env_id], actions_np[env_id], pc_np[env_id],
                                                None if priv_np is None else priv_np[env_id]))

                    # 7) episode boundaries
                    any_done = False
                    for env_id in range(args.num_envs):
                        if is_done_np[env_id]:
                            any_done = True
                            if env_started[env_id] and buffers[env_id] and success_np[env_id]:
                                vid = int(env_unwrapped._drill_variant_indices[env_id].item())
                                states, acts, pcs, privs = zip(*buffers[env_id])
                                flush_episode(list(states), list(acts), list(pcs), list(privs), vid,
                                              init_pose=(buf_init_pose[env_id] if args.save_init_poses else None))
                            buffers[env_id] = []
                            env_started[env_id] = True  # fresh episode begins now
                            if args.save_init_poses:
                                # env 已在本 step 内 auto-reset,读新一局的初始位姿留给下次 flush
                                buf_init_pose[env_id] = read_init_pose(env_id)
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
                        per_var = " ".join(f"v{vid}={n}" for vid, n in collected_per_variant.items())
                        print(f"  [{bar}] {total_collected}/{args.num_episodes} ({per_var}) | {total_steps}stp | "
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
        per_var = " ".join(f"v{vid}={n}" for vid, n in collected_per_variant.items())
        print(f"\nDone: collected={total_collected} episodes ({per_var}), steps={total_steps}, "
              f"dropped_abnormal={n_dropped_abnormal}, "
              f"zero_pc={total_zero_pc}, peak_gpu={gpu_mem_gb:.1f}GB, time={h}h{m}m{s}s", flush=True)
        print(f"Output: {args.output}", flush=True)

        # ---- overfit 测试:落盘电钻初始位姿(与 zarr episode 顺序 1:1)----
        if args.save_init_poses and len(flushed_init_poses) > 0:
            _pose_dir = args.output.rstrip("/") + "_init_poses"
            os.makedirs(_pose_dir, exist_ok=True)
            _vid = np.array([p[0] for p in flushed_init_poses], dtype=np.int64)
            _pos = np.stack([p[1] for p in flushed_init_poses]).astype(np.float32)
            _quat = np.stack([p[2] for p in flushed_init_poses]).astype(np.float32)
            _goal = np.stack([p[3] for p in flushed_init_poses]).astype(np.float32)
            _pose_file = os.path.join(_pose_dir, "init_poses.npz")
            np.savez(_pose_file, variant=_vid, pos_local=_pos, quat=_quat, goal_quat=_goal)
            print(f"[INFO] Saved {len(_vid)} episode init-poses -> {_pose_file}", flush=True)

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