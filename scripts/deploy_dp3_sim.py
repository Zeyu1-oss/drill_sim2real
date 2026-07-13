import argparse
import os
import sys
import time
import math

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (scripts/ -> ../..)
sys.path.insert(0, project_root)

import numpy as np
import torch

# Neutralize torch.compile BEFORE rl_games is imported.
def _safe_compile(model=None, *args, **kwargs):
    if callable(model):
        return model
    return lambda fn: fn
torch.compile = _safe_compile

DP3_ROOT = "/home/zeyu/3D-Diffusion-Policy/3D-Diffusion-Policy"
sys.path.insert(0, DP3_ROOT)
from perception.dp3_pointcloud import (depth_to_pointcloud_batch, build_fused_camera_pc,
                           camera_pc, camera_crop_bounds, init_fps_kernel)
init_fps_kernel("/home/zeyu/3D-Diffusion-Policy/third_party/pytorch3d_simplified")


_PERCEPTION_HP = None
def _load_perception_hp():
    """按文件路径加载 hyperparameters.py 取 PerceptionHyperparameters,绕过 tasks 包
    __init__(它 import grasp_drill_env 需要 IsaacLab)。与 collect 同一真源。"""
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
    _PERC = _load_perception_hp() 
    parser = argparse.ArgumentParser(description="DP3 仿真部署与性能验证")
    parser.add_argument("--dp3_ckpt", type=str,
                        default="/home/zeyu/3D-Diffusion-Policy/3D-Diffusion-Policy/data/outputs/inspire_drill-simple_dp3-simple_dp3_seed0/checkpoints/latest.ckpt",
                        help="DP3 checkpoint (.ckpt)")
    parser.add_argument("--data_path", type=str, default=None,
                        help="可选兜底:仅当 checkpoint 未内嵌 normalizer 时,用此 zarr 重算 "
                             "normalizer(须与该 checkpoint 的训练数据一致)。"
                             "正常 ckpt 自带 normalizer,无需提供本参数。")
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--num_episodes", type=int, default=1000,
                        help="评估的总 episode 数（达到即停止）")
    parser.add_argument("--num_inference_steps", type=int, default=None,
                        help="DDIM inference steps（默认从 checkpoint 读取）")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--drill_configs", type=str, default=None)
    parser.add_argument("--img_height", type=int, default=_PERC.img_height,
                        help="相机分辨率,默认来自 PerceptionHyperparameters(单一真源,与 collect 一致)")
    parser.add_argument("--img_width", type=int, default=_PERC.img_width)
    parser.add_argument("--pc_num_points", type=int, default=1024,
                        help="相机段点数=两相机合计(cam1/cam2 各 512;须与 collect 一致,默认 1024)")
    parser.add_argument("--robot_pc_points", type=int, default=160,
                        help="机器人 FK 点云降采样点数(须与 collect 一致;None/负数=不降)")
    parser.add_argument("--policy_pc_points", type=int, default=2048,
                        help="DP3 policy 期望的每帧点云数量（必须与训练数据一致）")
    parser.add_argument("--workspace", nargs=6, type=float,
                        default=list(_PERC.workspace),
                        help="[x_min..z_max] env-local (m);默认来自 hyperparameters.PerceptionHyperparameters.workspace(单一真源)")
    parser.add_argument("--save_traj", type=str, default=None,
                        help="保存每条轨迹的 state/action/reward 到 npz 文件")
    parser.add_argument("--pc_mode", type=str, default="robot",
                        choices=["camera", "robot", "robot_drill"],
                        help="点云组成(须与训练数据一致):camera / robot / robot_drill")
    parser.add_argument("--robot_pc_npz", type=str,
                        default="assets/inspire_tac/robot_canonical_points.npz",
                        help="FK 机器人 canonical 点(与 collect 用同一个文件)")
    parser.add_argument("--drill_pc_points", type=int, default=512,
                        help="电钻完整云点数(须与 collect --drill_pc_points 一致)")
    parser.add_argument("--init_pose_file", type=str, default=None,
                        help="overfit 测试用:加载 collect --save_init_poses 产出的 init_poses.npz,"
                             "每次 reset 按 variant 回放记录的电钻初始位姿,验证 DP3 能否在相同场景复刻。")
    parser.add_argument("--exec_horizon", type=int, default=None,
                        help="缓解开环累积误差;须 1<=exec_horizon<=n_action_steps。")
    parser.add_argument("--collect", action="store_true",
                        help="开启采集模式:DP3 开车跑 rollout,把轨迹录成 zarr(schema 同 collect_dp3_data.py),"
                             "用于 DAgger/recovery 数据。")
    parser.add_argument("--collect_output", type=str,
                        default="data/inspire_drill_dp3_deploy_collect.zarr",
                        help="采集模式输出 zarr 路径。")
    parser.add_argument("--collect_episodes", type=int, default=100,
                        help="采集模式:录满多少个 episode 即停止(默认 100)。")
    parser.add_argument("--collect_filter", type=str, default="all",
                        choices=["all", "success", "failure"],
                        help="采集过滤:all=成败都录 | success=只录成功(同 collect_dp3_data.py) | "
                             "failure=只录失败(专门抓 DP3 失败模式/recovery 场景)。")
    parser.add_argument("--collect_label_smooth_k", type=int, default=1,
                        help="采集模式动作标签前向滑动平均窗口,1=关闭(默认,与 collect 一致)。"
                             "注:当前 DeployCollector 未使用该参数,deploy 采集本就不平滑。")
    parser.add_argument("--collect_success_only", action="store_true",
                        help="[旧参数兼容] 等价于 --collect_filter success。")
    args = parser.parse_args()
    if args.collect_success_only:
        args.collect_filter = "success"
    args.save_robot_pc = args.pc_mode in ("robot", "robot_drill")
    args.save_drill_pc = args.pc_mode == "robot_drill"
    return args



def compute_normalizer_from_zarr(data_path: str, device: str):

    import dill
    _cache_path = data_path.rstrip("/") + ".normalizer.pkl"
    _zarray = os.path.join(data_path, "data", "point_cloud", ".zarray")
    _src_mtime = os.path.getmtime(_zarray) if os.path.exists(_zarray) else None
    if os.path.exists(_cache_path):
        try:
            with open(_cache_path, "rb") as _f:
                _blob = dill.load(_f)
            if _blob.get("src_mtime") == _src_mtime:
                print(f"[Normalizer] Loaded from cache (skipped zarr scan): {_cache_path}")
                return _blob["normalizer"]
            if _src_mtime is None:
                # 源 zarr 缺失/不完整(常见:磁盘清理删了点云数组),但缓存的 normalizer 仍有效。
                # 无法重算,直接用缓存——这正是该数据集当初算出的 normalizer。
                print(f"[Normalizer] source zarr missing/incomplete; using cached normalizer: {_cache_path}", flush=True)
                return _blob["normalizer"]
            print("[Normalizer] cache stale (zarr changed), recomputing", flush=True)
        except Exception as _e:
            print(f"[Normalizer] cache load failed ({_e}), recomputing", flush=True)

    import zarr
    from diffusion_policy_3d.model.common.normalizer import (
        LinearNormalizer, SingleFieldLinearNormalizer)

    z = zarr.open_group(data_path)
    state_data = z['data']['state'][:]           
    action_data = z['data']['action'][:]         
    pc = z['data']['point_cloud']              

    normalizer = LinearNormalizer()
    normalizer['action'] = SingleFieldLinearNormalizer.create_fit(
        action_data, last_n_dims=1, mode='limits')
    normalizer['agent_pos'] = SingleFieldLinearNormalizer.create_fit(
        state_data, last_n_dims=1, mode='limits')

    # point_cloud 分块流式:逐 xyz 维 min/max/mean/std
    T = int(pc.shape[0]); D = int(pc.shape[-1]); block = 4096
    run_min = run_max = None
    s = torch.zeros(D, dtype=torch.float64)
    ss = torch.zeros(D, dtype=torch.float64)
    n = 0
    for start in range(0, T, block):
        t = torch.from_numpy(np.asarray(pc[start:start + block], dtype=np.float32)).reshape(-1, D)
        bmin = t.min(dim=0).values; bmax = t.max(dim=0).values
        run_min = bmin if run_min is None else torch.minimum(run_min, bmin)
        run_max = bmax if run_max is None else torch.maximum(run_max, bmax)
        td = t.double(); s += td.sum(dim=0); ss += (td * td).sum(dim=0); n += t.shape[0]
    mean = (s / max(n, 1)).float()
    std = ((ss / max(n, 1)).float() - mean * mean).clamp_min(0).sqrt()
    omin, omax, eps = -1.0, 1.0, 1e-4
    rng = (run_max - run_min).clone(); ignore = rng < eps; rng[ignore] = omax - omin
    scale = (omax - omin) / rng
    offset = omin - scale * run_min
    offset[ignore] = (omax + omin) / 2 - run_min[ignore]
    normalizer['point_cloud'] = SingleFieldLinearNormalizer.create_manual(
        scale, offset, {'min': run_min, 'max': run_max, 'mean': mean, 'std': std})

    print(f"  action  range: {action_data.min():.3f} ~ {action_data.max():.3f}")
    print(f"  agent_pos range: {state_data.min():.3f} ~ {state_data.max():.3f}")
    print(f"  point_cloud range: {run_min.min():.3f} ~ {run_max.max():.3f}")

    try:
        with open(_cache_path, "wb") as _f:
            dill.dump({"src_mtime": _src_mtime, "normalizer": normalizer}, _f)
        print(f"[Normalizer] Cached for next run: {_cache_path}")
    except Exception as _e:
        print(f"[Normalizer] cache save failed (non-fatal): {_e}", flush=True)
    return normalizer


class DeployCollector:

    _CHUNK_FRAMES = 16

    def __init__(self, output, num_envs, total_pc, filter_mode, state_dim=13):
        import shutil
        import zarr
        import numcodecs
        self.total_pc = total_pc
        self.state_dim = state_dim        
        self.filter_mode = filter_mode    
        self.output = output
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        # zarr v3(env_isaaclab)下 zarr.group() 不覆盖已存在的 store,create_group 会抛
        # ContainsGroupError(常见:上一次 Ctrl-C 留下半截 zarr)。直接整体清掉再重建。
        if os.path.exists(output):
            print(f"[COLLECT] 输出已存在,覆盖: {output}", flush=True)
            shutil.rmtree(output) if os.path.isdir(output) else os.remove(output)
        comp = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=1)
        root = zarr.group(output, zarr_format=2)
        self.data = root.create_group("data")
        self.meta = root.create_group("meta")
        cf = self._CHUNK_FRAMES
        self.data.create_dataset("state", shape=(0, state_dim), dtype="float32",
                                 compressor=comp, chunks=(cf, state_dim), overwrite=True)
        self.data.create_dataset("action", shape=(0, 13), dtype="float32",
                                 compressor=comp, chunks=(cf, 13), overwrite=True)
        self.data.create_dataset("point_cloud", shape=(0, total_pc, 3), dtype="float32",
                                 compressor=comp, chunks=(cf, total_pc, 3), overwrite=True)
        self.meta.create_dataset("episode_ends", shape=(0,), dtype="int64",
                                 compressor=comp, chunks=(100,), overwrite=True)
        self.episode_ends = []
        self.buffers = [[] for _ in range(num_envs)]
        self.started = np.zeros(num_envs, dtype=bool)   # 与 collect 的 env_started 同义:跳过开局半截
        self.collected = 0

    def record_frame(self, env_id, state_np, action_np, pc_np):
        self.buffers[env_id].append((state_np, action_np, pc_np))

    def _keep(self, success):
        if self.filter_mode == "success":
            return success
        if self.filter_mode == "failure":
            return not success
        return True   # all

    def flush(self, env_id, success):
        """一个 episode 结束:满足过滤条件则写盘,然后清 buffer。"""
        buf = self.buffers[env_id]
        if buf and self._keep(success):
            states, actions, pcs = zip(*buf)
            states_arr = np.stack(states).astype(np.float32)
            actions_arr = np.stack(actions).astype(np.float32)
            pcs_arr = np.stack(pcs).astype(np.float32)
            n = len(buf)
            idx = self.data["state"].shape[0]
            self.data["state"].resize((idx + n, self.state_dim))
            self.data["action"].resize((idx + n, 13))
            self.data["point_cloud"].resize((idx + n, self.total_pc, 3))
            self.data["state"][idx:idx + n] = states_arr
            self.data["action"][idx:idx + n] = actions_arr
            self.data["point_cloud"][idx:idx + n] = pcs_arr
            self.episode_ends.append(idx + n)
            ends = np.array(self.episode_ends, dtype=np.int64)
            self.meta["episode_ends"].resize((len(ends),))
            self.meta["episode_ends"][:] = ends
            self.collected += 1
        self.buffers[env_id] = []


def main():
    args = parse_args()

    if not os.path.exists(args.dp3_ckpt):
        print(f"[ERROR] DP3 checkpoint not found: {args.dp3_ckpt}")
        return

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=args.headless, enable_cameras=True)
    simulation_app = app_launcher.app

    try:
        import carb
        import omni.timeline
        import omni.usd
        from tasks.grasp_drill_env import GraspDrillEnv, create_grasp_drill_env_cfg

        # Import rl_games AFTER torch.compile is neutralized (see top of file).
        from rl_games.common import env_configurations, vecenv
        from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

        # 从干净 stage 开始（与采集脚本一致）
        omni.usd.get_context().new_stage()

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

        for _cn in ("cam1", "cam2"):  
            _c = getattr(cfg.scene, _cn, None)
            if _c is not None and "rgb" not in _c.data_types:
                _c.data_types = ["rgb"] + list(_c.data_types)

        base_env = GraspDrillEnv(cfg=cfg, debug=args.debug)
        env_unwrapped = base_env.unwrapped

        if args.init_pose_file is not None:
            _ip = np.load(args.init_pose_file)
            _vids = _ip["variant"]; _pos = _ip["pos_local"]; _quat = _ip["quat"]; _goal = _ip["goal_quat"]
            _by_variant = {}
            for _k in np.unique(_vids):
                _m = _vids == _k
                _by_variant[int(_k)] = (
                    torch.from_numpy(_pos[_m]).float(),
                    torch.from_numpy(_quat[_m]).float(),
                    torch.from_numpy(_goal[_m]).float())
            # 每个 variant 从自己保存的位姿里【随机】抽一条(不再按保存顺序游标读)。
            # 用 args.seed 固定的独立 RNG,保证随机但可复现(同 seed 同序列)。
            _ip_rng = np.random.default_rng(args.seed)
            print(f"[INFO] init-pose replay (RANDOM per-variant, seed={args.seed}): "
                  f"{len(_vids)} poses, per-variant counts="
                  f"{ {k: v[0].shape[0] for k, v in _by_variant.items()} }", flush=True)

            def _drill_pose_override_fn(env_ids, variant_ids):
                n = len(env_ids)
                pos = torch.zeros(n, 3); quat = torch.zeros(n, 4); goal = torch.zeros(n, 4)
                for i in range(n):
                    vid = int(variant_ids[i].item())
                    if vid not in _by_variant:
                        raise RuntimeError(f"init_pose_file 缺少 variant {vid} 的记录位姿;"
                                           f"deploy 的 active variants 须与 collect 一致")
                    P, Q, G = _by_variant[vid]
                    c = int(_ip_rng.integers(0, P.shape[0]))   # 从该 variant 的全部保存位姿里随机抽
                    pos[i] = P[c]; quat[i] = Q[c]; goal[i] = G[c]
                return pos, quat, goal

            env_unwrapped._drill_pose_override_fn = _drill_pose_override_fn

        # 删除 MultiAssetSpawner 留下的模板原型，避免场景中出现多余的物体
        from isaaclab.sim.utils import delete_prim
        try:
            delete_prim("/World/Template")
        except Exception as e:
            print(f"[WARN] 删除 /World/Template 失败: {e}")

        settings = carb.settings.get_settings()
        settings.set("/app/player/useFixedTimeStepping", True)
        settings.set("/app/player/targetFrameRate", int(1.0 / env_unwrapped.step_dt))
        settings.set("/app/runLoops/rendering_0/rateLimitEnabled", True)
        settings.set("/app/runLoops/rendering_0/rateLimit", 60)
        settings.set("/physics/updateToUsd", False)
        settings.set("/physics/updateParticlesToUsd", False)
        settings.set("/app/runLoops/main/enabled", True)
        settings.set("/app/runLoops/main/syncTooFast", True)

        # reset BEFORE timeline.play()
        print("[INFO] Resetting environment (before play)...", flush=True)
        env_unwrapped.reset()

        # Create RlGamesVecEnvWrapper (matches collect_dp3_data.py order)
        clip_obs = math.inf
        clip_actions = 1.0
        env = RlGamesVecEnvWrapper(base_env, args.device, clip_obs, clip_actions)

        vecenv.register(
            "IsaacRlgWrapper",
            lambda config_name, num_actors, **kw: RlGamesGpuEnv(config_name, num_actors, **kw),
        )
        env_configurations.register(
            "rlgpu",
            {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env},
        )

        dt = env_unwrapped.step_dt
        cam1 = env_unwrapped.scene.sensors.get("cam1")
        cam2 = env_unwrapped.scene.sensors.get("cam2")
        if cam1 is None or cam2 is None:
            raise RuntimeError(
                f"Expected env-managed cameras 'cam1'/'cam2', got "
                f"sensors={list(env_unwrapped.scene.sensors.keys())}"
            )
        # cam1=30cm跟随框, cam2=固定工作区(不跟随)

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()

        if not simulation_app.is_running():
            print("[ERROR] simulation_app.is_running() is False before loop!")
            return

        # reset AFTER play: use wrapped env (env), NOT env_unwrapped.
        # env_unwrapped.reset() would also work here but the wrapped version
        # is consistent with collect_dp3_data.py and properly initializes cameras.
        _obs = env.reset()

        # Also call env_unwrapped.reset() once more after play() to get raw obs
        # and to finalize camera initialization. This is safe because cameras
        # are now initialized and the render loop is running.
        _, _ = env_unwrapped.reset()

        # ---------- 完整机器人/电钻点云(按 --pc_mode,与 collect_dp3_data.py 对齐)----------
        from perception.robot_pointcloud import RobotPointCloudFK, _quat_to_mat
        robot_fk = None
        robot_pc_M = 0
        if args.save_robot_pc:
            _npz = args.robot_pc_npz
            if not os.path.isabs(_npz):
                _npz = os.path.join(project_root, _npz)
            _rmax = args.robot_pc_points if (args.robot_pc_points and args.robot_pc_points > 0) else None
            robot_fk = RobotPointCloudFK(_npz, list(env_unwrapped.franka.body_names), args.device,
                                         max_points=_rmax)
            robot_pc_M = robot_fk.num_points
        drill_M = 0
        drill_canonical = None
        if args.save_drill_pc:
            drill_M = args.drill_pc_points
            _va = env_unwrapped._variant_attrs
            _vids = env_unwrapped._drill_variant_indices
            _var_canon = {}
            for _vid, _vdata in _va.items():
                _mp = _vdata.get("mesh_points")
                _scale = _vdata["scale"]
                if _mp is None or len(_mp) == 0:
                    _pts = torch.zeros(drill_M, 3, device=args.device)
                else:
                    _N = _mp.shape[0]
                    _idx = (torch.randperm(_N, device=_mp.device)[:drill_M] if _N >= drill_M
                            else torch.randint(0, _N, (drill_M,), device=_mp.device))
                    _pts = (_mp[_idx] * _scale).to(args.device)
                _var_canon[int(_vid)] = _pts
            drill_canonical = torch.stack([_var_canon[int(v.item())] for v in _vids], dim=0)
        # 合成地面/桌面点云(默认开,与 collect 同一套参数+同一抖动分布)
        from perception.robot_pointcloud import make_ground_points, jitter_ground_xy
        _PERC = _load_perception_hp()
        ground_M = _PERC.ground_points
        _ground_pts = make_ground_points(tuple(args.workspace), ground_M, _PERC.ground_z, args.device)
        ground_batch = _ground_pts.unsqueeze(0).expand(args.num_envs, -1, -1)  # (B,G,3) 网格基准
        _ground_xy_std = float(getattr(_PERC, "ground_xy_noise_std", 0.0))     # xy 每帧高斯抖动 σ(同 collect)
        total_pc = args.pc_num_points + robot_pc_M + drill_M + ground_M
        print(f"[INFO] pc_mode={args.pc_mode} -> total_pc={total_pc} "
              f"(camera={args.pc_num_points} + robot={robot_pc_M} + drill={drill_M} + ground={ground_M}) "
              f"| 须与训练数据点数一致", flush=True)

        def append_robot_drill(pc_cam):
            """pc_cam:(B,pc_num_points,3) env-local 相机云。按 collect 的顺序
            [camera | robot | drill | ground] 拼接(机器人 FK + 真值位姿电钻 + 合成地面)。"""
            parts = [pc_cam]
            if args.save_robot_pc:
                parts.append(robot_fk(env_unwrapped.franka.data.body_pos_w,
                                      env_unwrapped.franka.data.body_quat_w,
                                      env_unwrapped.scene.env_origins))
            if args.save_drill_pc:
                _dq = env_unwrapped.drill.data.root_quat_w
                _dp = env_unwrapped.drill.data.root_pos_w
                _Rd = _quat_to_mat(_dq)
                _dw = torch.einsum('bij,bnj->bni', _Rd, drill_canonical) + _dp.unsqueeze(1)
                parts.append(_dw - env_unwrapped.scene.env_origins.unsqueeze(1))
            # 合成地面(默认开,接最后):xy 每帧高斯抖动(z 不变,clamp 回区域),与 collect 同分布
            _ws = args.workspace
            parts.append(jitter_ground_xy(ground_batch, _ground_xy_std,
                                          _ws[0], _ws[1], _ws[2], _ws[3]))
            return torch.cat(parts, dim=1)

        # Warm-up: capture initial camera readings for observation histories.
        # This mirrors collect_dp3_data.py where cam.update() is called right
        # after reset to get the first valid point cloud.
        _sim_pc_per_cam = args.pc_num_points // 2    # 每相机 1024
        _workspace = tuple(args.workspace)
        cam1.update(dt)
        cam2.update(dt)
        # 裁剪框由 _PERC.camera_follow_drill 决定(当前=False -> 两相机都用固定 workspace);
        # 单一真源 camera_crop_bounds + camera_pc,与 collect 完全一致。
        _ws_init = camera_crop_bounds(env_unwrapped.drill.data.root_pos_w,
                                      env_unwrapped.scene.env_origins, _PERC, _workspace)
        pc1_init, _, _ = camera_pc(cam1, _ws_init, _sim_pc_per_cam, env_unwrapped.scene.env_origins)
        pc2_init, _, _ = camera_pc(cam2, _ws_init, _sim_pc_per_cam, env_unwrapped.scene.env_origins)
        # 相机空帧"沿用上一帧"的 per-cam 滚动缓存,初值取 warm-up 云(见主循环说明)。
        _last_pc1 = pc1_init
        _last_pc2 = pc2_init
        pc_fused_init = torch.zeros(args.num_envs, args.pc_num_points, 3, device=args.device, dtype=torch.float32)
        pc_fused_init[:, :_sim_pc_per_cam] = pc1_init   # cam1 30cm 跟随
        pc_fused_init[:, _sim_pc_per_cam:] = pc2_init   # cam2 30cm 跟随(同框,异视角)
        # 相机云(pc_num_points)后追加 FK机器人/电钻云 -> 与训练数据同布局
        _initial_pc = append_robot_drill(pc_fused_init)

        # ---------- Load DP3 ----------
        from diffusion_policy_3d.policy.dp3 import DP3
        import dill
        from hydra.utils import instantiate as hydra_instantiate
        from omegaconf import OmegaConf
        OmegaConf.register_new_resolver("eval", eval, replace=True)

        print(f"\nLoading DP3 checkpoint: {args.dp3_ckpt}", flush=True)
        payload = torch.load(args.dp3_ckpt, pickle_module=dill, map_location="cpu")
        dp3_cfg = payload["cfg"]

        # agent_pos 维度从 checkpoint 自适应:13(仅位置)或 26(位置+接触力)。
        # 自动判定,collect 用 --force_state 训出来的就走 force 模式,无需手动对齐。
        from perception.student_obs import build_agent_pos
        _agent_dim = None
        for _path in (lambda: dp3_cfg.task.shape_meta.obs.agent_pos.shape[0],
                      lambda: dp3_cfg.shape_meta.obs.agent_pos.shape[0]):
            try:
                _agent_dim = int(_path()); break
            except Exception:
                continue
        if _agent_dim is None:
            _agent_dim = 13
        with_force = _agent_dim > 13
        print(f"  agent_pos dim = {_agent_dim} (force_state={with_force})", flush=True)

        dp3_policy: DP3 = hydra_instantiate(dp3_cfg.policy)
        dp3_policy.to(args.device)
        dp3_policy.eval()
        dp3_policy.load_state_dict(payload["state_dicts"]["model"], strict=False)

        if args.num_inference_steps is not None:
            dp3_policy.num_inference_steps = args.num_inference_steps
            print(f"  Override num_inference_steps -> {args.num_inference_steps}")

        # 优先使用 EMA 权重
        if "ema_model" in payload["state_dicts"]:
            dp3_policy_ema: DP3 = hydra_instantiate(dp3_cfg.policy)
            dp3_policy_ema.to(args.device)
            dp3_policy_ema.load_state_dict(payload["state_dicts"]["ema_model"], strict=False)
            if args.num_inference_steps is not None:
                dp3_policy_ema.num_inference_steps = args.num_inference_steps
            dp3_policy_ema.eval()
            print("  Using EMA model.")
        else:
            dp3_policy_ema = dp3_policy

        _has_ema = "ema_model" in payload["state_dicts"]

        def _norm_ready(p):
            return len(p.normalizer.params_dict) > 0

        if _norm_ready(dp3_policy):
            dp3_policy.normalizer.to(args.device)
            if _has_ema:
                if not _norm_ready(dp3_policy_ema):
                    dp3_policy_ema.set_normalizer(dp3_policy.normalizer)
                dp3_policy_ema.normalizer.to(args.device)
            print("  Using normalizer embedded in checkpoint state_dict (--data_path not needed).")
        elif "pickles" in payload and "normalizer" in payload["pickles"]:
            normalizer = dill.loads(payload["pickles"]["normalizer"])
            dp3_policy.set_normalizer(normalizer)
            dp3_policy.normalizer.to(args.device)
            if _has_ema:
                dp3_policy_ema.set_normalizer(normalizer)
                dp3_policy_ema.normalizer.to(args.device)
            print("  Loaded normalizer from checkpoint pickles.")
        else:
            if not args.data_path:
                raise RuntimeError(
                    "Checkpoint 无内嵌 normalizer 且未提供 --data_path,无法构建 normalizer。")
            normalizer = compute_normalizer_from_zarr(args.data_path, args.device)
            dp3_policy.set_normalizer(normalizer)
            dp3_policy.normalizer.to(args.device)
            if _has_ema:
                dp3_policy_ema.set_normalizer(normalizer)
                dp3_policy_ema.normalizer.to(args.device)
            print("  Computed normalizer from dataset (fallback, used --data_path).")

        n_obs = dp3_policy.n_obs_steps          # 2
        n_act = dp3_policy.n_action_steps       # 从 ckpt 读(训练用 4)
        exec_n = n_act if args.exec_horizon is None else max(1, min(args.exec_horizon, n_act))
        horizon = dp3_policy.horizon           # 16
        num_inf_steps = args.num_inference_steps or dp3_policy.num_inference_steps
        policy_pc_points = total_pc  # camera + robot + drill(按 --pc_mode),须与训练数据一致
        sim_pc_per_cam = args.pc_num_points // 2  # 每相机 1024(cam1 跟随 + cam2 工作区)

        print(f"\n  DP3 config: horizon={horizon}, n_obs={n_obs}, n_act={n_act}")
        print(f"  inference steps={num_inf_steps}")

        from perception.target_drive import install_direct_drive
        install_direct_drive(env_unwrapped)

        # ---------- State helpers ----------
        controlled_indices = env_unwrapped.controlled_joint_indices.cpu()
        workspace = tuple(args.workspace)
        env_origins = env_unwrapped.scene.env_origins

        from collections import deque
        env_obs_histories = [deque(maxlen=n_obs + n_act) for _ in range(args.num_envs)]
        env_episode_rewards = torch.zeros(args.num_envs, device=args.device)
        total_episodes = 0
        successful_episodes = 0
        success_log = []
        episode_rewards = []
        total_zero_pc = 0
        total_steps = 0
        _zpc_dbg_count = 0   # 一次性诊断:已打印的 zero_pc 成因条数(封顶后不再刷屏)

        # ---------- 轨迹保存（可选）----------
        traj_buffers = [[] for _ in range(args.num_envs)] if args.save_traj else None
        env_started = np.zeros(args.num_envs, dtype=bool)

        # ---------- 部署采集(DP3 开车录 zarr,schema 同 collect_dp3_data.py)----------
        collect_mode = args.collect
        collector = (DeployCollector(args.collect_output, args.num_envs, total_pc,
                                     args.collect_filter, state_dim=_agent_dim)
                     if collect_mode else None)
        if collect_mode:
            print(f"[COLLECT] ON -> {args.collect_output} | target={args.collect_episodes} eps "
                  f"| filter={args.collect_filter} | drive=direct-target "
                  f"| state_dim={_agent_dim} | total_pc={total_pc}", flush=True)

        # ---------- 预填充每个 env 的观测历史 ----------
        initial_state = build_agent_pos(env_unwrapped, with_force)   # 13 或 26 维 agent_pos
        for env_id in range(args.num_envs):
            for _ in range(n_obs):
                env_obs_histories[env_id].append({
                    "agent_pos": initial_state[env_id],
                    "point_cloud": _initial_pc[env_id],
                })

        pending_actions = [None] * args.num_envs
        pending_idx = [0] * args.num_envs


        # ---- 分段性能计时（诊断瓶颈用）----
        _prof = {"infer": 0.0, "step": 0.0, "render": 0.0, "pc": 0.0}
        _prof_infer_calls = 0
        _last_print_step = 0
        _use_cuda = str(args.device).startswith("cuda")
        def _sync():
            if _use_cuda:
                torch.cuda.synchronize(args.device)

        start_time = time.time()
        print(f"\nStarting DP3 rollout (target {args.num_episodes} episodes, "
              f"{args.num_envs} envs)...")
        print(f"  workspace(env-local)={workspace}")
        print(f"  obs: agent_pos[{_agent_dim}] + point_cloud[{policy_pc_points},3]")
        sys.stdout.flush()

        try:
            with torch.inference_mode():
                while simulation_app.is_running() and (
                        (collector.collected < args.collect_episodes) if collect_mode
                        else (total_episodes < args.num_episodes)):
                    try:
                        # ---- 1) 哪些 env 需要新的 action chunk ----
                        need_policy_call = [
                            env_id for env_id in range(args.num_envs)
                            if pending_actions[env_id] is None or pending_idx[env_id] >= exec_n
                        ]

                        # ---- 2) DP3 推理（批处理）----
                        if need_policy_call:
                            _sync(); _t0 = time.perf_counter()
                            obs_batch_list, pc_batch_list = [], []
                            for env_id in need_policy_call:
                                hist = list(env_obs_histories[env_id])
                                if len(hist) < n_obs:
                                    hist = [hist[0]] * (n_obs - len(hist)) + hist
                                obs_timestep = hist[-n_obs:]
                                obs_batch_list.append(torch.stack([s["agent_pos"] for s in obs_timestep]))
                                pc_batch_list.append(torch.stack([s["point_cloud"] for s in obs_timestep]))

                            obs_dict = {
                                "agent_pos": torch.stack(obs_batch_list),       # (B, n_obs, 26) GPU
                                "point_cloud": torch.stack(pc_batch_list),      # (B, n_obs, P, 3) GPU
                            }
                            result = dp3_policy_ema.predict_action(obs_dict)
                            action_chunks = result["action"]                    # (B, n_act, 13) GPU

                            for i, env_id in enumerate(need_policy_call):
                                pending_actions[env_id] = action_chunks[i]
                                pending_idx[env_id] = 0
                            _sync(); _prof["infer"] += time.perf_counter() - _t0
                            _prof_infer_calls += 1

                        # ---- 3) 执行一帧动作 ----
                        # actions_policy: policy 直接输出（target 模式下即关节目标角 rad），全程留在 GPU
                        actions_policy = torch.zeros((args.num_envs, 13),
                                                     device=args.device, dtype=torch.float32)
                        for env_id in range(args.num_envs):
                            if pending_actions[env_id] is not None:
                                actions_policy[env_id] = pending_actions[env_id][pending_idx[env_id]]
                                pending_idx[env_id] += 1

                        # DP3 的关节目标角 q* 直接下发(patched _apply_action 用 _direct_target);
                        # actions 仅作占位传给 step(被 patched _apply_action 忽略)。
                        env_unwrapped._direct_target = actions_policy
                        actions = actions_policy
                        _sync(); _t0 = time.perf_counter()
                        obs_dict_step, rewards, terminated, truncated, _ = env_unwrapped.step(actions)
                        _sync(); _prof["step"] += time.perf_counter() - _t0
                        total_steps += 1

                        # ---- 4) 更新相机：在 step() 之后（与 collect_dp3_data.py 一致）----
                        _sync(); _t0 = time.perf_counter()
                        cam1.update(dt)
                        cam2.update(dt)
                        _sync(); _prof["render"] += time.perf_counter() - _t0

                        # ---- 5) 在 step() 之后读取 agent_pos（与 collect_dp3_data.py 同一真源）----
                        state_13 = build_agent_pos(env_unwrapped, with_force)   # 13 或 26 维

                        # ---- 6) 深度图 -> 点云(单一真源 camera_crop_bounds + camera_pc;
                        #         裁剪框由 _PERC.camera_follow_drill 决定,当前=固定 workspace)----
                        _sync(); _t0 = time.perf_counter()
                        _ws = camera_crop_bounds(env_unwrapped.drill.data.root_pos_w,
                                                 env_origins, _PERC, workspace)
                        pc1_t, zmask1, depth1 = camera_pc(cam1, _ws, sim_pc_per_cam, env_origins)
                        pc2_t, zmask2, _ = camera_pc(cam2, _ws, sim_pc_per_cam, env_origins)
                        total_zero_pc += int(zmask1.sum().item() + zmask2.sum().item())

                        # ---- 一次性诊断:zero_pc 究竟为什么空(最多打 30 条)----
                        # valid_depth_px≈0 -> 相机本帧深度整张是空的(渲染滞后/没渲染);
                        # valid_depth_px 很大但仍空 -> 框对不上(电钻真值位姿 vs 深度不同帧,
                        #   或电钻被移出相机视野)。reset_this_step 区分是不是 reset 瞬态。
                        if zmask1.any() and _zpc_dbg_count < 30:
                            _done_now = terminated.bool() | truncated.bool()
                            _dpos = (env_unwrapped.drill.data.root_pos_w - env_origins)  # (B,3) env-local
                            # 反投影(不裁剪)cam1 的全部有效点,看它们落在哪
                            _pl1, _vmsk1 = depth_to_pointcloud_batch(
                                depth1, cam1.data.intrinsic_matrices, cam1.data.pos_w,
                                cam1.data.quat_w_ros, _ws, sim_pc_per_cam, env_origins, diag_only=True)
                            def _bbox(pl, vm, e):
                                p = pl[e][vm[e]]
                                if p.numel() == 0:
                                    return "EMPTY"
                                lo = p.min(0).values; hi = p.max(0).values
                                return (f"x[{lo[0]:.2f},{hi[0]:.2f}] y[{lo[1]:.2f},{hi[1]:.2f}] "
                                        f"z[{lo[2]:.2f},{hi[2]:.2f}] n={p.shape[0]}")
                            for _e in range(args.num_envs):
                                if zmask1[_e] and _zpc_dbg_count < 30:
                                    _cp1 = cam1.data.pos_w[_e] - env_origins[_e]
                                    print(f"[ZPC] step={total_steps} env={_e} cam1_empty "
                                          f"reset={bool(_done_now[_e])} "
                                          f"drill(env-local)=[{_dpos[_e,0]:.2f},{_dpos[_e,1]:.2f},{_dpos[_e,2]:.2f}]",
                                          flush=True)
                                    print(f"       cam1@[{_cp1[0]:.2f},{_cp1[1]:.2f},{_cp1[2]:.2f}] pts {_bbox(_pl1,_vmsk1,_e)}", flush=True)
                                    _zpc_dbg_count += 1

                        # collect 采数据时跳过相机空帧(bad_pc),训练数据里没有全 0 的点云。
                        # 部署不能跳帧,改为沿用 cam1 上一帧的有效云(等价于"本帧无新信息"),
                        # 避免喂 0 这种严重 off-manifold 输入。空帧多发生在 reset 后渲染滞后,
                        # 或电钻被手遮挡的帧;下一帧拿到新云即自动纠正。
                        pc1_t = torch.where(zmask1.view(-1, 1, 1), _last_pc1, pc1_t)
                        pc2_t = torch.where(zmask2.view(-1, 1, 1), _last_pc2, pc2_t)
                        _last_pc1 = pc1_t
                        _last_pc2 = pc2_t

                        pc_fused_2048 = torch.zeros(args.num_envs, args.pc_num_points, 3,
                                                   device=args.device, dtype=torch.float32)
                        pc_fused_2048[:, :sim_pc_per_cam] = pc1_t   # cam1 30cm 跟随
                        pc_fused_2048[:, sim_pc_per_cam:] = pc2_t   # cam2 30cm 跟随(同框,异视角)

                        # 相机云(pc_num_points)后追加 FK机器人/电钻云 -> 与训练数据同布局
                        pc_for_policy = append_robot_drill(pc_fused_2048)
                        _sync(); _prof["pc"] += time.perf_counter() - _t0

                        # ---- 一次性 OBS 自检:对比 deploy 观测 range 与训练 zarr ----
                        # 训练数据(collect)范围参考(state 只剩 13 关节位置):
                        #   state pos[-2.84,3.19]
                        #   camera x[0.00,1.51] y[-0.60,0.94] z[0.02,0.78]
                        #   robot  x[-0.15,1.03] y[-0.34,0.56] z[-0.00,0.74]
                        #   drill  x[0.40,1.10] y[-0.26,0.58] z[0.02,0.78]
                        if total_steps == 1:
                            _s = state_13[0].detach().cpu().numpy()
                            _p = pc_for_policy[0].detach().cpu().numpy()
                            _nc = args.pc_num_points
                            print(f"[OBS-CHECK] pc.shape={tuple(pc_for_policy.shape)} | "
                                  f"state pos[{_s.min():.2f},{_s.max():.2f}]", flush=True)
                            def _bb(name, q):
                                print(f"  {name}: x[{q[:,0].min():.2f},{q[:,0].max():.2f}] "
                                      f"y[{q[:,1].min():.2f},{q[:,1].max():.2f}] "
                                      f"z[{q[:,2].min():.2f},{q[:,2].max():.2f}]", flush=True)
                            _bb("camera", _p[:_nc])
                            if robot_pc_M:
                                _bb("robot ", _p[_nc:_nc + robot_pc_M])
                            if drill_M:
                                _bb("drill ", _p[_nc + robot_pc_M:_nc + robot_pc_M + drill_M])
                            _bb("ground", _p[_nc + robot_pc_M + drill_M:])
                            # 重叠自检:相机段每帧唯一点数。<pc_num_points 即有重复点
                            # (有效像素 < pool_size 时,缺口用 first_valid 重复填充)。
                            _cam = pc_for_policy[:, :_nc, :]
                            _uniq = torch.tensor([torch.unique(_cam[b], dim=0).shape[0]
                                                  for b in range(_cam.shape[0])])
                            _vpx = (depth1[..., 0] > 0).flatten(1).sum(dim=1)  # cam1 总有效像素(裁剪前)
                            print(f"  [DUP-CHECK] camera unique/frame: min={int(_uniq.min())} "
                                  f"mean={float(_uniq.float().mean()):.0f} max={int(_uniq.max())} of {_nc} "
                                  f"| cam1 valid_px(裁剪前) min={int(_vpx.min())} mean={int(_vpx.float().mean())} "
                                  f"| pool_size=2048 -> unique<{_nc} 即有重复点", flush=True)

                        env_episode_rewards += rewards
                        is_done = terminated.bool() | truncated.bool()

                        try:
                            lenient_success = env_unwrapped._cached_lenient_success
                        except AttributeError:
                            lenient_success = torch.zeros(args.num_envs, dtype=torch.bool,
                                                          device=args.device)

                        # ---- (采集模式) 录当前帧 + episode 边界,逻辑逐条复刻 collect_dp3_data.py ----
                        if collect_mode:
                            # 直接 target 驱动:动作标签 = step 后的 cur_targets(= DP3 下发的 q*)
                            rec_act_np = env_unwrapped.cur_targets.detach().cpu().numpy()
                            rec_state_np = state_13.detach().cpu().numpy()
                            rec_pc_np = pc_for_policy.detach().cpu().numpy()
                            rec_bad_np = (zmask1 | zmask2).detach().cpu().numpy()   # 相机空帧,跳过不存
                            is_done_np = is_done.detach().cpu().numpy()
                            succ_np = lenient_success.detach().cpu().numpy().astype(bool)
                            # 6) 录活跃、未终止、非空帧(终止步 state/pc 已属下一局 -> 跳过)
                            for env_id in range(args.num_envs):
                                if (not collector.started[env_id] or is_done_np[env_id]
                                        or rec_bad_np[env_id]):
                                    continue
                                collector.record_frame(env_id, rec_state_np[env_id],
                                                        rec_act_np[env_id], rec_pc_np[env_id])
                            # 7) episode 边界:done 即 flush(成功过滤在 flush 内),开新局
                            for env_id in range(args.num_envs):
                                if is_done_np[env_id]:
                                    if collector.started[env_id]:
                                        collector.flush(env_id, bool(succ_np[env_id]))
                                    collector.buffers[env_id] = []
                                    collector.started[env_id] = True
                                elif not collector.started[env_id]:
                                    collector.buffers[env_id] = []
                                    collector.started[env_id] = True

                        finished = torch.where(is_done)[0]
                        if finished.numel() > 0:
                            reset_state = build_agent_pos(env_unwrapped, with_force)   # 13 或 26 维

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
                                        "state": state_13[env_id].cpu().numpy(),
                                        "action": actions_policy[env_id].cpu().numpy(),
                                        "reward": rewards[env_id].item(),
                                        "done": True,
                                    })
                                    save_traj_as_npz(traj_buffers[env_id], args.save_traj, env_id)

                                env_episode_rewards[env_id] = 0.0
                                pending_actions[env_id] = None
                                pending_idx[env_id] = 0
                                env_obs_histories[env_id].clear()
                                # 用刚 step 后读到的新点云重填历史，而不是冻结的 _initial_pc。
                                # env 在 step() 内部已 auto-reset 并 rerender 3 次
                                # (num_rerenders_on_reset=3)，所以 pc_for_policy[env_id] 已经
                                # 反映新一局随机后的电钻位姿；reset_state==state_13（都是 step 后读的）。
                                for _ in range(n_obs):
                                    env_obs_histories[env_id].append({
                                        "agent_pos": reset_state[env_id],
                                        "point_cloud": pc_for_policy[env_id],
                                    })
                                env_started[env_id] = True
                                if traj_buffers is not None:
                                    traj_buffers[env_id] = []

                        # ---- 8) 更新未结束 env 的观测历史（用 step 后读取的新 pc）----
                        for env_id in range(args.num_envs):
                            if not is_done[env_id]:
                                if not env_started[env_id]:
                                    env_started[env_id] = True
                                env_obs_histories[env_id].append({
                                    "agent_pos": state_13[env_id],
                                    "point_cloud": pc_for_policy[env_id],
                                })
                                if traj_buffers is not None:
                                    traj_buffers[env_id].append({
                                        "state": state_13[env_id].cpu().numpy(),
                                        "action": actions_policy[env_id].cpu().numpy(),
                                        "reward": rewards[env_id].item(),
                                        "done": False,
                                    })

                        # ---- 9) 进度打印 ----
                        if total_steps % 100 == 0:
                            elapsed = time.time() - start_time
                            sr = successful_episodes / max(total_episodes, 1) * 100
                            _win = max(total_steps - _last_print_step, 1)
                            _ms = lambda k: _prof[k] / _win * 1000.0
                            _infer_ms = _prof["infer"] / max(_prof_infer_calls, 1) * 1000.0
                            print(f"  step={total_steps} | eps={total_episodes}/{args.num_episodes} "
                                  f"| succ={successful_episodes} ({sr:.1f}%) "
                                  f"| zero_pc={total_zero_pc} | {total_steps/elapsed:.1f} step/s")
                            print(f"    [prof] render={_ms('render'):.1f}ms  pc+fps={_ms('pc'):.1f}ms  "
                                  f"phys={_ms('step'):.1f}ms /step | "
                                  f"infer={_infer_ms:.1f}ms/call ({_prof_infer_calls} calls / {_win} steps)")
                            sys.stdout.flush()
                            _prof = {k: 0.0 for k in _prof}
                            _prof_infer_calls = 0
                            _last_print_step = total_steps

                    except Exception as loop_err:
                        import traceback
                        print(f"[ERROR] loop body failed at step {total_steps}: {loop_err}", flush=True)
                        traceback.print_exc()
                        raise

        except Exception as main_err:
            import traceback
            print(f"[FATAL] main loop error: {main_err}", flush=True)
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
        if collect_mode:
            _frames = collector.data["state"].shape[0]
            print(f"[COLLECT] saved {collector.collected} episodes / {_frames} frames "
                  f"-> {args.collect_output} (filter={args.collect_filter})")
        if episode_rewards:
            ep_np = np.array(episode_rewards)
            print(f"Reward: mean={ep_np.mean():.1f}, max={ep_np.max():.1f}, min={ep_np.min():.1f}, std={ep_np.std():.1f}")
        print(f"Total steps: {total_steps}")
        print(f"Total zero-pc frames: {total_zero_pc}")
        print(f"Time: {h}h{m}m{s}s, {total_steps/elapsed:.0f} steps/s")
        if success_log:
            print(f"Success log: {success_log}")
        print(f"{'='*60}")

    except Exception:
        # 关键修复：在 finally 调用 close() 之前先把真实异常打印出来。
        # 否则 simulation_app.close() 在 IsaacSim 的 on-stop 渲染回调里死锁时，
        # 异常会被永久挂起、永远看不到 traceback（只表现为“卡住”）。
        import traceback
        print("\n[FATAL] Exception in main():", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        # 确保所有输出先落盘
        sys.stdout.flush()
        sys.stderr.flush()
        # IsaacSim 的 simulation_app.close() 可能在 on-stop 渲染回调里死锁，
        # 先尝试 stop timeline，再 close；最后用 os._exit 强制退出绕开死锁，
        # 避免每次都得手动 Ctrl-C 杀进程。
        try:
            import omni.timeline
            omni.timeline.get_timeline_interface().stop()
        except Exception:
            pass
        try:
            simulation_app.close()
        except Exception:
            pass
        os._exit(0)


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