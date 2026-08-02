"""Residual TD3 fine-tuning on top of a frozen DP3 checkpoint (see perception/residual_td3.py and
perception/residual_env_wrapper.py). DP3 (encoder + diffusion head) is frozen throughout; only a
small residual actor+critic (TD3) are trained, added on top of DP3's own per-step action.

Env/checkpoint-loading setup mirrors scripts/deploy_dp3_sim.py closely (same with_mask/with_force/
disable_cam2 auto-detection from the checkpoint, same total_pc consistency check) so a checkpoint
that deploys correctly there also fine-tunes correctly here.

Usage:
  python scripts/train_residual_td3.py --headless --stage1_only \
      --dp3_ckpt <path/to/latest.ckpt> --pc_num_points 2048 --disable_cam2 --robot_pc_points 512 \
      --num_envs 64 --total_timesteps 500000
"""
import argparse
import os
import sys
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import torch


def _safe_compile(model=None, *args, **kwargs):
    if callable(model):
        return model
    return lambda fn: fn
torch.compile = _safe_compile

DP3_ROOT = "/home/zeyu/3D-Diffusion-Policy/3D-Diffusion-Policy"
sys.path.insert(0, DP3_ROOT)
from perception.dp3_pointcloud import (camera_pc, camera_crop_bounds, build_plate_cam_pc,
                                        init_fps_kernel, wrist_cam_pose_w, raise_z_floor)
from perception.camera_setup import (load_perception_hp as _load_perception_hp,
                                     ensure_rgb_aov, get_cameras, detect_wrist_cam,
                                     apply_render_settings, setup_ground)
from perception.groundtruth_mask import DEFAULT_MASK_THRESHOLD as _GT_MASK_THRESHOLD
init_fps_kernel("/home/zeyu/3D-Diffusion-Policy/third_party/pytorch3d_simplified")


def parse_args():
    _PERC = _load_perception_hp()
    p = argparse.ArgumentParser()
    p.add_argument('--dp3_ckpt', type=str, required=True)
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--drill_configs", type=str, default=None)
    p.add_argument('--drill_variants', type=str, default=None)
    p.add_argument('--img_height', type=int, default=_PERC.img_height)
    p.add_argument("--img_width", type=int, default=_PERC.img_width)
    p.add_argument('--pc_num_points', type=int, default=1024)
    p.add_argument('--disable_cam2', action='store_true')
    p.add_argument('--robot_pc_points', type=int, default=160)
    p.add_argument('--ground_points', type=int, default=0)
    p.add_argument('--workspace', nargs=6, type=float, default=None)
    p.add_argument('--pc_mode', type=str, default='robot', choices=['camera', 'robot', 'robot_drill'])
    p.add_argument('--robot_pc_npz', type=str, default='assets/inspire_tac/robot_canonical_points.npz')
    p.add_argument('--mask_threshold', type=float, default=_GT_MASK_THRESHOLD,
                   help="is_handle radius (m); default comes from "
                        "perception/groundtruth_mask.DEFAULT_MASK_THRESHOLD, shared with "
                        "collect_dp3_data.py and deploy_dp3_sim.py")
    p.add_argument('--chained', action='store_true')
    p.add_argument('--stage1_only', action='store_true')
    p.add_argument('--plate_pc_points', type=int, default=512)
    p.add_argument('--drill_mesh_points', type=int, default=512)
    p.add_argument('--episode_length_s', type=float, default=10.0)

    # ---- TD3 / residual RL ----
    p.add_argument('--total_timesteps', type=int, default=500_000, help="total transitions across all envs")
    p.add_argument('--learning_starts', type=int, default=10_000)
    p.add_argument('--critic_warmup_steps', type=int, default=10_000,
                   help="transitions during which only the critic updates, actor stays untouched")
    p.add_argument('--buffer_size', type=int, default=300_000)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--updates_per_step', type=int, default=1, help="TD3 updates per parallel env-step")
    p.add_argument('--action_scale', type=float, default=0.1)
    p.add_argument('--actor_lr', type=float, default=1e-6)
    p.add_argument('--critic_lr', type=float, default=1e-4)
    p.add_argument('--gamma', type=float, default=0.995)
    p.add_argument('--tau', type=float, default=0.005)
    p.add_argument('--policy_noise', type=float, default=0.05)
    p.add_argument('--noise_clip', type=float, default=0.1)
    p.add_argument('--policy_delay', type=int, default=2)
    p.add_argument('--explore_noise', type=float, default=0.05)

    p.add_argument('--eval_interval', type=int, default=20_000, help="in transitions")
    p.add_argument('--save_interval', type=int, default=20_000, help="in transitions")
    p.add_argument('--output_dir', type=str, default=None,
                   help="default: <dp3_ckpt dir>/../residual_td3")
    args = p.parse_args()

    args.save_robot_pc = args.pc_mode in ("robot", "robot_drill")
    args.save_drill_mesh_pc = args.pc_mode == "robot_drill"
    if args.workspace is None:
        args.workspace = list(_PERC.chained_workspace if args.chained else _PERC.workspace)
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(os.path.dirname(args.dp3_ckpt)), "residual_td3")
    return args


def main():
    args = parse_args()

    if args.stage1_only and not args.chained:
        args.chained = True
        print("[INFO] --stage1_only: auto-enable ChainedEnv + wrist camera, cam3 off", flush=True)

    if not os.path.exists(args.dp3_ckpt):
        print(f"[ERROR] DP3 checkpoint not found: {args.dp3_ckpt}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=args.headless, enable_cameras=True)
    simulation_app = app_launcher.app

    try:
        import omni.timeline
        import omni.usd
        from tasks.grasp_drill_env import GraspDrillEnv, create_grasp_drill_env_cfg

        omni.usd.get_context().new_stage()

        cfg = create_grasp_drill_env_cfg(
            num_envs=args.num_envs, device=args.device, headless=args.headless, debug=args.debug,
            drill_config_path=args.drill_configs, drill_variants_path=args.drill_variants,
            img_height=args.img_height, img_width=args.img_width, enable_cameras=True,
            include_plate=args.chained, include_plate_camera=args.chained,
            include_cam3=args.chained and not args.stage1_only,
        )
        cfg.seed = args.seed
        if args.chained:
            cfg.episode_length_s = args.episode_length_s
        ensure_rgb_aov(cfg.scene)

        if args.chained:
            from tasks.chained_env import get_chained_env_class
            ChainedEnv = get_chained_env_class()
            base_env = ChainedEnv(cfg=cfg, debug=args.debug, success_hold_stop=1,
                                  stage1_only=args.stage1_only)
        else:
            base_env = GraspDrillEnv(cfg=cfg, debug=args.debug)
        env_unwrapped = base_env.unwrapped

        from isaaclab.sim.utils import delete_prim
        delete_prim("/World/Template")
        apply_render_settings(env_unwrapped.step_dt)
        env_unwrapped.reset()

        dt = env_unwrapped.step_dt
        cam1, cam2, cam3 = get_cameras(env_unwrapped, chained=args.chained,
                                       need_cam3=args.chained and not args.stage1_only)
        _use_plate = args.chained and (cam3 is not None)
        _cam2_on_body = detect_wrist_cam(env_unwrapped, cam2)

        def cam2_pose_fn():
            return wrist_cam_pose_w(env_unwrapped.franka, cam2) if _cam2_on_body else None

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        if not simulation_app.is_running():
            print("[ERROR] simulation_app not running before loop!")
            return
        env_unwrapped.reset()

        # ---- robot / drill-mesh / ground point-cloud modules (same as deploy_dp3_sim.py) ----
        from perception.robot_pointcloud import RobotPointCloudFK, jitter_ground_xy
        robot_fk = None
        robot_pc_M = 0
        if args.save_robot_pc:
            _npz = args.robot_pc_npz
            if not os.path.isabs(_npz):
                _npz = os.path.join(project_root, _npz)
            _rmax = args.robot_pc_points if args.robot_pc_points > 0 else None
            robot_fk = RobotPointCloudFK(_npz, list(env_unwrapped.franka.body_names), args.device,
                                         max_points=_rmax)
            robot_pc_M = robot_fk.num_points
        drill_mesh_fk = None
        drill_M = 0
        if args.save_drill_mesh_pc and args.drill_mesh_points > 0:
            from perception.drill_pointcloud import DrillMeshPointCloudFK
            drill_mesh_fk = DrillMeshPointCloudFK(env_unwrapped, num_points=args.drill_mesh_points,
                                                  device=args.device)
            drill_M = drill_mesh_fk.num_points
        _PERC = _load_perception_hp()
        ground_batch, ground_M, _ground_xy_std = setup_ground(args.num_envs, args.workspace, args.device,
                                                              num_points=args.ground_points)
        plate_M = args.plate_pc_points if _use_plate else 0
        total_pc = args.pc_num_points + plate_M + robot_pc_M + drill_M + ground_M

        _last_pc3 = None

        def append_robot_drill(pc_cam):
            nonlocal _last_pc3
            parts = [pc_cam]
            if _use_plate:
                pc_plate, _zm3, _ = build_plate_cam_pc(cam3, env_unwrapped.plate.data.root_pos_w,
                                                       env_unwrapped.scene.env_origins, plate_M)
                if _last_pc3 is not None:
                    pc_plate = torch.where(_zm3.view(-1, 1, 1), _last_pc3, pc_plate)
                _last_pc3 = pc_plate
                parts.append(pc_plate)
            if args.save_robot_pc:
                parts.append(robot_fk(env_unwrapped.franka.data.body_pos_w,
                                      env_unwrapped.franka.data.body_quat_w,
                                      env_unwrapped.scene.env_origins))
            if drill_mesh_fk is not None:
                parts.append(drill_mesh_fk(env_unwrapped.drill.data.root_pos_w,
                                           env_unwrapped.drill.data.root_quat_w,
                                           env_unwrapped._drill_variant_indices,
                                           env_unwrapped.scene.env_origins))
            _ws = args.workspace
            parts.append(jitter_ground_xy(ground_batch, _ground_xy_std, _ws[0], _ws[1], _ws[2], _ws[3]))
            return torch.cat(parts, dim=1)

        # ---- load frozen DP3 checkpoint (with_mask / with_force / total_pc check, same as deploy) ----
        from diffusion_policy_3d.policy.dp3 import DP3
        import dill
        from hydra.utils import instantiate as hydra_instantiate
        from omegaconf import OmegaConf
        OmegaConf.register_new_resolver("eval", eval, replace=True)

        payload = torch.load(args.dp3_ckpt, pickle_module=dill, map_location="cpu")
        dp3_cfg = payload["cfg"]

        from perception.student_obs import build_agent_pos
        _agent_dim = None
        for _path in (lambda: dp3_cfg.task.shape_meta.obs.agent_pos.shape[0],
                      lambda: dp3_cfg.shape_meta.obs.agent_pos.shape[0]):
            try:
                _agent_dim = int(_path()); break
            except Exception:
                continue
        with_force = (_agent_dim or 13) > 13
        print(f"  agent_pos dim = {_agent_dim} (force_state={with_force})", flush=True)

        _ckpt_pc = None
        for _path in (lambda: dp3_cfg.task.shape_meta.obs.point_cloud.shape[0],
                      lambda: dp3_cfg.shape_meta.obs.point_cloud.shape[0]):
            try:
                _ckpt_pc = int(_path()); break
            except Exception:
                continue
        if _ckpt_pc is not None and _ckpt_pc != total_pc:
            raise RuntimeError(f"ckpt point_cloud={_ckpt_pc} != built total_pc={total_pc}; "
                               f"check --pc_num_points/--disable_cam2/--robot_pc_points/--chained match training")

        _ckpt_ch = None
        for _path in (lambda: dp3_cfg.task.shape_meta.obs.point_cloud.shape[-1],
                      lambda: dp3_cfg.shape_meta.obs.point_cloud.shape[-1]):
            try:
                _ckpt_ch = int(_path()); break
            except Exception:
                continue
        with_mask = (_ckpt_ch or 3) == 4
        print(f"  point_cloud: total_pc={total_pc} channels={_ckpt_ch} with_mask={with_mask}", flush=True)

        from perception.handle_mask import compute_handle_mask

        def add_mask_channel(pc_now):
            if not with_mask:
                return pc_now
            _cam = pc_now[:, :args.pc_num_points]
            _rs = args.pc_num_points + plate_M
            _robot_seg = pc_now[:, _rs:_rs + robot_pc_M] if args.save_robot_pc and robot_pc_M > 0 else None
            _ish = compute_handle_mask(_cam, env_unwrapped, robot_pc_local=_robot_seg,
                                       threshold=args.mask_threshold)
            m = torch.zeros(pc_now.shape[0], pc_now.shape[1], 1, dtype=pc_now.dtype, device=pc_now.device)
            m[:, :args.pc_num_points, 0] = _ish.to(pc_now.dtype)
            return torch.cat([pc_now, m], dim=-1)

        dp3_policy: DP3 = hydra_instantiate(dp3_cfg.policy)
        dp3_policy.to(args.device)
        dp3_policy.eval()
        dp3_policy.load_state_dict(payload["state_dicts"]["model"], strict=False)
        if "ema_model" in payload["state_dicts"]:
            dp3_policy_ema: DP3 = hydra_instantiate(dp3_cfg.policy)
            dp3_policy_ema.to(args.device)
            dp3_policy_ema.load_state_dict(payload["state_dicts"]["ema_model"], strict=False)
            dp3_policy_ema.eval()
        else:
            dp3_policy_ema = dp3_policy

        def _norm_ready(pol):
            return len(pol.normalizer.params_dict) > 0
        if _norm_ready(dp3_policy):
            dp3_policy.normalizer.to(args.device)
            if dp3_policy_ema is not dp3_policy and not _norm_ready(dp3_policy_ema):
                dp3_policy_ema.set_normalizer(dp3_policy.normalizer)
            dp3_policy_ema.normalizer.to(args.device)
        elif "pickles" in payload and "normalizer" in payload["pickles"]:
            normalizer = dill.loads(payload["pickles"]["normalizer"])
            dp3_policy.set_normalizer(normalizer)
            dp3_policy.normalizer.to(args.device)
            dp3_policy_ema.set_normalizer(normalizer)
            dp3_policy_ema.normalizer.to(args.device)
        else:
            raise RuntimeError("Checkpoint has no embedded normalizer.")

        # freeze DP3 completely -- only the residual actor/critic ever get gradients
        for p in dp3_policy_ema.parameters():
            p.requires_grad_(False)
        dp3_policy_ema.eval()

        n_obs = dp3_policy_ema.n_obs_steps
        feat_dim = dp3_policy_ema.obs_encoder.output_shape()
        obs_dim = feat_dim + 13
        print(f"  DP3Encoder feat_dim={feat_dim} -> residual obs_dim={obs_dim} (feat + dp3_action)", flush=True)

        from perception.target_drive import install_direct_drive
        install_direct_drive(env_unwrapped)   # so env_unwrapped._direct_target drives _apply_action directly

        from perception.residual_env_wrapper import ResidualDP3EnvWrapper
        wrapper = ResidualDP3EnvWrapper(
            env_unwrapped=env_unwrapped, dp3_policy=dp3_policy_ema,
            cam1=cam1, cam2=cam2, cam3=cam3, cam2_pose_fn=cam2_pose_fn,
            build_agent_pos_fn=build_agent_pos,
            camera_crop_bounds_fn=camera_crop_bounds, camera_pc_fn=camera_pc, raise_z_floor_fn=raise_z_floor,
            add_mask_channel_fn=add_mask_channel, append_robot_drill_fn=append_robot_drill,
            perc=_PERC, workspace=tuple(args.workspace), pc_num_points=args.pc_num_points,
            disable_cam2=args.disable_cam2, with_force=with_force, chained=args.chained,
            n_obs_steps=n_obs, device=args.device, num_envs=args.num_envs, dt=dt,
        )

        from perception.residual_td3 import TD3ResidualAgent, ReplayBuffer
        agent = TD3ResidualAgent(
            obs_dim=obs_dim, action_dim=13, device=args.device, action_scale=args.action_scale,
            actor_lr=args.actor_lr, critic_lr=args.critic_lr, gamma=args.gamma, tau=args.tau,
            policy_noise=args.policy_noise, noise_clip=args.noise_clip, policy_delay=args.policy_delay,
            explore_noise=args.explore_noise,
        )
        buffer = ReplayBuffer(args.buffer_size, obs_dim, 13, args.device)

        def cat_obs(obs):
            return torch.cat([obs["feat"], obs["dp3_action"]], dim=-1)

        print(f"\n[RESIDUAL-TD3] starting: num_envs={args.num_envs} total_timesteps={args.total_timesteps} "
              f"(transitions) learning_starts={args.learning_starts} critic_warmup={args.critic_warmup_steps} "
              f"action_scale={args.action_scale}", flush=True)

        obs = wrapper.reset()
        ep_reward = torch.zeros(args.num_envs, device=args.device)
        ep_count = 0
        success_count = 0
        recent_success = []
        transitions = 0
        start_time = time.time()
        last_log_t = 0

        # NOTE: deliberately NOT wrapped in torch.inference_mode() -- unlike deploy_dp3_sim.py's
        # pure-inference loop, this loop also runs agent.update() (needs normal autograd). Tensors
        # created under inference_mode() are marked "inference tensors" and a nested
        # torch.enable_grad() does not reliably make them usable in a backward() graph again, so
        # the rollout parts that don't need grad (DP3 inference, action selection) instead guard
        # themselves internally with their own torch.no_grad() (see residual_env_wrapper.py /
        # residual_td3.py's select_action) rather than relying on an outer context here.
        while transitions < args.total_timesteps and simulation_app.is_running():
            with torch.no_grad():
                obs_cat = cat_obs(obs)
                explore = transitions < args.learning_starts
                if explore:
                    residual = (torch.rand(args.num_envs, 13, device=args.device) * 2 - 1) * args.action_scale
                else:
                    residual = agent.select_action(obs_cat, explore=True)

                next_obs, rewards, terminated, truncated, _ = wrapper.step(residual)
                done = terminated.bool() | truncated.bool()
                next_obs_cat = cat_obs(next_obs)

                buffer.add_batch(obs_cat, residual, rewards, next_obs_cat, done.float())
                transitions += args.num_envs
                ep_reward += rewards

                try:
                    lenient_success = env_unwrapped._cached_lenient_success
                except AttributeError:
                    lenient_success = torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)

                finished = torch.where(done)[0]
                for e in finished.tolist():
                    ep_count += 1
                    if lenient_success[e]:
                        success_count += 1
                    recent_success.append(bool(lenient_success[e].item()))
                    recent_success[:] = recent_success[-200:]
                    ep_reward[e] = 0.0

                obs = next_obs

            c_loss = a_loss = None
            if transitions >= args.learning_starts:
                critic_only = transitions < args.critic_warmup_steps
                for _ in range(args.updates_per_step):
                    c_loss, a_loss = agent.update(buffer, args.batch_size, critic_only=critic_only)

            if transitions - last_log_t >= 2000:
                elapsed = time.time() - start_time
                sr = (sum(recent_success) / len(recent_success) * 100) if recent_success else 0.0
                _cl = f"{c_loss:.4f}" if c_loss is not None else "n/a"
                _al = f"{a_loss:.4f}" if a_loss is not None else "n/a"
                print(f"  transitions={transitions}/{args.total_timesteps} eps={ep_count} "
                      f"succ_recent200={sr:.1f}% critic_loss={_cl} actor_loss={_al} "
                      f"| {transitions/elapsed:.1f} transitions/s", flush=True)
                last_log_t = transitions

            if transitions % args.save_interval < args.num_envs:
                ck_path = os.path.join(args.output_dir, f"residual_{transitions}.pt")
                agent.save(ck_path)
                agent.save(os.path.join(args.output_dir, "latest.pt"))
                print(f"  [SAVE] {ck_path}", flush=True)

        print(f"\n[DONE] {ep_count} episodes, overall success {success_count}/{max(ep_count,1)} "
              f"= {success_count/max(ep_count,1)*100:.1f}%", flush=True)

    except Exception:
        import traceback
        traceback.print_exc()
    finally:
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


if __name__ == "__main__":
    main()
