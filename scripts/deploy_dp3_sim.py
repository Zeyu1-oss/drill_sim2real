import argparse
import os
import sys
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (scripts/ -> ../..)
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
from perception.groundtruth_mask import DEFAULT_MASK_THRESHOLD as _GT_MASK_THRESHOLD
from perception.camera_setup import (load_perception_hp as _load_perception_hp,
                                     ensure_rgb_aov, get_cameras, detect_wrist_cam,
                                     apply_render_settings, setup_ground)
init_fps_kernel("/home/zeyu/3D-Diffusion-Policy/third_party/pytorch3d_simplified")


def parse_args():
    _PERC = _load_perception_hp() 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dp3_ckpt', type=str, default='/home/zeyu/3D-Diffusion-Policy/3D-Diffusion-Policy/data/outputs/inspire_drill-simple_dp3-simple_dp3_seed0/checkpoints/latest.ckpt')
    parser.add_argument("--num_envs", type=int, default=9)
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--num_inference_steps', type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--drill_configs", type=str, default=None)
    parser.add_argument('--drill_variants', type=str, default=None)
    parser.add_argument('--img_height', type=int, default=_PERC.img_height)
    parser.add_argument("--img_width", type=int, default=_PERC.img_width)
    parser.add_argument('--pc_num_points', type=int, default=1024)
    parser.add_argument('--disable_cam2', action='store_true',
                        help="cam1 alone supplies the whole camera segment (all pc_num_points); "
                             "cam2 (wrist cam) is not sampled into the point cloud. Must match "
                             "how the checkpoint's training data was collected "
                             "(collect_dp3_data.py --disable_cam2).")
    parser.add_argument('--robot_pc_points', type=int, default=160)
    parser.add_argument('--ground_points', type=int, default=0)
    parser.add_argument('--workspace', nargs=6, type=float, default=None)
    parser.add_argument('--pc_mode', type=str, default='robot', choices=['camera', 'robot', 'robot_drill'])
    parser.add_argument('--robot_pc_npz', type=str, default='assets/inspire_tac/robot_canonical_points.npz')
    parser.add_argument('--mask_threshold', type=float, default=_GT_MASK_THRESHOLD,
                        help="is_handle radius (m) for the 4th point-cloud channel. Single source of "
                             "truth is perception/groundtruth_mask.DEFAULT_MASK_THRESHOLD, shared "
                             "with collect_dp3_data.py; a different radius here is a different label "
                             "definition, i.e. an out-of-distribution 4th channel at deploy time.")
    parser.add_argument('--chained', action='store_true')
    parser.add_argument('--stage1_only', action='store_true')
    parser.add_argument('--stage2_only', action='store_true',
                        help="deploy DP3 on the alignment-only task: drill starts already grasped "
                             "(teleported in via --success_dataset, same pkl collect_dp3_data.py "
                             "--stage2_only / play_stage2.py use). Point cloud = cam2(wrist)+cam3(plate), "
                             "no separate oracle plate-crop segment. Uses Stage2Env directly (not "
                             "ChainedEnv). Must match the checkpoint's training data "
                             "(collect_dp3_data.py --stage2_only). Not compatible with --dual/--policy rl.")
    parser.add_argument('--success_dataset', type=str, default='collected_data/success_data_dp3.pkl',
                        help="pkl of successful stage1 grasp end-states (drill pose + joint pos), used "
                             "by --stage2_only to reset directly into an already-grasped configuration")
    parser.add_argument('--plate_pc_points', type=int, default=512)
    parser.add_argument('--drill_mesh_points', type=int, default=512)
    parser.add_argument('--success_hold_stop', type=int, default=1)
    parser.add_argument('--episode_length_s', type=float, default=10.0)
    parser.add_argument('--init_pose_file', type=str, default=None,)
    parser.add_argument('--exam_n', type=int, default=200)   # sequential-exam poses per variant (only used with --init_pose_file)
    parser.add_argument('--fixed_plate', action='store_true')
    parser.add_argument('--exec_horizon', type=int, default=None)
    parser.add_argument('--lag_comp', type=int, default=0)
    parser.add_argument('--timeout_only', type=lambda s: str(s).lower() in ('1', 'true', 't', 'yes'), nargs='?', const=False, default=False)
    parser.add_argument('--policy', type=str, default='dp3', choices=['dp3', 'rl'])
    parser.add_argument('--stage1_checkpoint', type=str, default='runs/inspire_hand_grasp_drill_26-06-13-21-10/inspire_hand_grasp_drill_26-06-13-21-10/nn/inspire_hand_grasp_drill_26-06-13-21-10.pth')
    parser.add_argument('--stage2_checkpoint', type=str, default='runs/stage2_26-07-11-18-36/stage2_26-07-11-18-36/nn/stage2_26-07-11-18-36.pth')
    parser.add_argument('--rl_checkpoint', type=str, default=None)
    parser.add_argument('--dual', action='store_true')
    parser.add_argument('--driver', type=str, default='rl', choices=['rl', 'dp3'])
    parser.add_argument('--dual_trace', type=str, default=None)
    parser.add_argument('--stage2_dp3_ckpt', type=str, default=None,
                        help="two-stage DP3 state machine: --dp3_ckpt drives GRASP, and once "
                             "ChainedEnv flips an env's phase 0->1 (20-in-last-50 lenient success "
                             "window, same criterion as --collect_success_data), THIS checkpoint "
                             "takes over driving ALIGN. "
                             "Both states' camera compositions + workspaces are hardcoded inside "
                             "run_dp3_two_stage to match exactly how the two real checkpoints were "
                             "collected. Requires --chained (auto-enabled), not --stage1_only/"
                             "--stage2_only/--dual/--policy rl.")
    parser.add_argument('--stage2_rl', action='store_true',
                        help="hybrid state machine: --dp3_ckpt (DP3, point cloud) drives GRASP, and "
                             "once ChainedEnv flips phase 0->1 the PRIVILEGED RL stage2 teacher "
                             "(--stage2_checkpoint, 78-dim state obs) takes over ALIGN. Same env, "
                             "same GRASP composition and same phase-switch criterion as "
                             "--stage2_dp3_ckpt, so the two runs differ only in who drives ALIGN: "
                             "the gap between them is the ALIGN distillation gap, and this run's "
                             "end-to-end rate is an upper bound on what the Grasp Student's "
                             "hand-off supports. Requires --chained (auto-enabled); mutually "
                             "exclusive with --stage2_dp3_ckpt/--stage1_only/--stage2_only/--dual/"
                             "--policy rl/--collect_success_data.")
    parser.add_argument('--collect_success_data', action='store_true',
                        help="")
    parser.add_argument('--num_samples', type=int, default=1000,
                        help="target number of success samples for --collect_success_data "
                             "(ignored when --samples_per_variant is set)")
    parser.add_argument('--samples_per_variant', type=int, default=None,
                        help="--collect_success_data: exact per-variant quota (e.g. 500 -> "
                             "500 x num_active_variants total); once a variant hits its quota its "
                             "further successes are not collected (episodes keep running/resetting "
                             "normally, mirrors collect_dp3_data.py's --episodes_per_variant). "
                             "Overrides --num_samples.")
    parser.add_argument('--output', type=str, default='collected_data/success_data.pkl',
                        help="pkl output path for --collect_success_data")
    # ---- removable block: --rate_limit (see perception/target_drive.install_direct_drive) ----
    parser.add_argument('--rate_limit',
                        type=lambda s: str(s).lower() in ('1', 'true', 't', 'yes'),
                        nargs='?', const=True, default=True,
                        help="clamp the commanded joint target to the same per-substep slew limit "
                             "raw_to_target applies when generating training labels (arm "
                             "+/-max_joint_delta, fingers +/-max_finger_delta). Training labels "
                             "never exceed it (0.00%% of steps) but raw DP3 output does (measured "
                             "15-26%% of arm steps, up to 16x), because DP3's q* never passes "
                             "through raw_to_target. Default on; pass '--rate_limit false' to "
                             "restore the old unclamped behaviour for an A/B.")
    parser.add_argument('--dump_obs_zarr', type=str, default=None,
                        help="debug: write the observations the DEPLOYED policy actually sees into a "
                             "zarr with collect_dp3_data.py's exact layout (data/point_cloud, "
                             "data/state, data/action, data/pc_mask when 4-channel, meta/episode_ends) "
                             "-- same timing convention (row = obs at decision time + the q* executed "
                             "that step) and the same bad-frame skipping, so it can be diffed 1:1 "
                             "against the training zarr. Only --dump_obs_env's first "
                             "--dump_obs_episodes episode(s). Main single-policy path only.")
    parser.add_argument('--dump_obs_env', type=int, default=0,
                        help="--dump_obs_zarr: which env to record (default 0)")
    parser.add_argument('--dump_obs_episodes', type=int, default=1,
                        help="--dump_obs_zarr: how many episodes of that env to record (default 1)")
    args = parser.parse_args()
    args.save_robot_pc = args.pc_mode in ("robot", "robot_drill")
    args.save_drill_mesh_pc = args.pc_mode == "robot_drill"
    if args.workspace is None:
        args.workspace = list(_PERC.chained_workspace
                              if (args.chained or args.stage2_only or args.stage2_dp3_ckpt is not None)
                              else _PERC.workspace)
    return args



def _build_teachers(env_unwrapped, args):
    """Build the RL teacher player (chained=two / non-chained=one); give obs/act space explicitly, no self-built env.
    Return (mode, agent1, agent2, d1, d2); in single mode agent2/d2 are None. Same as collect."""
    import yaml
    import importlib.util as _ilu

    def _resolve(p):
        return p if (p is None or os.path.isabs(p)) else os.path.join(project_root, p)

    _pc_spec = _ilu.spec_from_file_location(
        "_play_chained", os.path.join(project_root, "scripts", "play_chained.py"))
    _pc = _ilu.module_from_spec(_pc_spec)
    _pc_spec.loader.exec_module(_pc)

    with open(os.path.join(project_root, "config/agents/rl_games_ppo_cfg.yaml")) as f:
        agent_cfg = yaml.safe_load(f)
    agent_cfg["params"]["config"]["num_actors"] = args.num_envs
    agent_cfg["params"]["config"]["device"] = args.device
    agent_cfg["params"]["config"]["device_name"] = args.device
    act_dim = int(env_unwrapped.cfg.action_space)

    if args.chained:
        s1, s2 = _resolve(args.stage1_checkpoint), _resolve(args.stage2_checkpoint)
        for _p in (s1, s2):
            if not os.path.exists(_p):
                raise FileNotFoundError(f"RL teacher checkpoint not found: {_p}")
        d1 = _pc._ckpt_obs_dim(s1)
        d2 = _pc._ckpt_obs_dim(s2)
        agent1 = _pc._build_player(agent_cfg, s1, d1, act_dim, args.num_envs, args.device)
        agent2 = _pc._build_player(agent_cfg, s2, d2, act_dim, args.num_envs, args.device)
        print(f"[RL-TEACHER] chained dual teacher: stage1 obs={d1}, stage2 obs={d2}", flush=True)
        return "chained", agent1, agent2, d1, d2
    rlck = _resolve(args.rl_checkpoint or args.stage1_checkpoint)
    if not os.path.exists(rlck):
        raise FileNotFoundError(f"RL teacher checkpoint not found: {rlck}")
    d1 = _pc._ckpt_obs_dim(rlck)
    agent = _pc._build_player(agent_cfg, rlck, d1, act_dim, args.num_envs, args.device)
    print(f"[RL-TEACHER] single teacher: obs={d1}", flush=True)
    return "single", agent, None, d1, None


def _teacher_raw_action(mode, agent1, agent2, d1, obs_priv, env_unwrapped, device):
    """Pick teacher raw action ([-1,1]) by phase, same as collect/play_chained."""
    if mode == "chained":
        a1 = agent1.get_action(agent1.obs_to_torch(obs_priv[:, :d1]), is_deterministic=True)
        a2 = agent2.get_action(agent2.obs_to_torch(obs_priv), is_deterministic=True)
        traw = torch.where((env_unwrapped.phase == 1).unsqueeze(1), a2, a1)
    else:
        traw = agent1.get_action(agent1.obs_to_torch(obs_priv), is_deterministic=True)
    if not isinstance(traw, torch.Tensor):
        traw = torch.from_numpy(np.array(traw)).float().to(device)
    return traw


def _build_stage2_rl_player(env_unwrapped, args):
    """Build ONLY the stage2 RL teacher player (--stage2_checkpoint), for --stage2_rl's ALIGN state.

    _build_teachers() cannot be reused here: with args.chained it insists on both stage1 and stage2
    checkpoints, and --stage2_rl has no use for the stage1 teacher (DP3 drives GRASP).
    Returns (player, ckpt_obs_dim)."""
    import yaml
    import importlib.util as _ilu

    _pc_spec = _ilu.spec_from_file_location(
        "_play_chained", os.path.join(project_root, "scripts", "play_chained.py"))
    _pc = _ilu.module_from_spec(_pc_spec)
    _pc_spec.loader.exec_module(_pc)

    ckpt = args.stage2_checkpoint
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(project_root, ckpt)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"--stage2_rl needs --stage2_checkpoint, not found: {ckpt}")

    with open(os.path.join(project_root, "config/agents/rl_games_ppo_cfg.yaml")) as f:
        agent_cfg = yaml.safe_load(f)
    agent_cfg["params"]["config"]["num_actors"] = args.num_envs
    agent_cfg["params"]["config"]["device"] = args.device
    agent_cfg["params"]["config"]["device_name"] = args.device

    obs_dim = _pc._ckpt_obs_dim(ckpt)
    player = _pc._build_player(agent_cfg, ckpt, obs_dim, int(env_unwrapped.cfg.action_space),
                               args.num_envs, args.device)
    return player, obs_dim


def run_rl_teacher_eval(env_unwrapped, simulation_app, args):
    """Drive with the RL teacher (privileged obs) inside the exact same eval env as DP3 deploy.
    Only the policy changes: env / init_pose replay / success counting / episode management are byte-for-byte identical to the DP3 branch,
    so the difference in success rate between the two runs = pure teacher-student gap (same eval criterion)."""
    from perception.target_drive import install_direct_drive, raw_to_target

    mode, agent1, agent2, d1, d2 = _build_teachers(env_unwrapped, args)
    _drive_p = install_direct_drive(env_unwrapped)   # teacher raw -> q*, same source as collect

    if args.timeout_only:
        _orig_get_dones = env_unwrapped._get_dones

        def _get_dones_timeout_only():
            # still call the original _get_dones to update the success cache (_cached_lenient_success etc.). In the termination condition
            # keep only NaN (physically diverged envs must be recycled, else bad poses spam USD warnings each frame and pollute stats),
            # all other conditions (success/drop/joint-limit) do not terminate, letting the episode run to timeout (truncated).
            terminated, truncated = _orig_get_dones()
            nan_mask = getattr(env_unwrapped, "_pending_nan_mask", None)
            term = (nan_mask.clone() if nan_mask is not None else torch.zeros_like(terminated))
            return term, truncated

        env_unwrapped._get_dones = _get_dones_timeout_only

    obs = env_unwrapped._get_observations()["policy"]
    if mode == "chained":
        agent1.get_batch_size(obs[:, :d1], 1); agent1.reset()
        agent2.get_batch_size(obs, 1); agent2.reset()
    else:
        agent1.get_batch_size(obs, 1); agent1.reset()

    # ---- success counting: same criterion as the DP3 branch ----
    env_episode_rewards = torch.zeros(args.num_envs, device=args.device)
    total_episodes = successful_episodes = grasp_successful_episodes = stable_grasp_episodes = 0
    ever_grasp = np.zeros(args.num_envs, dtype=bool)
    grasp_switch_steps = []
    success_log = []
    episode_rewards = []
    total_steps = 0
    start_time = time.time()
    print(f"\n[RL-TEACHER] rollout (target {args.num_episodes} episodes, {args.num_envs} envs); "
          f"eval: episode={args.episode_length_s}s success_hold_stop={args.success_hold_stop} "
          f"init_pose_file={'yes' if args.init_pose_file else 'no'}", flush=True)
    sys.stdout.flush()

    with torch.inference_mode():
        while simulation_app.is_running() and total_episodes < args.num_episodes:
            teacher_raw = _teacher_raw_action(mode, agent1, agent2, d1, obs,
                                              env_unwrapped, args.device)
            cur0 = env_unwrapped.cur_targets.clone()
            env_unwrapped._direct_target = raw_to_target(teacher_raw, cur0, _drive_p)
            obs_dict, rewards, terminated, truncated, _ = env_unwrapped.step(teacher_raw)
            obs = obs_dict["policy"]
            total_steps += 1

            env_episode_rewards += rewards
            is_done = terminated.bool() | truncated.bool()
            try:
                lenient_success = env_unwrapped._cached_lenient_success
            except AttributeError:
                lenient_success = torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)
            if args.chained:
                _inst = getattr(env_unwrapped, "_cached_instant_success", None)
                if _inst is not None:
                    ever_grasp |= _inst.detach().cpu().numpy().astype(bool)

            finished = torch.where(is_done)[0]
            for env_idx in finished:
                env_id = env_idx.item()
                total_episodes += 1
                episode_rewards.append(env_episode_rewards[env_id].item())
                if lenient_success[env_id]:
                    successful_episodes += 1; success_log.append(1)
                else:
                    success_log.append(0)
                if args.chained:
                    if ever_grasp[env_id]:
                        grasp_successful_episodes += 1
                    ever_grasp[env_id] = False
                    _sw = int(env_unwrapped._last_episode_switch_step[env_id].item())
                    if _sw >= 0:
                        stable_grasp_episodes += 1; grasp_switch_steps.append(_sw)
                env_episode_rewards[env_id] = 0.0
            if is_done.any():
                agent1.reset()
                if mode == "chained":
                    agent2.reset()

            if total_steps % 100 == 0:
                elapsed = time.time() - start_time
                sr = successful_episodes / max(total_episodes, 1) * 100
                _grasp = (f"| grasp={grasp_successful_episodes} "
                          f"({grasp_successful_episodes / max(total_episodes, 1) * 100:.1f}%) "
                          f"stable={stable_grasp_episodes} " if args.chained else "")
                print(f"  step={total_steps} | eps={total_episodes}/{args.num_episodes} "
                      f"| succ={successful_episodes} ({sr:.1f}%) {_grasp}"
                      f"| {total_steps/elapsed:.1f} step/s", flush=True)
                sys.stdout.flush()

    print(f"\n{'='*60}")
    print("[RL-TEACHER] === teacher (privileged obs) results in the DP3 eval env ===")
    if total_episodes > 0:
        print(f"Success rate: {successful_episodes/total_episodes*100:.1f}%")
        if args.chained:
            print(f"Grasp success (>=1 step): {grasp_successful_episodes} "
                  f"({grasp_successful_episodes/total_episodes*100:.1f}%)")
            print(f"Grasp stable (entered stage2): {stable_grasp_episodes} "
                  f"({stable_grasp_episodes/total_episodes*100:.1f}%)")
            if stable_grasp_episodes > 0:
                print(f"Align | stable: {successful_episodes/stable_grasp_episodes*100:.1f}%")
    if episode_rewards:
        ep = np.array(episode_rewards)
        print(f"Reward: mean={ep.mean():.1f} max={ep.max():.1f} min={ep.min():.1f}")
    if success_log:
        print(f"Success log: {success_log}")
    print(f"{'='*60}")


def _load_dp3_ckpt_for_two_stage(ckpt_path, expected_pc, device, num_inference_steps_override, tag):
    """Minimal, self-contained DP3 checkpoint loader for run_dp3_two_stage (separate from the main
    single-checkpoint loading code below to avoid touching that already-working path)."""
    from diffusion_policy_3d.policy.dp3 import DP3
    import dill
    from hydra.utils import instantiate as hydra_instantiate
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    payload = torch.load(ckpt_path, pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]

    def _get(path_fns, default=None):
        for fn in path_fns:
            try:
                return fn()
            except Exception:
                continue
        return default

    agent_dim = _get([lambda: int(cfg.task.shape_meta.obs.agent_pos.shape[0]),
                      lambda: int(cfg.shape_meta.obs.agent_pos.shape[0])], 13)
    with_force = agent_dim > 13
    ckpt_pc = _get([lambda: int(cfg.task.shape_meta.obs.point_cloud.shape[0]),
                    lambda: int(cfg.shape_meta.obs.point_cloud.shape[0])], None)
    if ckpt_pc is not None and ckpt_pc != expected_pc:
        raise RuntimeError(
            f"[{tag}] {ckpt_path}: point_cloud={ckpt_pc} != expected {expected_pc} "
            f"(run_dp3_two_stage's hardcoded camera composition for this state)")

    policy: DP3 = hydra_instantiate(cfg.policy)
    policy.to(device); policy.eval()
    policy.load_state_dict(payload["state_dicts"]["model"], strict=False)
    if num_inference_steps_override is not None:
        policy.num_inference_steps = num_inference_steps_override

    has_ema = "ema_model" in payload["state_dicts"]
    if has_ema:
        policy_ema: DP3 = hydra_instantiate(cfg.policy)
        policy_ema.to(device)
        policy_ema.load_state_dict(payload["state_dicts"]["ema_model"], strict=False)
        if num_inference_steps_override is not None:
            policy_ema.num_inference_steps = num_inference_steps_override
        policy_ema.eval()
    else:
        policy_ema = policy

    def _norm_ready(p):
        return len(p.normalizer.params_dict) > 0

    if _norm_ready(policy):
        policy.normalizer.to(device)
        if has_ema:
            if not _norm_ready(policy_ema):
                policy_ema.set_normalizer(policy.normalizer)
            policy_ema.normalizer.to(device)
    elif "pickles" in payload and "normalizer" in payload["pickles"]:
        normalizer = dill.loads(payload["pickles"]["normalizer"])
        policy.set_normalizer(normalizer); policy.normalizer.to(device)
        if has_ema:
            policy_ema.set_normalizer(normalizer); policy_ema.normalizer.to(device)
    else:
        raise RuntimeError(f"[{tag}] {ckpt_path}: no embedded normalizer, cannot deploy")

    print(f"  [{tag}] point_cloud={ckpt_pc} agent_pos={agent_dim} (force_state={with_force}) "
          f"n_obs={policy.n_obs_steps} n_act={policy.n_action_steps} horizon={policy.horizon}", flush=True)
    return dict(policy_ema=policy_ema, with_force=with_force, agent_dim=agent_dim,
               n_obs=policy.n_obs_steps, n_act=policy.n_action_steps, horizon=policy.horizon)


def run_collect_success_data(env_unwrapped, simulation_app, args, cam1, PERC):
    """Run the GRASP/stage1 DP3 checkpoint (--dp3_ckpt) and record the final drill pose + robot
    joint pose of every successful grasp into a pkl -- the SAME format scripts/collect_success_data.py
    produces (consumed by SuccessDataDataset / Stage2Env / --stage2_only), just sourced from DP3
    instead of the RL teacher, so stage2's starting-state distribution matches what a real DP3
    stage1 hand-off actually produces.

    Camera composition is hardcoded to match --stage1_only exactly: cam1 alone 2048 + robot 512 =
    2560, PERC.workspace (the narrow box) -- see run_dp3_two_stage's docstring for why the workspace
    choice matters (a standalone --stage1_only run has always resolved to the narrow box, and that's
    what the checkpoint's training data was collected under).

    Caching convention mirrors collect_success_data.py's history_len=0 case: every step, snapshot
    (agent_pos, joint_pos, drill pos/quat) BEFORE stepping; whenever the step's resulting
    `_cached_lenient_success` is True, overwrite that env's cache with the pre-step snapshot. The
    cache is refreshed every success step, so by the time the episode actually ends the cache holds
    the LAST state that was still in success -- the same "final grasp pose" semantics as the
    original collector, just with DP3 driving instead of the RL teacher."""
    from perception.target_drive import install_direct_drive
    from perception.student_obs import build_agent_pos
    from perception.robot_pointcloud import RobotPointCloudFK
    from collections import deque, Counter
    import pickle

    CAM_PTS = 2048
    ROBOT_PTS = 512
    workspace = tuple(PERC.workspace)   # narrow box -- matches --stage1_only, see run_dp3_two_stage

    install_direct_drive(env_unwrapped, rate_limit=args.rate_limit)

    _npz = args.robot_pc_npz if os.path.isabs(args.robot_pc_npz) else os.path.join(project_root, args.robot_pc_npz)
    robot_fk = RobotPointCloudFK(_npz, list(env_unwrapped.franka.body_names), args.device, max_points=ROBOT_PTS)
    total_pc = CAM_PTS + robot_fk.num_points

    print(f"[COLLECT] GRASP composition (hardcoded): cam1 alone {CAM_PTS} + robot {robot_fk.num_points} "
          f"= {total_pc} pts, workspace={workspace} | ckpt={args.dp3_ckpt}", flush=True)
    if args.pc_num_points != 1024 or args.disable_cam2 or args.robot_pc_points != 160:
        print("[COLLECT] NOTE: --pc_num_points/--disable_cam2/--robot_pc_points are ignored in "
              "--collect_success_data mode (camera config is hardcoded, see above)", flush=True)

    ck = _load_dp3_ckpt_for_two_stage(args.dp3_ckpt, total_pc, args.device, args.num_inference_steps, "GRASP")
    n_obs, n_act, horizon = ck["n_obs"], ck["n_act"], ck["horizon"]
    if args.lag_comp < 0 or n_obs - 1 + args.lag_comp + n_act > horizon:
        raise ValueError(f"lag_comp={args.lag_comp} invalid (need 0<=lag_comp and "
                         f"(n_obs-1)+lag_comp+n_act<=horizon)")
    exec_n = n_act if args.exec_horizon is None else max(1, min(args.exec_horizon, n_act))

    env_origins = env_unwrapped.scene.env_origins
    controlled = env_unwrapped.controlled_joint_indices
    dt = env_unwrapped.step_dt

    def build_obs():
        ws = camera_crop_bounds(env_unwrapped.drill.data.root_pos_w, env_origins, PERC, workspace)
        pc1, _, _ = camera_pc(cam1, ws, CAM_PTS, env_origins)
        pcf = torch.cat([pc1, robot_fk(env_unwrapped.franka.data.body_pos_w,
                                       env_unwrapped.franka.data.body_quat_w, env_origins)], dim=1)
        agent_pos = build_agent_pos(env_unwrapped, ck["with_force"])
        return pcf, agent_pos

    env_obs_histories = [deque(maxlen=n_obs + n_act) for _ in range(args.num_envs)]
    pending_actions = [None] * args.num_envs
    pending_idx = [0] * args.num_envs

    pc_all, agent_all = build_obs()
    for e in range(args.num_envs):
        for _ in range(n_obs):
            env_obs_histories[e].append({"agent_pos": agent_all[e], "point_cloud": pc_all[e]})

    # per-env success cache (overwritten every step current_success holds; read out at episode end)
    cache_valid = torch.zeros(args.num_envs, dtype=torch.bool)
    cache_agent = agent_all.detach().cpu().clone()
    cache_jp = env_unwrapped.franka.data.joint_pos[:, controlled].detach().cpu().clone()
    cache_dp = (env_unwrapped.drill.data.root_pos_w - env_origins).detach().cpu().clone()
    cache_dq = env_unwrapped.drill.data.root_quat_w.detach().cpu().clone()
    cache_vid = env_unwrapped._drill_variant_indices.detach().cpu().clone()

    env_episode_rewards = torch.zeros(args.num_envs, device=args.device)
    collected_samples = []
    total_steps = 0
    start_time = time.time()

    active_vids = sorted(v.variant_index for v in env_unwrapped.drill_variants)
    per_variant_target = args.samples_per_variant
    per_variant_count = {vid: 0 for vid in active_vids}
    target_total = (per_variant_target * len(active_vids)) if per_variant_target is not None else args.num_samples

    def _collection_done():
        if per_variant_target is not None:
            return all(per_variant_count[v] >= per_variant_target for v in active_vids)
        return len(collected_samples) >= args.num_samples

    if per_variant_target is not None:
        print(f"\n[COLLECT] rollout starting (target {per_variant_target} x {len(active_vids)} "
              f"variants = {target_total} success samples, {args.num_envs} envs)", flush=True)
    else:
        print(f"\n[COLLECT] rollout starting (target {args.num_samples} success samples, "
              f"{args.num_envs} envs)", flush=True)
    sys.stdout.flush()

    with torch.inference_mode():
        while simulation_app.is_running() and not _collection_done():
            need = [e for e in range(args.num_envs)
                   if pending_actions[e] is None or pending_idx[e] >= exec_n]
            if need:
                obs_batch, pc_batch = [], []
                for e in need:
                    hist = list(env_obs_histories[e])[-n_obs:]
                    if len(hist) < n_obs:
                        hist = [hist[0]] * (n_obs - len(hist)) + hist
                    obs_batch.append(torch.stack([h["agent_pos"] for h in hist]))
                    pc_batch.append(torch.stack([h["point_cloud"] for h in hist]))
                obs_dict = {"agent_pos": torch.stack(obs_batch), "point_cloud": torch.stack(pc_batch)}
                result = ck["policy_ema"].predict_action(obs_dict)
                cs = n_obs - 1 + args.lag_comp
                chunks = result["action_pred"][:, cs:cs + n_act]
                for i, e in enumerate(need):
                    pending_actions[e] = chunks[i]
                    pending_idx[e] = 0

            actions = torch.zeros((args.num_envs, 13), device=args.device, dtype=torch.float32)
            for e in range(args.num_envs):
                actions[e] = pending_actions[e][pending_idx[e]]
                pending_idx[e] += 1

            # snapshot BEFORE stepping (this step's action is about to be applied to THIS state)
            pre_agent = agent_all.detach().cpu().clone()
            pre_jp = env_unwrapped.franka.data.joint_pos[:, controlled].detach().cpu().clone()
            pre_dp = (env_unwrapped.drill.data.root_pos_w - env_origins).detach().cpu().clone()
            pre_dq = env_unwrapped.drill.data.root_quat_w.detach().cpu().clone()
            pre_vid = env_unwrapped._drill_variant_indices.detach().cpu().clone()

            env_unwrapped._direct_target = actions
            obs_dict_step, rewards, terminated, truncated, _ = env_unwrapped.step(actions)
            total_steps += 1
            env_episode_rewards += rewards

            cam1.update(dt)

            try:
                lenient_success = env_unwrapped._cached_lenient_success
            except AttributeError:
                lenient_success = torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)
            is_done = terminated.bool() | truncated.bool()

            success_now = lenient_success.detach().cpu()
            if success_now.any():
                cache_valid |= success_now
                cache_agent[success_now] = pre_agent[success_now]
                cache_jp[success_now] = pre_jp[success_now]
                cache_dp[success_now] = pre_dp[success_now]
                cache_dq[success_now] = pre_dq[success_now]
                cache_vid[success_now] = pre_vid[success_now]

            pc_all, agent_all = build_obs()

            finished = torch.where(is_done)[0]
            for env_idx in finished:
                e = env_idx.item()
                vid = int(cache_vid[e].item())
                _quota_ok = (per_variant_count[vid] < per_variant_target
                            if per_variant_target is not None else len(collected_samples) < args.num_samples)
                if cache_valid[e] and _quota_ok:
                    vname = env_unwrapped._variant_attrs.get(vid, {}).get("name", f"variant_{vid}")
                    win_sum = (int(env_unwrapped._success_window[e].sum().item())
                              if hasattr(env_unwrapped, "_success_window") else 0)
                    collected_samples.append({
                        # DP3's own agent_pos vector (13 or 26-dim), NOT rl_games' policy obs vector
                        "full_obs": cache_agent[e].numpy(),
                        "joint_pos": cache_jp[e].numpy(),
                        "joint_vel": np.zeros_like(cache_jp[e].numpy()),   # not tracked by this collector
                        "drill_pos": cache_dp[e].numpy(),
                        "drill_quat": cache_dq[e].numpy(),
                        "drill_lin_vel": np.zeros(3, dtype=np.float32),
                        "drill_ang_vel": np.zeros(3, dtype=np.float32),
                        "variant_idx": vid,
                        "variant_name": vname,
                        "episode_reward": float(env_episode_rewards[e].item()),
                        "success_hold_steps": win_sum,
                        "success_window_sum": win_sum,
                    })
                    per_variant_count[vid] += 1
                cache_valid[e] = False
                env_episode_rewards[e] = 0.0

            for e in range(args.num_envs):
                if not is_done[e]:
                    env_obs_histories[e].append({"agent_pos": agent_all[e], "point_cloud": pc_all[e]})
                else:
                    pending_actions[e] = None
                    pending_idx[e] = 0
                    env_obs_histories[e].clear()
                    for _ in range(n_obs):
                        env_obs_histories[e].append({"agent_pos": agent_all[e], "point_cloud": pc_all[e]})

            if total_steps % 100 == 0:
                elapsed = time.time() - start_time
                n = len(collected_samples)
                eta = (elapsed / max(n, 1)) * (target_total - n)
                em, es = divmod(int(eta), 60); eh, em = divmod(em, 60)
                _pv = (" | " + " ".join(f"v{v}={per_variant_count[v]}/{per_variant_target}"
                                       for v in active_vids)
                      if per_variant_target is not None else "")
                print(f"  step={total_steps} | collected={n}/{target_total}{_pv} "
                      f"| {total_steps/elapsed:.1f} step/s | ETA={eh}h{em}m{es}s", flush=True)
                sys.stdout.flush()

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    metadata = {
        "num_samples": len(collected_samples),
        "obs_dim": collected_samples[0]["full_obs"].shape[-1] if collected_samples else 0,
        "joint_pos_dim": collected_samples[0]["joint_pos"].shape[-1] if collected_samples else 0,
        "joint_vel_dim": collected_samples[0]["joint_vel"].shape[-1] if collected_samples else 0,
        "version": "dp3-1.0",
        "history_len": 0,
        "description": (
            "Success episode data for grasp_drill task, collected by DP3 (--dp3_ckpt) via "
            "deploy_dp3_sim.py --collect_success_data, NOT the RL teacher. full_obs is DP3's own "
            "agent_pos vector (13 or 26-dim), not rl_games' policy obs vector. joint_vel/"
            "drill_lin_vel/drill_ang_vel are not tracked by this collector (zero-filled) -- "
            "Stage2Env._reset_idx never reads them anyway. Stores the LAST state at which "
            "lenient_success still held (cache refreshed every success step). drill_pos is in "
            "LOCAL coordinates (relative to env_origin). joint_pos is 13-dim controlled joints only."
        ),
    }
    with open(args.output, "wb") as f:
        pickle.dump({"metadata": metadata, "samples": collected_samples}, f)

    elapsed = time.time() - start_time
    m, s = divmod(int(elapsed), 60); h, m = divmod(m, 60)
    print(f"\n{'='*60}")
    print(f"[COLLECT] collection complete: {len(collected_samples)}/{target_total} samples")
    print(f"  total steps: {total_steps:,}  time: {h}h{m}m{s}s  rate: {total_steps/elapsed:.0f} step/s")
    print(f"  output: {args.output}")
    print(f"{'='*60}")
    if collected_samples:
        counts = Counter(s["variant_name"] for s in collected_samples)
        print("Per-variant collection stats:")
        for name, count in sorted(counts.items()):
            print(f"  {name}: {count} ({count/len(collected_samples)*100:.1f}%)")


GRASP, ALIGN = 0, 1   # state-machine states; match ChainedEnv.phase's own 0/1 convention exactly


def run_dp3_two_stage(env_unwrapped, simulation_app, args, cam1, cam2, cam3, cam2_pose_fn, dt, PERC,
                      align_driver="dp3"):
    """State machine over ChainedEnv.phase (0=GRASP, 1=ALIGN), one policy per state.

    align_driver selects who drives ALIGN:
      "dp3" (--stage2_dp3_ckpt): the align DP3 Student, on its own point cloud.
      "rl"  (--stage2_rl):       the privileged RL stage2 teacher, on the env's 78-dim state obs.
                                 GRASP is untouched, so this run and the "dp3" run differ ONLY in
                                 who drives ALIGN -- the difference between their end-to-end rates
                                 is the ALIGN distillation gap, and this run's rate is an upper
                                 bound on what the Grasp Student's hand-off can support. No ALIGN
                                 point cloud is built and cam2/cam3 are not updated in this mode
                                 (the RL teacher reads simulator state, not depth).

    Each DP3 state's camera composition AND workspace are HARDCODED to be byte-for-byte identical
    to running that state standalone (--stage1_only / --stage2_only), since the two real checkpoints
    were collected under exactly those standalone commands:

      GRASP (0): cam1 alone (cam2 disabled) 2048 pts + robot 512 pts = 2560.
                 workspace = PERC.workspace (the NARROW box). This is not a style choice: it is the
                 actual box --stage1_only alone resolves to, because parse_args() computes the
                 workspace default from args.chained, which is still False at that point even
                 though --stage1_only later auto-enables it inside main() -- so a standalone
                 `--stage1_only` run (no explicit --chained) has ALWAYS used the narrow box, and
                 that's what the checkpoint's training data was collected under too. Using the WIDE
                 chained_workspace here (an earlier version of this function did) crops a visibly
                 different region of space and corrupts the point cloud from step 1.
      ALIGN (1): cam2+cam3 fused 1024 pts (512 each) + robot 160 pts = 1184.
                 workspace = PERC.chained_workspace (the WIDE box -- what --stage2_only/--chained
                 resolve to; ALIGN needs to see the plate, which sits outside the narrow box).

    Neither composition includes an oracle plate-crop segment (matching how both were collected).
    ChainedEnv's own phase 0->1 switch (20-in-last-50 lenient success window, same criterion as
    --collect_success_data, see tasks/chained_env.py) drives the transition; this function only
    reacts to it -- swap which
    checkpoint/camera-builder answers an env's next policy call, and reseed that env's observation
    history from a clean read of the NEW state's own point cloud (a history built from GRASP's 2560
    2-cam1-only points can't be reused for ALIGN's 1184 cam2+cam3 points, or vice versa)."""
    from perception.target_drive import install_direct_drive, raw_to_target
    from perception.student_obs import build_agent_pos
    from perception.robot_pointcloud import RobotPointCloudFK
    from collections import deque

    align_is_rl = (align_driver == "rl")
    tag = "HYBRID" if align_is_rl else "2STAGE"
    # DP3 action_pred is already q*; the RL teacher's raw action needs raw_to_target(_drive_p).
    # rate_limit only ever bites on the DP3 (GRASP) half -- the RL half's q* comes out of
    # raw_to_target and already satisfies the same bound.
    _drive_p = install_direct_drive(env_unwrapped, rate_limit=args.rate_limit)

    _npz = args.robot_pc_npz if os.path.isabs(args.robot_pc_npz) else os.path.join(project_root, args.robot_pc_npz)
    _body_names = list(env_unwrapped.franka.body_names)

    # ---- per-state config, hardcoded -- see docstring ----
    dp3_states = (GRASP,) if align_is_rl else (GRASP, ALIGN)
    _fk_pts = {GRASP: 512, ALIGN: 160}
    robot_fk = {s: RobotPointCloudFK(_npz, _body_names, args.device, max_points=_fk_pts[s])
                for s in dp3_states}
    cam_pts = {GRASP: 2048, ALIGN: 1024}
    total_pc = {s: cam_pts[s] + robot_fk[s].num_points for s in dp3_states}
    workspace = {GRASP: tuple(PERC.workspace), ALIGN: tuple(PERC.chained_workspace)}
    ckpt_path = {GRASP: args.dp3_ckpt, ALIGN: args.stage2_dp3_ckpt}
    state_name = {GRASP: "GRASP(stage1)", ALIGN: "ALIGN(stage2)"}

    for s in dp3_states:
        print(f"[{tag}] {state_name[s]}: {cam_pts[s]} cam pts + {robot_fk[s].num_points} robot pts = "
              f"{total_pc[s]} pts, workspace={workspace[s]} | ckpt={ckpt_path[s]}", flush=True)
    if args.pc_num_points != 1024 or args.disable_cam2 or args.robot_pc_points != 160:
        print(f"[{tag}] NOTE: --pc_num_points/--disable_cam2/--robot_pc_points are ignored here "
              "(each DP3 state's camera config is hardcoded, see above)", flush=True)

    ck = {s: _load_dp3_ckpt_for_two_stage(ckpt_path[s], total_pc[s], args.device,
                                          args.num_inference_steps, state_name[s])
          for s in dp3_states}
    for s in dp3_states:
        n_obs, n_act, horizon = ck[s]["n_obs"], ck[s]["n_act"], ck[s]["horizon"]
        if args.lag_comp < 0 or n_obs - 1 + args.lag_comp + n_act > horizon:
            raise ValueError(f"lag_comp={args.lag_comp} invalid for {state_name[s]} ckpt "
                             f"(need 0<=lag_comp and (n_obs-1)+lag_comp+n_act<=horizon)")
    exec_n = {s: (ck[s]["n_act"] if args.exec_horizon is None
                 else max(1, min(args.exec_horizon, ck[s]["n_act"]))) for s in dp3_states}

    # ---- ALIGN driven by the privileged RL stage2 teacher ----
    rl_agent = rl_obs = None
    if align_is_rl:
        rl_agent, _rl_obs_dim = _build_stage2_rl_player(env_unwrapped, args)
        rl_obs = env_unwrapped._get_observations()["policy"]
        if rl_obs.shape[1] != _rl_obs_dim:
            raise ValueError(f"stage2 RL checkpoint expects obs dim {_rl_obs_dim} but the env "
                             f"produces {rl_obs.shape[1]} -- is this really a stage2 (plate-pose "
                             f"extended) checkpoint?")
        rl_agent.get_batch_size(rl_obs, 1)
        rl_agent.reset()
        print(f"[{tag}] {state_name[ALIGN]}: privileged RL teacher, obs={_rl_obs_dim} dims, "
              f"per-step (no action chunking) | ckpt={args.stage2_checkpoint}", flush=True)

    env_origins = env_unwrapped.scene.env_origins

    def build_obs(state):
        """One frame of (point_cloud, agent_pos) for `state`, using ONLY that state's own hardcoded
        camera composition + workspace -- never mixed with the other state's."""
        ws = camera_crop_bounds(env_unwrapped.drill.data.root_pos_w, env_origins, PERC, workspace[state])
        pcf = torch.zeros(args.num_envs, cam_pts[state], 3, device=args.device, dtype=torch.float32)
        if state == GRASP:
            pc1, _, _ = camera_pc(cam1, ws, cam_pts[GRASP], env_origins)
            pcf[:, :] = pc1
        else:
            half = cam_pts[ALIGN] // 2
            ws2 = raise_z_floor(ws, getattr(PERC, "wrist_cam_z_floor", None))
            pc2, _, _ = camera_pc(cam2, ws2, half, env_origins, pose_w=cam2_pose_fn())
            pc3, _, _ = camera_pc(cam3, ws, half, env_origins)
            pcf[:, :half] = pc2
            pcf[:, half:] = pc3
        robot_pc = robot_fk[state](env_unwrapped.franka.data.body_pos_w, env_unwrapped.franka.data.body_quat_w,
                                   env_origins)
        pcf = torch.cat([pcf, robot_pc], dim=1)
        agent_pos = build_agent_pos(env_unwrapped, ck[state]["with_force"])
        return pcf, agent_pos

    _hist_len = (max(ck[s]["n_obs"] for s in dp3_states)
                 + max(ck[s]["n_act"] for s in dp3_states))
    env_obs_histories = [deque(maxlen=_hist_len) for _ in range(args.num_envs)]
    pending_actions = [None] * args.num_envs
    pending_idx = [0] * args.num_envs

    def reseed_history(e, state, pc_batch, agent_batch):
        """Drop env e's leftover action chunk and refill its history from `state`'s own observation.
        For an RL-driven ALIGN there is no history to refill (the teacher is per-step and memoryless);
        clearing the pending chunk is the whole job, so the stale GRASP q* is never issued again."""
        env_obs_histories[e].clear()
        pending_actions[e] = None
        pending_idx[e] = 0
        if state not in dp3_states:
            return
        for _ in range(ck[state]["n_obs"]):
            env_obs_histories[e].append({"agent_pos": agent_batch[e], "point_cloud": pc_batch[e]})

    def update_cams():
        cam1.update(dt)
        if not align_is_rl:      # cam2/cam3 only feed the ALIGN point cloud
            cam2.update(dt); cam3.update(dt)

    def build_all_obs():
        return {s: build_obs(s) for s in dp3_states}

    update_cams()
    state_now = env_unwrapped.phase.clone()
    _obs_of = build_all_obs()
    pc_of = {s: v[0] for s, v in _obs_of.items()}
    agent_of = {s: v[1] for s, v in _obs_of.items()}
    for e in range(args.num_envs):
        s = int(state_now[e].item())
        reseed_history(e, s, pc_of.get(s), agent_of.get(s))

    total_episodes = successful_episodes = grasp_successful_episodes = stable_grasp_episodes = 0
    ever_grasp = np.zeros(args.num_envs, dtype=bool)
    grasp_switch_steps = []
    success_log = []
    episode_rewards = []
    env_episode_rewards = torch.zeros(args.num_envs, device=args.device)
    total_steps = 0
    start_time = time.time()

    print(f"\n[{tag}] rollout starting (target {args.num_episodes} episodes, {args.num_envs} envs)", flush=True)
    sys.stdout.flush()

    with torch.inference_mode():
        while simulation_app.is_running() and total_episodes < args.num_episodes:
            state_before = env_unwrapped.phase.clone()

            need = {s: [] for s in dp3_states}
            for e in range(args.num_envs):
                s = int(state_before[e].item())
                if s in dp3_states and (pending_actions[e] is None or pending_idx[e] >= exec_n[s]):
                    need[s].append(e)

            for s in dp3_states:
                if not need[s]:
                    continue
                n_obs, n_act = ck[s]["n_obs"], ck[s]["n_act"]
                obs_batch, pc_batch = [], []
                for e in need[s]:
                    hist = list(env_obs_histories[e])[-n_obs:]
                    if len(hist) < n_obs:
                        hist = [hist[0]] * (n_obs - len(hist)) + hist
                    obs_batch.append(torch.stack([h["agent_pos"] for h in hist]))
                    pc_batch.append(torch.stack([h["point_cloud"] for h in hist]))
                obs_dict = {"agent_pos": torch.stack(obs_batch), "point_cloud": torch.stack(pc_batch)}
                result = ck[s]["policy_ema"].predict_action(obs_dict)
                cs = n_obs - 1 + args.lag_comp
                chunks = result["action_pred"][:, cs:cs + n_act]
                for i, e in enumerate(need[s]):
                    pending_actions[e] = chunks[i]
                    pending_idx[e] = 0

            # ALIGN-by-RL: the teacher outputs raw [-1,1], so it needs the same raw->q* mapping the
            # collector/teacher-eval use. Computed for the whole batch (raw_to_target is pure and
            # cheap); only the rows of envs currently in ALIGN are actually taken below.
            rl_q = None
            if align_is_rl:
                traw = rl_agent.get_action(rl_agent.obs_to_torch(rl_obs), is_deterministic=True)
                if not isinstance(traw, torch.Tensor):
                    traw = torch.from_numpy(np.array(traw)).float().to(args.device)
                rl_q = raw_to_target(traw, env_unwrapped.cur_targets.clone(), _drive_p)

            actions = torch.zeros((args.num_envs, 13), device=args.device, dtype=torch.float32)
            for e in range(args.num_envs):
                if align_is_rl and int(state_before[e].item()) == ALIGN:
                    actions[e] = rl_q[e]
                elif pending_actions[e] is not None:
                    actions[e] = pending_actions[e][pending_idx[e]]
                    pending_idx[e] += 1

            env_unwrapped._direct_target = actions
            obs_dict_step, rewards, terminated, truncated, _ = env_unwrapped.step(actions)
            total_steps += 1
            if align_is_rl:
                rl_obs = obs_dict_step["policy"] if isinstance(obs_dict_step, dict) else obs_dict_step

            update_cams()

            state_after = env_unwrapped.phase.clone()
            just_switched = (state_before == GRASP) & (state_after == ALIGN)

            _obs_of = build_all_obs()
            for s, v in _obs_of.items():
                pc_of[s], agent_of[s] = v

            env_episode_rewards += rewards
            is_done = terminated.bool() | truncated.bool()
            try:
                lenient_success = env_unwrapped._cached_lenient_success
            except AttributeError:
                lenient_success = torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)
            _inst = getattr(env_unwrapped, "_cached_instant_success", None)
            if _inst is not None:
                ever_grasp |= _inst.detach().cpu().numpy().astype(bool)

            finished = torch.where(is_done)[0]
            for env_idx in finished:
                e = env_idx.item()
                total_episodes += 1
                episode_rewards.append(env_episode_rewards[e].item())
                if lenient_success[e]:
                    successful_episodes += 1; success_log.append(1)
                else:
                    success_log.append(0)
                if ever_grasp[e]:
                    grasp_successful_episodes += 1
                ever_grasp[e] = False
                sw = int(env_unwrapped._last_episode_switch_step[e].item())
                if sw >= 0:
                    stable_grasp_episodes += 1; grasp_switch_steps.append(sw)
                env_episode_rewards[e] = 0.0
                reseed_history(e, GRASP, pc_of[GRASP], agent_of[GRASP])   # reset -> always back to GRASP
            if align_is_rl and len(finished) > 0:
                rl_agent.reset()   # same per-episode player reset as run_rl_teacher_eval

            for e in range(args.num_envs):
                if is_done[e]:
                    continue
                if just_switched[e]:
                    # GRASP->ALIGN this step: discard GRASP's leftover chunk + history, reseed with
                    # ALIGN's OWN camera composition so the next policy call is ALIGN's.
                    # (align_driver="rl": nothing to reseed, the chunk is just dropped.)
                    reseed_history(e, ALIGN, pc_of.get(ALIGN), agent_of.get(ALIGN))
                else:
                    s = int(state_after[e].item())
                    if s in dp3_states:
                        env_obs_histories[e].append({"agent_pos": agent_of[s][e],
                                                     "point_cloud": pc_of[s][e]})

            if total_steps % 100 == 0:
                elapsed = time.time() - start_time
                sr = successful_episodes / max(total_episodes, 1) * 100
                n_align = int((state_after == ALIGN).sum().item())
                print(f"  step={total_steps} | eps={total_episodes}/{args.num_episodes} "
                      f"| succ={successful_episodes} ({sr:.1f}%) | grasp={grasp_successful_episodes} "
                      f"stable={stable_grasp_episodes} | in_align={n_align}/{args.num_envs} "
                      f"| {total_steps/elapsed:.1f} step/s", flush=True)
                sys.stdout.flush()

    elapsed = time.time() - start_time
    m, sec = divmod(int(elapsed), 60); h, m = divmod(m, 60)
    print(f"\n{'='*60}")
    print(f"[{tag}] === state-machine results: GRASP=DP3, "
          f"ALIGN={'privileged RL teacher' if align_is_rl else 'DP3'} ===")
    if total_episodes > 0:
        print(f"Success rate: {successful_episodes/total_episodes*100:.1f}%")
        _gr = grasp_successful_episodes / total_episodes * 100
        _sr2 = stable_grasp_episodes / total_episodes * 100
        print(f"Grasp success (>=1 step): {grasp_successful_episodes} ({_gr:.1f}%)")
        print(f"Grasp stable (entered ALIGN): {stable_grasp_episodes} ({_sr2:.1f}%)")
        if stable_grasp_episodes > 0:
            print(f"Align | stable: {successful_episodes/stable_grasp_episodes*100:.1f}%")
        if grasp_switch_steps:
            _ss = np.array(grasp_switch_steps)
            print(f"Switch step (time to stable grasp): mean={_ss.mean():.0f} "
                  f"median={np.median(_ss):.0f} max={_ss.max()} steps")
    if episode_rewards:
        ep = np.array(episode_rewards)
        print(f"Reward: mean={ep.mean():.1f} max={ep.max():.1f} min={ep.min():.1f}")
    print(f"Time: {h}h{m}m{sec}s, {total_steps/elapsed:.0f} steps/s")
    if success_log:
        print(f"Success log: {success_log}")
    print(f"{'='*60}")


def main():
    args = parse_args()

    if args.stage2_rl:
        if args.stage2_dp3_ckpt is not None:
            print("[ERROR] --stage2_rl and --stage2_dp3_ckpt both claim the ALIGN state; pick one "
                  "(--stage2_rl = privileged RL teacher, --stage2_dp3_ckpt = align DP3 student)",
                  flush=True)
            return
        if args.stage1_only or args.stage2_only or args.dual or args.policy == "rl" \
                or args.collect_success_data:
            print("[ERROR] --stage2_rl is mutually exclusive with --stage1_only/--stage2_only/"
                  "--dual/--policy rl/--collect_success_data (it is its own state machine: DP3 "
                  "drives GRASP, the RL stage2 teacher drives ALIGN)", flush=True)
            return
        _s2 = args.stage2_checkpoint if os.path.isabs(args.stage2_checkpoint) \
            else os.path.join(project_root, args.stage2_checkpoint)
        if not os.path.exists(_s2):
            print(f"[ERROR] --stage2_rl needs the RL stage2 checkpoint, not found: {_s2}", flush=True)
            return
        if not args.chained:
            args.chained = True
            print("[INFO] --stage2_rl: auto-enable --chained (needed for the phase 0->1 switch)",
                  flush=True)
        print("[INFO] --stage2_rl: hybrid deploy. GRASP driven by DP3 (--dp3_ckpt) via cam1 alone "
              "(2048+robot512=2560, hardcoded, same as --stage2_dp3_ckpt); on the same phase 0->1 "
              "switch, ALIGN is driven by the PRIVILEGED RL stage2 teacher on the env's state obs. "
              "End-to-end success here is an upper bound on what the Grasp Student's hand-off "
              "supports; the gap to a --stage2_dp3_ckpt run is the ALIGN distillation gap",
              flush=True)

    if args.stage2_dp3_ckpt is not None:
        if args.stage1_only or args.stage2_only:
            print("[ERROR] --stage2_dp3_ckpt (two-stage DP3 deploy) is mutually exclusive with "
                  "--stage1_only/--stage2_only", flush=True)
            return
        if args.dual or args.policy == "rl":
            print("[ERROR] --stage2_dp3_ckpt does not support --dual/--policy rl (those paths assume "
                  "a single checkpoint/dual-RL-teacher setup); use plain DP3 inference "
                  "(default --policy dp3, no --dual)", flush=True)
            return
        if not os.path.exists(args.stage2_dp3_ckpt):
            print(f"[ERROR] stage2 DP3 checkpoint not found: {args.stage2_dp3_ckpt}", flush=True)
            return
        if not args.chained:
            args.chained = True
            print("[INFO] --stage2_dp3_ckpt: auto-enable --chained (needed for the phase 0->1 switch)",
                  flush=True)
        print("[INFO] --stage2_dp3_ckpt: two-stage DP3 deploy. stage1 (--dp3_ckpt) drives grasp via "
              "cam1 alone (2048+robot512=2560, hardcoded); on stage1 success, stage2 "
              "(--stage2_dp3_ckpt) takes over via cam2+cam3 (1024+robot160=1184, hardcoded)",
              flush=True)

    if args.collect_success_data:
        if args.stage2_dp3_ckpt is not None or args.stage2_only:
            print("[ERROR] --collect_success_data is mutually exclusive with "
                  "--stage2_dp3_ckpt/--stage2_only", flush=True)
            return
        if args.dual or args.policy == "rl":
            print("[ERROR] --collect_success_data does not support --dual/--policy rl "
                  "(needs the DP3 GRASP checkpoint driving directly)", flush=True)
            return
        if not args.stage1_only:
            args.stage1_only = True
            print("[INFO] --collect_success_data: auto-enable --stage1_only "
                  "(same env/checkpoint/camera composition as that path)", flush=True)

    if args.stage2_only:
        if args.chained or args.stage1_only:
            print("[ERROR] --stage2_only uses Stage2Env directly and is mutually exclusive with "
                  "--chained/--stage1_only (those use ChainedEnv)", flush=True)
            return
        if args.dual or args.policy == "rl":
            print("[ERROR] --stage2_only does not support --dual/--policy rl (those paths assume "
                  "--chained's cam1+cam2+plate layout and stage1/stage2 dual-teacher checkpoints); "
                  "use plain DP3 inference (default --policy dp3, no --dual)", flush=True)
            return
        if not os.path.exists(args.success_dataset):
            print(f"[ERROR] --success_dataset not found: {args.success_dataset} "
                  f"(generate with scripts/collect_success_data.py, or pass --success_dataset)", flush=True)
            return
        print("[INFO] --stage2_only: Stage2Env, drill starts already grasped (from --success_dataset), "
              "DP3 policy drives alignment, camera segment = cam2+cam3", flush=True)
    # stage1_only reuses ChainedEnv + wrist camera (same as collect), but cam3 is off (no plate segment).
    # passing --stage1_only alone auto-forces chained (otherwise it uses GraspDrillEnv and cam2 becomes a fixed camera).
    elif args.stage1_only and not args.chained:
        args.chained = True
        _cam_desc = "cam1 alone" if args.disable_cam2 else "cam1+cam2 fused"
        print(f"[INFO] --stage1_only: auto-enable ChainedEnv + wrist camera, cam3 off (no plate segment). "
              f"camera segment={args.pc_num_points} pts ({_cam_desc}) + robot (up to {args.robot_pc_points} pts "
              f"if --pc_mode robot) -- exact total printed once the checkpoint's expected point_cloud is "
              f"checked against it below", flush=True)

    if args.policy == "dp3" and not os.path.exists(args.dp3_ckpt):
        print(f"[ERROR] DP3 checkpoint not found: {args.dp3_ckpt}")
        return

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=args.headless, enable_cameras=True)
    simulation_app = app_launcher.app

    _exit_code = 0                             # for hard exit; set to 1 on exception so the outer script can tell
    try:
        import carb
        import omni.timeline
        import omni.usd
        from tasks.grasp_drill_env import GraspDrillEnv, create_grasp_drill_env_cfg

        omni.usd.get_context().new_stage()

        cfg = create_grasp_drill_env_cfg(
            num_envs=args.num_envs,
            device=args.device,
            headless=args.headless,
            debug=args.debug,
            drill_config_path=args.drill_configs,
            drill_variants_path=args.drill_variants,
            img_height=args.img_height,
            img_width=args.img_width,
            enable_cameras=True,
            # stage2_only: needs plate + cam3 too (cam3 is one of the two main camera views, fused with cam2).
            include_plate=args.chained or args.stage2_only,
            include_plate_camera=args.chained or args.stage2_only,
            # stage1_only: cam2 is still the wrist camera, plate object present, but cam3 not built -> no plate segment
            include_cam3=(args.chained and not args.stage1_only) or args.stage2_only,
        )
        cfg.seed = args.seed
        cfg.log_dir = os.path.dirname(args.dp3_ckpt)
        if args.chained or args.stage2_only:
            cfg.episode_length_s = args.episode_length_s

        ensure_rgb_aov(cfg.scene)   # depth-only TiledCamera may deadlock on the first frame

        if args.stage2_only:
            from tasks.stage2_env import get_stage2_env_class, SuccessDataDataset
            Stage2Env = get_stage2_env_class()
            success_dataset = SuccessDataDataset(args.success_dataset)
            base_env = Stage2Env(cfg=cfg, debug=args.debug, success_dataset=success_dataset,
                                 success_hold_stop=args.success_hold_stop)
            print(f"[STAGE2] Stage2Env: drill starts already grasped ({len(success_dataset)} pkl samples), "
                  f"DP3 policy drives alignment, terminates early after {args.success_hold_stop} "
                  f"consecutive aligned steps (episode cap={args.episode_length_s}s)", flush=True)
        elif args.chained:
            from tasks.chained_env import get_chained_env_class
            ChainedEnv = get_chained_env_class()
            base_env = ChainedEnv(cfg=cfg, debug=args.debug,
                                  success_hold_stop=args.success_hold_stop,
                                  stage1_only=args.stage1_only)
        else:
            base_env = GraspDrillEnv(cfg=cfg, debug=args.debug)
        env_unwrapped = base_env.unwrapped

        _exam_res = None   # set below when --init_pose_file given: {vid: int8 array of -1(not run)/0(fail)/1(success)}
        _EXAM_N = int(args.exam_n)   # sequential exam: run each variant's first N saved poses in stored order, exactly once
        if args.init_pose_file is not None:
            _ip = np.load(args.init_pose_file)
            _vids = _ip["variant"]; _pos = _ip["pos_local"]; _quat = _ip["quat"]; _goal = _ip["goal_quat"]
            _by_variant = {}
            for _k in np.unique(_vids):
                _m = _vids == _k
                _by_variant[int(_k)] = (
                    torch.from_numpy(_pos[_m]).float()[:_EXAM_N],
                    torch.from_numpy(_quat[_m]).float()[:_EXAM_N],
                    torch.from_numpy(_goal[_m]).float()[:_EXAM_N])
            # sequential-exam state: each variant runs its first N poses in stored order, exactly once.
            _exam_res = {vid: -np.ones(P.shape[0], dtype=np.int8) for vid, (P, Q, G) in _by_variant.items()}
            _exam_next = {vid: 0 for vid in _by_variant}          # per-variant cursor into its pose list
            _env_item = [(-1, -1)] * args.num_envs                 # exam item each env is currently running (-1,-1=filler)
            _exam_total = sum(int(P.shape[0]) for (P, Q, G) in _by_variant.values())

            def _exam_pending():
                # the actual stop condition for the exam: True until every variant's first
                # _EXAM_N poses have been graded. Deliberately independent of args.num_episodes
                # (that flag only bounds the non-exam / no-init_pose_file fallback path below).
                return not all((r != -1).all() for r in _exam_res.values())

            def _exam_graded_count():
                return sum(int((r != -1).sum()) for r in _exam_res.values())

            print(f"[INFO] init-pose replay (SEQUENTIAL exam, first {_EXAM_N}/variant, deterministic): "
                  f"{_exam_total} poses total, per-variant counts="
                  f"{ {k: v[0].shape[0] for k, v in _by_variant.items()} }", flush=True)

            # env<->variant assignment is fixed for the whole run (env_id % num_variants, see
            # GraspDrillEnv._reset_idx), so once a variant's exam poses run out, the same envs keep
            # getting reassigned to it. Freeze those envs instead of re-running them on a filler pose:
            # force terminated/truncated=False forever (never resets again -> never draws a new pose,
            # never starts another episode) and hold their current joint targets in the main loop below.
            _frozen = torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)
            _orig_get_dones_exam = env_unwrapped._get_dones

            def _get_dones_freeze_exhausted():
                terminated, truncated = _orig_get_dones_exam()
                terminated = terminated & ~_frozen
                truncated = truncated & ~_frozen
                return terminated, truncated

            env_unwrapped._get_dones = _get_dones_freeze_exhausted

            def _drill_pose_override_fn(env_ids, variant_ids):
                # called at each reset. For each env: (1) record the result of the exam item it just
                # finished (via _cached_lenient_success, already set by _get_dones before this reset),
                # (2) hand it the next sequential pose of its variant, or freeze it once that variant
                # is done (see _get_dones_freeze_exhausted / the main loop's action override).
                n = len(env_ids)
                pos = torch.zeros(n, 3); quat = torch.zeros(n, 4); goal = torch.zeros(n, 4)
                _suc = getattr(env_unwrapped, "_cached_lenient_success", None)   # just-ended success per env
                for i in range(n):
                    e = int(env_ids[i].item())
                    vp, ip = _env_item[e]
                    if vp >= 0 and _exam_res[vp][ip] == -1:      # record once per item
                        _exam_res[vp][ip] = 1 if (_suc is not None and bool(_suc[e].item())) else 0
                    vid = int(variant_ids[i].item())
                    if vid not in _by_variant:
                        raise RuntimeError(f"init_pose_file is missing recorded poses for variant {vid};"
                                           f"deploy's active variants must match collect")
                    P, Q, G = _by_variant[vid]
                    j = _exam_next[vid]
                    if j < P.shape[0]:
                        _exam_next[vid] += 1
                        _env_item[e] = (vid, j)
                        idx = j
                    else:
                        _env_item[e] = (-1, -1)   # this variant's exam done -> freeze this env
                        _frozen[e] = True
                        idx = 0
                    pos[i] = P[idx]; quat[i] = Q[idx]; goal[i] = G[idx]
                return pos, quat, goal

            env_unwrapped._drill_pose_override_fn = _drill_pose_override_fn

        if args.fixed_plate:
            _fp_pos = torch.tensor([0.0, 1.0, 0.5], device=args.device)   # matches _randomize_plate default base
            _fp_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=args.device)

            def _plate_pose_override_fn(env_ids):
                n = len(env_ids)
                return _fp_pos.unsqueeze(0).expand(n, -1), _fp_quat.unsqueeze(0).expand(n, -1)

            env_unwrapped._plate_pose_override_fn = _plate_pose_override_fn
            print("[INFO] --fixed_plate: plate pinned at (0,1,0.5) default base, no random jitter each reset",
                  flush=True)

        from isaaclab.sim.utils import delete_prim
        delete_prim("/World/Template")


        apply_render_settings(env_unwrapped.step_dt)   # without these carb settings, the first step() render deadlocks

        env_unwrapped.reset()

        dt = env_unwrapped.step_dt
        # for stage1_only cam3 is not built (need_cam3=False), cam2 is still the wrist camera.
        # stage2_only needs cam3 too (fused with cam2 as the camera segment).
        cam1, cam2, cam3 = get_cameras(env_unwrapped, chained=args.chained,
                                       need_cam3=(args.chained and not args.stage1_only) or args.stage2_only)
        # whether there is a plate segment: chained and cam3 actually exists. stage1_only -> cam3=None -> no plate segment.
        _use_plate = args.chained and (cam3 is not None)
        _cam2_on_body = detect_wrist_cam(env_unwrapped, cam2)

        def _cam2_pose():
            return wrist_cam_pose_w(env_unwrapped.franka, cam2) if _cam2_on_body else None

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()

        if not simulation_app.is_running():
            print("[ERROR] simulation_app.is_running() is False before loop!")
            return

        _, _ = env_unwrapped.reset()

        # ---- RL teacher mode: same eval env, same init_pose, same success counting, only policy swapped, measure pure gap ----
        if args.policy == "rl":
            run_rl_teacher_eval(env_unwrapped, simulation_app, args)
            return   # trigger the outer finally (timeline.stop + app.close + os._exit)

        # ---- two-stage DP3 state machine: GRASP ckpt (cam1 alone) drives grasp; on ChainedEnv's
        # phase 0->1 switch (stage1 success) ALIGN ckpt (cam2+cam3) takes over driving alignment.
        # Camera composition + workspace per state are hardcoded inside run_dp3_two_stage (not
        # derived from args.workspace here) -- see that function's docstring for why.
        if args.stage2_dp3_ckpt is not None or args.stage2_rl:
            run_dp3_two_stage(env_unwrapped, simulation_app, args, cam1, cam2, cam3, _cam2_pose,
                              dt, _load_perception_hp(),
                              align_driver=("rl" if args.stage2_rl else "dp3"))
            return   # trigger the outer finally (timeline.stop + app.close + os._exit)

        if args.collect_success_data:
            run_collect_success_data(env_unwrapped, simulation_app, args, cam1, _load_perception_hp())
            return   # trigger the outer finally (timeline.stop + app.close + os._exit)

        from perception.robot_pointcloud import RobotPointCloudFK
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
        drill_mesh_fk = None
        drill_M = 0
        if args.save_drill_mesh_pc and args.drill_mesh_points > 0:
            from perception.drill_pointcloud import DrillMeshPointCloudFK
            drill_mesh_fk = DrillMeshPointCloudFK(env_unwrapped, num_points=args.drill_mesh_points,
                                                  device=args.device)
            drill_M = drill_mesh_fk.num_points
        from perception.robot_pointcloud import jitter_ground_xy
        _PERC = _load_perception_hp()
        ground_batch, ground_M, _ground_xy_std = setup_ground(args.num_envs, args.workspace, args.device,
                                                              num_points=args.ground_points)
        plate_M = args.plate_pc_points if _use_plate else 0   # stage1_only (cam3 off) -> 0, no plate segment
        total_pc = args.pc_num_points + plate_M + robot_pc_M + drill_M + ground_M

        _last_pc3 = None   # reuse cache for empty cam3 frames (only when there is a plate segment)

        def append_robot_drill(pc_cam):
            """pc_cam: (B,pc_num_points,3) env-local camera cloud. Concatenated in collect's order:
            [camera | plate (only when there is a plate segment) | robot | drill | ground]."""
            nonlocal _last_pc3
            parts = [pc_cam]
            if _use_plate:
                pc_plate, _zm3, _ = build_plate_cam_pc(
                    cam3, env_unwrapped.plate.data.root_pos_w,
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
            parts.append(jitter_ground_xy(ground_batch, _ground_xy_std,
                                          _ws[0], _ws[1], _ws[2], _ws[3]))
            return torch.cat(parts, dim=1)

        # ---- 4-channel (xyz+mask) support: when ckpt point cloud channels==4, add GT handle mask to the camera segment ----
        # with_mask is auto-set True by ckpt point_cloud.shape[-1]==4 (see detection after ckpt load below),
        # same approach as auto-detecting force; 3-channel ckpt is unaffected (add_mask_channel returns as-is).
        with_mask = False
        # same module collect uses -> byte-identical label definition and the same default threshold
        from perception.groundtruth_mask import add_mask_channel as _gt_add_mask_channel

        def add_mask_channel(pc_now):
            """pc_now: (B,total_pc,3) -> (B,total_pc,4)=[xyz|is_handle], only when with_mask.
            Sim-only: the label comes from the drill's GT pose + the variant's body_mask (on the real
            robot replace this call with a segmentation network's output)."""
            if not with_mask:
                return pc_now
            return _gt_add_mask_channel(pc_now, env_unwrapped, args.pc_num_points,
                                        plate_M=plate_M, robot_pc_M=robot_pc_M,
                                        threshold=args.mask_threshold,
                                        use_robot_seg=args.save_robot_pc)

        _sim_pc_per_cam = args.pc_num_points if args.disable_cam2 else args.pc_num_points // 2
        _workspace = tuple(args.workspace)
        cam1.update(dt)
        cam2.update(dt)
        if cam3 is not None:
            cam3.update(dt)
        _ws_init = camera_crop_bounds(env_unwrapped.drill.data.root_pos_w,
                                      env_unwrapped.scene.env_origins, _PERC, _workspace)
        pc_fused_init = torch.zeros(args.num_envs, args.pc_num_points, 3, device=args.device, dtype=torch.float32)
        if args.stage2_only:
            # camera segment = cam2 (wrist) + cam3 (plate), cam1 not sampled (same convention as
            # collect_dp3_data.py --stage2_only).
            _half = args.pc_num_points // 2
            _sim_pc_per_cam = _half
            _ws_init_cam2 = raise_z_floor(_ws_init, getattr(_PERC, "wrist_cam_z_floor", None))
            pc2_init, _, _ = camera_pc(cam2, _ws_init_cam2, _half, env_unwrapped.scene.env_origins,
                                       pose_w=_cam2_pose())
            pc3_init, _, _ = camera_pc(cam3, _ws_init, _half, env_unwrapped.scene.env_origins)
            _last_pc1 = None
            _last_pc2 = pc2_init
            _last_pc3_stage2 = pc3_init
            pc_fused_init[:, :_half] = pc2_init
            pc_fused_init[:, _half:] = pc3_init
        else:
            pc1_init, _, _ = camera_pc(cam1, _ws_init, _sim_pc_per_cam, env_unwrapped.scene.env_origins)
            _last_pc1 = pc1_init
            pc_fused_init[:, :_sim_pc_per_cam] = pc1_init   # cam1 30cm follow
            if args.disable_cam2:
                _last_pc2 = None
            else:
                _ws_init_cam2 = raise_z_floor(_ws_init, getattr(_PERC, "wrist_cam_z_floor", None))
                pc2_init, _, _ = camera_pc(cam2, _ws_init_cam2, _sim_pc_per_cam, env_unwrapped.scene.env_origins,
                                           pose_w=_cam2_pose())
                _last_pc2 = pc2_init
                pc_fused_init[:, _sim_pc_per_cam:] = pc2_init   # cam2 30cm follow (same frame, different view)
        _initial_pc = append_robot_drill(pc_fused_init)

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
        if _agent_dim is None:
            _agent_dim = 13
        with_force = _agent_dim > 13
        print(f"  agent_pos dim = {_agent_dim} (force_state={with_force})", flush=True)

        _ckpt_pc = None
        for _path in (lambda: dp3_cfg.task.shape_meta.obs.point_cloud.shape[0],
                      lambda: dp3_cfg.shape_meta.obs.point_cloud.shape[0]):
            try:
                _ckpt_pc = int(_path()); break
            except Exception:
                continue
        if _ckpt_pc is not None and _ckpt_pc != total_pc:
            raise RuntimeError(
                f"check that --chained/--pc_num_points/--plate_pc_points/--robot_pc_points/"
                f"--drill_mesh_points match the collection")
        print(f"  point_cloud: ckpt={_ckpt_pc} deploy={total_pc} ✓", flush=True)

        # point cloud channels: ckpt shape[-1]==4 -> 4 channels (xyz+mask), compute GT handle mask live in sim
        _ckpt_ch = None
        for _path in (lambda: dp3_cfg.task.shape_meta.obs.point_cloud.shape[-1],
                      lambda: dp3_cfg.shape_meta.obs.point_cloud.shape[-1]):
            try:
                _ckpt_ch = int(_path()); break
            except Exception:
                continue
        if _ckpt_ch is None:
            _ckpt_ch = 3
        with_mask = (_ckpt_ch == 4)
        print(f"  point_cloud channels = {_ckpt_ch} "
              f"({'xyz+mask(GT handle)' if with_mask else 'xyz'})", flush=True)
        if with_mask:
            # printed loudly because a threshold mismatch with the collection run is silent otherwise:
            # the channel is still 0/1, just labelled by a different radius than the policy was trained on
            print(f"[MASK] 4th channel = GT is_handle, threshold={args.mask_threshold}m, applied to "
                  f"the camera segment [0,{args.pc_num_points}) only (robot/plate/ground = 0). "
                  f"This MUST match collect_dp3_data.py --mask_threshold for this checkpoint's data.",
                  flush=True)
            _initial_pc = add_mask_channel(_initial_pc)   # add the 4th channel to the initial obs history too

        # ---- --dump_obs_zarr: record the DEPLOYED observation stream in collect's exact layout ----
        # Purpose: diff "what the policy sees at deploy" against "what it was trained on", byte for
        # byte. Mirrors collect_dp3_data.py exactly: same row pairing (obs at decision time + the q*
        # executed that step), same bad-frame skipping, point_cloud stored as xyz with the mask split
        # back out into its own pc_mask array, same float16 point cloud.
        _dump = None
        if args.dump_obs_zarr:
            import shutil
            import zarr as _zarr
            from numcodecs import Blosc as _Blosc
            _dpath = args.dump_obs_zarr
            if os.path.exists(_dpath):
                shutil.rmtree(_dpath)
            # zarr_format=2 like collect: zarr v3's native codecs reject numcodecs.Blosc and don't
            # resize incrementally, and the dump must be readable by the same tools as a collect zarr
            _droot = _zarr.group(_dpath, zarr_format=2)
            _dd = _droot.create_group("data")
            _dm = _droot.create_group("meta")
            _cmp = _Blosc(cname="zstd", clevel=3, shuffle=1)
            _dd.create_dataset("point_cloud", shape=(0, total_pc, 3), dtype="float16",
                               chunks=(64, total_pc, 3), compressor=_cmp, overwrite=True)
            _dd.create_dataset("state", shape=(0, _agent_dim), dtype="float32",
                               chunks=(1024, _agent_dim), compressor=_cmp, overwrite=True)
            _dd.create_dataset("action", shape=(0, 13), dtype="float32",
                               chunks=(1024, 13), compressor=_cmp, overwrite=True)
            if with_mask:
                _dd.create_dataset("pc_mask", shape=(0, total_pc), dtype="uint8",
                                   chunks=(64, total_pc), compressor=_cmp, overwrite=True)
            _dm.create_dataset("episode_ends", shape=(0,), dtype="int64",
                               chunks=(64,), compressor=_cmp, overwrite=True)
            _dump = dict(root=_droot, data=_dd, meta=_dm, env=int(args.dump_obs_env),
                         want=int(args.dump_obs_episodes), done=0, rows=[], ends=[],
                         prev_pc=None, prev_state=None, prev_bad=False, skipped=0)
            print(f"[DUMP] recording env {_dump['env']}'s first {_dump['want']} episode(s) -> "
                  f"{_dpath} | point_cloud({total_pc},3) float16"
                  f"{' + pc_mask' if with_mask else ''} + state({_agent_dim}) + action(13), "
                  f"collect's layout & timing", flush=True)

        def _dump_row(action_row):
            """Called right before env.step(): pair the previous post-step observation (= what the
            policy saw when it decided) with the q* about to be executed, exactly like collect's 1c."""
            if _dump is None or _dump["done"] >= _dump["want"] or _dump["prev_pc"] is None:
                return
            if _dump["prev_bad"]:            # collect drops empty-camera frames; do the same
                _dump["skipped"] += 1
                return
            _dump["rows"].append((_dump["prev_state"], action_row, _dump["prev_pc"]))

        def _dump_flush():
            """Episode finished for the recorded env -> append it to the zarr."""
            if _dump is None or not _dump["rows"]:
                return
            st, ac, pc = zip(*_dump["rows"])
            n = len(st)
            pc_arr = np.stack(pc)                                   # (n, total_pc, 3 or 4)
            xyz = pc_arr[..., :3].astype(np.float16)
            base = _dump["data"]["point_cloud"].shape[0]
            _dump["data"]["point_cloud"].resize((base + n, total_pc, 3))
            _dump["data"]["point_cloud"][base:base + n] = xyz
            _dump["data"]["state"].resize((base + n, _agent_dim))
            _dump["data"]["state"][base:base + n] = np.stack(st).astype(np.float32)
            _dump["data"]["action"].resize((base + n, 13))
            _dump["data"]["action"][base:base + n] = np.stack(ac).astype(np.float32)
            if with_mask:
                _dump["data"]["pc_mask"].resize((base + n, total_pc))
                _dump["data"]["pc_mask"][base:base + n] = (pc_arr[..., 3] > 0.5).astype(np.uint8)
            _dump["ends"].append(base + n)
            _dump["meta"]["episode_ends"].resize((len(_dump["ends"]),))
            _dump["meta"]["episode_ends"][:] = np.array(_dump["ends"], dtype=np.int64)
            _dump["done"] += 1
            _dump["rows"] = []
            print(f"[DUMP] episode {_dump['done']}/{_dump['want']} written: {n} frames "
                  f"(skipped {_dump['skipped']} empty-camera frames) -> {args.dump_obs_zarr}",
                  flush=True)

        dp3_policy: DP3 = hydra_instantiate(dp3_cfg.policy)
        dp3_policy.to(args.device)
        dp3_policy.eval()
        dp3_policy.load_state_dict(payload["state_dicts"]["model"], strict=False)

        if args.num_inference_steps is not None:
            dp3_policy.num_inference_steps = args.num_inference_steps

        if "ema_model" in payload["state_dicts"]:
            dp3_policy_ema: DP3 = hydra_instantiate(dp3_cfg.policy)
            dp3_policy_ema.to(args.device)
            dp3_policy_ema.load_state_dict(payload["state_dicts"]["ema_model"], strict=False)
            if args.num_inference_steps is not None:
                dp3_policy_ema.num_inference_steps = args.num_inference_steps
            dp3_policy_ema.eval()
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
            raise RuntimeError("Checkpoint has no embedded normalizer, cannot deploy (check that training saved correctly).")

        n_obs = dp3_policy.n_obs_steps          # 2
        n_act = dp3_policy.n_action_steps       # read from ckpt (training used 4)
        exec_n = n_act if args.exec_horizon is None else max(1, min(args.exec_horizon, n_act))
        horizon = dp3_policy.horizon           # 16
        if args.lag_comp < 0 or n_obs - 1 + args.lag_comp + n_act > horizon:
            raise ValueError(f"lag_comp={args.lag_comp} invalid: need 0 <= lag_comp and "
                             f"(To-1)+lag_comp+n_act <= horizon "
                             f"({n_obs - 1}+{args.lag_comp}+{n_act} > {horizon})")
        num_inf_steps = args.num_inference_steps or dp3_policy.num_inference_steps
        sim_pc_per_cam = args.pc_num_points if args.disable_cam2 else args.pc_num_points // 2

        print(f"\n  DP3 config: horizon={horizon}, n_obs={n_obs}, n_act={n_act}, "
              f"lag_comp={args.lag_comp} (execution slot from {n_obs - 1 + args.lag_comp};"
              f"0=new-gen official-timing ckpt, 1=old-gen action-shifted ckpt)")
        print(f"  inference steps={num_inf_steps}")

        from perception.target_drive import install_direct_drive, raw_to_target
        # main single-checkpoint DP3 path (and dual mode's teacher-q* reference)
        _drive_p = install_direct_drive(env_unwrapped, rate_limit=args.rate_limit)

        if args.timeout_only:
            _orig_get_dones = env_unwrapped._get_dones

            def _get_dones_timeout_only():
                # still call the original _get_dones to update the success cache. Keep only NaN in the termination condition (diverged envs must be recycled,
                # else bad poses spam USD warnings each frame and pollute stats), all others (success/drop/limit) do not terminate, run to timeout.
                terminated, truncated = _orig_get_dones()
                nan_mask = getattr(env_unwrapped, "_pending_nan_mask", None)
                term = (nan_mask.clone() if nan_mask is not None
                        else torch.zeros_like(terminated))
                return term, truncated

            env_unwrapped._get_dones = _get_dones_timeout_only

        controlled_indices = env_unwrapped.controlled_joint_indices.cpu()
        workspace = tuple(args.workspace)
        env_origins = env_unwrapped.scene.env_origins

        from collections import deque
        env_obs_histories = [deque(maxlen=n_obs + n_act) for _ in range(args.num_envs)]
        env_episode_rewards = torch.zeros(args.num_envs, device=args.device)
        total_episodes = 0
        successful_episodes = 0
        grasp_successful_episodes = 0          # >=1 step check_success (main criterion)
        ever_grasp = np.zeros(args.num_envs, dtype=bool)   # whether this episode has had at least 1 grasp-success step
        stable_grasp_episodes = 0              # stable grasp entering stage2 (20/50 criterion)
        grasp_switch_steps = []      
        success_log = []
        episode_rewards = []
        total_zero_pc = 0
        total_steps = 0


        initial_state = build_agent_pos(env_unwrapped, with_force)   # 13 or 26 dim agent_pos
        for env_id in range(args.num_envs):
            for _ in range(n_obs):
                env_obs_histories[env_id].append({
                    "agent_pos": initial_state[env_id],
                    "point_cloud": _initial_pc[env_id],
                })

        pending_actions = [None] * args.num_envs
        pending_idx = [0] * args.num_envs


        _prof = {"infer": 0.0, "step": 0.0, "render": 0.0, "pc": 0.0}
        _prof_infer_calls = 0
        _last_print_step = 0
        _use_cuda = str(args.device).startswith("cuda")
        def _sync():
            if _use_cuda:
                torch.cuda.synchronize(args.device)

        def _run_dual():
            """Co-trajectory dual inference: each step runs DP3 (shadow, replans every step) and the RL teacher on the same observation,
            the driver actually drives (env follows its trajectory), printing the per-step q* action difference (rad->deg), optionally saved to disk.
            driver=rl: env follows the success trajectory, diff=student's deviation on the good trajectory (cleanest diagnostic).
            driver=dp3: env follows the student trajectory, diff=how the teacher would correct at the state the student drifted to."""
            nonlocal _last_pc1, _last_pc2
            mode, agent1, agent2, d1, d2 = _build_teachers(env_unwrapped, args)
            _cs = n_obs - 1 + args.lag_comp
            obs_priv = env_unwrapped._get_observations()["policy"]
            if mode == "chained":
                agent1.get_batch_size(obs_priv[:, :d1], 1); agent1.reset()
                agent2.get_batch_size(obs_priv, 1); agent2.reset()
            else:
                agent1.get_batch_size(obs_priv, 1); agent1.reset()

            _deg = 57.29578
            _steps = 0; _eps = 0
            _tr_phase, _tr_dp3, _tr_tea, _tr_diff = [], [], [], []
            print(f"\n[DUAL] co-trajectory dual inference: driver={args.driver} "
                  f"({'teacher drives, measure student deviation' if args.driver=='rl' else 'DP3 drives, measure teacher correction'})"
                  f" | per-step: |dq*| arm/finger mean|max (deg)", flush=True)
            sys.stdout.flush()

            with torch.inference_mode():
                while simulation_app.is_running() and _eps < args.num_episodes:
                    # DP3 shadow: replan each step with the current history, take the current step's action q*
                    obs_batch_list, pc_batch_list = [], []
                    for env_id in range(args.num_envs):
                        hist = list(env_obs_histories[env_id])
                        if len(hist) < n_obs:
                            hist = [hist[0]] * (n_obs - len(hist)) + hist
                        obs_ts = hist[-n_obs:]
                        obs_batch_list.append(torch.stack([s["agent_pos"] for s in obs_ts]))
                        pc_batch_list.append(torch.stack([s["point_cloud"] for s in obs_ts]))
                    obs_dict = {"agent_pos": torch.stack(obs_batch_list),
                                "point_cloud": torch.stack(pc_batch_list)}
                    dp3_q = dp3_policy_ema.predict_action(obs_dict)["action_pred"][:, _cs]  # (B,13) q*

                    # teacher inference -> q* (same obs_priv, raw->target same as collect)
                    traw = _teacher_raw_action(mode, agent1, agent2, d1, obs_priv,
                                               env_unwrapped, args.device)
                    cur0 = env_unwrapped.cur_targets.clone()
                    teacher_q = raw_to_target(traw, cur0, _drive_p)

                    diff = dp3_q - teacher_q                          # (B,13) rad, same scale so direct subtraction
                    _drive = teacher_q if args.driver == "rl" else dp3_q
                    env_unwrapped._direct_target = _drive
                    obs_dict_step, rewards, terminated, truncated, _ = env_unwrapped.step(_drive)
                    obs_priv = obs_dict_step["policy"]
                    _steps += 1

                    _ad = diff[:, :7].abs(); _fd = diff[:, 7:].abs()
                    _ph = int((env_unwrapped.phase == 1).sum().item()) if args.chained else 0
                    print(f"  step={_steps:4d} s2={_ph}/{args.num_envs} | "
                          f"arm |Δ| mean={_ad.mean().item()*_deg:5.2f} max={_ad.max().item()*_deg:6.2f} | "
                          f"finger |Δ| mean={_fd.mean().item()*_deg:5.2f} max={_fd.max().item()*_deg:6.2f}",
                          flush=True)
                    if args.dual_trace:
                        _tr_phase.append(env_unwrapped.phase.detach().cpu().numpy().copy())
                        _tr_dp3.append(dp3_q.detach().cpu().numpy())
                        _tr_tea.append(teacher_q.detach().cpu().numpy())
                        _tr_diff.append(diff.detach().cpu().numpy())

                    # update DP3 obs history (post-step, byte-for-byte same as the main loop)
                    cam1.update(dt); cam2.update(dt)
                    if cam3 is not None:
                        cam3.update(dt)
                    state_now = build_agent_pos(env_unwrapped, with_force)
                    _ws = camera_crop_bounds(env_unwrapped.drill.data.root_pos_w,
                                             env_origins, _PERC, workspace)
                    pc1_t, zmask1, _ = camera_pc(cam1, _ws, sim_pc_per_cam, env_origins)
                    if not args.disable_cam2:
                        _ws_cam2 = raise_z_floor(_ws, getattr(_PERC, "wrist_cam_z_floor", None))
                        pc2_t, zmask2, _ = camera_pc(cam2, _ws_cam2, sim_pc_per_cam, env_origins,
                                                     pose_w=_cam2_pose())
                    if not args.chained:
                        pc1_t = torch.where(zmask1.view(-1, 1, 1), _last_pc1, pc1_t)
                        _last_pc1 = pc1_t
                        if not args.disable_cam2:
                            pc2_t = torch.where(zmask2.view(-1, 1, 1), _last_pc2, pc2_t)
                            _last_pc2 = pc2_t
                    pcf = torch.zeros(args.num_envs, args.pc_num_points, 3,
                                     device=args.device, dtype=torch.float32)
                    pcf[:, :sim_pc_per_cam] = pc1_t
                    if not args.disable_cam2:
                        pcf[:, sim_pc_per_cam:] = pc2_t
                    pc_now = add_mask_channel(append_robot_drill(pcf))

                    is_done = terminated.bool() | truncated.bool()
                    if is_done.any():
                        _eps += int(is_done.sum().item())
                        agent1.reset()
                        if mode == "chained":
                            agent2.reset()
                        reset_state = build_agent_pos(env_unwrapped, with_force)
                        for env_idx in torch.where(is_done)[0]:
                            e = env_idx.item()
                            env_obs_histories[e].clear()
                            for _ in range(n_obs):
                                env_obs_histories[e].append(
                                    {"agent_pos": reset_state[e], "point_cloud": pc_now[e]})
                    for env_id in range(args.num_envs):
                        if not is_done[env_id]:
                            env_obs_histories[env_id].append(
                                {"agent_pos": state_now[env_id], "point_cloud": pc_now[env_id]})

            if args.dual_trace and _tr_diff:
                np.savez(args.dual_trace, phase=np.stack(_tr_phase), dp3_q=np.stack(_tr_dp3),
                         teacher_q=np.stack(_tr_tea), diff=np.stack(_tr_diff))
                print(f"[DUAL] trace -> {args.dual_trace} ({len(_tr_diff)} steps, "
                      f"shape/step=({args.num_envs},13))", flush=True)
            _all = np.concatenate([d.reshape(-1, 13) for d in _tr_diff], 0) if _tr_diff else None
            if _all is not None:
                print(f"[DUAL] overall mean |dq*|: arm={np.abs(_all[:, :7]).mean()*_deg:.2f}deg "
                      f"finger={np.abs(_all[:, 7:]).mean()*_deg:.2f}deg over {_steps} steps", flush=True)

        if args.dual:
            _run_dual()
            return

        start_time = time.time()
        _target_desc = (f"{_exam_total} exam poses (sequential, fixed from init_pose_file)"
                        if _exam_res is not None else f"{args.num_episodes} episodes")
        print(f"\nStarting DP3 rollout (target {_target_desc}, "
              f"{args.num_envs} envs)...")
        print(f"  workspace(env-local)={workspace}")
        print(f"  obs: agent_pos[{_agent_dim}] + point_cloud[{total_pc},{_ckpt_ch}]"
              f"{' (4th channel=GT handle mask)' if with_mask else ''}")
        if args.chained:
            print(f"  eval: chained={args.chained} stage1_only={args.stage1_only} "
                  f"(must match collect, else the success criterion mismatches the training data)")
        if args.stage2_only:
            print(f"  eval: stage2_only=True, camera segment=cam2+cam3 ({args.pc_num_points} points, "
                  f"{args.pc_num_points//2} each), drill starts already grasped (must match collect, "
                  f"else the success criterion mismatches the training data)")
        sys.stdout.flush()

        try:
            with torch.inference_mode():
                while simulation_app.is_running() and (
                        _exam_pending() if _exam_res is not None else total_episodes < args.num_episodes):
                    try:
                        need_policy_call = [
                            env_id for env_id in range(args.num_envs)
                            if pending_actions[env_id] is None or pending_idx[env_id] >= exec_n
                        ]

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
                            _cs = n_obs - 1 + args.lag_comp
                            action_chunks = result["action_pred"][:, _cs:_cs + n_act]  # (B, n_act, 13) GPU

                            for i, env_id in enumerate(need_policy_call):
                                pending_actions[env_id] = action_chunks[i]
                                pending_idx[env_id] = 0
                            _sync(); _prof["infer"] += time.perf_counter() - _t0
                            _prof_infer_calls += 1

                        actions_policy = torch.zeros((args.num_envs, 13),
                                                     device=args.device, dtype=torch.float32)
                        for env_id in range(args.num_envs):
                            if pending_actions[env_id] is not None:
                                actions_policy[env_id] = pending_actions[env_id][pending_idx[env_id]]
                                pending_idx[env_id] += 1

                        if _exam_res is not None and _frozen.any():
                            # frozen envs (their variant's exam is done): hold current joint targets
                            # instead of the policy's action, so they just sit still (never reset,
                            # via _get_dones_freeze_exhausted above -> never draw a new pose either).
                            actions_policy[_frozen] = env_unwrapped.cur_targets[_frozen]

                        env_unwrapped._direct_target = actions_policy
                        actions = actions_policy
                        # collect's 1c timing: pair the obs the policy just decided on with this q*
                        if _dump is not None:
                            _dump_row(actions_policy[_dump["env"]].detach().cpu().numpy())
                        _sync(); _t0 = time.perf_counter()
                        obs_dict_step, rewards, terminated, truncated, _ = env_unwrapped.step(actions)
                        _sync(); _prof["step"] += time.perf_counter() - _t0
                        total_steps += 1


                        _sync(); _t0 = time.perf_counter()
                        cam1.update(dt)
                        cam2.update(dt)
                        if cam3 is not None:
                            cam3.update(dt)
                        _sync(); _prof["render"] += time.perf_counter() - _t0

                        state_13 = build_agent_pos(env_unwrapped, with_force)   # 13 or 26 dims
                        _sync(); _t0 = time.perf_counter()
                        _ws = camera_crop_bounds(env_unwrapped.drill.data.root_pos_w,
                                                 env_origins, _PERC, workspace)
                        pc_fused_2048 = torch.zeros(args.num_envs, args.pc_num_points, 3,
                                                   device=args.device, dtype=torch.float32)
                        if args.stage2_only:
                            # camera segment = cam2 (wrist) + cam3 (plate), cam1 not sampled (same
                            # convention as collect_dp3_data.py --stage2_only).
                            _ws_cam2 = raise_z_floor(_ws, getattr(_PERC, "wrist_cam_z_floor", None))
                            pc2_t, zmask2, _ = camera_pc(cam2, _ws_cam2, sim_pc_per_cam, env_origins,
                                                         pose_w=_cam2_pose())
                            pc3_t, zmask3, _ = camera_pc(cam3, _ws, sim_pc_per_cam, env_origins)
                            total_zero_pc += int(zmask2.sum().item() + zmask3.sum().item())

                            # reuse-last-good-frame substitution: stage2_only is never `chained`
                            # (mutually exclusive), so this always applies here, matching collect's
                            # convention (only chained/stage1_only skip it).
                            pc2_t = torch.where(zmask2.view(-1, 1, 1), _last_pc2, pc2_t)
                            _last_pc2 = pc2_t
                            pc3_t = torch.where(zmask3.view(-1, 1, 1), _last_pc3_stage2, pc3_t)
                            _last_pc3_stage2 = pc3_t

                            pc_fused_2048[:, :sim_pc_per_cam] = pc2_t
                            pc_fused_2048[:, sim_pc_per_cam:] = pc3_t
                        else:
                            pc1_t, zmask1, _ = camera_pc(cam1, _ws, sim_pc_per_cam, env_origins)
                            if not args.disable_cam2:
                                _ws_cam2 = raise_z_floor(_ws, getattr(_PERC, "wrist_cam_z_floor", None))
                                pc2_t, zmask2, _ = camera_pc(cam2, _ws_cam2, sim_pc_per_cam, env_origins,
                                                             pose_w=_cam2_pose())
                                total_zero_pc += int(zmask1.sum().item() + zmask2.sum().item())
                            else:
                                total_zero_pc += int(zmask1.sum().item())

                            if not args.chained:
                                pc1_t = torch.where(zmask1.view(-1, 1, 1), _last_pc1, pc1_t)
                                _last_pc1 = pc1_t
                                if not args.disable_cam2:
                                    pc2_t = torch.where(zmask2.view(-1, 1, 1), _last_pc2, pc2_t)
                                    _last_pc2 = pc2_t

                            pc_fused_2048[:, :sim_pc_per_cam] = pc1_t   # cam1 30cm follow
                            if not args.disable_cam2:
                                pc_fused_2048[:, sim_pc_per_cam:] = pc2_t   # cam2 30cm follow (same frame, different view)

                        pc_for_policy = add_mask_channel(append_robot_drill(pc_fused_2048))
                        # empty-camera mask, same composition rule collect uses to drop a frame
                        # (stage2_only: cam2|cam3; disable_cam2: cam1 alone; else cam1|cam2)
                        if _dump is not None:
                            _l = locals()
                            if args.stage2_only:
                                _bad_now = _l["zmask2"] | _l["zmask3"]
                            elif args.disable_cam2:
                                _bad_now = _l["zmask1"]
                            else:
                                _bad_now = _l["zmask1"] | _l["zmask2"]
                        _sync(); _prof["pc"] += time.perf_counter() - _t0

                        env_episode_rewards += rewards
                        is_done = terminated.bool() | truncated.bool()

                        try:
                            lenient_success = env_unwrapped._cached_lenient_success
                        except AttributeError:
                            lenient_success = torch.zeros(args.num_envs, dtype=torch.bool,
                                                          device=args.device)

                        if args.chained:
                            _inst = getattr(env_unwrapped, "_cached_instant_success", None)
                            if _inst is not None:
                                ever_grasp |= _inst.detach().cpu().numpy().astype(bool)

                        finished = torch.where(is_done)[0]
                        if finished.numel() > 0:
                            reset_state = build_agent_pos(env_unwrapped, with_force)   # 13 or 26 dims

                            for env_idx in finished:
                                env_id = env_idx.item()
                                total_episodes += 1
                                episode_rewards.append(env_episode_rewards[env_id].item())

                                if lenient_success[env_id]:
                                    successful_episodes += 1
                                    success_log.append(1)
                                else:
                                    success_log.append(0)

                                if args.chained:
                                    if ever_grasp[env_id]:
                                        grasp_successful_episodes += 1
                                    ever_grasp[env_id] = False   # new episode, re-accumulate
                                    _sw = int(env_unwrapped._last_episode_switch_step[env_id].item())
                                    if _sw >= 0:
                                        stable_grasp_episodes += 1
                                        grasp_switch_steps.append(_sw)

                                env_episode_rewards[env_id] = 0.0
                                pending_actions[env_id] = None
                                pending_idx[env_id] = 0
                                env_obs_histories[env_id].clear()
                                for _ in range(n_obs):
                                    env_obs_histories[env_id].append({
                                        "agent_pos": reset_state[env_id],
                                        "point_cloud": pc_for_policy[env_id],
                                    })

                        for env_id in range(args.num_envs):
                            if not is_done[env_id]:
                                env_obs_histories[env_id].append({
                                    "agent_pos": state_13[env_id],
                                    "point_cloud": pc_for_policy[env_id],
                                })

                        # --dump_obs_zarr bookkeeping: episode boundary first (so the finished
                        # episode is written before prev_* is overwritten by the post-reset frame),
                        # then cache this step's post-step observation as the next row's "obs".
                        if _dump is not None and _dump["done"] < _dump["want"]:
                            _e = _dump["env"]
                            if is_done[_e]:
                                _dump_flush()
                            _dump["prev_pc"] = pc_for_policy[_e].detach().cpu().numpy()
                            _dump["prev_state"] = state_13[_e].detach().cpu().numpy()
                            # collect skips frames whose camera segment came back empty
                            _dump["prev_bad"] = bool(_bad_now[_e].item())

                        # sequential-exam mode: stop once every variant's first N poses are graded
                        # (this is the actual stop condition -- see the while-loop guard above --
                        # not args.num_episodes; this inner check just lets us stop mid-iteration
                        # instead of running one extra step before the outer guard re-checks)
                        if _exam_res is not None and not _exam_pending():
                            print("[EXAM] all sequential init-pose items graded -> stopping", flush=True)
                            break

                        if total_steps % 100 == 0:
                            elapsed = time.time() - start_time
                            sr = successful_episodes / max(total_episodes, 1) * 100
                            _win = max(total_steps - _last_print_step, 1)
                            _ms = lambda k: _prof[k] / _win * 1000.0
                            _infer_ms = _prof["infer"] / max(_prof_infer_calls, 1) * 1000.0
                            _grasp = (f"| grasp={grasp_successful_episodes} "
                                      f"({grasp_successful_episodes / max(total_episodes, 1) * 100:.1f}%) "
                                      f"stable={stable_grasp_episodes} "
                                      if args.chained else "")
                            _eps_desc = (f"{total_episodes} (exam {_exam_graded_count()}/{_exam_total}, "
                                        f"frozen={int(_frozen.sum().item())}/{args.num_envs})"
                                        if _exam_res is not None
                                        else f"{total_episodes}/{args.num_episodes}")
                            print(f"  step={total_steps} | eps={_eps_desc} "
                                  f"| succ={successful_episodes} ({sr:.1f}%) {_grasp}"
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

        elapsed = time.time() - start_time
        m, s = divmod(int(elapsed), 60)
        h, m = divmod(m, 60)

        if total_episodes > 0:
            print(f"Success rate: {successful_episodes/total_episodes*100:.1f}%")
        if args.chained and total_episodes > 0:
            _gr = grasp_successful_episodes / total_episodes * 100
            _sr2 = stable_grasp_episodes / total_episodes * 100
            print(f"Grasp success (>=1 step check_success): {grasp_successful_episodes} ({_gr:.1f}%)")
            print(f"Grasp stable (20/50 stable, entered stage2): {stable_grasp_episodes} ({_sr2:.1f}%)")
            if stable_grasp_episodes > 0:
                _ar = successful_episodes / stable_grasp_episodes * 100
                print(f"Align success | stable (alignment rate given stable grasp): {_ar:.1f}%")
            if grasp_switch_steps:
                _ss = np.array(grasp_switch_steps)
                print(f"Switch step (time to stable grasp): mean={_ss.mean():.0f} "
                      f"median={np.median(_ss):.0f} max={_ss.max()} steps")
        if _exam_res is not None:
            _et = sum(len(r) for r in _exam_res.values())
            _es = sum(int((r == 1).sum()) for r in _exam_res.values())
            _er = sum(int((r != -1).sum()) for r in _exam_res.values())
            print(f"\n[EXAM] sequential init-pose eval (first {_EXAM_N}/variant, each pose run once):")
            print(f"  OVERALL success {_es}/{_er} = {_es/max(_er,1)*100:.1f}%  (exam poses total {_et})")
            for vid in sorted(_exam_res):
                r = _exam_res[vid]; s = int((r == 1).sum()); nn = int((r != -1).sum())
                print(f"  variant {vid}: {s}/{nn} = {s/max(nn,1)*100:.1f}%")
        if episode_rewards:
            ep_np = np.array(episode_rewards)
            print(f"Reward: mean={ep_np.mean():.1f}, max={ep_np.max():.1f}, min={ep_np.min():.1f}, std={ep_np.std():.1f}")
        print(f"Time: {h}h{m}m{s}s, {total_steps/elapsed:.0f} steps/s")
        if success_log:
            print(f"Success log: {success_log}")
        print(f"{'='*60}")

    except Exception:
        _exit_code = 1
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
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