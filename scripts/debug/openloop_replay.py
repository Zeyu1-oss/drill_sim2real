"""Open-loop replay test: load the drill initial poses + action sequences recorded by collect, reset to the same start,
ignore observations and feed the recorded q* frame by frame (open-loop), comparing the replayed vs recorded joint trajectory deviation.

Purpose: isolate \"is the data/label reproducible\" from \"the closed-loop policy\".
  - replay deviation always small -> recorded actions + deterministic dynamics suffice to re-walk -> data reproducible, the problem is the closed-loop policy.
  - replay diverges somewhere -> the task is chaotically sensitive to micro-state, even exact open-loop actions cannot follow the original trajectory.

No camera needed (open-loop replay does not read observations). Data source: any collect output (action/state in the zarr +
drill poses in <output>_init_poses/init_poses.npz), the two 1:1 by episode order.

Usage:
  conda activate env_isaaclab && source .../isaac-sim/setup_conda_env.sh
  python scripts/openloop_replay.py \
      --zarr data/inspire_drill_dp3_chained_v4.zarr \
      --init_poses data/inspire_drill_dp3_chained_v4_init_poses/init_poses.npz \
      --chained --num_replays 3 --headless
"""
import argparse
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root (scripts/debug/ -> ../../..)
sys.path.insert(0, project_root)

import numpy as np
import torch


def _safe_compile(model=None, *a, **k):
    return model if callable(model) else (lambda fn: fn)
torch.compile = _safe_compile


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--zarr', type=str, required=True)
    p.add_argument('--init_poses', type=str, required=True)
    p.add_argument('--num_replays', type=int, default=30)
    p.add_argument('--chained', action='store_true')
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument('--max_steps', type=int, default=None)
    p.add_argument('--out_trace', type=str, default=None)
    p.add_argument('--settle_probe', type=int, default=0)
    p.add_argument('--shuffle_init', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()

    # ---------- read replay data (before starting sim, precompute env->episode assignment info) ----------
    import zarr
    z = zarr.open(args.zarr, mode="r")
    actions_all = np.asarray(z["data/action"][:]).astype(np.float32)       # (T,13) q* absolute target angles
    states_all = np.asarray(z["data/state"][:, :13]).astype(np.float32)    # (T,13) recorded joint positions
    ends = np.asarray(z["meta/episode_ends"][:]).astype(np.int64)
    starts = np.concatenate([[0], ends[:-1]])
    ip = np.load(args.init_poses)
    ip_var = ip["variant"].astype(np.int64)
    ip_pos, ip_quat, ip_goal = (ip["pos_local"].astype(np.float32),
                                ip["quat"].astype(np.float32),
                                ip["goal_quat"].astype(np.float32))
    n_ep = len(ends)
    assert len(ip_var) == n_ep, f"init_poses({len(ip_var)}) and zarr episodes({n_ep}) not 1:1"
    eps_by_var = {}
    for e in range(n_ep):
        eps_by_var.setdefault(int(ip_var[e]), []).append(e)
    print(f"[DATA] {n_ep} episodes, variants={ {v: len(l) for v, l in eps_by_var.items()} }", flush=True)

    num_envs = args.num_replays

    # ---------- start sim (no camera) ----------
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=args.headless, enable_cameras=False)
    simulation_app = app_launcher.app

    _exit_code = 0
    try:
        import omni.timeline
        import omni.usd
        from tasks.grasp_drill_env import GraspDrillEnv, create_grasp_drill_env_cfg
        from perception.target_drive import install_direct_drive
        from perception.camera_setup import apply_render_settings

        omni.usd.get_context().new_stage()
        cfg = create_grasp_drill_env_cfg(
            num_envs=num_envs, device=args.device, headless=args.headless,
            enable_cameras=False, include_plate=args.chained, include_plate_camera=False)
        cfg.seed = args.seed
        cfg.episode_length_s = 1e6   # do not end on timeout during replay

        # env->variant is originally sampled by torch.randint when building the env (grasp_drill_env.py:570, before scene setup
        # ), so with small num_replays it often does not cover all variants. Here we only temporarily, at the env-build step,
        # replace torch.randint with a deterministic loop (env i -> variant i % num_drill_variants), restored right after;
        # after scene setup nothing else depends on it being random (the other two randint in _reset_idx go through
        # the _drill_pose_override_fn-overridden branch, unused by this script), so the replacement is safe and local.
        _orig_randint = torch.randint
        def _det_variant_randint(low, high, size, **kw):
            n = size[0] if isinstance(size, (tuple, list)) else int(size)
            dev = kw.get("device", None)
            return (torch.arange(n, device=dev) % high).long()
        torch.randint = _det_variant_randint
        try:
            if args.chained:
                from tasks.chained_env import get_chained_env_class
                ChainedEnv = get_chained_env_class()
                base_env = ChainedEnv(cfg=cfg, success_hold_stop=0)
            else:
                base_env = GraspDrillEnv(cfg=cfg)
        finally:
            torch.randint = _orig_randint
        env = base_env.unwrapped

        # ---- read each env's variant (fixed by randint at init, reset only reads it) ----
        # env->variant is randomly sampled, not controlled by this script; a naive in-order assignment may have multiple envs pick the same
        # variant while some variants get no env, causing \"3 different recordings exist but the same one is loaded repeatedly\"
        # a misleading result. Here we use greedy: prioritize covering every episode in the data with at least one env,
        # and explicitly warn about episodes that cannot be covered (rather than silently duplicating).
        env_variants = env._drill_variant_indices.detach().cpu().numpy().astype(int)
        env_to_ep = np.full(num_envs, -1, dtype=np.int64)
        env_pool_by_var = {}
        for e in range(num_envs):
            env_pool_by_var.setdefault(int(env_variants[e]), []).append(e)

        uncovered = []
        for vid, eps in eps_by_var.items():
            pool = list(env_pool_by_var.get(vid, []))
            if not pool:
                uncovered.extend(eps)
                continue
            for i, ep in enumerate(eps):
                if i < len(pool):
                    env_to_ep[pool[i]] = ep
                else:
                    uncovered.append(ep)   # this variant's env slots are used up, this recording cannot be covered this time
        # remaining envs without an episode (this variant has spare slots): reassign already-covered episodes, leave no -1
        for vid, pool in env_pool_by_var.items():
            eps = eps_by_var.get(vid)
            if not eps:
                continue
            for e in pool:
                if env_to_ep[e] < 0:
                    env_to_ep[e] = eps[0]

        print(f"[MAP] env→episode(action): {list(env_to_ep)}  (env variants={list(env_variants)})", flush=True)
        if uncovered:
            print(f"[WARN] this run's random variant draw with num_replays={num_envs} did not cover all recorded episodes,"
                  f"the following episodes had no env draw the matching variant and were not replayed: {sorted(set(uncovered))}."
                  f"to see all {n_ep} at once, increase --num_replays and rerun (higher hit probability per variant)"
                  f", or run a few times until [WARN] no longer appears.", flush=True)

        # pose episode: normal = same as action; --shuffle_init = another one of the same variant (deliberate mismatch, as a control)
        env_to_pose = env_to_ep.copy()
        if args.shuffle_init:
            for e in range(num_envs):
                v = int(env_variants[e])
                alt = [ep for ep in eps_by_var[v] if ep != env_to_ep[e]]
                if alt:
                    env_to_pose[e] = alt[e % len(alt)]
            print(f"[SHUFFLE] mismatch control: pose episode={list(env_to_pose)} (action still uses {list(env_to_ep)})", flush=True)

        ep_pos_t = torch.from_numpy(ip_pos[env_to_pose]).to(args.device)
        ep_quat_t = torch.from_numpy(ip_quat[env_to_pose]).to(args.device)
        ep_goal_t = torch.from_numpy(ip_goal[env_to_pose]).to(args.device)

        def _override(env_ids, variant_ids):
            idx = env_ids.detach().cpu().numpy()
            return ep_pos_t[idx].cpu(), ep_quat_t[idx].cpu(), ep_goal_t[idx].cpu()
        env._drill_pose_override_fn = _override

        from isaaclab.sim.utils import delete_prim
        delete_prim("/World/Template")
        apply_render_settings(env.step_dt)

        env.reset()
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        env.reset()

        # ---- disable all termination during replay (else env.step auto-resets done envs and interrupts replay) ----
        def _no_dones():
            z0 = torch.zeros(num_envs, dtype=torch.bool, device=args.device)
            return z0, z0.clone()
        env._get_dones = _no_dones

        install_direct_drive(env)   # _apply_action directly commands _direct_target = recorded q*
        ci = env.controlled_joint_indices
        dt = env.step_dt

        # ---- self-check: drill actual pose in replay vs loaded init pose (env-local), proving pose loading is correct ----
        _actual_pos = (env.drill.data.root_pos_w - env.scene.env_origins).detach().cpu().numpy()
        _actual_quat = env.drill.data.root_quat_w.detach().cpu().numpy()   # wxyz
        _loaded_pos = ip_pos[env_to_pose]; _loaded_quat = ip_quat[env_to_pose]
        print("[POSE-CHECK] loaded init pose vs drill actual pose after reset (env-local):", flush=True)
        for e in range(num_envs):
            _dp = np.abs(_actual_pos[e] - _loaded_pos[e]).max() * 100
            _dq = np.abs(_actual_quat[e] - _loaded_quat[e]).max()
            print(f"  env{e} pose#{env_to_pose[e]}(v{env_variants[e]}): "
                  f"loaded_pos={np.round(_loaded_pos[e],4)} actual={np.round(_actual_pos[e],4)} "
                  f"|Δpos|={_dp:.2f}cm |Δquat|={_dq:.3f}", flush=True)
        print("  (|dpos|<~1cm, |dquat|<~0.02 = pose loading correct; env1/env2 with the same pose# should be identical)", flush=True)

        # ---- diagnostic probe: freeze the hand, let the drill sit at its init pose alone, measure how much it drops/drifts ----
        if args.settle_probe > 0:
            hold = env.cur_targets.clone()   # hand initial target after reset, frozen
            p0 = (env.drill.data.root_pos_w - env.scene.env_origins).clone()
            print(f"\n[SETTLE-PROBE] freeze the hand, let the drill free-run at its init pose for {args.settle_probe} steps, see how much it drifts:", flush=True)
            print(f"  initial z(cm): {np.round(p0[:,2].cpu().numpy()*100,1)}  (variant={list(env_variants)})", flush=True)
            with torch.inference_mode():
                for t in range(args.settle_probe):
                    if not simulation_app.is_running():
                        break
                    env._direct_target = hold
                    env.step(hold)
                    pnow = env.drill.data.root_pos_w - env.scene.env_origins
                    drift = (pnow - p0).norm(dim=1).detach().cpu().numpy() * 100
                    if t % 5 == 0 or t == args.settle_probe - 1:
                        print(f"  step={t:3d} | drill displacement(cm)={np.round(drift,2)} "
                              f"| current z(cm)={np.round(pnow[:,2].detach().cpu().numpy()*100,1)}", flush=True)
            print("  [read] displacement>~2cm or z clearly dropping = init is an unsettled spawn pose -> actions do not match;"
                  "displacement~0 = init pose is stable, look elsewhere", flush=True)
            return

        # per-env recorded action/state slices
        ep_actions = [actions_all[starts[e]:ends[e]] for e in env_to_ep]
        ep_states = [states_all[starts[e]:ends[e]] for e in env_to_ep]
        ep_len = np.array([len(a) for a in ep_actions])
        max_len = int(ep_len.max()) if args.max_steps is None else min(args.max_steps, int(ep_len.max()))
        print(f"[REPLAY] per-env episode len={list(ep_len)}  replay {max_len} steps open-loop\n", flush=True)

        drill_z0 = env.drill.data.root_pos_w[:, 2].clone()   # initial drill height (for lift detection)
        div_arm = np.full((max_len, num_envs), np.nan, np.float32)
        div_fin = np.full((max_len, num_envs), np.nan, np.float32)
        rec_tr = np.full((max_len, num_envs, 13), np.nan, np.float32)
        rep_tr = np.full((max_len, num_envs, 13), np.nan, np.float32)
        drill_lift = np.full((max_len, num_envs), np.nan, np.float32)
        hand_drill = np.full((max_len, num_envs), np.nan, np.float32)   # nearest hand-body to drill distance (for alignment)
        dead = np.zeros(num_envs, dtype=bool)   # freeze an env after it diverges (NaN), does not affect others

        with torch.inference_mode():
            for t in range(max_len):
                if not simulation_app.is_running():
                    break
                # measured joint position at decision time t (before applying action[t]), aligned with recorded state[t]
                pos_now = env.franka.data.joint_pos[:, ci]                # (num_envs,13)
                nan_env = torch.isnan(pos_now).any(dim=1).detach().cpu().numpy()
                for i in np.where(nan_env & ~dead)[0]:
                    print(f"[REPLAY] env{i} NaN@step {t} -> freeze (other envs continue)", flush=True)
                dead = dead | nan_env
                if dead.all():
                    print(f"[REPLAY] all envs diverged, stopping @step {t}", flush=True); break
                pos_np = torch.nan_to_num(pos_now, nan=0.0).detach().cpu().numpy()

                tgt = torch.zeros(num_envs, 13, device=args.device, dtype=torch.float32)
                for i in range(num_envs):
                    ti = min(t, ep_len[i] - 1)
                    tgt[i] = torch.from_numpy(ep_actions[i][ti]).to(args.device)
                    if t < ep_len[i] and not dead[i]:
                        d = pos_np[i] - ep_states[i][t]
                        div_arm[t, i] = np.abs(d[:7]).mean()
                        div_fin[t, i] = np.abs(d[7:]).mean()
                        rec_tr[t, i] = ep_states[i][t]
                        rep_tr[t, i] = pos_np[i]

                env._direct_target = tgt
                env.step(tgt)
                _drill_pw = env.drill.data.root_pos_w                      # (num_envs,3)
                _dl = (_drill_pw[:, 2] - drill_z0).detach().cpu().numpy()
                _hd = (env.franka.data.body_pos_w - _drill_pw[:, None, :]).norm(dim=-1).min(dim=1).values
                _hdv = _hd.detach().cpu().numpy()
                _dl[dead] = np.nan; _hdv[dead] = np.nan   # frozen envs are excluded
                drill_lift[t] = _dl; hand_drill[t] = _hdv

                if t % 50 == 0 or t == max_len - 1:
                    _a = np.nanmean(div_arm[t]); _f = np.nanmean(div_fin[t])
                    _lz = np.nanmean(drill_lift[t]); _hdm = np.nanmean(hand_drill[t])
                    print(f"  step={t:4d} | deviation(deg) arm={_a*57.3:5.2f} finger={_f*57.3:5.2f} "
                          f"| drill lift {_lz*100:5.1f}cm | hand<->drill nearest {_hdm*100:4.1f}cm", flush=True)

        # ---------- summary ----------
        print(f"\n{'='*64}\n[OPENLOOP] per trajectory replay vs recorded joint deviation(deg) and drill lift(cm):", flush=True)
        for i in range(num_envs):
            L = int(ep_len[i]) if args.max_steps is None else min(int(ep_len[i]), max_len)
            a = div_arm[:L, i] * 57.3; f = div_fin[:L, i] * 57.3
            lz = drill_lift[:L, i] * 100; hd = hand_drill[:L, i] * 100
            _tag = f"pose#{env_to_pose[i]}" if args.shuffle_init else ""
            print(f"  env{i} act#{env_to_ep[i]}{_tag}(v{env_variants[i]}) len={L}: "
                  f"arm last={a[-1]:.2f} max={np.nanmax(a):.2f} | "
                  f"finger last={f[-1]:.2f} max={np.nanmax(f):.2f} | "
                  f"lift max={np.nanmax(lz):.1f}cm | hand<->drill min={np.nanmin(hd):.1f}cm", flush=True)
        print("  [align] hand<->drill min small (~1-3cm) = hand actually grasps the drill -> init aligned; if always large (>10cm) = mismatch", flush=True)
        print("  [reproduce] deviation always <~2-3deg = reproducible (problem is closed-loop); a step blowing up = chaotic sensitivity there", flush=True)
        print('='*64, flush=True)

        if args.out_trace:
            np.savez(args.out_trace, div_arm=div_arm, div_fin=div_fin,
                     rec_traj=rec_tr, rep_traj=rep_tr, drill_lift=drill_lift,
                     hand_drill=hand_drill, env_to_ep=env_to_ep, env_to_pose=env_to_pose,
                     env_variants=env_variants, ep_len=ep_len)
            print(f"[OPENLOOP] trace -> {args.out_trace}", flush=True)

    except Exception:
        _exit_code = 1
        import traceback
        traceback.print_exc()
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        try:
            import omni.timeline
            omni.timeline.get_timeline_interface().stop()
        except Exception:
            pass
        try:
            simulation_app.close()
        except Exception:
            pass
        os._exit(_exit_code)


if __name__ == "__main__":
    main()
