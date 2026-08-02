#!/usr/bin/env python3
"""
Stage2 environment — inherits GraspDrillEnv for scene setup and observations,
overrides _reset_idx (pkl sampling), _get_rewards, _get_dones (custom logic).
"""

import os
import numpy as np
import torch

ISAAC_NUCLEUS_DIR = "/home/zeyu/inspire_drill/assets"


# ============================================================
# Dataset wrapper
# ============================================================
class SuccessDataDataset:
    def __init__(self, samples_or_path):
        if isinstance(samples_or_path, str):
            import pickle
            if not os.path.exists(samples_or_path):
                raise FileNotFoundError(f"[Stage2] Dataset not found: {samples_or_path}")
            with open(samples_or_path, "rb") as f:
                raw = pickle.load(f)
            self._samples = raw["samples"]
            print(f"[Stage2] Loaded {len(self._samples)} samples from {samples_or_path}")
        else:
            self._samples = samples_or_path

    def __len__(self):
        return len(self._samples)

    def get_batch(self, n: int):
        indices = np.random.randint(0, len(self._samples), size=n)
        return [self._samples[i] for i in indices]


# ============================================================
# Build Stage2Env class — inherits GraspDrillEnv
# ============================================================
def get_stage2_env_class():
    """Must be called AFTER AppLauncher is running."""
    from tasks.grasp_drill_env import GraspDrillEnv, GraspDrillEnvCfg
    return _make_stage2_env(GraspDrillEnv, GraspDrillEnvCfg)


def _make_stage2_env(GraspDrillEnv, GraspDrillEnvCfg):
    """Stage2Env inherits GraspDrillEnv for scene + obs, overrides reward/done/reset."""

    class Stage2Env(GraspDrillEnv):
        """
        Stage2: inherits all scene setup and observation logic from GraspDrillEnv.
        Custom: _reset_idx (pkl sampling), _get_rewards, _get_dones,
        _get_observations (stage1 obs + plate pose 7 dims pos+quat).
        """

        # episode-level alignment success: staying in the aligned region (dist<1cm and angle<5deg)
        # for >= this many steps sets this episode's sticky success flag (~0.67s @30Hz). Written to
        # _cached_lenient_success, reused by collection filtering / deploy stats / failure stats (same interface as stage1).
        ALIGN_SUCCESS_HOLD = 20

        def __init__(
            self,
            cfg: GraspDrillEnvCfg,
            success_dataset=None,
            target_pos=(0.0, 0.0, 0.85),
            target_quat=(1.0, 0.0, 0.0, 0.0),
            debug=False,
            success_hold_stop: int = 0,
            **kwargs,
        ):
            # ── PKL dataset (before super().__init__) ───────────
            self.success_dataset = success_dataset
            self.debug = debug
            # >0: after holding alignment this many steps, terminate early and mark success
            #     (same convention as ChainedEnv's success_hold_stop); 0: run to timeout (default,
            #     matches stage2 RL training, which wants the fixed episode length for reward shaping).
            self.success_hold_stop = int(success_hold_stop)
            # obs = all stage1 obs + plate pose 7 dims (pos_local 3 + quat wxyz 4).
            # plate is randomized +/-10cm/+/-10deg each episode; without it in the obs the target is invisible and the task unsolvable.
            # when loading a stage1 checkpoint, train2.py zero-pads the input layer to extend it (auto by dim difference).
            cfg.observation_space = int(cfg.observation_space) + 7

            if target_pos is not None:
                cfg.target_pos = tuple(target_pos)
            if target_quat is not None:
                cfg.target_quat = tuple(target_quat)

            # ── Delegate ALL scene + obs setup to GraspDrillEnv ──
            super().__init__(cfg, debug=debug, **kwargs)

            # alignment success tracking: consecutive-hold counter + this episode's sticky flag
            self._align_hold_steps = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device)
            self._align_achieved = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device)

            # ── Target alignment (used in Stage2 rewards) ──────
            self._target_pos = torch.tensor(
                getattr(cfg, "target_pos", (0.0, 0.0, 0.85)),
                dtype=torch.float32, device=self.device,
            )
            self._target_quat = torch.nn.functional.normalize(
                torch.tensor(
                    getattr(cfg, "target_quat", (1.0, 0.0, 0.0, 0.0)),
                    dtype=torch.float32, device=self.device,
                ).unsqueeze(0), dim=1
            ).squeeze(0)

            # ── PKL variant index → sample pool ───────────────
            self._pkl_by_variant = {}
            if self.success_dataset is not None:
                for s in self.success_dataset._samples:
                    vid = s["variant_idx"]
                    self._pkl_by_variant.setdefault(vid, []).append(s)
                for vid, pool in sorted(self._pkl_by_variant.items()):
                    print(f"[Stage2] PKL variant {vid}: {len(pool)} samples")

            # ── Plate: registered in scene by create_stage2_env_cfg ──
            try:
                self.plate = self.scene["plate"]
                print(f"[Stage2] Plate loaded from scene, num_instances={self.plate.num_instances}")
            except KeyError:
                self.plate = None
                print("[WARN] Stage2: 'plate' not found in scene, check cfg.scene.plate")

            # Pre-computed local offset of /scene/Meshes/Xform1 inside plate.usd (from USD inspection)
            # This is the constant translation between plate root and Xform1 in the USD file.
            self._plate_xform1_local_offset = torch.tensor(
                [0.10775, 0.0076, 0.1088], dtype=torch.float32, device=self.device
            )

            # ── Plate xform frames visualization (debug) ──
            if self.debug:
                self._add_trigger_offset_visualization()
                self._add_plate_xform_visualization()
                self._add_drill_bit_frame_visualization()

        def _setup_scene(self):
            """Set up the scene - set plate collision API before sensor init"""
            from tasks.grasp_drill_env import GraspDrillEnv
            GraspDrillEnv._setup_scene(self)

            # === Plate collision physics setup (before sensors init in sim.reset()) ===
            try:
                self.plate = self.scene["plate"]
                self._setup_plate_contact_physics()
            except KeyError:
                print("[WARN] Stage2: 'plate' not found in scene")

        # ================================================================
        def _randomize_plate(self, env_ids: torch.Tensor):
            """Plate pose randomization (+/-10cm / +/-10deg about z) and rebuild Xform cache.

            Extracted from _reset_idx, shared by Stage2 normal reset and ChainedEnv (chained play).

            If self._plate_pose_override_fn is given, skip the random jitter and use the env-local
            (pos, quat) it returns to place the plate (collect/deploy use it to pin the plate to a fixed
            position, removing the \"alignment target changes every reset\" variable -- init_pose_file only
            reproduces the drill, not the plate; a fixed plate lets the whole scene reproduce byte-for-byte).
            """
            if self.plate is None:
                return
            n = len(env_ids)
            env_origins = self.scene.env_origins[env_ids]

            _override = getattr(self, "_plate_pose_override_fn", None)
            if _override is not None:
                _pl, _q = _override(env_ids)
                plate_pos_w = _pl.to(self.device).float() + env_origins
                plate_quat = _q.to(self.device).float()
            else:
                # Base pos from init_state: (0, 1, 0.5), jitter ±10cm on x, y, z
                plate_default_pos = torch.tensor([0.0, 1, 0.5], device=self.device)
                rx = (torch.rand(n, device=self.device) * 0.20 - 0.10)  # [-0.10, 0.10]
                ry = (torch.rand(n, device=self.device) * 0.20 - 0.10)  # [-0.10, 0.10]
                rz = (torch.rand(n, device=self.device) * 0.20 - 0.10)  # [-0.10, 0.10]
                jitter_pos = torch.stack([rx, ry, rz], dim=1)  # [n, 3]
                plate_pos = plate_default_pos.unsqueeze(0) + jitter_pos  # [n, 3]
                plate_pos_w = plate_pos + env_origins  # world coords

                # Base rot from init_state, add ±10° on z only
                base_quat = torch.tensor([0, 0.0, 0.0, 1], device=self.device)
                delta_yaw = (torch.rand(n, device=self.device) - 0.5) * (2 * 0.1745)  # [-10°, +10°]
                delta_quat = self._euler_to_quat(
                    torch.zeros(n, device=self.device),  # no roll
                    torch.zeros(n, device=self.device),  # no pitch
                    delta_yaw
                )  # [n, 4]
                plate_quat = self._quat_mul_torch(delta_quat, base_quat.unsqueeze(0).expand(n, -1))  # [n, 4]
                plate_quat = torch.nn.functional.normalize(plate_quat, dim=1)  # normalize to unit quat

            plate_root_state = torch.cat([
                plate_pos_w, plate_quat,
                torch.zeros(n, 6, device=self.device),
            ], dim=1)
            self.plate.write_root_state_to_sim(plate_root_state, env_ids=env_ids)
            # Invalidate cached Xform1 positions
            if hasattr(self, "_xform1_pos_cache"):
                delattr(self, "_xform1_pos_cache")
            if hasattr(self, "_xform1_rot_cache"):
                delattr(self, "_xform1_rot_cache")
            # Always rebuild with freshly computed plate state so kinematics lag doesn't matter.
            # For partial resets, we need full [num_envs] arrays: get unchanged values from cache
            # (which we just deleted above, so read root_pos_w for unchanged envs).
            if n < self.num_envs:
                # Partial reset: merge unchanged + new
                full_plate_pos = self.plate.data.root_pos_w.clone()
                full_plate_quat = self.plate.data.root_quat_w.clone()
                full_plate_pos[env_ids] = plate_pos_w
                full_plate_quat[env_ids] = plate_quat
                self._build_xform1_cache(full_plate_pos, full_plate_quat)
            else:
                # Full reset
                self._build_xform1_cache(plate_pos_w, plate_quat)

        # ================================================================
        def _reset_idx(self, env_ids: torch.Tensor):
            import random

            # ── Clear state buffers ──
            if hasattr(self, "_nan_reset_mask"):
                self._nan_reset_mask[:] = False
            if hasattr(self, "_nan_env_mask"):
                self._nan_env_mask[:] = False
            if hasattr(self, "contact_history_buffer"):
                self.contact_history_buffer[env_ids] = False
            if hasattr(self, "_hand_base_no_contact_steps"):
                self._hand_base_no_contact_steps[env_ids] = 0
            if hasattr(self, "_success_window_idx"):
                self._success_window_idx[env_ids] = 0
                self._success_window[env_ids] = False
            if hasattr(self, "_align_hold_steps"):
                self._align_hold_steps[env_ids] = 0
                self._align_achieved[env_ids] = False

            # ── Statistics (mirrors GraspDrillEnv pattern) ──
            if len(env_ids) > 0:
                failure_reasons = getattr(
                    self, "_cached_failure_mask",
                    torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                )[env_ids].clone()
                nan_batch = self._pending_nan_mask[env_ids].clone()
                normal_timeout_batch = self._normal_timeout_mask[env_ids]
                cached_lenient = getattr(
                    self, "_cached_lenient_success",
                    torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                )[env_ids].clone()
                success_batch = cached_lenient

                n_batch = len(success_batch)
                indices = torch.arange(
                    self._recent_idx, self._recent_idx + n_batch
                ) % self._recent_success_buffer_size
                self._recent_success[indices] = success_batch.cpu()
                self._recent_total[indices] = True
                self._recent_idx += n_batch
                self._recent_filled = min(
                    self._recent_filled + n_batch, self._recent_success_buffer_size
                )

                n = len(env_ids)
                self._failure_stats["total"] += n
                self._failure_stats["physics_nan"] += nan_batch.sum().item()
                self._failure_stats["normal_total"] += normal_timeout_batch.sum().item()
                self._failure_stats["lenient_success_count"] += (
                    success_batch & normal_timeout_batch
                ).sum().item()

                if failure_reasons.any():
                    flipped = getattr(
                        self, "_cached_drill_flipped_mask",
                        torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                    )[env_ids]
                    self._failure_stats["drill_flipped"] += (flipped & failure_reasons).sum().item()
                    self._failure_stats["hand_base_timeout"] += (
                        (self._hand_base_no_contact_steps[env_ids] > self._hand_base_contact_timeout)
                        & failure_reasons
                    ).sum().item()
                    knocked = getattr(
                        self, "_cached_knocked_away_mask",
                        torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                    )[env_ids]
                    self._failure_stats["drill_knocked_away"] += (knocked & failure_reasons).sum().item()

            # ── Variant assignment (before parent reset) ──────
            if self.num_drill_variants > 1:
                torch.manual_seed(
                    self._sim_step_counter // self.cfg.decimation
                    if self.cfg.decimation > 0 else 0
                )
                local_ids = env_ids % self.num_drill_variants
                variant_ids = self._active_indices_tensor[local_ids]
                self._drill_variant_indices[env_ids] = variant_ids
                if self.debug and len(env_ids) <= 8:
                    print(f"[DEBUG reset step={self._sim_step_counter}] "
                          f"env_ids={env_ids.tolist()[:8]} -> "
                          f"variants={variant_ids.tolist()[:8]}")

            # ── Parent reset (physics + Franka default state) ──
            super()._reset_idx(env_ids)

            n = len(env_ids)
            env_origins = self.scene.env_origins[env_ids]

            # ── Plate position and rotation randomization ──
            self._randomize_plate(env_ids)

            variant_ids = self._drill_variant_indices[env_ids]
            controlled = self.controlled_joint_indices

            # ── One-hot variant attributes ──
            _VA = self._variant_attrs
            nv = self.total_num_variants
            vid_oh = torch.nn.functional.one_hot(
                variant_ids.long(), num_classes=nv
            ).float()

            _flat_toff  = torch.zeros(nv * 3, device=self.device)
            _flat_thumb = torch.zeros(nv * 3, device=self.device)
            _flat_bmin  = torch.zeros(nv,    device=self.device)
            _flat_bmax  = torch.zeros(nv,    device=self.device)
            _flat_up    = torch.zeros(nv * 3, device=self.device)
            _flat_scale = torch.zeros(nv * 3, device=self.device)
            _flat_pos   = torch.zeros(nv * 3, device=self.device)
            _flat_rot   = torch.zeros(nv * 4, device=self.device)

            for vid, vdata in _VA.items():
                _flat_toff[vid*3:(vid+1)*3]  = vdata["trigger_offset"]
                _flat_thumb[vid*3:(vid+1)*3]  = vdata["thumb_target_local"]
                _flat_bmin[vid]               = vdata["body_mask_min"]
                _flat_bmax[vid]               = vdata["body_mask_max"]
                _flat_up[vid*3:(vid+1)*3]    = vdata["up_axis"]
                _flat_scale[vid*3:(vid+1)*3]  = vdata["scale"]
                # parent now stores a multi-pose list (initial_pos_list); fallback takes the first base pose
                _flat_pos[vid*3:(vid+1)*3]    = vdata["initial_pos_list"][0]
                _flat_rot[vid*4:(vid+1)*4]    = vdata["initial_rot_list"][0]

            self._trigger1_offset[env_ids] = vid_oh @ _flat_toff.view(nv, 3)
            self._thumb_target_local[env_ids] = vid_oh @ _flat_thumb.view(nv, 3)
            self._body_mask_y_min[env_ids] = vid_oh.float() @ _flat_bmin
            self._body_mask_y_max[env_ids] = vid_oh.float() @ _flat_bmax
            self._up_axis[env_ids] = vid_oh @ _flat_up.view(nv, 3)
            self._drill_scale[env_ids] = vid_oh @ _flat_scale.view(nv, 3)

            # ── PKL sampling: drill pose + robot joints ──
            if self.success_dataset is not None and len(self.success_dataset) > 0:
                drill_pos_world = torch.zeros(n, 3, device=self.device)
                drill_quat_world = torch.zeros(n, 4, device=self.device)
                joint_pos = self.franka.data.default_joint_pos[env_ids].clone()
                joint_vel = self.franka.data.default_joint_vel[env_ids].clone()

                for i in range(n):
                    vid = variant_ids[i].item()
                    pool = self._pkl_by_variant.get(vid, [])
                    if not pool:
                        # no success sample for this variant: fall back to this variant's base initial pose,
                        # to avoid drill_pos_world staying zero (drill placed at the env origin).
                        if not hasattr(self, "_warned_missing_pkl_variant"):
                            self._warned_missing_pkl_variant = set()
                        if vid not in self._warned_missing_pkl_variant:
                            self._warned_missing_pkl_variant.add(vid)
                            print(f"[WARN Stage2] pkl has no samples for variant {vid},"
                                  f"this variant will use the base initial pose (not a grasp pose)")
                        drill_pos_world[i] = _flat_pos.view(nv, 3)[vid] + env_origins[i]
                        drill_quat_world[i] = _flat_rot.view(nv, 4)[vid]
                        continue
                    sample = pool[np.random.randint(len(pool))]

                    drill_pos_world[i] = (
                        torch.as_tensor(sample["drill_pos"], dtype=torch.float32, device=self.device)
                        + env_origins[i]
                    )
                    drill_quat_world[i] = torch.as_tensor(
                        sample["drill_quat"], dtype=torch.float32, device=self.device
                    )

                    jp = sample.get("joint_pos")
                    if jp is not None:
                        joint_pos[i, controlled] = torch.as_tensor(
                            jp, dtype=torch.float32, device=self.device
                        )
            else:
                # Fallback: use variant initial poses
                drill_initial_pos = vid_oh @ _flat_pos.view(nv, 3)
                drill_initial_rot = vid_oh @ _flat_rot.view(nv, 4)
                drill_pos_world = drill_initial_pos + env_origins
                drill_quat_world = drill_initial_rot
                joint_pos = self.franka.data.default_joint_pos[env_ids].clone()
                joint_vel = self.franka.data.default_joint_vel[env_ids].clone()

            # ── Write drill state ──
            drill_root_state = torch.cat([
                drill_pos_world, drill_quat_world,
                torch.zeros(n, 6, device=self.device),
            ], dim=1)
            self._drill.write_root_state_to_sim(drill_root_state, env_ids=env_ids)

            if not hasattr(self, "initial_drill_pos") or self.initial_drill_pos.shape[0] != self.num_envs:
                self.initial_drill_pos = torch.zeros(self.num_envs, 3, device=self.device)
                self.initial_drill_pos[:, 2] = 0.5
            self.initial_drill_pos[env_ids] = drill_pos_world.clone()

            # ── Initial bit tip world position (for drop detection) ──
            bit_offset = self._trigger1_offset[env_ids]  # [n, 3], drill-local
            R_init = self._quat_to_rot_matrix(drill_quat_world)
            bit_pos_init = drill_pos_world + (R_init @ bit_offset.unsqueeze(-1)).squeeze(-1)
            if not hasattr(self, "_initial_bit_pos") or self._initial_bit_pos.shape[0] != self.num_envs:
                self._initial_bit_pos = torch.zeros(self.num_envs, 3, device=self.device)
            self._initial_bit_pos[env_ids] = bit_pos_init

            # ── Initial hand-link → drill-body distances (for drop detection) ──
            # Will be computed AFTER franka joints are written (so hand is at pkl pose).
            pass  # deferred below

            # ── Write robot joints (from pkl) ──
            self.franka.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
            if hasattr(self, "cur_targets"):
                self.cur_targets[env_ids] = joint_pos[:, controlled].clone()

            # ── Compute link distances AFTER hand is at pkl pose ──
            link_dists = self._compute_link_body_dists()  # [num_envs, num_obs_links]
            if not hasattr(self, "_initial_link_body_dists") or self._initial_link_body_dists.shape[0] != self.num_envs:
                self._initial_link_body_dists = torch.zeros_like(link_dists)
            self._initial_link_body_dists[env_ids] = link_dists[env_ids].clone()

            # -- fingertip initial positions in the drill local frame (baseline for whole-hand grasp-drift penalty) --
            # computed after writing the pkl pose into franka joints; baseline = each fingertip's
            # configuration relative to the drill at a successful grasp; _get_rewards compares drift with the same transform each step.
            tip_idx_init = self._get_fingertip_indices()
            if tip_idx_init is not None:
                tips_w_init = self.franka.data.body_pos_w[env_ids][:, tip_idx_init, :]  # [n, F, 3]
                R_init = self._quat_to_rot_matrix(drill_quat_world)                     # [n, 3, 3]
                rel_init = tips_w_init - drill_pos_world.unsqueeze(1)                   # [n, F, 3]
                tips_local_init = torch.einsum(
                    'nij,nfj->nfi', R_init.transpose(1, 2), rel_init)
                if not hasattr(self, "_initial_tips_drill_local") or \
                        self._initial_tips_drill_local.shape[0] != self.num_envs:
                    self._initial_tips_drill_local = torch.zeros(
                        self.num_envs, tips_local_init.shape[1], 3, device=self.device)
                self._initial_tips_drill_local[env_ids] = tips_local_init

            # Debug: print initial min distance per env
            init_min_per_env = link_dists[env_ids].min(dim=1).values


        # ================================================================
        # Observation override: stage1 obs + plate pose 6 dims
        # ================================================================

        def _get_observations(self) -> dict:
            parent_obs_dict = super()._get_observations()
            parent_obs = parent_obs_dict["policy"]

            plate = getattr(self, "plate", None)
            if plate is not None:
                plate_pos_local = plate.data.root_pos_w - self.scene.env_origins   # [N, 3]
                plate_quat = plate.data.root_quat_w                                 # [N, 4] wxyz
                plate_obs = torch.cat([plate_pos_local, plate_quat], dim=1)         # [N, 7]
            else:
                plate_obs = torch.zeros(parent_obs.shape[0], 7, device=parent_obs.device)

            return {"policy": torch.cat([parent_obs, plate_obs], dim=1)}

        # ================================================================
        # Grasp success check
        # ================================================================

        @staticmethod
        def _quat_to_rot_matrix(q: torch.Tensor) -> torch.Tensor:
            """Convert [w,x,y,z] quaternions to rotation matrices [N,3,3].
            Matches isaaclab convention: root_quat_w is (w, x, y, z)."""
            w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
            R = torch.zeros(q.shape[0], 3, 3, device=q.device)
            R[:, 0, 0] = 1 - 2 * (y * y + z * z)
            R[:, 0, 1] = 2 * (x * y - w * z)
            R[:, 0, 2] = 2 * (x * z + w * y)
            R[:, 1, 0] = 2 * (x * y + w * z)
            R[:, 1, 1] = 1 - 2 * (x * x + z * z)
            R[:, 1, 2] = 2 * (y * z - w * x)
            R[:, 2, 0] = 2 * (x * z - w * y)
            R[:, 2, 1] = 2 * (y * z + w * x)
            R[:, 2, 2] = 1 - 2 * (x * x + y * y)
            return R

        @staticmethod
        def _quat_to_axis_angle(q: torch.Tensor) -> torch.Tensor:
            """Convert [w,x,y,z] quaternions to axis-angle [N, 3] (unit_axis * angle).
            Matches isaaclab convention: root_quat_w is (w, x, y, z)."""
            # Flatten extra dims: [N, 4] or [N, 1, 4] -> [N, 4]
            q = q.reshape(-1, 4)
            w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
            # Normalize quaternion
            norm = torch.sqrt(w * w + x * x + y * y + z * z + 1e-8)
            wn, xn, yn, zn = w / norm, x / norm, y / norm, z / norm
            # Angle = 2 * acos(w), clamp for numerical stability
            angle = 2.0 * torch.acos(wn.clamp(-1.0, 1.0))
            # Axis = (x, y, z) / sin(angle/2); avoid division by zero
            sin_half = torch.sqrt(1.0 - wn * wn).clamp(min=1e-8)
            axis = torch.stack([xn, yn, zn], dim=-1) / sin_half.unsqueeze(-1)
            return axis * angle.unsqueeze(-1)

        @staticmethod
        def _euler_to_quat(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
            """Convert Euler angles (roll, pitch, yaw) in radians to quaternions [w,x,y,z].
            Assumes ZYX convention (yaw first, pitch second, roll last)."""
            cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
            cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
            cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
            w = cr * cp * cy + sr * sp * sy
            x = sr * cp * cy - cr * sp * sy
            y = cr * sp * cy + sr * cp * sy
            z = cr * cp * sy - sr * sp * cy
            return torch.stack([w, x, y, z], dim=-1)

        @staticmethod
        def _quat_mul_torch(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
            """Multiply two quaternions [w,x,y,z]. q1 * q2 = apply q2 then q1.
            q1, q2: [..., 4]"""
            w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
            w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
            return torch.stack([w, x, y, z], dim=-1)

        def _compute_drill_bit_pos_w(self) -> tuple:
            """World position of the drill bit tip and its world-space direction for each env."""
            drill_pos = self.drill.data.root_pos_w          # [num_envs, 3]
            drill_quat = self.drill.data.root_quat_w        # [num_envs, 4]  (w,x,y,z)
            variant_ids = self._drill_variant_indices.long()

            bit_offset = torch.stack([
                self._variant_attrs[vid.item()]["drill_bit_offset"] for vid in variant_ids
            ])
            bit_scale = torch.stack([
                self._variant_attrs[vid.item()]["scale"] for vid in variant_ids
            ])
            scaled_offset = bit_offset * bit_scale  # still in drill-local coords

            R_drill = self._quat_to_rot_matrix(drill_quat)  # [N, 3, 3]

            # Per-variant drill_bit_rot: applies to the local frame
            # drill2: identity (bit along drill +Z)
            # drill_blue/yellow: 90° around Y (bit along drill +X)
            bit_rot = torch.stack([
                self._variant_attrs[vid.item()]["drill_bit_rot"] for vid in variant_ids
            ])  # [N, 4] wxyz
            R_bit = self._quat_to_rot_matrix(bit_rot)  # [N, 3, 3]
            # Bit local +Z after R_bit rotation → this is the bit tip direction in drill-local space
            bit_dir_local = R_bit[:, :, 2]  # [N, 3]
            # Transform to world space
            bit_dir_world = torch.bmm(R_drill, bit_dir_local.unsqueeze(-1)).squeeze(-1)  # [N, 3]
            bit_dir_world = torch.nn.functional.normalize(bit_dir_world, dim=1)

            bit_pos = drill_pos + (R_drill @ scaled_offset.unsqueeze(-1)).squeeze(-1)
            return bit_pos, bit_dir_world, R_drill

        def _compute_plate_xform1_pos_w(self) -> torch.Tensor:
            """World position of plate's /scene/Meshes/Xform1 for each env (live, no cache)."""
            plate_pos = self.plate.data.root_pos_w          # [num_envs, 3]
            plate_quat = self.plate.data.root_quat_w        # [num_envs, 4]  (w,x,y,z)
            R = self._quat_to_rot_matrix(plate_quat)
            offset = self._plate_xform1_local_offset.unsqueeze(0)  # [1, 3]
            xform1_pos = plate_pos + (R @ offset.unsqueeze(-1)).squeeze(-1)
            return xform1_pos

        def _build_xform1_cache(self, plate_pos=None, plate_quat=None):
            """Build Xform1 world positions from plate root state (handles randomized plates).

            Args:
                plate_pos:  [num_envs, 3] world positions. If None, reads from plate.data.root_pos_w.
                plate_quat: [num_envs, 4] quaternions (w,x,y,z). If None, reads from plate.data.root_quat_w.
            """
            if plate_pos is None:
                plate_pos = self.plate.data.root_pos_w.clone()
            if plate_quat is None:
                plate_quat = self.plate.data.root_quat_w
            R = self._quat_to_rot_matrix(plate_quat)
            offset = self._plate_xform1_local_offset.unsqueeze(0)  # [1, 3]
            self._xform1_pos_cache = plate_pos + (R @ offset.unsqueeze(-1)).squeeze(-1)
            self._xform1_rot_cache = R.clone()

        # ── All hand-contact sensors that contribute to grasp success ──
        _ALL_HAND_SENSORS = [
            "contact_index_intermediate",
            "contact_thumb_distal",
            "contact_middle_intermediate",
            "contact_ring_intermediate",
            "contact_pinky_intermediate",
            "contact_thumb_intermediate",
            "contact_thumb_proximal_base",
            "contact_thumb_proximal",
            "contact_index_proximal",
            "contact_middle_proximal",
            "contact_ring_proximal",
            "contact_pinky_proximal",
            "contact_hand_base",
        ]

        def _check_grasp_success(self) -> torch.Tensor:
            """Return binary success per env: tip near trigger (3cm) AND ≥7 contacts."""
            num_envs = self.num_envs
            device = self.device

            # ── 1. Tip → trigger distance ──────────────────────────
            drill_pos = self.drill.data.root_pos_w
            drill_quat = self.drill.data.root_quat_w
            variant_ids = self._drill_variant_indices.long()

            trigger_offset = torch.stack([
                self._variant_attrs[vid.item()]["drill_bit_offset"]
                for vid in variant_ids
            ])
            scale = torch.stack([
                self._variant_attrs[vid.item()]["scale"]
                for vid in variant_ids
            ])
            scaled_trigger = trigger_offset * scale

            R = self._quat_to_rot_matrix(drill_quat)
            trigger_world = drill_pos + (R @ scaled_trigger.unsqueeze(-1)).squeeze(-1)

            body_names = self.franka.data.body_names
            tip_name = "R_index_intermediate"
            if tip_name in body_names:
                tip_idx = body_names.index(tip_name)
                tip_pos = self.franka.data.body_pos_w[:, tip_idx, :]
                tip_dist = torch.norm(tip_pos - trigger_world, dim=1)
            else:
                tip_dist = torch.full((num_envs,), 999.0, device=device)
            tip_ok = tip_dist < 0.03  # 3 cm

            # ── 2. Contact count ───────────────────────────────────
            contact_count = torch.zeros(num_envs, device=device, dtype=torch.float32)
            if hasattr(self, "scene") and hasattr(self.scene, "sensors"):
                for sname in self._ALL_HAND_SENSORS:
                    if sname in self.scene.sensors:
                        s = self.scene.sensors[sname]
                        fm = s.data.force_matrix_w
                        fm = torch.nan_to_num(fm, nan=0.0, posinf=0.0, neginf=0.0)
                        fm = torch.clamp(fm, -50.0, 50.0)
                        force_mag = torch.norm(fm.sum(dim=(1, 2)), dim=1)
                        contact_count += (force_mag > 0.1).float()
            multi_contact_ok = contact_count >= 7

            return tip_ok & multi_contact_ok



        # fingertip bodies tracked for grasp-drift penalty (missing ones auto-skipped)
        _FINGERTIP_BODIES = ("R_index_intermediate", "R_middle_intermediate",
                             "R_ring_intermediate", "R_pinky_intermediate",
                             "R_thumb_distal")

        def _get_fingertip_indices(self):
            """Lazily resolve fingertip body indices, return LongTensor or None (if none found)."""
            if not hasattr(self, "_fingertip_body_idx"):
                names = list(self.franka.data.body_names)
                idx = [names.index(n) for n in self._FINGERTIP_BODIES if n in names]
                if idx:
                    self._fingertip_body_idx = torch.tensor(
                        idx, dtype=torch.long, device=self.device)
                    print(f"[Stage2] tip_drift tracking {len(idx)} fingertips: "
                          f"{[names[i] for i in idx]}")
                else:
                    self._fingertip_body_idx = None
                    print("[WARN Stage2] no fingertip body found, tip_drift penalty disabled")
            return self._fingertip_body_idx

        def _get_rewards(self) -> torch.Tensor:
            """reward = proximity (dual-scale) + orientation (distance-gated)
            + success (+30 every step inside the success region, continuous, non-terminating)
            + tip_drift (whole-hand fingertip configuration drift relative to the drill, single-finger clamp 0.2m)
            + plate_contact penalty (anti-collision plate)."""
            if self.debug:
                try:
                    import isaacsim.util.debug_draw._debug_draw as omni_debug_draw
                    di = omni_debug_draw.acquire_debug_draw_interface()
                    di.clear_points()
                    di.clear_lines()
                except Exception:
                    pass


            self._reward_components.clear()  # keep only Stage2-related reward components

            # ── Proximity reward: drill bit → plate Xform1 distance ──
            bit_pos, bit_dir_world, R_drill = self._compute_drill_bit_pos_w()

            # Use cached Xform1 world pos (built once on first call)
            if not hasattr(self, "_xform1_pos_cache"):
                self._build_xform1_cache()
            xform1_pos = self._xform1_pos_cache   # [num_envs, 3]
            xform1_rot = self._xform1_rot_cache    # [num_envs, 3, 3]

            dist = torch.norm(bit_pos - xform1_pos, dim=1)
            # dual-scale: coarse term (tau=0.5m) gives guidance far away, fine term (tau=8cm) has a steep gradient near.
            # cap is still 20, but the far-away \"free\" baseline is halved, so the marginal gain of getting close is much larger
            r_proximity = 10.0 * torch.exp(-dist / 0.5) + 10.0 * torch.exp(-dist / 0.08)
            plate_x_axis = xform1_rot[:, :, 0]  # [N, 3]
            plate_y_axis = xform1_rot[:, :, 1]  # [N, 3]
            plate_z_axis = xform1_rot[:, :, 2]  # [N, 3]
            plate_neg_y = torch.nn.functional.normalize(-plate_y_axis, dim=1)  # [N, 3]
            if self.debug:
                print(f"[DEBUG plate axes @step={self._sim_step_counter}]  "
                      f"env_0: X=({plate_x_axis[0,0]:.3f},{plate_x_axis[0,1]:.3f},{plate_x_axis[0,2]:.3f})  "
                      f"Y=({plate_y_axis[0,0]:.3f},{plate_y_axis[0,1]:.3f},{plate_y_axis[0,2]:.3f})  "
                      f"Z=({plate_z_axis[0,0]:.3f},{plate_z_axis[0,1]:.3f},{plate_z_axis[0,2]:.3f})  "
                      f"-Y=({plate_neg_y[0,0]:.3f},{plate_neg_y[0,1]:.3f},{plate_neg_y[0,2]:.3f})")
            variant_ids = self._drill_variant_indices.long()  # [N]
            bit_dir_world = torch.zeros_like(R_drill[:, :, 0])  # [N, 3]

            for vid in self._variant_attrs.keys():
                mask = (variant_ids == vid).unsqueeze(-1)  # [N, 1]
                fwd = self._variant_attrs[vid].get("forward_axis", "Z")
                if fwd == "Z":
                    axis_vec = R_drill[:, :, 2]
                elif fwd == "-Z":
                    axis_vec = -R_drill[:, :, 2]
                elif fwd == "X":
                    axis_vec = R_drill[:, :, 0]
                elif fwd == "-X":
                    axis_vec = -R_drill[:, :, 0]
                elif fwd == "Y":
                    axis_vec = R_drill[:, :, 1]
                elif fwd == "-Y":
                    axis_vec = -R_drill[:, :, 1]
                else:
                    axis_vec = R_drill[:, :, 2]
                bit_dir_world = torch.where(mask, axis_vec, bit_dir_world)

            dot_zy = (bit_dir_world * plate_neg_y).sum(dim=1)   # [N]
            angle = torch.acos(torch.clamp(dot_zy, -1.0, 1.0))   # [N] in radians [0, pi/2]
            r_align = torch.exp(-angle / 0.2)  # exp decay: 1 at 0deg, ~0.04 at 90deg

            # Print axes for each env every 120 steps (non-debug mode)。
            # muted for stage1_only (ChainedEnv collects grasp segment only): alignment diagnostics are meaningless, just log noise.
            if not hasattr(self, "_axes_debug_counter"):
                self._axes_debug_counter = 0
            self._axes_debug_counter += 1
            if self._axes_debug_counter % 120 == 0 and not getattr(self, "stage1_only", False):
                for env_i in range(min(3, self.num_envs)):
                    vid = variant_ids[env_i].item()
                    vattr = self._variant_attrs.get(vid, {})
                    vname = vattr.get("name", f"vid{vid}")
                    fwd = vattr.get("forward_axis", "Z")

            gate = torch.exp(-dist / 0.5)  # same decay scale as r_proximity
            r_orientation = r_align * gate * 20  # scale up to compensate (was *5, now gate<=1)
            reward = r_proximity + r_orientation

            angle_deg = torch.rad2deg(angle)
            success_mask = (dist < 0.01) & (angle_deg < 5.0)
            # +30 every step inside the success region, continuous, non-terminating (encourages holding alignment to timeout)
            r_success = success_mask.float() * 30.0
            reward = reward + r_success

            # episode-level success flag: held ALIGN_SUCCESS_HOLD consecutive steps -> sticky set
            # (only for success detection / data filtering, not part of reward)
            self._align_hold_steps = torch.where(
                success_mask,
                self._align_hold_steps + 1,
                torch.zeros_like(self._align_hold_steps))
            self._align_achieved |= self._align_hold_steps >= self.ALIGN_SUCCESS_HOLD

            self._r_tip_drift = None
            mean_tip_dev = None
            tip_idx = self._get_fingertip_indices()
            if tip_idx is not None and hasattr(self, "_initial_tips_drill_local"):
                drill_pos = self.drill.data.root_pos_w
                tips_w = self.franka.data.body_pos_w[:, tip_idx, :]                # [N, F, 3]
                rel = tips_w - drill_pos.unsqueeze(1)                              # [N, F, 3]
                tips_local = torch.einsum('nij,nfj->nfi', R_drill.transpose(1, 2), rel)
                dev = torch.norm(tips_local - self._initial_tips_drill_local, dim=2)  # [N, F]
                mean_tip_dev = dev.clamp(max=0.2).mean(dim=1)                      # [N]
                self._r_tip_drift = mean_tip_dev * (-100.0)
                reward = reward + self._r_tip_drift

            # ── Penalty: drill-plate collision via contact sensor ─
            contact_force = None
            try:
                contact_force = self.scene["contact_plate"].data.net_forces_w  # [N, 1, 3]
                contact_mag = contact_force.squeeze(1).norm(dim=1)   # [N]
                self._r_plate_contact = torch.where(
                    contact_mag > 0.01,
                    torch.full_like(contact_mag, -10.0),
                    torch.zeros_like(contact_mag)
                )
                reward = reward + self._r_plate_contact
            except Exception as e:
                self._r_plate_contact = None

            if self.debug:
                _step = getattr(self, "_debug_step", 0)
                self._debug_step = _step + 1
                for i in range(min(3, self.num_envs)):
                    r_prox_raw = torch.exp(-dist[i] / 0.05).item()
                    r_orient_raw = r_align[i].item()
                    tip_dist_str = f"  tip_dev={mean_tip_dev[i].item():.4f}m" if mean_tip_dev is not None else ""
                    bit_dist_str = f"  bit_xform1_dist={dist[i].item():.4f}m"
                    contact_str = f"  plate_contact={self._r_plate_contact[i].item():.2f}" if getattr(self, "_r_plate_contact", None) is not None else ""
                    print(f"[DEBUG reward #{_step}] env_{i}  "
                          f"total={reward[i].item():.4f}  "
                          f"proximity={r_proximity[i].item():.4f}/{r_prox_raw:.4f}{bit_dist_str}  "
                          f"orientation={r_orientation[i].item():.4f}/{r_orient_raw:.4f} (dot={dot_zy[i].item():.4f}){tip_dist_str}{contact_str}")

            self._reward_components["proximity"] = r_proximity
            self._reward_components["orientation"] = r_orientation
            self._reward_components["success"] = r_success
            self._reward_components["tip_drift"] = getattr(self, "_r_tip_drift", None)
            self._reward_components["plate_contact"] = getattr(self, "_r_plate_contact", None)

            if "log" not in self.extras:
                self.extras["log"] = dict()
            for name, tensor in self._reward_components.items():
                if tensor is not None and tensor.numel() > 0:
                    self.extras["log"][f"reward_{name}"] = tensor.mean()
            # alignment success monitoring: instant aligned ratio + episode-level rolling success rate (sticky criterion)
            self.extras["log"]["align_rate_now"] = success_mask.float().mean()
            if getattr(self, "_recent_filled", 0) > 0:
                self.extras["log"]["success_rate_recent"] = \
                    self._recent_success[:self._recent_filled].float().mean().item()

            if self.debug:
                try:
                    self._draw_trigger_offset_viz()
                except Exception:
                    pass
                if not hasattr(self, "_grasp_debug_counter"):
                    self._grasp_debug_counter = 0
                self._grasp_debug_counter += 1
                if self._grasp_debug_counter % 30 == 0:
                    angles_deg = torch.rad2deg(torch.acos(torch.clamp(dot_zy, -1.0, 1.0)))
                    num_success = success_mask.sum().item()
                    print(f"[Stage2 reward] proximity={r_proximity.mean().item():.3f}  "
                          f"orient={r_orientation.mean().item():.3f}  "
                          f"success={r_success.mean().item():.3f} ({num_success}/{self.num_envs} envs)  "
                          f"dist={dist.mean().item():.3f}m  "
                          f"angle={angles_deg.mean().item():.1f}°")
                    # Per-variant angle summary every 30 steps
                    if self._grasp_debug_counter % 30 == 0:
                        variant_angle_sums = {}
                        for env_i in range(self.num_envs):
                            vid = self._drill_variant_indices[env_i].item()
                            vname = self._variant_attrs.get(vid, {}).get("name", f"vid{vid}")
                            if vid not in variant_angle_sums:
                                variant_angle_sums[vid] = {"name": vname, "angles": [], "dots": []}
                            variant_angle_sums[vid]["angles"].append(angles_deg[env_i].item())
                            variant_angle_sums[vid]["dots"].append(dot_zy[env_i].item())
                        # Per-variant drill root axes for verification
                        for vid in sorted(variant_angle_sums.keys()):
                            info = variant_angle_sums[vid]
                            avg_angle = sum(info["angles"]) / len(info["angles"])
                            avg_dot   = sum(info["dots"])   / len(info["dots"])
                            vattr = self._variant_attrs.get(vid, {})
                            fwd = vattr.get("forward_axis", "Z")
                            # Show which root axis is used
                            if fwd == "Z":
                                axis_label = "root +Z"
                            elif fwd == "-Z":
                                axis_label = "root -Z"
                            elif fwd == "X":
                                axis_label = "root +X"
                            elif fwd == "-X":
                                axis_label = "root -X"
                            elif fwd == "Y":
                                axis_label = "root +Y"
                            elif fwd == "-Y":
                                axis_label = "root -Y"
                            else:
                                axis_label = f"root {fwd}"
                            print(f"  [{info['name']}] fwd_axis={axis_label}  "
                                  f"avg_angle={avg_angle:.1f}°  avg_dot={avg_dot:.3f}")
                        # Per-env detail every 120 steps
                        if self._grasp_debug_counter % 120 == 0:
                            for env_i in range(min(3, self.num_envs)):
                                vid = self._drill_variant_indices[env_i].item()
                                vattr = self._variant_attrs.get(vid, {})
                                vname = vattr.get("name", f"vid{vid}")
                                fwd = vattr.get("forward_axis", "Z")
                                print(f"  env_{env_i} ({vname}, fwd={fwd}): "
                                      f"drill_root={R_drill[env_i, :, :].cpu().tolist()}  "
                                      f"plate_y_axis={plate_y_axis[env_i].cpu().tolist()}  "
                                      f"dot={dot_zy[env_i].item():.3f}  angle={angles_deg[env_i].item():.1f}°")

            return reward

        def _get_dones(self) -> tuple:
            time_out = (
                self.episode_length_buf >= self.max_episode_length_s / self.step_dt
            )

            # Drop detection: any hand-link to drill-body distance increased >5cm vs initial
            cur_dists = self._cached_link_body_dists  # [num_envs, num_obs_links]
            cur_max = cur_dists.min(dim=1).values       # [num_envs]
            init_max = self._initial_link_body_dists.min(dim=1).values  # [num_envs]
            dist_delta = cur_max - init_max
            drop = dist_delta > 0.05

            terminated = drop
            # early stop: aligned for success_hold_stop consecutive steps -> terminate and mark success,
            # instead of always running to time_out. Gated to a standalone Stage2Env instance only
            # (e.g. --stage2_only data collection): ChainedEnv subclasses Stage2Env and calls this
            # method directly (Stage2Env._get_dones(self)) purely to get the raw `drop` signal, then
            # layers its OWN independent success_hold_stop early-stop on top -- if this block ran
            # there too it would fold success_stop into the returned value under the same name
            # `drop`/`terminated` that ChainedEnv relies on being the pure physical-drop signal,
            # corrupting its `~drop` gating and its _cached_lenient_success bookkeeping.
            # _align_hold_steps is updated in _get_rewards (one step after dones); the 1-step lag is fine.
            if type(self) is Stage2Env and getattr(self, "success_hold_stop", 0) > 0:
                success_stop = ~drop & (self._align_hold_steps >= self.success_hold_stop)
                if success_stop.any():
                    self._align_achieved |= success_stop
                terminated = terminated | success_stop

            # episode-level success = was stably aligned (sticky) and did not drop this step.
            # written to the same interface as stage1; collection filtering / deploy stats / _reset_idx's
            # failure stats and recent-success-rate tracking all reuse it with no changes.
            self._cached_lenient_success = self._align_achieved & ~drop

            # if not hasattr(self, "_done_log_counter"):
            #     self._done_log_counter = 0
            # self._done_log_counter += 1
            # if self._done_log_counter % 30 == 0:
            #     print(f"[Stage2 done] timeout={time_out.float().mean().item()*100:.1f}%  "
            #           f"dropped={drop.float().mean().item()*100:.1f}%  "
            #           f"dist_delta_mean={dist_delta.mean().item()*100:.2f}cm  "
            #           f"dist_delta_max={dist_delta.max().item()*100:.2f}cm  "
            #           f"(link_dist increased >5cm)")

            return terminated, time_out


        def _draw_trigger_offset_viz(self):
            """Draw colored spheres at drill_bit_offset (world-space) for env_0..2.
            Matches rewards.py tip_trigger_reward exactly:
              bit_scaled = drill_bit_offset * scale
              bit_world  = R_drill @ bit_scaled + drill_pos

            Also reads Xform1 world position from _xform1_pos_cache (follows plate randomization).

            Colors: env_0=red, env_1=green, env_2=blue
            """
            try:
                import isaacsim.util.debug_draw._debug_draw as omni_debug_draw
                import torch.nn.functional as F

                if not hasattr(self, "_trigger_viz_counter"):
                    self._trigger_viz_counter = 0
                self._trigger_viz_counter += 1
                step = self._trigger_viz_counter

                draw_interface = omni_debug_draw.acquire_debug_draw_interface()

                # ── Per-env variant attributes via one-hot (same as _reset_idx) ──
                _VA = self._variant_attrs
                nv = self.total_num_variants
                variant_ids = self._drill_variant_indices  # [num_envs]
                vid_oh = F.one_hot(variant_ids.long(), num_classes=nv).float()

                _flat_bit_off = torch.zeros(nv * 3, device=self.device)
                for vid, vdata in _VA.items():
                    _flat_bit_off[vid*3:(vid+1)*3] = vdata["drill_bit_offset"]

                bit_offset_per_env = vid_oh @ _flat_bit_off.view(nv, 3)    # [num_envs, 3]

                # ── Drill root state ──
                drill_pos  = self.drill.data.root_pos_w      # [num_envs, 3]
                drill_quat = self.drill.data.root_quat_w     # [num_envs, 4] (w,x,y,z)

                # ── Scale (per-env) ──
                bit_offset_scaled = bit_offset_per_env * self._drill_scale  # [num_envs, 3]

                # ── R_drill: drill-local → world (matches rewards.py quat_to_rotmat) ──
                q = F.normalize(drill_quat, dim=1)
                w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
                N = q.shape[0]
                R_drill = torch.zeros((N, 3, 3), device=drill_quat.device, dtype=drill_quat.dtype)
                R_drill[:, 0, 0] = 1 - 2 * (y * y + z * z)
                R_drill[:, 0, 1] = 2 * (x * y - w * z)
                R_drill[:, 0, 2] = 2 * (x * z + w * y)
                R_drill[:, 1, 0] = 2 * (x * y + w * z)
                R_drill[:, 1, 1] = 1 - 2 * (x * x + z * z)
                R_drill[:, 1, 2] = 2 * (y * z - w * x)
                R_drill[:, 2, 0] = 2 * (x * z - w * y)
                R_drill[:, 2, 1] = 2 * (y * z + w * x)
                R_drill[:, 2, 2] = 1 - 2 * (x * x + y * y)

                # ── World position (same formula as rewards.py) ──
                bit_world = torch.bmm(R_drill, bit_offset_scaled.unsqueeze(-1)).squeeze(-1) + drill_pos

                # ── Resolve Xform1 world pos from cache (follows plate randomization) ──
                if not hasattr(self, "_xform1_pos_cache"):
                    self._build_xform1_cache()
                xform1_world_pos = self._xform1_pos_cache[:3].cpu().tolist()


                # ── Colors ──
                env_colors = [
                    [1.0, 0.15, 0.15, 1.0],   # red   — env_0
                    [0.15, 1.0, 0.15, 1.0],   # green — env_1
                    [0.15, 0.15, 1.0, 1.0],   # blue  — env_2
                ]
                white = [1.0, 1.0, 1.0, 1.0]

                line_src, line_tgt, line_colors, line_widths = [], [], [], []

                for env_idx in range(3):
                    wp = bit_world[env_idx].cpu().tolist()
                    xp = xform1_world_pos[env_idx]
                    col = env_colors[env_idx]
                    dist = ((wp[0]-xp[0])**2 + (wp[1]-xp[1])**2 + (wp[2]-xp[2])**2) ** 0.5

                    # bit sphere
                    draw_interface.draw_points([[wp[0], wp[1], wp[2]]], [col], [120.0])

                    # bit cross
                    r = 0.02
                    src = [[wp[0]-r,wp[1],wp[2]],[wp[0],wp[1]-r,wp[2]],[wp[0],wp[1],wp[2]-r]]
                    tgt = [[wp[0]+r,wp[1],wp[2]],[wp[0],wp[1]+r,wp[2]],[wp[0],wp[1],wp[2]+r]]
                    draw_interface.draw_lines(src, tgt, [col] * 3, [6.0] * 3)

                    # Xform1 cross (white)
                    draw_interface.draw_points([[xp[0], xp[1], xp[2]]], [white], [100.0])
                    xsrc = [[xp[0]-r,xp[1],xp[2]],[xp[0],xp[1]-r,xp[2]],[xp[0],xp[1],xp[2]-r]]
                    xtgt = [[xp[0]+r,xp[1],xp[2]],[xp[0],xp[1]+r,xp[2]],[xp[0],xp[1],xp[2]+r]]
                    draw_interface.draw_lines(xsrc, xtgt, [white] * 3, [6.0] * 3)

                    # bit → Xform1 line (distance reward proxy)
                    line_src.append([wp[0], wp[1], wp[2]])
                    line_tgt.append([xp[0], xp[1], xp[2]])
                    line_colors.append(col)
                    line_widths.append(4.0)

                # draw all bit→xform1 lines in one batch
                draw_interface.draw_lines(line_src, line_tgt, line_colors, line_widths)

            except Exception:
                pass  # silent in headless training

        def _add_trigger_offset_visualization(self):
            """DEPRECATED — kept for compat. Visualization is now done via draw_points in _draw_trigger_offset_viz."""
            pass

        def _add_plate_xform_visualization(self):
            """Discover all Xform prims inside the plate USD and attach coordinate-frame
            debug visuals to each one. Works regardless of how the plate is cloned /
            instantiated by the scene system."""
            try:
                from pxr import UsdGeom, Gf, Usd
                import omni.usd

                stage = omni.usd.get_context().get_stage()
                if stage is None:
                    print("[WARN] plate xform vis: no USD stage")
                    return

                frame_usd_path = f"{ISAAC_NUCLEUS_DIR}/frame_prim.usd"
                frame_scale = 0.01

                # After scene init, cfg.prim_path is expanded to a regex path like
                # "/World/envs/env_.*/Plate". Replace the regex token with "env_0"
                # to target the first environment's plate for visualisation.
                plate_cfg_path = self.plate.cfg.prim_path if self.plate is not None else ""
                if not plate_cfg_path:
                    print("[WARN] plate xform vis: plate not loaded")
                    return

                # Handle both raw template "{ENV_REGEX_NS}/Plate" and the resolved
                # regex form "/World/envs/env_.*/Plate"
                import re as _re
                plate_root_path = _re.sub(r"env_\.\*", "env_0",
                                         plate_cfg_path.replace("{ENV_REGEX_NS}", "env_0"))
                plate_prim = stage.GetPrimAtPath(plate_root_path)
                if not plate_prim.IsValid():
                    print(f"[WARN] plate xform vis: plate prim not found at {plate_root_path}")
                    return

                print(f"[DEBUG] Plate root prim: {plate_root_path}")

                # Traverse the plate USD hierarchy to find all Xform prims
                xform_prims = []
                for prim in Usd.PrimRange(plate_prim):
                    if prim.IsA(UsdGeom.Xform) and prim.GetName() not in ("Looks", "Render", "Materials"):
                        xform_prims.append(prim)

                print(f"[DEBUG] Found {len(xform_prims)} Xform prims under plate root")
                for xp in xform_prims:
                    print(f"       {xp.GetPath()}")

                for prim in xform_prims:
                    prim_path = str(prim.GetPath())
                    frame_path = f"{prim_path}/DebugFrame"

                    if stage.GetPrimAtPath(frame_path).IsValid():
                        continue

                    frame_xform = UsdGeom.Xform.Define(stage, frame_path)
                    frame_prim = frame_xform.GetPrim()
                    frame_prim.GetReferences().AddReference(frame_usd_path)

                    xformable = UsdGeom.Xformable(frame_prim)
                    xformable.ClearXformOpOrder()
                    xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
                        Gf.Vec3d(0.0, 0.0, 0.0)
                    )
                    xformable.AddScaleOp(UsdGeom.XformOp.PrecisionFloat).Set(
                        Gf.Vec3f(frame_scale, frame_scale, frame_scale)
                    )

                    world_xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                        Usd.TimeCode.Default()
                    )
                    t = world_xf.ExtractTranslation()
                    print(f"[Stage2] PlateXform '{prim.GetName()}' @ {prim_path}  "
                          f"world=({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f})")

            except Exception as e:
                pass  # silent in headless training

        def _add_drill_bit_frame_visualization(self):
            """Add a persistent coordinate frame at the drill_bit point for each variant.

            The bit position is computed as: bit_world = R_drill @ (bit_offset * scale) + drill_pos
            We create a static Xform prim parented under each drill's root prim.
            """
            try:
                from pxr import UsdGeom, Gf, Usd, Sdf
                import omni.usd

                stage = omni.usd.get_context().get_stage()
                if stage is None:
                    print("[WARN] drill_bit frame vis: no USD stage")
                    return

                frame_usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd"
                frame_scale = 0.05

                drill_cfg_path = self.drill.cfg.prim_path if hasattr(self, "drill") and self.drill is not None else ""
                if not drill_cfg_path:
                    print("[WARN] drill_bit frame vis: drill not loaded")
                    return

                import re as _re
                drill_root_path = _re.sub(r"env_\.\*", "env_0",
                                         drill_cfg_path.replace("{ENV_REGEX_NS}", "env_0"))
                drill_prim = stage.GetPrimAtPath(drill_root_path)
                if not drill_prim.IsValid():
                    print(f"[WARN] drill_bit frame vis: drill prim not found at {drill_root_path}")
                    return

                # Compute drill_bit_offset from variant attrs (use variant 0 as reference)
                bit_offset = torch.tensor(
                    self._variant_attrs[0]["drill_bit_offset"],
                    dtype=torch.float32, device=self.device
                )
                scale = getattr(self, "_drill_scale", torch.ones(self.num_envs, device=self.device))
                bit_offset_scaled = bit_offset * scale[0]  # scale[0] is [3] tensor for env_0

                # Convert to Gf.Vec3d for USD
                bit_offset_usd = Gf.Vec3d(bit_offset_scaled[0].item(), bit_offset_scaled[1].item(), bit_offset_scaled[2].item())

                # Create a static Xform prim as child of the drill root
                bit_frame_path = f"{drill_root_path}/DrillBitFrame"
                bit_frame_prim = stage.GetPrimAtPath(bit_frame_path)
                if not bit_frame_prim.IsValid():
                    xf_def = UsdGeom.Xform.Define(stage, bit_frame_path)
                    bit_frame_prim = xf_def.GetPrim()
                    bit_frame_prim.GetReferences().AddReference(frame_usd_path)

                    # Set translation to bit_offset
                    xformable = UsdGeom.Xformable(bit_frame_prim)
                    xformable.ClearXformOpOrder()
                    translate_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
                    translate_op.Set(bit_offset_usd)

                    # Set scale
                    scale_op = xformable.AddScaleOp(UsdGeom.XformOp.PrecisionFloat)
                    scale_op.Set(Gf.Vec3f(frame_scale, frame_scale, frame_scale))

                    print(f"[Stage2] DrillBitFrame added at {bit_frame_path}  "
                          f"offset=({bit_offset_usd[0]:.4f}, {bit_offset_usd[1]:.4f}, {bit_offset_usd[2]:.4f})")
                else:
                    print(f"[DEBUG] DrillBitFrame already exists at {bit_frame_path}")

            except Exception as e:
                pass  # silent in headless training

        def _setup_plate_contact_physics(self):
            """Add contact report API to plate Mesh(es) rigid body for contact sensor detection (all envs)."""
            try:
                from pxr import PhysxSchema
                import omni.usd
                import re

                stage = omni.usd.get_context().get_stage()
                if stage is None:
                    return

                # Target: /World/envs/env_*/Plate/Meshes (the rigid body prim, parent of Cube__0)
                pattern = r"^/World/envs/env_\d+/Plate/Meshes$"
                
                count = 0
                for prim in stage.TraverseAll():
                    prim_path = prim.GetPath().pathString
                    if re.match(pattern, prim_path):
                        count += 1
                        # Add ContactReportAPI for contact sensor
                        if not prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                            PhysxSchema.PhysxContactReportAPI.Apply(prim)
                
                print(f"[Stage2] Plate contact physics setup: processed {count} Plate/Meshes prims")

            except Exception:
                import traceback
                traceback.print_exc()

    return Stage2Env


def create_stage2_env_cfg(
    num_envs: int = 256,
    device: str = "cuda:0",
    headless: bool = False,
    debug: bool = False,
    target_pos=None,
    target_quat=None,
    drill_variants_path: str = None,

):
    """
    Build a GraspDrillEnvCfg (scene + observations from GraspDrillEnv).
    Stage2Env inherits it and overrides reward/done/reset only.
    """
    from tasks.grasp_drill_env import create_grasp_drill_env_cfg

    cfg = create_grasp_drill_env_cfg(
        num_envs=num_envs,
        device=device,
        headless=headless,
        debug=debug,
        drill_variants_path=drill_variants_path,
        # stage2 is pure RL (state observation), no camera needed; default True would spawn a TiledCamera,
        # an App without --enable_cameras crashes at sim.reset(), and enabling rendering is very VRAM-heavy.
        enable_cameras=False,
        # plate + contact_plate sensor registration moved up to create_grasp_drill_env_cfg (globally shared)
        include_plate=True,
    )

    if target_pos is not None:
        cfg.target_pos = tuple(target_pos)
    if target_quat is not None:
        cfg.target_quat = tuple(target_quat)

    return cfg
