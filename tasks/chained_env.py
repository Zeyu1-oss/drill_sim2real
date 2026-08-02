
import torch


def get_chained_env_class():
    """Must be called AFTER AppLauncher is running."""
    from tasks.grasp_drill_env import GraspDrillEnv
    from tasks.stage2_env import get_stage2_env_class
    Stage2Env = get_stage2_env_class()

    class ChainedEnv(Stage2Env):

        def __init__(self, cfg, success_hold_stop: int = 0, stage1_only: bool = False,
                     **kwargs):
            kwargs.pop("success_dataset", None)   # chained mode does not use the pkl
            super().__init__(cfg, success_dataset=None, **kwargs)
            # >0: after holding alignment this many steps, terminate early and mark success;
            #     0: run to timeout (matches stage2 training)
            self.success_hold_stop = int(success_hold_stop)
            # True: collect grasp segment only -- terminate and mark success once grasped (the
            # moment stage2 would switch), phase stays 0. Scene (plate/cam3/randomization) is
            # identical to chained, same data layout, mixable with full chained data.
            self.stage1_only = bool(stage1_only)
            self._init_phase_buffers()

        def _init_phase_buffers(self):
            if hasattr(self, "phase"):
                return
            # 0 = stage1 (grasp), 1 = stage2 (align); sticky until reset
            self.phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            # step at which this episode switched (-1 = not yet)
            self._switch_step_buf = torch.full(
                (self.num_envs,), -1, dtype=torch.long, device=self.device)
            # snapshot of last (already reset) episode's switch step; play scripts read it after done
            self._last_episode_switch_step = torch.full(
                (self.num_envs,), -1, dtype=torch.long, device=self.device)

        # ------------------------------------------------------------
        def _reset_idx(self, env_ids: torch.Tensor):
            self._init_phase_buffers()   # first reset may precede the end of __init__
            self._last_episode_switch_step[env_ids] = self._switch_step_buf[env_ids]
            self.phase[env_ids] = 0
            self._switch_step_buf[env_ids] = -1
            if hasattr(self, "_align_hold_steps"):
                self._align_hold_steps[env_ids] = 0
                self._align_achieved[env_ids] = False

            # full stage1 reset: drill initial_pos_list random + jitter, hand back to init pose,
            # thumb yaw preset. Bypasses Stage2Env._reset_idx's pkl / fixed-pose logic.
            GraspDrillEnv._reset_idx(self, env_ids)

            self._randomize_plate(env_ids)

            # placeholder grasp reference (current state): ensures stage2 done/reward buffers exist.
            # phase 0 never uses its result to terminate; re-captured with the real grasp on switch.
            self._capture_grasp_reference(env_ids)

        # ------------------------------------------------------------
        def _capture_grasp_reference(self, env_ids: torch.Tensor):
            """Rebuild stage2 drop-detection / grasp-drift references from the current sim state."""
            link_dists = self._compute_link_body_dists()  # [num_envs, L]
            if not hasattr(self, "_initial_link_body_dists") or \
                    self._initial_link_body_dists.shape[0] != self.num_envs:
                self._initial_link_body_dists = torch.zeros_like(link_dists)
            self._initial_link_body_dists[env_ids] = link_dists[env_ids].clone()

            drill_pos = self.drill.data.root_pos_w[env_ids]      # [n, 3]
            drill_quat = self.drill.data.root_quat_w[env_ids]    # [n, 4] wxyz
            R = self._quat_to_rot_matrix(drill_quat)              # [n, 3, 3]

            bit_offset = self._trigger1_offset[env_ids]           # [n, 3] drill-local
            bit_pos = drill_pos + (R @ bit_offset.unsqueeze(-1)).squeeze(-1)
            if not hasattr(self, "_initial_bit_pos") or \
                    self._initial_bit_pos.shape[0] != self.num_envs:
                self._initial_bit_pos = torch.zeros(self.num_envs, 3, device=self.device)
            self._initial_bit_pos[env_ids] = bit_pos

            tip_idx = self._get_fingertip_indices()
            if tip_idx is not None:
                tips_w = self.franka.data.body_pos_w[env_ids][:, tip_idx, :]   # [n, F, 3]
                rel = tips_w - drill_pos.unsqueeze(1)                          # [n, F, 3]
                tips_local = torch.einsum('nij,nfj->nfi', R.transpose(1, 2), rel)
                if not hasattr(self, "_initial_tips_drill_local") or \
                        self._initial_tips_drill_local.shape[0] != self.num_envs:
                    self._initial_tips_drill_local = torch.zeros(
                        self.num_envs, tips_local.shape[1], 3, device=self.device)
                self._initial_tips_drill_local[env_ids] = tips_local

        # ------------------------------------------------------------
        def _switch_to_stage2(self, env_ids: torch.Tensor):
            self.phase[env_ids] = 1
            self._switch_step_buf[env_ids] = self.episode_length_buf[env_ids]
            self._align_hold_steps[env_ids] = 0
            self._align_achieved[env_ids] = False
            self._capture_grasp_reference(env_ids)

        # ------------------------------------------------------------
        def _get_dones(self) -> tuple:
            # stage1 check: refresh _success_window / _pending_nan_mask / failure cache;
            # _cached_lenient_success is now stage1 semantics (>=20 grasped steps in window)
            term1, trunc1 = GraspDrillEnv._get_dones(self)
            stage1_success = self._cached_lenient_success

            if self.stage1_only:
                # grasp-only collection: terminate on grasp (same moment as stage2 switch, so the
                # behavior distribution matches); no phase switch, no stage2 check;
                # success = stage1 lenient success and no NaN.
                self._cached_lenient_success = stage1_success & ~self._pending_nan_mask
                return term1 | stage1_success, trunc1

            # phase 0->1 switch criterion: the same 20-in-last-50 sliding-window lenient success
            # (`stage1_success` above) used by --collect_success_data / stage1_only, so the switch
            # fires at the same point that pkl-based stage2 hand-off states are captured from.
            in_stage1 = self.phase == 0
            newly = in_stage1 & stage1_success
            if newly.any():
                self._switch_to_stage2(newly.nonzero(as_tuple=True)[0])

            # stage2 check (drop); overwrites _cached_lenient_success with
            # align_achieved & ~drop (stage2 semantics)
            drop, _ = Stage2Env._get_dones(self)

            in_stage2 = self.phase == 1
            terminated_s2 = drop | self._pending_nan_mask

            # early stop: aligned for success_hold_stop consecutive steps -> terminate and mark success.
            # _align_hold_steps is updated in _get_rewards (one step after dones); the 1-step lag is fine.
            if getattr(self, "success_hold_stop", 0) > 0:
                success_stop = in_stage2 & ~drop & \
                    (self._align_hold_steps >= self.success_hold_stop)
                if success_stop.any():
                    self._align_achieved |= success_stop
                    # recompute same as Stage2Env._get_dones so early-stopped envs are marked success
                    self._cached_lenient_success = self._align_achieved & ~drop
                terminated_s2 = terminated_s2 | success_stop

            # phase0 (not yet switched) envs must NOT terminate via the base 20-in-50 window
            # (folded into term1 as `lenient_success`) -- that is now exactly the switch criterion
            # above, and should move the env into phase 1 rather than end the episode.
            # Only genuine failure/nan/proximal-limit conditions end a still-phase0 episode.
            term1_phase0 = self._cached_hard_fail_mask
            terminated = torch.where(in_stage2, terminated_s2, term1_phase0)
            # chained success = entered stage2 and aligned and did not drop; physical-NaN episodes never count
            # (the sticky align flag may have been set before the NaN, which could otherwise store a bad episode as success)
            self._cached_lenient_success = (
                self._cached_lenient_success & in_stage2 & ~self._pending_nan_mask)

            return terminated, trunc1

        # ------------------------------------------------------------
        def _get_rewards(self) -> torch.Tensor:
            reward = Stage2Env._get_rewards(self)
            # do not accumulate alignment during phase 0 (unreachable geometrically; defensive zeroing)
            in_stage2 = self.phase == 1
            self._align_hold_steps *= in_stage2.long()
            self._align_achieved &= in_stage2
            return reward

    return ChainedEnv
