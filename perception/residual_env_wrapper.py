"""Wraps an Isaac Lab grasp env + a frozen DP3 policy into a single reset()/step() interface,
so a standard off-policy RL trainer (TD3) can train a small residual correction on top of DP3
without knowing anything about point clouds, camera crops, or diffusion inference.

Mirrors the pattern in Amazon's ResFiT (residual-offpolicy-rl) BasePolicyVecEnvWrapper: every
step, get the frozen base policy's (DP3's) proposed action for the CURRENT observation, add the
RL residual, step the real env with the combined action, then compute the NEXT observation and
the next base action for the following step. DP3 replans every single env step (no chunk
execution) so the whole thing is a clean single-step MDP, matching what TD3 (and ResFiT's own
wrapper) expects.

DP3 (encoder + diffusion head) must already be frozen (requires_grad_(False), .eval()) by the
caller before constructing this wrapper -- this class never updates it.
"""
from collections import deque

import torch


class ResidualDP3EnvWrapper:
    """
    obs returned by reset()/step() is a dict:
        {"feat": (B, C) frozen DP3Encoder features for the CURRENT single frame,
         "dp3_action": (B, 13) DP3's own proposed action for the CURRENT step}
    Residual RL trains on ("feat" concatenated with "dp3_action" as the actor/critic input);
    the actual action taken by the environment is dp3_action + residual (computed by the caller
    and passed into step()).
    """

    def __init__(
        self,
        env_unwrapped,
        dp3_policy,          # frozen DP3 (ema) policy: .obs_encoder, .normalizer, .predict_action, .use_pc_color
        cam1, cam2, cam3,
        cam2_pose_fn,        # () -> (pos_w, quat_w) or None; wrist-cam FK pose getter
        build_agent_pos_fn,  # (env_unwrapped, with_force) -> (B,13 or 26)
        camera_crop_bounds_fn, camera_pc_fn, raise_z_floor_fn,
        add_mask_channel_fn, append_robot_drill_fn,
        perc, workspace, pc_num_points, disable_cam2, with_force, chained,
        n_obs_steps, device, num_envs, dt,
    ):
        self.env = env_unwrapped
        self.dp3 = dp3_policy
        self.cam1, self.cam2, self.cam3 = cam1, cam2, cam3
        self._cam2_pose_fn = cam2_pose_fn
        self._build_agent_pos = build_agent_pos_fn
        self._camera_crop_bounds = camera_crop_bounds_fn
        self._camera_pc = camera_pc_fn
        self._raise_z_floor = raise_z_floor_fn
        self._add_mask_channel = add_mask_channel_fn
        self._append_robot_drill = append_robot_drill_fn
        self.perc = perc
        self.workspace = workspace
        self.pc_num_points = pc_num_points
        self.disable_cam2 = disable_cam2
        self.with_force = with_force
        self.chained = chained
        self.n_obs = n_obs_steps
        self.device = device
        self.num_envs = num_envs
        self.dt = dt
        self.sim_pc_per_cam = pc_num_points if disable_cam2 else pc_num_points // 2

        # per-env (agent_pos, point_cloud) history, length n_obs, for DP3's predict_action
        self._hist = [deque(maxlen=self.n_obs) for _ in range(num_envs)]
        self._last_pc1 = None
        self._last_pc2 = None

    # ------------------------------------------------------------------
    def _build_pc_and_state(self):
        """One frame of (point_cloud, agent_pos), same construction as deploy_dp3_sim.py's main loop."""
        env_origins = self.env.scene.env_origins
        ws = self._camera_crop_bounds(self.env.drill.data.root_pos_w, env_origins, self.perc, self.workspace)
        pc1, zmask1, _ = self._camera_pc(self.cam1, ws, self.sim_pc_per_cam, env_origins)
        if not self.disable_cam2:
            ws_cam2 = self._raise_z_floor(ws, getattr(self.perc, "wrist_cam_z_floor", None))
            pc2, zmask2, _ = self._camera_pc(self.cam2, ws_cam2, self.sim_pc_per_cam, env_origins,
                                             pose_w=self._cam2_pose_fn())
        # reuse-last-good-frame substitution only applies in non-chained mode, exactly matching
        # deploy_dp3_sim.py's `if not args.chained:` gating -- with --stage1_only (chained=True,
        # what training actually uses) DP3 was trained WITHOUT this substitution, so applying it
        # here would feed DP3 an input distribution it never saw during training.
        if not self.chained and self._last_pc1 is not None:
            pc1 = torch.where(zmask1.view(-1, 1, 1), self._last_pc1, pc1)
        self._last_pc1 = pc1
        if not self.disable_cam2:
            if not self.chained and self._last_pc2 is not None:
                pc2 = torch.where(zmask2.view(-1, 1, 1), self._last_pc2, pc2)
            self._last_pc2 = pc2

        pcf = torch.zeros(self.num_envs, self.pc_num_points, 3, device=self.device, dtype=torch.float32)
        pcf[:, :self.sim_pc_per_cam] = pc1
        if not self.disable_cam2:
            pcf[:, self.sim_pc_per_cam:] = pc2

        pc_full = self._add_mask_channel(self._append_robot_drill(pcf))   # (B, total_pc, 3 or 4)
        agent_pos = self._build_agent_pos(self.env, self.with_force)      # (B, 13 or 26)
        return pc_full, agent_pos

    def _encode_current_frame(self, agent_pos, point_cloud):
        """Frozen DP3Encoder features for a SINGLE current frame (no time dim) -- separate,
        cheap forward pass; does not touch predict_action's own internal encoder call."""
        with torch.no_grad():
            nobs = self.dp3.normalizer.normalize({"agent_pos": agent_pos, "point_cloud": point_cloud})
            if not self.dp3.use_pc_color:
                nobs["point_cloud"] = nobs["point_cloud"][..., :3]
            feat = self.dp3.obs_encoder(nobs)     # (B, C)
        return feat

    def _dp3_action_for_current_step(self, env_id_list=None):
        """Frozen DP3 replans from the current n_obs-length history -> (B,13) action for THIS step."""
        obs_batch, pc_batch = [], []
        ids = range(self.num_envs) if env_id_list is None else env_id_list
        for e in ids:
            hist = list(self._hist[e])
            if len(hist) < self.n_obs:
                hist = [hist[0]] * (self.n_obs - len(hist)) + hist
            obs_batch.append(torch.stack([s["agent_pos"] for s in hist]))
            pc_batch.append(torch.stack([s["point_cloud"] for s in hist]))
        obs_dict = {"agent_pos": torch.stack(obs_batch), "point_cloud": torch.stack(pc_batch)}
        with torch.no_grad():
            result = self.dp3.predict_action(obs_dict)
        _cs = self.n_obs - 1   # lag_comp=0 convention, same as deploy_dp3_sim.py default
        return result["action_pred"][:, _cs]   # (B,13)

    def _push_history(self, agent_pos, point_cloud, env_ids=None):
        ids = range(self.num_envs) if env_ids is None else env_ids
        for e in ids:
            self._hist[e].append({"agent_pos": agent_pos[e], "point_cloud": point_cloud[e]})

    # ------------------------------------------------------------------
    def reset(self):
        self.env.reset()
        for h in self._hist:
            h.clear()
        self._last_pc1 = None
        self._last_pc2 = None
        pc_full, agent_pos = self._build_pc_and_state()
        for _ in range(self.n_obs):
            self._push_history(agent_pos, pc_full)
        feat = self._encode_current_frame(agent_pos, pc_full)
        dp3_action = self._dp3_action_for_current_step()
        self._cur_dp3_action = dp3_action
        return {"feat": feat, "dp3_action": dp3_action}

    def step(self, residual_action):
        """residual_action: (B,13), already scaled by the actor (tanh*action_scale). Combines with
        the CURRENT dp3_action (from the previous reset()/step() call) and steps the real env."""
        combined_action = self._cur_dp3_action + residual_action
        self.env._direct_target = combined_action
        obs_dict, rewards, terminated, truncated, extras = self.env.step(combined_action)

        self.cam1.update(self.dt)
        self.cam2.update(self.dt)
        if self.cam3 is not None:
            self.cam3.update(self.dt)

        pc_full, agent_pos = self._build_pc_and_state()
        done = (terminated.bool() | truncated.bool())
        finished = torch.where(done)[0].tolist()
        for e in finished:
            self._hist[e].clear()
        self._push_history(agent_pos, pc_full)
        for e in finished:
            for _ in range(self.n_obs - 1):
                self._push_history(agent_pos, pc_full, env_ids=[e])

        feat = self._encode_current_frame(agent_pos, pc_full)
        dp3_action = self._dp3_action_for_current_step()
        self._cur_dp3_action = dp3_action

        obs = {"feat": feat, "dp3_action": dp3_action}
        return obs, rewards, terminated, truncated, extras
