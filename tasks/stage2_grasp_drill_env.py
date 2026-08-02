"""
Stage2 grasp-manipulate drill environment

Differences from grasp_drill_env.py:
1. initial pose loaded from a dataset (instead of random)
2. adds dedicated rewards: trigger-press reward, tilt penalty, etc.
3. adds manipulation-related termination conditions
4. reserves an operating board (operating_board)
"""

import torch
import numpy as np
from typing import Tuple

from .grasp_drill_env import (
    GraspDrillEnv,
    GraspDrillEnvCfg,
    GraspDrillSceneCfg,
    create_grasp_drill_env_cfg as _create_grasp_drill_env_cfg,
)

# =============================================================================
# Stage2-specific Scene config: must define a subclass to add fields
# =============================================================================
try:
    from isaaclab.utils import configclass
    from isaaclab.assets import AssetBaseCfg
except ImportError:
    from omni.isaac.lab.utils import configclass
    from omni.isaac.lab.assets import AssetBaseCfg


@configclass
class Stage2SceneCfg(GraspDrillSceneCfg):
    """Stage2-specific scene: pre-declare operating_board"""
    operating_board: AssetBaseCfg = MISSING


# =============================================================================
# env config
# =============================================================================

def create_stage2_env_cfg(
    num_envs: int = 256,
    device: str = "cuda:0",
    headless: bool = False,
    hyperparameters=None,
    drill_config_path: str = None,
    debug: bool = False,
    dataset_path: str = None,
) -> GraspDrillEnvCfg:
    """Create the Stage2 env config"""
    cfg = _create_grasp_drill_env_cfg(
        num_envs=num_envs,
        device=device,
        headless=headless,
        hyperparameters=hyperparameters,
        drill_config_path=drill_config_path,
        debug=debug,
    )
    cfg.dataset_path = dataset_path

    # operating board config
    try:
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.sim import CuboidCfg, PreviewSurfaceCfg, CollisionPropertiesCfg, RigidBodyPropertiesCfg, MassPropertiesCfg
        import isaaclab.sim as sim_utils
    except ImportError:
        from omni.isaac.lab.assets import AssetBaseCfg
        from omni.isaac.lab.sim import CuboidCfg, PreviewSurfaceCfg, CollisionPropertiesCfg, RigidBodyPropertiesCfg, MassPropertiesCfg
        import omni.isaac.lab.sim as sim_utils

    _BOARD_USD = GraspDrillStage2Env.OPERATING_BOARD_USD
    _BOARD_POS = GraspDrillStage2Env.OPERATING_BOARD_POS
    _BOARD_SCALE = GraspDrillStage2Env.OPERATING_BOARD_SCALE

    if _BOARD_USD:
        board_spawn = sim_utils.UsdFileCfg(usd_path=_BOARD_USD, scale=_BOARD_SCALE)
    else:
        board_spawn = CuboidCfg(
            size=(0.3, 0.3, 0.02),
            visual_material=PreviewSurfaceCfg(diffuse_color=(0.6, 0.6, 0.7), metallic=0.5, roughness=0.3),
            collision_props=CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.005, rest_offset=0.001),
            rigid_props=RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=True),
            mass_props=MassPropertiesCfg(mass=0.0),
        )

    operating_board_cfg = AssetBaseCfg(
        prim_path="/World/OperatingBoard",
        spawn=board_spawn,
        init_state=AssetBaseCfg.InitialStateCfg(pos=_BOARD_POS),
    )

    # replace scene with Stage2SceneCfg (with operating_board)
    cfg.scene = Stage2SceneCfg(
        num_envs=num_envs,
        env_spacing=cfg.scene.env_spacing,
        replicate_physics=cfg.scene.replicate_physics,
        clone_in_fabric=cfg.scene.clone_in_fabric,
        drill=cfg.scene.drill,
        franka=cfg.scene.franka,
        table=cfg.scene.table,
        operating_board=operating_board_cfg,
    )

    return cfg


# =============================================================================
# env class
# =============================================================================

class GraspDrillStage2Env(GraspDrillEnv):
    """Stage2 env: load grasp-pose data, focus on the manipulation task"""

    # === operating board config (fill in once provided) ===
    # TODO: fill in the actual USD path and coordinates
    OPERATING_BOARD_USD = None   # "path/to/operating_board.usd"
    OPERATING_BOARD_POS = (0.0, 0.0, 0.0)  # (x, y, z) world coordinates
    OPERATING_BOARD_SCALE = (1.0, 1.0, 1.0)

    def __init__(self, cfg: GraspDrillEnvCfg, render_mode: str = None, debug: bool = False, **kwargs):
        self._stage2_dataset = None
        self._stage2_dataset_path = getattr(cfg, 'dataset_path', None)
        super().__init__(cfg, render_mode=render_mode, debug=debug, **kwargs)

        # operating board reference
        self.operating_board = self.scene["operating_board"] if "operating_board" in self.scene else None

        # === Stage2 state ===
        self._trigger_pressed_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._trigger_press_threshold = 15  # 15 consecutive press steps count as a successful press

        self._drill_orientation_buffer_size = 20
        self._drill_orientation_buffer = torch.zeros(
            (self.num_envs, self._drill_orientation_buffer_size),
            dtype=torch.float32,
            device=self.device,
        )
        self._drill_orientation_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    # -------------------------------------------------------------------------
    # fix 1: override _get_rewards and _get_dones (do not rewrite step())
    # -------------------------------------------------------------------------
    def _get_rewards(self):
        return self._get_stage2_rewards()

    def _get_dones(self):
        return self._get_stage2_dones()

    # -------------------------------------------------------------------------
    # fix 3: _reset_idx order
    # super()._reset_idx() already wrote the drill/franka default state (including env_origins)
    # dataset poses are world coordinates, overwrite directly, do not add env_origins
    # joint_pos dim match: the dataset stores franka.data.joint_pos (all joint DOFs)
    # -------------------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)  # write default state first

        if self._stage2_dataset is None and self._stage2_dataset_path:
            self.load_dataset(self._stage2_dataset_path)

        if self._stage2_dataset is not None and len(self._stage2_dataset) > 0:
            idx = np.random.randint(0, len(self._stage2_dataset))
            pose = self._stage2_dataset[idx]

            drill_pos = torch.as_tensor(pose["drill_pos"], dtype=torch.float32, device=self.device)
            drill_quat = torch.as_tensor(pose["drill_quat"], dtype=torch.float32, device=self.device)
            drill_lin_vel = torch.as_tensor(pose.get("drill_lin_vel"), dtype=torch.float32, device=self.device) \
                if pose.get("drill_lin_vel") is not None else None
            drill_ang_vel = torch.as_tensor(pose.get("drill_ang_vel"), dtype=torch.float32, device=self.device) \
                if pose.get("drill_ang_vel") is not None else None
            joint_pos = torch.as_tensor(pose["joint_pos"], dtype=torch.float32, device=self.device)
        else:
            drill_pos = None
            drill_quat = None
            drill_lin_vel = None
            drill_ang_vel = None
            joint_pos = None

        n = len(env_ids)

        # --- override drill state (world coordinates) ---
        if drill_pos is not None:
            drill_root_state = self.drill.data.default_root_state[env_ids].clone()
            drill_root_state[:, 0:3] = drill_pos.unsqueeze(0).expand(n, -1)
            drill_root_state[:, 3:7] = drill_quat.unsqueeze(0).expand(n, -1)
            drill_root_state[:, 7:10] = drill_lin_vel.unsqueeze(0).expand(n, -1) if drill_lin_vel is not None else 0.0
            drill_root_state[:, 10:13] = drill_ang_vel.unsqueeze(0).expand(n, -1) if drill_ang_vel is not None else 0.0
            self.drill.write_root_pose_to_sim(drill_root_state[:, :7], env_ids=env_ids)
            self.drill.write_root_velocity_to_sim(drill_root_state[:, 7:], env_ids=env_ids)
            self.initial_drill_pos[env_ids] = drill_root_state[:, :3].clone()
            self.drill_initial_rot_tensor[env_ids] = drill_root_state[:, 3:7].clone()

        # --- override arm joint state (dataset stores all joints, dims match) ---
        if joint_pos is not None:
            franka_joint_pos = self.franka.data.joint_pos.clone()
            franka_joint_vel = self.franka.data.joint_vel.clone()
            franka_joint_pos[env_ids] = joint_pos.unsqueeze(0).expand(n, -1)
            franka_joint_vel[env_ids] = 0.0
            self.franka.write_joint_state_to_sim(franka_joint_pos, franka_joint_vel, env_ids=env_ids)
            if hasattr(self, 'cur_targets'):
                self.cur_targets[env_ids] = joint_pos.unsqueeze(0).expand(n, -1)[:, self.controlled_joint_indices]
                self.prev_targets[env_ids] = joint_pos.unsqueeze(0).expand(n, -1)[:, self.controlled_joint_indices]

        # --- reset Stage2 state ---
        self._trigger_pressed_steps[env_ids] = 0
        self._drill_orientation_buffer[env_ids] = 0.0
        self._drill_orientation_idx[env_ids] = 0

    # -------------------------------------------------------------------------
    # the logic below is unchanged
    # -------------------------------------------------------------------------
    def _check_grasp_success(self) -> torch.Tensor:
        return self._check_success()

    def load_dataset(self, dataset_path: str):
        """Load the pose dataset"""
        import pickle
        with open(dataset_path, 'rb') as f:
            self._stage2_dataset = pickle.load(f)
        print(f"[Stage2] loaded dataset: {dataset_path}, containing {len(self._stage2_dataset)} samples")

    def _check_trigger_pressed(self) -> torch.Tensor:
        """Check whether the trigger is pressed: R_index_intermediate within < 2cm of the trigger"""
        drill_pos = self.drill.data.root_pos_w
        drill_quat = self.drill.data.root_quat_w
        body_names = self.franka.data.body_names

        if "R_index_intermediate" not in body_names:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        index_idx = body_names.index("R_index_intermediate")
        index_pos = self.franka.data.body_pos_w[:, index_idx, :]

        trigger_offset = self._trigger1_offset.to(self.device)
        scale_tensor = self._drill_scale  # [num_envs, 3]
        trigger_offset = trigger_offset * scale_tensor  # broadcast

        q = drill_quat
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        norm = torch.sqrt(w*w + x*x + y*y + z*z + 1e-8)
        w, x, y, z = w/norm, x/norm, y/norm, z/norm

        R = torch.zeros((self.num_envs, 3, 3), device=self.device)
        R[:, 0, 0] = 1 - 2*(y*y + z*z)
        R[:, 0, 1] = 2*(x*y - w*z)
        R[:, 0, 2] = 2*(x*z + w*y)
        R[:, 1, 0] = 2*(x*y + w*z)
        R[:, 1, 1] = 1 - 2*(x*x + z*z)
        R[:, 1, 2] = 2*(y*z - w*x)
        R[:, 2, 0] = 2*(x*z - w*y)
        R[:, 2, 1] = 2*(y*z + w*x)
        R[:, 2, 2] = 1 - 2*(x*x + y*y)

        trigger_world = torch.bmm(
            R, trigger_offset.unsqueeze(0).unsqueeze(-1).expand(self.num_envs, -1, -1)
        ).squeeze(-1) + drill_pos

        dist = torch.norm(index_pos - trigger_world, dim=1)
        return dist < 0.02

    def _get_drill_uprightness(self) -> torch.Tensor:
        """Compute how upward the drill points: cosine of the angle between its Y axis and world Z axis"""
        drill_quat = self.drill.data.root_quat_w
        qw, qx, qy, qz = drill_quat[:, 0], drill_quat[:, 1], drill_quat[:, 2], drill_quat[:, 3]
        upright = 2.0 * (qy * qz + qw * qx)
        return torch.clamp(upright, -1.0, 1.0)

    def _update_orientation_buffer(self):
        """Update the sliding window of drill orientation"""
        upright = self._get_drill_uprightness()
        idx = self._drill_orientation_idx.long()
        self._drill_orientation_buffer.scatter_(1, idx.unsqueeze(1), upright.unsqueeze(1))
        self._drill_orientation_idx = (self._drill_orientation_idx + 1) % self._drill_orientation_buffer_size

    # -------------------------------------------------------------------------
    # fix 3: _get_stage2_rewards writes to extras["log"] (TensorBoard)
    # -------------------------------------------------------------------------
    def _get_stage2_rewards(self) -> torch.Tensor:
        """Stage2-specific reward function"""
        total_reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # === 0. maintain grasp success reward ===
        grasp_success = self._check_success()
        grasp_reward = grasp_success.float() * 50.0
        total_reward += grasp_reward

        # === 1. trigger press reward ===
        trigger_pressed = self._check_trigger_pressed()
        self._trigger_pressed_steps += 1
        self._trigger_pressed_steps[~trigger_pressed] = 0

        trigger_reward = trigger_pressed.float() * 50.0
        trigger_hold_bonus = (self._trigger_pressed_steps >= self._trigger_press_threshold).float() * 500.0
        total_reward += trigger_reward + trigger_hold_bonus

        # === 2. keep drill pointing up reward ===
        upright = self._get_drill_uprightness()
        upright_reward = upright * 20.0
        total_reward += upright_reward

        # === 3. drill-up stability reward ===
        self._update_orientation_buffer()
        mean_upright = self._drill_orientation_buffer.mean(dim=1)
        upright_stability_reward = (mean_upright > 0.8).float() * 100.0
        total_reward += upright_stability_reward

        # === 4. hand near the drill ===
        drill_pos = self.drill.data.root_pos_w
        hand_base_pos = self.franka.data.root_pos_w
        hand_drill_dist = torch.norm(hand_base_pos - drill_pos, dim=1)
        proximity_reward = torch.clamp(1.0 - hand_drill_dist / 0.5, 0.0, 1.0) * 5.0
        total_reward += proximity_reward

        # === 5. joint velocity penalty ===
        joint_vel = self.franka.data.joint_vel
        vel_penalty = -torch.sum(joint_vel ** 2, dim=1) * 0.01
        total_reward += vel_penalty

        # === 6. contact-maintenance reward ===
        contact_ok = self._get_contact_ok()
        contact_reward = contact_ok.float() * 20.0
        total_reward += contact_reward

        # --- write to extras["log"] (TensorBoard) ---
        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["reward_grasp"] = grasp_reward.mean()
        self.extras["log"]["reward_trigger"] = trigger_reward.mean()
        self.extras["log"]["reward_trigger_hold"] = trigger_hold_bonus.mean()
        self.extras["log"]["reward_upright"] = upright_reward.mean()
        self.extras["log"]["reward_upright_stability"] = upright_stability_reward.mean()
        self.extras["log"]["reward_proximity"] = proximity_reward.mean()
        self.extras["log"]["reward_vel_penalty"] = vel_penalty.mean()
        self.extras["log"]["reward_contact"] = contact_reward.mean()
        self.extras["log"]["reward_total"] = total_reward.mean()
        self.extras["log"]["trigger_hold_success"] = (
            self._trigger_pressed_steps >= self._trigger_press_threshold
        ).float().mean()
        self.extras["log"]["drill_uprightness"] = upright.mean()

        return total_reward

    def _get_contact_ok(self) -> torch.Tensor:
        """Check whether there is enough contact"""
        contact_force_threshold = 0.1
        all_hand_sensors = [
            "contact_index_intermediate", "contact_thumb_distal",
            "contact_middle_intermediate", "contact_ring_intermediate",
            "contact_pinky_intermediate", "contact_thumb_intermediate",
            "contact_thumb_proximal_base", "contact_thumb_proximal",
            "contact_index_proximal", "contact_middle_proximal",
            "contact_ring_proximal", "contact_pinky_proximal",
            "contact_hand_base",
        ]
        contact_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        try:
            if hasattr(self, 'scene') and hasattr(self.scene, 'sensors'):
                for sensor_name in all_hand_sensors:
                    if sensor_name in self.scene.sensors:
                        sensor = self.scene.sensors[sensor_name]
                        force_matrix = sensor.data.force_matrix_w
                        if force_matrix is not None:
                            force_matrix = torch.nan_to_num(force_matrix, nan=0.0, posinf=0.0, neginf=0.0)
                            force = force_matrix.sum(dim=(1, 2))
                        else:
                            force = sensor.data.net_forces_w.sum(dim=1)
                        force_mag = torch.norm(force, dim=1)
                        force_mag = torch.nan_to_num(force_mag, nan=0.0)
                        contact_count += (force_mag > contact_force_threshold).float()
        except Exception:
            pass
        return contact_count >= 4

    # -------------------------------------------------------------------------
    # fix 2: _get_stage2_dones reuses the parent's NaN detection
    # -------------------------------------------------------------------------
    def _get_stage2_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stage2 termination: reuse parent NaN detection + Stage2-specific logic"""
        # call the parent first to get NaN termination (physical NaN causes crashes, must detect)
        parent_terminated, _ = super()._get_dones()

        failure = self._check_stage2_failure()
        time_outs = self.episode_length_buf >= self.max_episode_length
        trigger_hold_success = self._trigger_pressed_steps >= self._trigger_press_threshold

        terminated = failure | parent_terminated  # keep NaN termination
        truncated = time_outs | trigger_hold_success

        self._cached_failure_mask = failure
        self._cached_lenient_success = trigger_hold_success
        return terminated, truncated

    def _check_stage2_failure(self) -> torch.Tensor:
        """Stage2 failure detection"""
        drill_pos = self.drill.data.root_pos_w
        upright = self._get_drill_uprightness()

        drill_flipped = upright < 0.6

        if hasattr(self, 'initial_drill_pos'):
            drill_xy_distance = torch.norm(drill_pos[:, :2] - self.initial_drill_pos[:, :2], dim=1)
            drill_knocked_away = drill_xy_distance > 0.3
        else:
            drill_knocked_away = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        drill_fell = drill_pos[:, 2] < 0.05

        return drill_flipped | drill_knocked_away | drill_fell
