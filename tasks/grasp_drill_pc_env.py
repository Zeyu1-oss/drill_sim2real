"""
Point Cloud version of GraspDrillEnv.

This module provides a drop-in replacement for GraspDrillEnv that:
  - Adds a TiledCamera above the workspace
  - Converts depth images to point clouds
  - Replaces drill pose observations with point cloud observations
  - Keeps all other functionality (rewards, terminations, actions) identical

No changes to the original grasp_drill_env.py are made.
"""

import torch
import numpy as np

# Re-use all the same imports as the parent env
from tasks.grasp_drill_env import (
    _import_isaac_lab,
    GraspDrillEnv,
    GraspDrillEnvCfg,
    GraspDrillSceneCfg,
    create_grasp_drill_env_cfg,
    load_drill_variants_from_yaml,
    DrillVariantCfg,
)

(
    DirectRLEnv, DirectRLEnvCfg, ObsGroup, ObsTerm,
    RewTerm, DoneTerm, EventTerm, Articulation, RigidObject,
    InteractiveScene, InteractiveSceneCfg, configclass, sim_utils, math_utils,
    ArticulationCfg, RigidObjectCfg, mdp, ImplicitActuatorCfg, AssetBaseCfg,
    ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR,
    ContactSensorCfg,
    DifferentialIKControllerCfg, DifferentialIKController, DifferentialInverseKinematicsActionCfg,
    FRANKA_PANDA_HIGH_PD_CFG, HAS_FRANKA_CFG,
) = _import_isaac_lab()


# ---------------------------------------------------------------------------
# Camera configuration
# ---------------------------------------------------------------------------

@configclass
class _PointCloudCameraSceneCfg(InteractiveSceneCfg):
    """Extended scene with a TiledCamera above the workspace."""

    # Inherit all contact sensors and objects from the base scene
    franka: ArticulationCfg = MISSING
    drill: RigidObjectCfg = MISSING

    # Re-declare contact sensors (must be present in subclass cfg too)
    contact_thumb_distal = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/fr3_inspire_hand/R_thumb_distal",
        update_period=0.02, history_length=0, debug_vis=False,
        force_threshold=0.01, filter_prim_paths_expr=["{ENV_REGEX_NS}/Drill*"],
    )
    contact_hand_base = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/fr3_inspire_hand/R_hand_base_link",
        update_period=0.02, history_length=0, debug_vis=False,
        force_threshold=0.01, filter_prim_paths_expr=["{ENV_REGEX_NS}/Drill*"],
    )
    contact_index_intermediate = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/fr3_inspire_hand/R_index_intermediate",
        update_period=0.02, history_length=0, debug_vis=False,
        force_threshold=0.01, filter_prim_paths_expr=["{ENV_REGEX_NS}/Drill*"],
    )
    contact_middle_intermediate = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/fr3_inspire_hand/R_middle_intermediate",
        update_period=0.02, history_length=0, debug_vis=False,
        force_threshold=0.01, filter_prim_paths_expr=["{ENV_REGEX_NS}/Drill*"],
    )
    contact_ring_intermediate = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/fr3_inspire_hand/R_ring_intermediate",
        update_period=0.02, history_length=0, debug_vis=False,
        force_threshold=0.01, filter_prim_paths_expr=["{ENV_REGEX_NS}/Drill*"],
    )
    contact_pinky_intermediate = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/fr3_inspire_hand/R_pinky_intermediate",
        update_period=0.02, history_length=0, debug_vis=False,
        force_threshold=0.01, filter_prim_paths_expr=["{ENV_REGEX_NS}/Drill*"],
    )
    contact_index_proximal = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/fr3_inspire_hand/R_index_proximal",
        update_period=0.02, history_length=0, debug_vis=False,
        force_threshold=0.01, filter_prim_paths_expr=["{ENV_REGEX_NS}/Drill*"],
    )
    contact_middle_proximal = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/fr3_inspire_hand/R_middle_proximal",
        update_period=0.02, history_length=0, debug_vis=False,
        force_threshold=0.01, filter_prim_paths_expr=["{ENV_REGEX_NS}/Drill*"],
    )
    contact_ring_proximal = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/fr3_inspire_hand/R_ring_proximal",
        update_period=0.02, history_length=0, debug_vis=False,
        force_threshold=0.01, filter_prim_paths_expr=["{ENV_REGEX_NS}/Drill*"],
    )
    contact_pinky_proximal = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/fr3_inspire_hand/R_pinky_proximal",
        update_period=0.02, history_length=0, debug_vis=False,
        force_threshold=0.01, filter_prim_paths_expr=["{ENV_REGEX_NS}/Drill*"],
    )
    contact_thumb_proximal_base = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/fr3_inspire_hand/R_thumb_proximal_base",
        update_period=0.02, history_length=0, debug_vis=False,
        force_threshold=0.01, filter_prim_paths_expr=["{ENV_REGEX_NS}/Drill*"],
    )
    contact_thumb_proximal = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/fr3_inspire_hand/R_thumb_proximal",
        update_period=0.02, history_length=0, debug_vis=False,
        force_threshold=0.01, filter_prim_paths_expr=["{ENV_REGEX_NS}/Drill*"],
    )
    contact_thumb_intermediate = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/fr3_inspire_hand/R_thumb_intermediate",
        update_period=0.02, history_length=0, debug_vis=False,
        force_threshold=0.01, filter_prim_paths_expr=["{ENV_REGEX_NS}/Drill*"],
    )

    # Point cloud camera: fixed above the workspace, looking down
    point_cloud_camera = sim_utils.TiledCameraCfg(
        prim_path="/World/envs/env_.*/PointCloudCamera",
        update_period=0.0,
        data_types=["distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=0.8,
            horizontal_aperture=20.955,
            vertical_aperture=20.955,
            clipping_range=(0.01, 10.0),
        ),
        # Camera is above the table, tilted to face the work area
        # pos: (x=0, y=0, z=0.8) relative to env origin
        # rot: looking down (-90deg around X axis), convention=world means rotation is in world frame
        offset=sim_utils.TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.8),
            rot=(0.7071068, 0.0, 0.7071068, 0.0),  # 90deg around X → looking down
            convention="world",
        ),
        width=224,
        height=224,
    )

    # Ground and table
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0]),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    table = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.2, 0.2),
                metallic=0.0, roughness=0.5,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
                rest_offset=0.001,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0.0, 0.0)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )


# ---------------------------------------------------------------------------
# Point Cloud Environment
# ---------------------------------------------------------------------------

class GraspDrillPCEnv(GraspDrillEnv):
    """
    Point Cloud version of GraspDrillEnv.

    Differences from GraspDrillEnv:
    - Adds a TiledCamera above the workspace
    - Observation space replaces drill pose with point cloud
    - All rewards, terminations, and actions are identical

    Observation layout:
      joint_pos (13) + joint_vel (13) + point_cloud (N x 3)
      = 26 + 3*N

    With N=512: obs_dim = 26 + 1536 = 1562
    """

    def __init__(self, cfg: GraspDrillEnvCfg, render_mode: str = None, debug: bool = False, **kwargs):
        # Swap the scene cfg class for our camera-augmented version
        original_scene = cfg.scene
        # Build the PC scene cfg from the same base parameters
        pc_scene = _PointCloudCameraSceneCfg()

        # Copy over all non-sensor fields from the original scene
        # (franka, drill configs come from the create_grasp_drill_env_cfg factory)
        # We need to assign the articulation/rigid-object cfgs from the original
        pc_scene.franka = original_scene.franka
        pc_scene.drill = original_scene.drill
        pc_scene.table = original_scene.table

        cfg.scene = pc_scene

        # Call parent __init__ (this sets up the scene including the camera)
        super().__init__(cfg, render_mode=render_mode, debug=debug, **kwargs)

        # Attach the camera sensor handle
        self._pc_camera = self.scene["point_cloud_camera"]

        # Camera intrinsic parameters (for point cloud computation)
        # These match the PinholeCameraCfg used in the scene
        self._cam_fx = 24.0
        self._cam_fy = 24.0
        self._cam_cx = float(self._pc_camera.cfg.width) / 2.0
        self._cam_cy = float(self._pc_camera.cfg.height) / 2.0
        self._cam_width = self._pc_camera.cfg.width
        self._cam_height = self._pc_camera.cfg.height

        # Point cloud parameters
        self._pc_num_points = cfg.pc_num_points

        # Observation space: joint(13) + joint_vel(13) + pointcloud(N*3)
        self._obs_dim_with_pc = 26 + 3 * self._pc_num_points

        print(f"[INFO] GraspDrillPCEnv: camera={self._cam_width}x{self._cam_height}, "
              f"pc_points={self._pc_num_points}, obs_dim={self._obs_dim_with_pc}")

    # ------------------------------------------------------------------
    # Point cloud computation
    # ------------------------------------------------------------------

    def _compute_pointcloud_from_depth(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Convert a batch of depth images to point clouds.

        Args:
            depth: [num_envs, height, width] depth in meters

        Returns:
            pointcloud: [num_envs, num_points, 3] xyz in camera frame (meters)
        """
        B, H, W = depth.shape
        N = self._pc_num_points

        # Create pixel grid
        u_grid = torch.arange(W, device=depth.device, dtype=torch.float32)
        v_grid = torch.arange(H, device=depth.device, dtype=torch.float32)
        u, v = torch.meshgrid(u_grid, v_grid, indexing="xy")
        # v is [H, W], u is [H, W]

        # Flatten
        u_flat = u.reshape(-1)      # [H*W]
        v_flat = v.reshape(-1)      # [H*W]
        depth_flat = depth.reshape(B, -1)  # [B, H*W]

        # Filter valid depth
        valid = depth_flat > 0.01   # exclude zero / very close depth

        pointclouds = []
        for b in range(B):
            d = depth_flat[b]       # [H*W]
            v_mask = valid[b]

            # Back-project to camera frame
            x = (u_flat - self._cam_cx) * d / self._cam_fx
            y = (v_flat - self._cam_cy) * d / self._cam_fy
            z = d

            pts = torch.stack([x, y, z], dim=1)  # [H*W, 3]

            # Keep only valid points
            pts_valid = pts[v_mask]               # [M, 3]

            if pts_valid.shape[0] == 0:
                pts_valid = torch.zeros((1, 3), device=depth.device)

            # Random subsample / upsampling to fixed num_points
            M = pts_valid.shape[0]
            if M >= N:
                # Random sample N points
                idx = torch.randperm(M, device=depth.device)[:N]
                pc = pts_valid[idx]
            else:
                # Random sample with replacement to reach N points
                idx = torch.randint(0, M, (N,), device=depth.device)
                pc = pts_valid[idx]

            pointclouds.append(pc)

        return torch.stack(pointclouds, dim=0)  # [B, N, 3]

    # ------------------------------------------------------------------
    # Observation override
    # ------------------------------------------------------------------

    def _get_observations(self) -> torch.Tensor:
        """
        Returns point cloud observations (no drill state).

        Shape: [num_envs, 26 + 3*pc_num_points]
         - joint_pos: 13
         - joint_vel: 13
         - pointcloud: pc_num_points x 3
        """
        joint_pos = self.franka.data.joint_pos
        joint_vel = self.franka.data.joint_vel

        proximal_joint_indices = self.controlled_joint_indices.tolist()
        joint_pos_proc = joint_pos[:, proximal_joint_indices]
        joint_vel_proc = joint_vel[:, proximal_joint_indices]

        # Point cloud from camera depth
        depth = self._pc_camera.data["distance_to_camera"]  # [num_envs, H, W]
        pointcloud = self._compute_pointcloud_from_depth(depth)  # [num_envs, N, 3]

        # Transform point cloud z to env-local frame
        # Camera is at z=0.8 in env frame, looking down.
        # depth is along camera optical axis; approximate env_z = camera_z - depth
        pc_env = pointcloud.clone()
        cam_offset_z = 0.8  # meters above env origin
        pc_env[:, :, 2] = pc_env[:, :, 2] - cam_offset_z

        # Flatten point cloud: [num_envs, N*3]
        pc_flat = pc_env.reshape(self.num_envs, -1)

        # Concatenate: joint(26) + pc(N*3)
        observations = torch.cat([
            joint_pos_proc,  # 13
            joint_vel_proc,  # 13
            pc_flat,         # N*3
        ], dim=1)

        # NaN guard
        nan_mask = torch.isnan(observations)
        if nan_mask.any():
            observations = torch.where(nan_mask, torch.zeros_like(observations), observations)

        return {"policy": observations}

    @property
    def observation_space_dim(self) -> int:
        """Return the point cloud observation dimension."""
        return self._obs_dim_with_pc


# ---------------------------------------------------------------------------
# Config factory for the PC environment
# ---------------------------------------------------------------------------

def create_grasp_drill_pc_env_cfg(
    num_envs: int = 256,
    device: str = "cuda:0",
    headless: bool = False,
    hyperparameters=None,
    drill_config_path: str = None,
    drill_variants_path: str = None,
    debug: bool = False,
    pc_num_points: int = 512,
    enable_pc_viz: bool = False,
) -> GraspDrillEnvCfg:
    """
    Create a configuration for GraspDrillPCEnv.

    This is a thin wrapper around create_grasp_drill_env_cfg that:
    - Sets pc_num_points
    - The actual scene swap happens in GraspDrillPCEnv.__init__
    """
    # Build the base config (this populates franka, drill, etc.)
    cfg = create_grasp_drill_env_cfg(
        num_envs=num_envs,
        device=device,
        headless=headless,
        hyperparameters=hyperparameters,
        drill_config_path=drill_config_path,
        drill_variants_path=drill_variants_path,
        debug=debug,
        enable_pc_viz=enable_pc_viz,
    )

    # Override observation space (no drill state)
    pc_obs_dim = 26 + 3 * pc_num_points
    cfg.observation_space = pc_obs_dim
    cfg.pc_num_points = pc_num_points

    print(f"[INFO] GraspDrillPCEnv config: obs_space={pc_obs_dim} ({pc_num_points} PC points)")

    return cfg
