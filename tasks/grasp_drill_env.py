import os
import ast
import sys
import torch
import numpy as np
import yaml
from dataclasses import MISSING, dataclass
from typing import Dict, Any, Tuple

def _import_isaac_lab():
    try:
        # 尝试新版本导入 (isaaclab)
        from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg  # 改为 DirectRL
        from isaaclab.managers import ObservationGroupCfg as ObsGroup
        from isaaclab.managers import ObservationTermCfg as ObsTerm
        from isaaclab.managers import RewardTermCfg as RewTerm
        from isaaclab.managers import TerminationTermCfg as DoneTerm
        from isaaclab.managers import EventTermCfg as EventTerm
        from isaaclab.managers import SceneEntityCfg
        from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg, AssetBaseCfg
        from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
        from isaaclab.sensors.camera import CameraCfg
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.scene import InteractiveScene
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.utils import configclass
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
        import isaaclab.sim as sim_utils
        import isaaclab.utils.math as math_utils
        import isaaclab.envs.mdp as mdp  # 奖励函数需要
        # IK 控制相关
        from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
        from isaaclab.controllers.differential_ik import DifferentialIKController
        from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
        # Franka 预配置
        try:
            from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
            HAS_FRANKA_CFG = True
        except ImportError:
            HAS_FRANKA_CFG = False
            FRANKA_PANDA_HIGH_PD_CFG = None

        try:
            from inspire_drill.tasks.config.inspire_hand_cfg import INSPIRE_HAND_CFG
            HAS_INSPIRE_HAND_CFG = True
        except ImportError:
            HAS_INSPIRE_HAND_CFG = False
            INSPIRE_HAND_CFG = None
    except ImportError:
        try:
            # 回退到旧版本导入 (omni.isaac.lab)
            from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
            from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
            from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
            from omni.isaac.lab.managers import RewardTermCfg as RewTerm
            from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
            from omni.isaac.lab.managers import EventTermCfg as EventTerm
            from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg, AssetBaseCfg
            from omni.isaac.lab.actuators import ImplicitActuatorCfg
            from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
            from omni.isaac.lab.utils import configclass
            from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
            import omni.isaac.lab.sim as sim_utils
            import omni.isaac.lab.utils.math as math_utils
            import omni.isaac.lab.envs.mdp as mdp
            from omni.isaac.lab.sensors import ContactSensorCfg, TiledCameraCfg
            from omni.isaac.lab.sensors.camera import CameraCfg
            from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
            from omni.isaac.lab.controllers.differential_ik import DifferentialIKController
            from omni.isaac.lab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
            try:
                from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
                HAS_FRANKA_CFG = True
            except ImportError:
                HAS_FRANKA_CFG = False
                FRANKA_PANDA_HIGH_PD_CFG = None
            try:
                from inspire_drill.tasks.config.inspire_hand_cfg import INSPIRE_HAND_CFG
                HAS_INSPIRE_HAND_CFG = True
            except ImportError:
                HAS_INSPIRE_HAND_CFG = False
                INSPIRE_HAND_CFG = None
        except ImportError as e:
            raise ImportError(
                "无法导入Isaac Lab模块。请确保：\n"
                "1. 已激活正确的conda环境\n"
                "2. 使用 isaaclab.sh 脚本运行，或\n"
                "3. 在IsaacLab目录下运行\n"
                f"错误详情: {e}"
            )
    
    return (
        DirectRLEnv, DirectRLEnvCfg, ObsGroup, ObsTerm,
        RewTerm, DoneTerm, EventTerm, Articulation, RigidObject,
        InteractiveScene, InteractiveSceneCfg, configclass, sim_utils, math_utils,
        ArticulationCfg, RigidObjectCfg, mdp, ImplicitActuatorCfg, AssetBaseCfg, ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR,
        ContactSensorCfg, TiledCameraCfg, CameraCfg,
        DifferentialIKControllerCfg, DifferentialIKController, DifferentialInverseKinematicsActionCfg,
        FRANKA_PANDA_HIGH_PD_CFG, HAS_INSPIRE_HAND_CFG,
    )

try:
    (
        DirectRLEnv, DirectRLEnvCfg, ObsGroup, ObsTerm,
        RewTerm, DoneTerm, EventTerm, Articulation, RigidObject,
        InteractiveScene, InteractiveSceneCfg, configclass, sim_utils, math_utils,
        ArticulationCfg, RigidObjectCfg, mdp, ImplicitActuatorCfg, AssetBaseCfg, ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR,
        ContactSensorCfg, TiledCameraCfg, CameraCfg,
        DifferentialIKControllerCfg, DifferentialIKController, DifferentialInverseKinematicsActionCfg,
        FRANKA_PANDA_HIGH_PD_CFG, HAS_INSPIRE_HAND_CFG,
    ) = _import_isaac_lab()
except ImportError as e:
    import sys
    import os
    isaac_sim_path = os.environ.get("ISAAC_SIM_PATH", None)
    
    python_path = os.environ.get("PYTHONPATH", "")


def _try_set_attribute(schema_api, name: str, value, camel_case: bool = True):
    """Set a USD schema attribute without raising if the attribute doesn't exist."""
    if value is None:
        return
    from isaaclab.sim.utils.prims import to_camel_case
    attr_name = to_camel_case(name, to="CC") if camel_case else name
    attr_creator = getattr(schema_api, f"Create{attr_name}Attr", None)
    if attr_creator is not None:
        attr_creator().Set(value)



@dataclass
class DrillVariantCfg:
    """Configuration for a single drill variant (geometry + grasp params)."""
    name: str
    usd_path: str
    enabled: bool = True        # set False to disable this variant (won't spawn)
    variant_index: int = -1    # global fixed index; must match training layout
    trigger_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    drill_bit_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    drill_bit_rot: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    body_mask_axis: str = "Z"
    forward_axis: str = "Z"
    body_mask_min: float = -0.03
    body_mask_max: float = 0.045
    body_mask_y_min: float = -0.03
    body_mask_y_max: float = 0.045
    thumb_target_local: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    # List of initial positions; one is randomly chosen at each reset
    initial_pos_list: Tuple[Tuple[float, float, float], ...] = ()
    # Corresponding list of initial rotations (same length as initial_pos_list)
    initial_rot_list: Tuple[Tuple[float, float, float, float], ...] = ()
    # Goal position list for mutual reward (e.g., initial_pos_1)
    goal_pos_list: Tuple[Tuple[float, float, float], ...] = ()
    # Goal rotation list for mutual reward (e.g., initial_rot_1)
    goal_rot_list: Tuple[Tuple[float, float, float, float], ...] = ()
    # Fallback single values (for compatibility with old YAMLs that have no _list fields)
    initial_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_rot: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    up_axis: str = "Y"


def load_drill_variants_from_yaml(yaml_path: str) -> tuple[list[DrillVariantCfg], int]:
    """Load drill variants from YAML.

    Returns:
        (active_variants, total_num_variants):
            - active_variants: only enabled variants (for spawning / resetting)
            - total_num_variants: total number of defined variants (including disabled).
              Used for one-hot encoding dimension — always matches training setup so that
              checkpoints remain compatible when variants are disabled.
    """
    import os
    if not os.path.exists(yaml_path):
        print(f"[WARN] Drill variants YAML not found: {yaml_path}")
        variants = _get_default_drill_variants()
        return variants, len(variants)

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    all_variants = []
    for i, v in enumerate(data.get("drill_variants", [])):
        # Collect all initial_pos / initial_rot entries, pairing by index suffix.
        # initial_pos     + initial_rot     -> index 0  (backwards compat with old YAMLs)
        # initial_pos_N   + initial_rot_N   -> index N  (N=1,2,3,...)
        # pos/rot with the same suffix index are paired together.
        _pos_by_idx: dict[int, tuple] = {}
        _rot_by_idx: dict[int, tuple] = {}
        _goal_pos_by_idx: dict[int, tuple] = {}
        _goal_rot_by_idx: dict[int, tuple] = {}
        for key in sorted(v.keys()):
            if key == "initial_pos":
                _pos_by_idx[0] = tuple(v[key])
            elif key == "initial_rot":
                _rot_by_idx[0] = tuple(v[key])
            elif key.startswith("initial_pos_") and key[12:].isdigit():
                _pos_by_idx[int(key[12:])] = tuple(v[key])
            elif key.startswith("initial_rot_") and key[12:].isdigit():
                _rot_by_idx[int(key[12:])] = tuple(v[key])
            elif key.startswith("goal_pos_") and key[9:].isdigit():
                _goal_pos_by_idx[int(key[9:])] = tuple(v[key])
            elif key.startswith("goal_rot_") and key[9:].isdigit():
                _goal_rot_by_idx[int(key[9:])] = tuple(v[key])

        # Build aligned pos_list / rot_list by merging on index
        all_indices = sorted(set(_pos_by_idx.keys()) | set(_rot_by_idx.keys()))
        pos_list = [_pos_by_idx[i] for i in all_indices]
        rot_list = [_rot_by_idx[i] for i in all_indices]

        # Build aligned goal_pos_list / goal_rot_list
        all_goal_indices = sorted(set(_goal_pos_by_idx.keys()) | set(_goal_rot_by_idx.keys()))
        goal_pos_list_out = [_goal_pos_by_idx[i] for i in all_goal_indices]
        goal_rot_list_out = [_goal_rot_by_idx[i] for i in all_goal_indices]

        cfg = DrillVariantCfg(
            name=str(v["name"]),
            usd_path=str(v["usd_path"]),
            enabled=bool(v.get("enabled", True)),
            variant_index=int(v.get("variant_index", i)),
            trigger_offset=tuple(v.get("trigger_offset", [0.0, 0.0, 0.0])),
            drill_bit_offset=tuple(v.get("drill_bit_offset", [0.0, 0.0, 0.0])),
            drill_bit_rot=tuple(v.get("drill_bit_rot", [1.0, 0.0, 0.0, 0.0])),
            body_mask_axis=str(v.get("body_mask_axis", "Z")),
            forward_axis=str(v.get("forward_axis", "Z")),
            body_mask_min=float(v.get("body_mask_min", v.get("body_mask_y_min", -0.03))),
            body_mask_max=float(v.get("body_mask_max", v.get("body_mask_y_max", 0.045))),
            thumb_target_local=tuple(v.get("thumb_target_local", [0.0, 0.0, 0.0])),
            initial_pos_list=tuple(pos_list) if pos_list else ((0.0, 0.0, 0.0),),
            initial_rot_list=tuple(rot_list) if rot_list else ((1.0, 0.0, 0.0, 0.0),),
            goal_pos_list=tuple(goal_pos_list_out) if goal_pos_list_out else ((0.0, 0.0, 0.0),),
            goal_rot_list=tuple(goal_rot_list_out) if goal_rot_list_out else ((1.0, 0.0, 0.0, 0.0),),
            initial_pos=tuple(v.get("initial_pos", [0.0, 0.0, 0.0])),
            initial_rot=tuple(v.get("initial_rot", [1.0, 0.0, 0.0, 0.0])),
            scale=tuple(v.get("scale", [1.0, 1.0, 1.0])),
            up_axis=str(v.get("up_axis", "Y")),
        )
        all_variants.append(cfg)

    if not all_variants:
        variants = _get_default_drill_variants()
        return variants, len(variants)

    # total_num_variants = max index + 1 across ALL defined variants (including disabled).
    # This determines the one-hot dimension — it MUST match training so that checkpoints load correctly.
    total_num_variants = max(v.variant_index for v in all_variants) + 1
    active_variants = [v for v in all_variants if v.enabled]

    if not active_variants:
        print("[WARN] No enabled drill variants, falling back to all variants")
        active_variants = all_variants

    print(f"[INFO] Loaded {len(active_variants)} active / {total_num_variants} total drill variants:")
    for v in all_variants:
        status = "active" if v.enabled else "DISABLED"
        print(f"  [{v.variant_index}] {status}: {v.name}")

    return active_variants, total_num_variants


def _get_default_drill_variants() -> list[DrillVariantCfg]:
    """Hardcoded fallback when no YAML is provided."""
    return [
        DrillVariantCfg(
            name="drill2",
            usd_path="assets/drill2.usd",
            trigger_offset=(0.0, 0.03864899, 0.0501017),
            body_mask_y_min=-0.03,
            body_mask_y_max=0.045,
            thumb_target_local=(0.013, 0.05064899, 0.02),
            initial_pos=(0.85, 0.2, 0.13),
            initial_rot=(0.559, 0.521, 0.430, 0.480),
            scale=(1.0, 1.2, 1.0),
        ),
    ]


@configclass
class GraspDrillSceneCfg(InteractiveSceneCfg):

    franka: ArticulationCfg = MISSING
    drill: RigidObjectCfg = MISSING

    contact_thumb_distal = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/fr3_inspire_hand/R_thumb_distal",
        update_period=0.02,
        history_length=0,
        debug_vis=False,
        force_threshold=0.01,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Drill*"],
    )
    contact_hand_base = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/fr3_inspire_hand/R_hand_base_link",
        update_period=0.02, history_length=0, debug_vis=False,
        force_threshold=0.01, filter_prim_paths_expr=["{ENV_REGEX_NS}/Drill*"],
    )
    contact_index_intermediate = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/fr3_inspire_hand/R_index_intermediate",
        update_period=0.02,
        history_length=0,
        debug_vis=False,
        force_threshold=0.01,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Drill*"],
    )
    contact_middle_intermediate = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/fr3_inspire_hand/R_middle_intermediate",
        update_period=0.02,
        history_length=0,
        debug_vis=False,
        force_threshold=0.01,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Drill*"],
    )
    contact_ring_intermediate = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/fr3_inspire_hand/R_ring_intermediate",
        update_period=0.02,
        history_length=0,
        debug_vis=False,
        force_threshold=0.01,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Drill*"],
    )
    contact_pinky_intermediate = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/fr3_inspire_hand/R_pinky_intermediate",
        update_period=0.02,
        history_length=0,
        debug_vis=False,
        force_threshold=0.01,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Drill*"],
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


    # ground plane
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
                diffuse_color=(0.2, 0.2, 0.2),  # 深灰色
                metallic=0.0,
                roughness=0.5,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
                rest_offset=0.001,     # 必须 < contact_offset
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),  # 0 = 静态物体
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0, 0.0, 0.0),  # 桌子高度将在 reset 时设置（基于 drill 初始位置）
        ),
    )
    
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    cam1: TiledCameraCfg | None = None
    cam2: TiledCameraCfg | None = None


@configclass
class GraspDrillEnvCfg(DirectRLEnvCfg):

    action_space = 13  # Panda 7臂关节(绝对) + Inspire Hand 6近端关节(增量) = 13维
    # 观测: 关节(13*2) + drill位置速度euler(12) + 关键距离(2) + link_dists(7) + hand_base(7) + goal_quat(4) + contact_forces(13) = 71维
    observation_space = 71
    total_num_variants: int = 3  # one-hot dim; set via create_grasp_drill_env_cfg; NEVER change after training
    scene: GraspDrillSceneCfg = MISSING
    observations: Dict[str, ObsGroup] = MISSING
    rewards: Dict[str, RewTerm] = MISSING
    terminations: Dict[str, DoneTerm] = MISSING
    events: Dict[str, EventTerm] = MISSING
    num_envs: int = 256 
    decimation: int = 3 
    episode_length_s: float = 10  
    fall_dist: float = 0.10 
    lift_z_threshold: float = 0.03 
    success_hold_steps: int = 20
    success_hold_reward: float = 500.0  

    action_smoothing: float = 0.1
    max_joint_delta: float = 0.01
    max_finger_delta: float = 0.03
    # 点云配置
    pc_num_points: int = 512
    tiled_camera_cfg: TiledCameraCfg | None = None  # for distillation

    def __post_init__(self):
        """Post initialization - 配置仿真参数"""
        from .config.hyperparameters import DEFAULT_HYPERPARAMETERS
        self.sim.dt = DEFAULT_HYPERPARAMETERS.simulation.dt

        self.sim.physx.solver_type = 1  # 0=PGS, 1=TGS

        self.sim.physx.gpu_max_rigid_patch_count = 2**19
        self.sim.physx.friction_correlation_distance = 0.001
        self.sim.physx.enable_stabilization = True
        self.sim.physx.bounce_threshold_velocity = 0.1  #）
        self.sim.physx.friction_offset_threshold = 0.01
        self.sim.physx.enableEnhancedDeterminism = False  #
        self.sim.physx.enableImmediateContactProjection = True  

        # Disable DLSS to prevent OmniGraph dtype=0 errors during TiledCamera init.
        # DLSS is forced on by isaaclab.python.headless.rendering.kit (execMode=0).
        # Setting antialiasing_mode="Off" uses rep.settings API which works correctly.
        self.sim.render.antialiasing_mode = "Off"

class _SingleDrillHandle:

    def __init__(self, env: "GraspDrillEnv"):
        self._env = env

    @property
    def data(self):
        """Delegate to the real RigidObject's data."""
        return self._env._drill.data

    @property
    def cfg(self):
        return self._env._drill.cfg

    def write_root_pose_to_sim(self, root_pose, env_ids=None):
        self._env._drill.write_root_pose_to_sim(root_pose, env_ids)

    def write_root_velocity_to_sim(self, root_vel, env_ids=None):
        self._env._drill.write_root_velocity_to_sim(root_vel, env_ids)

    def write_root_state_to_sim(self, root_state, env_ids=None):
        self._env._drill.write_root_state_to_sim(root_state, env_ids)


_failure_active_names_cache = None

def _get_active_failure_names() -> set:
    """解析源码中 _check_failure() 的 failure 赋值语句里实际参与的变量名。
    通过 AST 读取，注释掉的变量不会被 AST 解析出来，从而实现「注释即禁用」。
    结果会被缓存。"""
    global _failure_active_names_cache
    if _failure_active_names_cache is not None:
        return _failure_active_names_cache
    try:
        source_file = os.path.join(os.path.dirname(__file__), "tasks", "grasp_drill_env.py")
        with open(source_file) as f:
            source = f.read()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_check_failure":
                for child in ast.walk(node):
                    if isinstance(child, ast.Assign):
                        for target in child.targets:
                            if isinstance(target, ast.Name) and target.id == "failure":
                                names = set()
                                for elt in ast.walk(child.value):
                                    if isinstance(elt, ast.Name):
                                        names.add(elt.id)
                                _failure_active_names_cache = names
                                return names
        _failure_active_names_cache = {"drill_knocked_away"}
        return _failure_active_names_cache
    except Exception:
        _failure_active_names_cache = {"drill_knocked_away"}
        return _failure_active_names_cache


class GraspDrillEnv(DirectRLEnv):
    
    cfg: GraspDrillEnvCfg
    
    def __init__(self, cfg: GraspDrillEnvCfg, render_mode: str = None, debug: bool = False, **kwargs):

        self.drill_variants = cfg.drill_variants          # only active variants
        self.num_drill_variants = len(self.drill_variants)  # for spawn/reset loop
        self.total_num_variants = cfg.total_num_variants    # for one-hot dim; fixed at training value

        self._obs_print_counter = 0

        # print(f"[INFO] {self.num_drill_variants} active / {self.total_num_variants} total drill variants:")
        # for v in self.drill_variants:
        #     print(f"  [{v.variant_index}] {v.name}: usd={v.usd_path}, scale={v.scale}, num_init_poses={len(v.initial_pos_list)}")
        #     for pi, (p, r) in enumerate(zip(v.initial_pos_list, v.initial_rot_list)):
        #         print(f"    pose[{pi}] pos={p}  rot={r}")

        self._variant_attrs: dict[int, dict] = {}

        self._active_variant_indices_list = [v.variant_index for v in self.drill_variants]
        self._active_indices_tensor = torch.tensor(
            self._active_variant_indices_list,
            dtype=torch.long, device=cfg.device
        )

        axis_map = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}
        cfg._assets_dir = getattr(cfg, '_assets_dir',
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets"))

        for v in self.drill_variants:
            vid = v.variant_index  # global fixed index
            self._load_variant_mesh_from_usd(vid, assets_dir=cfg._assets_dir)
            mesh_data = getattr(self, '_variant_mesh_data', {}).get(vid, {})
            all_pts = mesh_data.get('all_points', [])
            mesh_info = mesh_data.get('mesh_info', [])
            up_vec = axis_map.get(v.up_axis.upper(), (0, 1, 0))
            self._variant_attrs[vid] = {
                "name":               v.name,
                "trigger_offset":     torch.tensor(v.trigger_offset, dtype=torch.float32, device=cfg.device),
                "drill_bit_offset":  torch.tensor(v.drill_bit_offset, dtype=torch.float32, device=cfg.device),
                "drill_bit_rot":     torch.tensor(v.drill_bit_rot, dtype=torch.float32, device=cfg.device),
                "thumb_target_local": torch.tensor(v.thumb_target_local, dtype=torch.float32, device=cfg.device),
                "forward_axis":     getattr(v, 'forward_axis', 'Z').upper(),
                "body_mask_axis":     getattr(v, 'body_mask_axis', 'Z').upper(),
                "body_mask_min":      getattr(v, 'body_mask_min', v.body_mask_y_min),
                "body_mask_max":     getattr(v, 'body_mask_max', v.body_mask_y_max),
                "initial_pos_list":   torch.tensor(v.initial_pos_list, dtype=torch.float32, device=cfg.device),
                "initial_rot_list":   torch.tensor(v.initial_rot_list, dtype=torch.float32, device=cfg.device),
                "goal_pos_list":      torch.tensor(v.goal_pos_list, dtype=torch.float32, device=cfg.device),
                "goal_rot_list":      torch.tensor(v.goal_rot_list, dtype=torch.float32, device=cfg.device),
                "scale":              torch.tensor(v.scale, dtype=torch.float32, device=cfg.device),
                "up_axis":            torch.tensor(up_vec, dtype=torch.float32, device=cfg.device),
                "mesh_points":        torch.tensor(all_pts, dtype=torch.float32, device=cfg.device) if all_pts else None,
                "mesh_info":          mesh_info,
            }

        self._drill_variant_indices = self._active_indices_tensor[
            torch.randint(0, self.num_drill_variants, (cfg.num_envs,), device=cfg.device)
        ]
        # ============================================================

        super().__init__(cfg, render_mode=render_mode, **kwargs)

        self.franka: Articulation = self.scene["franka"]
        self.debug = debug  # 调试模式

        self._drill = self.scene["drill"]
        self._drill._is_initialized = False  # force re-init on first reset
        self.drill = _SingleDrillHandle(self)

        # ============================================================
        # 摩擦参数设置（IsaacLab 官方方式：PhysX tensor API）
        # ============================================================
        try:
            # 读取配置参数（从 sim_params 中）
            # 灵巧手
            hand_s = getattr(self.cfg.sim_params, 'hand_static_friction', 1.0)
            hand_d = getattr(self.cfg.sim_params, 'hand_dynamic_friction', 1.0)
            hand_r = getattr(self.cfg.sim_params, 'hand_restitution', 0.0)
            # 电钻
            drill_s = getattr(self.cfg.sim_params, 'drill_static_friction', 1.0)
            drill_d = getattr(self.cfg.sim_params, 'drill_dynamic_friction', 1.0)
            drill_r = getattr(self.cfg.sim_params, 'drill_restitution', 0.0)
            # 桌子
            table_s = getattr(self.cfg.sim_params, 'table_static_friction', 0.5)
            table_d = getattr(self.cfg.sim_params, 'table_dynamic_friction', 0.4)
            table_r = getattr(self.cfg.sim_params, 'table_restitution', 0.0)

            cpu = "cpu"
            num_envs = self.num_envs
            all_env_ids = torch.arange(num_envs, device=cpu)

            # --- 灵巧手 ---
            franka_num_shapes = self.franka.root_physx_view.max_shapes
            franka_mats = torch.full((num_envs, franka_num_shapes, 3), 0.0, dtype=torch.float32, device=cpu)
            franka_mats[:, :, 0] = hand_s
            franka_mats[:, :, 1] = hand_d
            franka_mats[:, :, 2] = hand_r
            self.franka.root_physx_view.set_material_properties(franka_mats, all_env_ids)
            print(f"[FRICTION] Hand: s={hand_s}, d={hand_d}, r={hand_r}, {franka_num_shapes} shapes x {num_envs} envs", flush=True)

            # --- 电钻 ---
            drill_num_shapes = self._drill.root_physx_view.max_shapes
            drill_mats = torch.full((num_envs, drill_num_shapes, 3), 0.0, dtype=torch.float32, device=cpu)
            drill_mats[:, :, 0] = drill_s
            drill_mats[:, :, 1] = drill_d
            drill_mats[:, :, 2] = drill_r
            self._drill.root_physx_view.set_material_properties(drill_mats, all_env_ids)
            print(f"[FRICTION] Drill: s={drill_s}, d={drill_d}, r={drill_r}, {drill_num_shapes} shapes x {num_envs} envs", flush=True)

            # --- 桌子 ---
            try:
                table_asset = self.scene["table"]
                table_num_shapes = table_asset.root_physx_view.max_shapes
                table_mats = torch.full((num_envs, table_num_shapes, 3), 0.0, dtype=torch.float32, device=cpu)
                table_mats[:, :, 0] = table_s
                table_mats[:, :, 1] = table_d
                table_mats[:, :, 2] = table_r
                table_asset.root_physx_view.set_material_properties(table_mats, all_env_ids)
                print(f"[FRICTION] Table: s={table_s}, d={table_d}, r={table_r}, {table_num_shapes} shapes x {num_envs} envs", flush=True)
            except KeyError:
                print("[FRICTION] Table: not found in scene (KeyError), skipping", flush=True)
            except Exception as e:
                print(f"[FRICTION] Table: setup failed ({e}), skipping", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[FRICTION] === friction setup FAILED: {e} ===", flush=True)

        self._write_all_drill_initial_poses()


        body_names = self.franka.body_names
        self._body_name_to_idx = {name: i for i, name in enumerate(body_names)}
        thumb_candidates = [n for n in body_names if "thumb" in n and "distal" in n]
        self._thumb_distal_body_idx = (
            self._body_name_to_idx.get(thumb_candidates[0], -1) if thumb_candidates else -1
        )
        self._hand_link_obs_mask = torch.tensor(
            [name.startswith("R_")
             and "index" not in name
             and ("thumb_proximal_base" in name or "thumb" not in name)
             and name != "R_hand_base_link"
             for name in body_names],
            dtype=torch.bool, device=self.device
        )
        self._obs_link_names = [name for name in body_names if self._hand_link_obs_mask[self._body_name_to_idx[name]]]

        for v in self.drill_variants:
            vid = v.variant_index
            try:
                trigger_pts, body_filtered, body_all, _ = self._build_variant_surface_points(vid)
                self._variant_attrs[vid]["body_surface_points"] = body_filtered
                # 500点，不过滤，用于DP3点云
                all_pts = self._variant_attrs[vid]["mesh_points"]
                if all_pts is not None and len(all_pts) > 0:
                    n = min(500, len(all_pts))
                    idx = torch.randperm(len(all_pts), device=self.device)[:n]
                    self._variant_attrs[vid]["body_surface_points_500"] = all_pts[idx]
                else:
                    self._variant_attrs[vid]["body_surface_points_500"] = torch.zeros((500, 3), device=self.device)
            except Exception:
                self._variant_attrs[vid]["body_surface_points"] = torch.zeros((1, 3), device=self.device)
                self._variant_attrs[vid]["body_surface_points_500"] = torch.zeros((500, 3), device=self.device)

        drill_mass = self.cfg.drill_params.mass if hasattr(self.cfg, 'drill_params') else 0.5
        drill_gravity = drill_mass * 9.81
        self._table_impact_force_threshold = drill_gravity * 10
          # [N]

        # 成功保持计数器（滑动窗口）
        self._success_window_size = 50
        self._success_window_idx = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._success_window = torch.zeros(
            (self.num_envs, self._success_window_size), dtype=torch.bool, device=self.device
        )
        self._cached_lenient_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._trigger1_offset = torch.zeros(
            self.num_envs, 3, dtype=torch.float32, device=self.device
        )
        self._body_mask_y_min = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._body_mask_y_max = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._up_axis = torch.zeros(
            self.num_envs, 3, dtype=torch.float32, device=self.device
        )
        self._forward_axis = torch.zeros(
            self.num_envs, 3, dtype=torch.float32, device=self.device
        )
        self._thumb_target_local = torch.zeros(
            self.num_envs, 3, dtype=torch.float32, device=self.device
        )
        self._drill_scale = torch.zeros(
            self.num_envs, 3, dtype=torch.float32, device=self.device
        )
        if self.num_drill_variants > 0:
            scales = torch.stack([self._variant_attrs[v.variant_index]["scale"] for v in self.drill_variants])
            self._drill_scale_mean = scales.mean(dim=0)  # [3]
        else:
            self._drill_scale_mean = torch.ones(3, dtype=torch.float32, device=self.device)
        self._env_drill_attrs: dict[int, dict] = {}

        # if self.debug:
        #     print(f"[DEBUG] 最终 contact_sensors 数量: 0 (使用 external forces)")
        #     body_names = self.franka.body_names
        #     print(f"[DEBUG] fr3 body_names ({len(body_names)}):")
        #     for name in body_names[:15]:  # 只打印前15个
        #         print(f"    - {name}")
        #     print(f"[DEBUG] ... (共 {len(body_names)} 个 body)")

        force_attrs = [attr for attr in dir(self.franka.data) if 'force' in attr.lower()]


        self._recent_success_buffer_size = 2048
        self._recent_success = torch.zeros(self._recent_success_buffer_size, dtype=torch.bool, device="cpu")
        self._recent_total = torch.zeros(self._recent_success_buffer_size, dtype=torch.bool, device="cpu")
        self._recent_idx = 0
        self._recent_filled = 0

        self._failure_stats = {
            "drill_knocked_away": 0,
            "physics_nan": 0,
            "normal_total": 0,
            "lenient_success_count": 0,
            "total": 0,
        }
        self._normal_timeout_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._pending_nan_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._cached_failure_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._cached_knocked_away_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # _active_failure_reasons 会在 _check_failure() 首次调用时自动设置
        self._active_failure_reasons = set()

        self.initial_drill_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.initial_drill_pos[:, 2] = 0.15  # 默认高度 15cm (桌子表面上方)

        self.contact_history_len = 5 # 记录过去 10 步
        self.contact_history_buffer = torch.zeros(
            self.num_envs, self.contact_history_len, 
            dtype=torch.bool, device=self.device
        )
        
        self.obs_history_len = 3  # 记录过去 3 帧
        num_joints = len(self.franka.joint_names)
        self.joint_pos_history = torch.zeros(
            self.num_envs, self.obs_history_len, num_joints,
            dtype=torch.float32, device=self.device
        )
        self.drill_pos_history = torch.zeros(
            self.num_envs, self.obs_history_len, 3,
            dtype=torch.float32, device=self.device
        )
        self.finger_dist_history = torch.zeros(
            self.num_envs, self.obs_history_len, 5,  # 5个手指
            dtype=torch.float32, device=self.device
        )
        
        self._reward_print_count = 0  # 用于调试输出
        
        self._reward_components = {
            "approach": torch.zeros(self.num_envs, device=self.device),
            "contact": torch.zeros(self.num_envs, device=self.device),
        }
        self._initial_upright_score = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._drill_goal_quat_per_env = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self.device)  # per-env goal quat
        # 势函数塑形:存上一步的朝向接近度 Φ,朝向 reward = scale*(Φ_now - Φ_prev)。
        # reset 时被覆盖成新一局初始姿态的 Φ(见 _reset_idx 末尾),切断跨局残留,
        # 使站立/躺下都从各自初始 Φ 起步,消除初始姿态造成的 reward 基线差。
        self._upright_phi_prev = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._euler_phi_prev = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._episode_reward_sums = {
            "approach": 0.0,
            "lift": 0.0,
            "contact": 0.0,
        }

        if hasattr(self.cfg.drill_params, 'initial_rot') and self.cfg.drill_params.initial_rot is not None:
            initial_rot = torch.tensor(self.cfg.drill_params.initial_rot, dtype=torch.float32, device=self.device)
        else:
            initial_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
        self.drill_initial_rot_tensor = initial_rot.unsqueeze(0).repeat(self.num_envs, 1)
        self.fall_dist = cfg.fall_dist
        self.lift_z_threshold = cfg.lift_z_threshold
        self.joint_names = self.franka.joint_names

        self.finger_joint_names = [
            "R_index_proximal_joint",
            "R_middle_proximal_joint",
            "R_ring_proximal_joint",
            "R_pinky_proximal_joint",
            "R_thumb_proximal_yaw_joint",
            "R_thumb_proximal_pitch_joint",
            "R_index_intermediate_joint",
            "R_middle_intermediate_joint",
            "R_ring_intermediate_joint",
            "R_pinky_intermediate_joint",
            "R_thumb_intermediate_joint",
            "R_thumb_distal_joint",
        ]

        self.finger_tip_names = [
            "R_thumb_distal", "R_index_intermediate", "R_middle_intermediate",
            "R_ring_intermediate", "R_pinky_intermediate"
        ]

        limits = self.franka.data.joint_pos_limits
        self.joint_limits = {}
        for i, name in enumerate(self.franka.joint_names):
            if limits is not None and i < limits.shape[1]:
                self.joint_limits[name] = (limits[0, i, 0].item(), limits[0, i, 1].item())
            else:
                self.joint_limits[name] = (-1.0, 1.0)



        num_proximal_joints = 13  # 7 panda + 6 proximal hand joints
        self.pc_num_points = getattr(cfg, 'pc_num_points', 256)
        # (obs_dim set dynamically by RL wrapper from actual obs shape)


        # if self.debug:
        #     self._add_drill_frame_visualization()
        
        self.controlled_joint_names = [
            "fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
            "fr3_joint5", "fr3_joint6", "fr3_joint7",
        # Inspire Hand 6近端关节
        "R_index_proximal_joint", "R_middle_proximal_joint",
        "R_pinky_proximal_joint", "R_ring_proximal_joint",
        "R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint",
        ]
        
        self.controlled_joint_indices = []
        for name in self.controlled_joint_names:
            if name in self.franka.joint_names:
                idx = self.franka.joint_names.index(name)
                self.controlled_joint_indices.append(idx)
            else:
                print(f"[WARN]")
        
        self.controlled_joint_indices = torch.tensor(
            self.controlled_joint_indices, dtype=torch.long, device=self.device
        )
        self.num_controlled_joints = len(self.controlled_joint_indices)
        
        all_limits = self.franka.data.joint_pos_limits[0]  # [num_joints, 2]
        self.joint_lower_limits = all_limits[self.controlled_joint_indices, 0]  # [num_controlled]
        self.joint_upper_limits = all_limits[self.controlled_joint_indices, 1]  # [num_controlled]

        self.action_smoothing = getattr(cfg, 'action_smoothing', 0.3)
        self.max_joint_delta = getattr(cfg, 'max_joint_delta', 0.03)
        self.max_finger_delta = getattr(cfg, 'max_finger_delta', 0.02)

        self.num_arm_joints = 7
        self.num_finger_joints = 6
        self.arm_indices = self.controlled_joint_indices[: self.num_arm_joints]
        self.finger_indices = self.controlled_joint_indices[self.num_arm_joints :]

        self.cur_targets = torch.zeros(
            self.num_envs, self.num_controlled_joints, dtype=torch.float32, device=self.device
        )
        current_pos = self.franka.data.joint_pos[0, self.controlled_joint_indices]
        self.cur_targets[:] = current_pos.unsqueeze(0)

        self._drill_mesh_local_points = None
        self._drill_mesh_info = []

    def _find_rb_and_visual_quats(self, stage):

        from pxr import UsdGeom
        rb_quat = None
        vis_quat = None

        def collect(prim, depth=0):
            nonlocal rb_quat, vis_quat
            if depth > 15 or not prim.IsValid():
                return
            schemas = list(prim.GetAppliedSchemas())
            xf = UsdGeom.Xformable(prim)
            if xf:
                t = xf.ComputeLocalToWorldTransform(0)
                rot = t.ExtractRotation()
                q = rot.GetQuat()
                quat_arr = np.array([q.GetReal(), q.GetImaginary()[0], q.GetImaginary()[1], q.GetImaginary()[2]])

                if 'PhysicsRigidBodyAPI' in schemas and rb_quat is None:
                    rb_quat = quat_arr
                if prim.GetTypeName() == 'Mesh' and vis_quat is None:
                    # Only the collision mesh (with PhysicsCollisionAPI) is the visual mesh
                    if 'PhysicsCollisionAPI' in schemas:
                        vis_quat = quat_arr
            for c in prim.GetChildren():
                collect(c, depth + 1)

        collect(stage.GetPseudoRoot())
        return rb_quat, vis_quat

    def _quat_conj(self, q):
        """Quaternion conjugate: [w, -x, -y, -z]"""
        return np.array([q[0], -q[1], -q[2], -q[3]])

    def _quat_mul(self, a, b):
        """Quaternion multiplication (wxyz format)."""
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return np.array([
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ])

    def _load_variant_mesh_from_usd(self, variant_idx: int, assets_dir: str = None):
        import os
        import numpy as np
        from pxr import Usd, UsdGeom, Gf

        if hasattr(self, '_variant_mesh_data') and variant_idx in self._variant_mesh_data:
            return

        variant = next((v for v in self.drill_variants if v.variant_index == variant_idx), None)
        _assets = assets_dir if assets_dir else getattr(self.cfg, '_assets_dir', '')
        usd_path = os.path.join(_assets, variant.usd_path)

        if not os.path.exists(usd_path):
            print(f"[WARN] USD not found for variant {variant_idx} ({variant.name}): {usd_path}")
            return

        stage = Usd.Stage.Open(usd_path)
        unit_scale = 1.0  # hardcoded: mesh vertices are in meters
        print(f"[DEBUG-MESH] Variant {variant_idx} ({variant.name}): unit_scale={unit_scale} (hardcoded)")
        rb_quat, vis_quat = self._find_rb_and_visual_quats(stage)
        if rb_quat is not None and vis_quat is not None:
            # Correction: R_correction = R_RB^-1 * R_vis  =>  q_correction = q_RB^-1 * q_vis
            q_correction = self._quat_mul(self._quat_conj(rb_quat), vis_quat)
            correction_angle = 2 * np.arccos(np.clip(abs(q_correction[0]), 0, 1))
            print(f"[INFO] Variant {variant_idx} ({variant.name}): "
                  f"RB_quat=[{rb_quat[0]:.4f},{rb_quat[1]:.4f},{rb_quat[2]:.4f},{rb_quat[3]:.4f}] "
                  f"Vis_quat=[{vis_quat[0]:.4f},{vis_quat[1]:.4f},{vis_quat[2]:.4f},{vis_quat[3]:.4f}] "
                  f"Correction_angle={np.degrees(correction_angle):.1f}deg")
        else:
            q_correction = np.array([1.0, 0.0, 0.0, 0.0])
            if rb_quat is None:
                print(f"[WARN] Variant {variant_idx} ({variant.name}): no RigidBody prim found")
            if vis_quat is None:
                print(f"[WARN] Variant {variant_idx} ({variant.name}): no visual mesh prim found")

        # Build correction rotation matrix (wxyz -> mat3)
        q = q_correction
        w, x, y, z = q[0], q[1], q[2], q[3]
        R_correction = np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
        ])

        all_points = []
        mesh_info = []

        def collect(prim):
            if not prim.IsValid():
                return
            m = UsdGeom.Mesh(prim)
            if m:
                pa = m.GetPointsAttr()
                if pa:
                    pts = pa.Get()
                    if pts:
                        lp = [(p[0], p[1], p[2]) for p in pts]
                        path_lower = str(prim.GetPath()).lower()
                        if 'trigger' in path_lower:
                            part_type = 'Trigger'
                        elif 'chuck' in path_lower or 'handle' in path_lower or 'grip' in path_lower:
                            part_type = 'Handle'
                        elif 'motor' in path_lower or 'body' in path_lower or 'object_4' in path_lower or 'drill_01' in path_lower or 'drill_' in path_lower:
                            part_type = 'Body'
                        else:
                            part_type = 'Other'
                        mesh_info.append({
                            'name': str(prim.GetPath()),
                            'points': lp,
                            'type': part_type,
                            'start_idx': len(all_points)
                        })
                        all_points.extend(lp)
            for c in prim.GetChildren():
                collect(c)

        collect(stage.GetPseudoRoot())

        if all_points:
            pts_arr = np.array(all_points)
            avg_mag = np.mean(np.linalg.norm(pts_arr, axis=1))
            max_mag = np.max(np.linalg.norm(pts_arr, axis=1))
            if avg_mag > 2.0 or max_mag > 5.0:
                unit_scale = 0.01
                all_points = [(p[0] * 0.01, p[1] * 0.01, p[2] * 0.01) for p in all_points]
                for info in mesh_info:
                    info['points'] = [(p[0] * 0.01, p[1] * 0.01, p[2] * 0.01) for p in info['points']]
                print(f"[INFO] Variant {variant_idx} ({variant.name}): auto-detected CENTIMETER units "
                      f"(avg_mag={avg_mag:.2f}m, max_mag={max_mag:.2f}m) -> scale=0.01")
            else:
                unit_scale = 1.0
                print(f"[INFO] Variant {variant_idx} ({variant.name}): auto-detected METER units "
                      f"(avg_mag={avg_mag:.3f}m, max_mag={max_mag:.3f}m) -> scale=1.0")
        else:
            unit_scale = 1.0

        all_points = [tuple(R_correction @ np.array(pt)) for pt in all_points]
        for info in mesh_info:
            info['points'] = [tuple(R_correction @ np.array(pt)) for pt in info['points']]

        print(f"[DEBUG] Variant {variant_idx} ({variant.name}): {len(all_points)} pts, unit_scale={unit_scale:.4f}")
        for info in mesh_info:
            print(f"  - {info['name'].split('/')[-1]}: {len(info['points'])} pts, type={info['type']}")

        if not hasattr(self, '_variant_mesh_data'):
            self._variant_mesh_data = {}
        self._variant_mesh_data[variant_idx] = {
            'all_points': all_points,
            'mesh_info': mesh_info,
            'unit_scale': unit_scale,
        }

    def _init_drill_mesh_from_usd(self):
        if self._drill_mesh_local_points is not None:
            return

        vid0 = self._drill_variant_indices[0].item()
        self._load_variant_mesh_from_usd(vid0)

        if not hasattr(self, '_variant_mesh_data') or vid0 not in self._variant_mesh_data:
            print("[WARN] No mesh data loaded")
            return

        data = self._variant_mesh_data[vid0]
        if data['all_points']:
            self._drill_mesh_local_points = torch.tensor(
                data['all_points'], dtype=torch.float32, device=self.device
            )
            self._drill_mesh_info = data['mesh_info']
            self._mesh_variant_idx = vid0
            print(f"[INFO] Variant {vid0} ({self._variant_attrs.get(vid0, {}).get('name', str(vid0))}) mesh: {len(data['all_points'])} vertices")
        else:
            print("[WARN] No vertices in loaded mesh")

    def _get_drill_surface_points_from_usd(self) -> torch.Tensor:
        if self._drill_mesh_local_points is None:
            self._init_drill_mesh_from_usd()
        
        if self._drill_mesh_local_points is None or len(self._drill_mesh_local_points) == 0:
            return torch.zeros((1, 3), device=self.device)
        
        return self._drill_mesh_local_points

    def _sample_surface_points_uniform(
        self, local_points: torch.Tensor, num_pts: int = 512
    ) -> torch.Tensor:
        """Uniformly sample `num_pts` points from mesh vertices.

        local_points: [N, 3] in drill local frame (before scale).
        Returns: [num_pts, 3] sampled points in drill local frame.
        """
        N = local_points.shape[0]
        if N >= num_pts:
            indices = torch.linspace(0, N - 1, num_pts, dtype=torch.long, device=local_points.device)
            return local_points[indices]
        else:
            repeats = (num_pts + N - 1) // N
            repeated = local_points.repeat(repeats)
            indices = torch.linspace(0, len(repeated) - 1, num_pts, dtype=torch.long, device=local_points.device)
            return repeated[indices]

    def _build_variant_surface_points(self, vid: int) -> tuple:
        """Build (trigger_points, body_points, all_points) for a single variant from cached mesh data.

        Uses mesh data stored in self._variant_attrs[vid] during __init__.

        body_pts_filtered uses body_mask_axis + body_mask_min/max from the variant config
        (e.g. drill2: Z-axis body, drill_yellow: X-axis body).
        """
        vdata = self._variant_attrs[vid]
        all_points = vdata["mesh_points"]
        mesh_info = vdata["mesh_info"]
        trigger_offset = vdata["trigger_offset"]

        if all_points is None or len(all_points) == 0:
            zero = torch.zeros((1, 3), device=self.device)
            return zero, zero, zero, zero

        local_points = all_points

        # --- body mask from mesh_info (Body type), fallback to non-Handle ---
        body_mask = torch.zeros(len(local_points), dtype=torch.bool, device=self.device)
        for info in mesh_info:
            if info.get('type') == 'Body':
                s, n = info['start_idx'], len(info['points'])
                body_mask[s:s + n] = True

        body_pts = local_points[body_mask]
        if body_pts.shape[0] == 0:
            body_mask = torch.ones(len(local_points), dtype=torch.bool, device=self.device)
            for info in mesh_info:
                if info.get('type') == 'Handle':
                    s, n = info['start_idx'], len(info['points'])
                    body_mask[s:s + n] = False
            body_pts = local_points[body_mask]
            if body_pts.shape[0] == 0:
                body_pts = local_points

        # --- apply per-variant body-axis filter ---
        axis_str = vdata["body_mask_axis"]
        axis_idx = {"X": 0, "Y": 1, "Z": 2}[axis_str]
        bmin = vdata["body_mask_min"]
        bmax = vdata["body_mask_max"]
        body_axis_vals = body_pts[:, axis_idx]
        body_y_mask = (body_axis_vals > bmin) & (body_axis_vals < bmax)
        if body_y_mask.any():
            body_pts_filtered = body_pts[body_y_mask]
        else:
            # Mask too narrow or all points outside; fallback to wider range (10% margin)
            axis_min, axis_max = body_axis_vals.min().item(), body_axis_vals.max().item()
            margin = (axis_max - axis_min) * 0.1 + 0.01
            body_y_mask = (body_axis_vals > axis_min - margin) & (body_axis_vals < axis_max + margin)
            body_pts_filtered = body_pts[body_y_mask] if body_y_mask.any() else body_pts
            print(f"[WARN] Variant {vid}: body mask [{bmin},{bmax}] on axis {axis_str} "
                  f"filtered to 0 pts; used fallback range "
                  f"[{axis_min - margin:.4f}, {axis_max + margin:.4f}] -> {body_pts_filtered.shape[0]} pts")

        # --- avoided_points: body vertices OUTSIDE the mask (for penalty) ---
        avoided_pts = body_pts[~body_y_mask] if body_y_mask.any() and not body_y_mask.all() else torch.zeros((0, 3), device=self.device)

        # --- trigger: small sphere at trigger_offset (drill local coords) ---
        num_pts = 58
        theta = torch.linspace(0, 2 * torch.pi, num_pts // 2, device=self.device)
        phi = torch.linspace(0, torch.pi, 2, device=self.device)
        theta, phi = torch.meshgrid(theta, phi, indexing='ij')
        sphere = torch.stack([
            0.005 * torch.sin(phi) * torch.cos(theta),
            0.005 * torch.sin(phi) * torch.sin(theta),
            0.005 * torch.cos(phi),
        ], dim=2).reshape(-1, 3)
        trigger_pts = sphere + trigger_offset.unsqueeze(0)

        return trigger_pts, body_pts_filtered, body_pts, avoided_pts

    def _get_drill_surface_points_split(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (trigger_points, body_points, all_local_points) for the primary (env 0) variant.

        Deprecated for multi-variant use — kept for backward compat with non-reward callers.
        """
        vid0 = self._drill_variant_indices[0].item()
        return self._build_variant_surface_points(vid0)

    def _get_drill_surface_points_split_for_env(
        self, env_ids: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Return (trigger_points, body_filtered_points, body_all_points, avoided_points) per env, grouped by variant.

        Returns four lists of tensors (one per env):
          - trigger_points: sphere points at trigger_offset (drill local coords)
          - body_filtered_points: body-type mesh vertices filtered by body_mask_y_min/max
          - body_all_points: all body-type mesh vertices (unfiltered)
          - avoided_points: body-type vertices OUTSIDE the body_mask range
        """
        variant_ids = self._drill_variant_indices[env_ids]  # [n]
        unique_vids = variant_ids.unique().tolist()
        n = len(env_ids)

        # Pre-build for each unique variant
        var_data = {}
        for vid in unique_vids:
            var_data[vid] = self._build_variant_surface_points(vid)

        # Assign to envs
        trigger_list, body_filtered_list, body_all_list, avoided_list = [], [], [], []
        for i, vid in enumerate(variant_ids.tolist()):
            t, bf, ba, av = var_data[vid]
            trigger_list.append(t)
            body_filtered_list.append(bf)
            body_all_list.append(ba)
            avoided_list.append(av)

        return trigger_list, body_filtered_list, body_all_list, avoided_list

    def _add_drill_frame_visualization(self):
        """Add coordinate frame visualization for each drill variant.

        Adds a frame to one representative env per variant (env_0 for variant 0,
        env_1 for variant 1, etc.) so all variants are visible in debug mode.
        """
        try:
            from pxr import Usd, UsdGeom, Sdf, Gf
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            drill_prim_path_tpl = str(self.drill.cfg.prim_path)
            if "env_.*" not in drill_prim_path_tpl:
                print(f"[DEBUG-WARN] Drill prim path has no env wildcard: {drill_prim_path_tpl}")
                return

            # Show a frame for each ACTIVE variant at its corresponding env.
            # Spawner uses env_id % num_active → spawn_cfg[slot]. Use that mapping here.
            num_to_show = min(self.num_drill_variants, self.num_envs)
            for spawn_slot in range(num_to_show):
                vid = self._active_indices_tensor[spawn_slot].item()  # global index
                env_idx = spawn_slot  # spawn_slot = env_id % num_active
                variant_name = self._variant_attrs.get(vid, {}).get("name", str(vid))
                drill_prim_path = drill_prim_path_tpl.replace("env_.*", f"env_{env_idx}")
                print(f"[DEBUG] Drill variant {vid} ({variant_name}) prim path: {drill_prim_path}")

                drill_prim = stage.GetPrimAtPath(drill_prim_path)
                if not drill_prim:
                    print(f"[DEBUG-WARN] Drill prim not found: {drill_prim_path}")
                    continue

                frame_prim_path = f"{drill_prim_path}/FrameVisualizer"
                frame_prim = stage.GetPrimAtPath(frame_prim_path)
                if frame_prim:
                    continue

                xform = UsdGeom.Xform.Define(stage, frame_prim_path)
                frame_prim = xform.GetPrim()
                frame_usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd"
                frame_prim.GetReferences().AddReference(frame_usd_path)
                frame_xform = UsdGeom.Xformable(frame_prim)
                frame_xform.AddScaleOp().Set(Gf.Vec3f(0.2, 0.2, 0.2))

            self._add_approach_links_frame_visualization(stage, drill_prim_path_tpl)


        except Exception as e:
            import traceback
            traceback.print_exc()

    def _add_approach_links_frame_visualization(self, stage, drill_prim_path_tpl: str):
        """为所有 approach 相关的 link 添加坐标系可视化（只对 env_0 的 hand 加一次）"""
        try:
            from pxr import UsdGeom, Gf

            # 获取 hand 的 prim path (Articulation) - 只在 env_0 上加
            hand_prim_path = str(self.franka.cfg.prim_path)
            if "env_.*" in hand_prim_path:
                hand_prim_path = hand_prim_path.replace("env_.*", "env_0")

            # Inspire Hand 的实际路径: /fr3_inspire_hand/
            inspire_hand_path = hand_prim_path

            approach_links = [
                "R_index_intermediate",
                "R_middle_intermediate",
                "R_ring_intermediate",
                "R_pinky_intermediate",
                "R_thumb_distal",
            ]

            # 获取 hand 的 body 名称
            body_names = self.franka.body_names

            for link_name in approach_links:
                # 使用 inspire_hand_path 而不是 hand_prim_path
                link_prim_path = f"{inspire_hand_path}/{link_name}"
                link_prim = stage.GetPrimAtPath(link_prim_path)

                if not link_prim:
                    continue

                # 创建 frame prim
                frame_prim_path = f"{link_prim_path}/FrameVisualizer"
                frame_prim = stage.GetPrimAtPath(frame_prim_path)

                if frame_prim:
                    continue

                # 创建 Xform 并添加 frame
                xform = UsdGeom.Xform.Define(stage, frame_prim_path)
                frame_prim = xform.GetPrim()

                frame_usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd"
                references = frame_prim.GetReferences()
                references.AddReference(frame_usd_path)

                # 设置坐标系 scale
                frame_xform = UsdGeom.Xformable(frame_prim)
                frame_xform.AddScaleOp().Set(Gf.Vec3f(0.05, 0.05, 0.05))  # 较小



        except Exception as e:
            import traceback
            traceback.print_exc()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        # self.actions[:, -6:] = 0.0

    def _apply_action(self) -> None:
        # actions: [num_envs, 13]
        # 前 7 维: Panda 臂绝对位置 (action in [-1, 1])
        # 后 6 维: 手指关节增量 (action in [-1, 1] -> delta in [-max_finger_delta, max_finger_delta])
        range_len = self.joint_upper_limits - self.joint_lower_limits

        # --- Panda 臂: 绝对位置映射 ---
        arm_actions = self.actions[:, : self.num_arm_joints]
        arm_raw = 0.5 * (arm_actions + 1.0) * range_len[: self.num_arm_joints].unsqueeze(0) + self.joint_lower_limits[: self.num_arm_joints].unsqueeze(0)

        arm_smoothed = (
            self.action_smoothing * arm_raw
            + (1.0 - self.action_smoothing) * self.cur_targets[:, : self.num_arm_joints]
        )

        if self.max_joint_delta > 0:
            arm_delta = arm_smoothed - self.cur_targets[:, : self.num_arm_joints]
            arm_delta = torch.clamp(arm_delta, -self.max_joint_delta, self.max_joint_delta)
            self.cur_targets[:, : self.num_arm_joints] = self.cur_targets[:, : self.num_arm_joints] + arm_delta
        else:
            self.cur_targets[:, : self.num_arm_joints] = arm_smoothed

        # --- 手指: 增量控制 ---
        finger_actions = self.actions[:, self.num_arm_joints :]
        finger_delta = finger_actions * self.max_finger_delta
        self.cur_targets[:, self.num_arm_joints :] = self.cur_targets[:, self.num_arm_joints :] + finger_delta

        # --- 安全裁剪 ---
        self.cur_targets = math_utils.saturate(
            self.cur_targets,
            self.joint_lower_limits.unsqueeze(0),
            self.joint_upper_limits.unsqueeze(0),
        )

        self.franka.set_joint_position_target(self.cur_targets, joint_ids=self.controlled_joint_indices)

    def _get_rewards(self) -> torch.Tensor:
        self._reward_print_count += 1
        
        if not hasattr(self, '_joint_monitor_counter'):
            self._joint_monitor_counter = 0
        self._joint_monitor_counter += 1

        if self.debug:
            try:
                import isaacsim.util.debug_draw._debug_draw as omni_debug_draw
                draw_interface = omni_debug_draw.acquire_debug_draw_interface()
                draw_interface.clear_lines()
                draw_interface.clear_points()
            except Exception:
                pass        
        from .mdp import rewards_optimized as reward_funcs
        from isaaclab.managers import SceneEntityCfg
        
        total_reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # euler_gate:门控因子,由下面 r_euler 块用"到 goal_quat 的夹角"算出真值。
        # 默认 1.0,防止 r_euler 块异常时后续 thumb/tip/success 引用到未定义变量。
        euler_gate = torch.ones(self.num_envs, device=self.device)
        try:
            drill_quat = self.drill.data.root_quat_w
            qw, qx, qy, qz = drill_quat[:, 0], drill_quat[:, 1], drill_quat[:, 2], drill_quat[:, 3]
            r_x = 2.0 * (qx * qz - qy * qw)
            r_y = 2.0 * (qy * qz + qw * qx)
            r_z = 1.0 - 2.0 * (qx * qx + qy * qy)
            curr_up_z = (
                self._up_axis[:, 0] * r_x +
                self._up_axis[:, 1] * r_y +
                self._up_axis[:, 2] * r_z
            )
            curr_up_z = torch.clamp(curr_up_z, -1.0, 1.0)
            curr_up_z = torch.nan_to_num(curr_up_z, nan=0.0)
            self._cached_upright_score = curr_up_z

            # 势函数塑形(γ=1):r = scale*(Φ_now - Φ_prev)。只奖励"朝直立的进展":
            # 站立(Φ≈1 不变)→0、扶正→正、被碰歪(Φ 下降)→负。Φ_prev 在 reset 时
            # 重置为新一局初始 Φ,消除站立/躺下的初始基线差。Φ_upright 取绝对直立度[0,1]。
            ORIENT_SCALE = 500.0
            phi_upright = curr_up_z.clamp(0.0, 1.0)
            r_upright = ORIENT_SCALE * (phi_upright - self._upright_phi_prev)
            self._upright_phi_prev = phi_upright.detach()
            self._reward_components["r_upright"] = r_upright
            total_reward += r_upright
        except Exception as e:
            import traceback
            print(f"[WARN] r_upright failed: {e}")
            traceback.print_exc()

        try:
            vid = self._drill_variant_indices.long().flatten()
            drill_quat = self.drill.data.root_quat_w
            drill_quat_norm = torch.nn.functional.normalize(drill_quat, dim=1)
            goal_quats = self._drill_goal_quat_per_env  # [num_envs, 4]，已在 reset 时固定

            # 当前与目标四元数夹角（已归一化）
            dot = (drill_quat_norm * goal_quats).sum(dim=1).abs()  # abs 处理 q 和 -q 同姿态
            dot = dot.clamp(-1.0, 1.0)
            angle = 2.0 * torch.acos(dot)

            # euler_gate:用"到 goal_quat 的夹角"做门控(替代原 upright_gate 的绝对直立度)。
            # 越接近目标朝向 → 门越大 → thumb/tip/success 放大越多,直接激励"转到目标朝向";
            # 且"直立但朝向错"也会被压低门,比纯直立度更贴合任务目标。
            _GATE_ANG_MIN = 0.1745   # 10°:此内满门=1
            _GATE_ANG_MAX = 2.0944   # 120°:此外降到下限 0.01
            _gate_raw = torch.clamp((_GATE_ANG_MAX - angle) / (_GATE_ANG_MAX - _GATE_ANG_MIN), 0.0, 1.0)
            euler_gate = torch.clamp(_gate_raw ** 2, 0.01, 1.0)

            euler_temp = 0.5
            # 绝对值奖励:离 goal_quat 越近给分越多(每步都给,不是势差)。
            r_euler = torch.exp(-angle / euler_temp) * 20

            self._reward_components["r_euler"] = r_euler
            total_reward += r_euler

            if not hasattr(self, "_euler_debug_counter"):
                self._euler_debug_counter = 0
            self._euler_debug_counter += 1
            # if self._euler_debug_counter % 100 == 1:
            #     import math
            #     print(f"\n[DEBUG euler] step={self._euler_debug_counter}")
            #     for i in range(min(3, self.num_envs)):
            #         vid_i = vid[i].item()
            #         vdata = self._variant_attrs[vid_i]
            #         goal_q = goal_quats[i].cpu().numpy()
            #         curr_q = drill_quat_norm[i].cpu().numpy()
            #         angle_i = angle[i].item()
            #         r_euler_i = r_euler[i].item()
                    # print(f"  env{i} variant={vdata['name']}")
                    # print(f"    goal_quat:    ({goal_q[0]:+.4f}, {goal_q[1]:+.4f}, {goal_q[2]:+.4f}, {goal_q[3]:+.4f})")
                    # print(f"    curr_quat:    ({curr_q[0]:+.4f}, {curr_q[1]:+.4f}, {curr_q[2]:+.4f}, {curr_q[3]:+.4f})")
                    # print(f"    angle={math.degrees(angle_i):.1f}deg, r_euler={r_euler_i:.3f}")
        except Exception as e:
            import traceback
            print(f"[WARN] r_euler failed: {e}")
            traceback.print_exc()

        try:
            thumb_distal_reward = reward_funcs.thumb_approach_reward(
                self,
                finger_scale=20,
                finger_temp=0.05,
                hand_cfg=SceneEntityCfg("franka"),
                drill_scale=(1,1.2,1),
                verbose=self.debug,
            )
            self._reward_components["thumb_distal"] = thumb_distal_reward * euler_gate
            total_reward += thumb_distal_reward  * euler_gate
        except Exception as e:
            import traceback
            print(f"[WARN] thumb_distal_reward failed: {e}")
            traceback.print_exc()

        try:

            tip_trigger = reward_funcs.tip_trigger_reward(
                self,
                finger_scale=20,
                finger_temp=0.05,
                hand_cfg=SceneEntityCfg("franka"),
                drill_scale=(1, 1.2, 1),
                verbose=self.debug,
            )
            self._reward_components["tip_trigger"] = tip_trigger  * euler_gate
            total_reward += tip_trigger * euler_gate
        except Exception as e:
            print(f"[WARN] tip_trigger_reward failed: {e}")

        try:
            approach_reward = reward_funcs.approach_reward_improved(
                self,
                finger_scale=20,
                finger_temp=0.05,
                hand_cfg=SceneEntityCfg("franka"),
                verbose=self.debug,
                drill_scale=(1,1.2,1),
            )
            self._reward_components["approach"] = approach_reward * 0.5
            total_reward += approach_reward *  0.5 
        except Exception as e:
            import traceback
            print(f"[WARN] approach_reward failed: {e}")
            traceback.print_exc()

        # try:
        #     success_reward = reward_funcs.success_reward(
        #         self,
        #         base_weight=200.0,       
        #         bonus_per_contact =10,
        #         success_contact_force_threshold=0.01,  # 接触力阈值
        #         drill_scale=(1, 1.2, 1),     # 电钻缩放
        #     )
        #     self._reward_components["success"] = success_reward * upright_gate
        #     total_reward += success_reward * upright_gate          
        # except Exception as e:
        #     print(f"[WARN] success_reward failed: {e}")

        # success_now reward
        try:
            instant_success_raw = self._check_success().float()
            instant_success = instant_success_raw * 100

            self._reward_components["success_now"] = instant_success * euler_gate
            total_reward += instant_success * euler_gate
        except Exception as e:
            import traceback
            print(f"[WARN] success_now instant reward failed: {e}")
            traceback.print_exc()

        try:
            drill_ang_vel = self.drill.data.root_ang_vel_w
            ang_vel_norm = torch.norm(drill_ang_vel, dim=1)  # [num_envs]
            max_ang_vel = 2.0  # 2 rad/s 以上开始惩罚
            ang_vel_penalty = torch.where(
                ang_vel_norm > max_ang_vel,
                -1.0 * (ang_vel_norm - max_ang_vel),
                torch.zeros_like(ang_vel_norm)
            )
            ang_vel_penalty = ang_vel_penalty * 10
            self._reward_components["drill_ang_vel_penalty"] = ang_vel_penalty
            # print(f"  ang_vel_penalty = {ang_vel_penalty.cpu().numpy()}")
            total_reward += ang_vel_penalty
        except Exception as e:
            import traceback
            print(f"[WARN] drill_ang_vel_penalty failed: {e}")
            traceback.print_exc()

        # try:
        #     contact_reward = reward_funcs.contact_reward_detailed(
        #         self,
        #         fingertip_contact_bonus=5, 
        #         other_link_contact_bonus=3,
        #         hand_base_contact_bonus=5,
        #         force_threshold=0.00001,
        #         dist_temp=0.03,
        #         drill_scale=(1,1.2,1),
        #     )
        #     self._reward_components["contact"] = contact_reward * upright_gate
        #     total_reward += contact_reward * upright_gate
        # except Exception as e:
        #     print(f"[WARN] contact_reward failed: {e}")



        # === 电钻线速度惩罚 ===
        try:
            drill_lin_vel = self.drill.data.root_lin_vel_w
            lin_vel_norm = torch.norm(drill_lin_vel, dim=1)
            max_lin_vel = 0.3  # 0.3 m/s 以上开始惩罚
            lin_vel_penalty = torch.where(
                lin_vel_norm > max_lin_vel,
                -1.0 * (lin_vel_norm - max_lin_vel),
                torch.zeros_like(lin_vel_norm)
            )
            self._reward_components["drill_lin_vel_penalty"] = lin_vel_penalty
            total_reward += lin_vel_penalty
        except Exception as e:
            import traceback
            print(f"[WARN] drill_lin_vel_penalty failed: {e}")
            traceback.print_exc()

        if hasattr(self, '_pending_nan_mask') and self._pending_nan_mask.any():
            nan_count = self._pending_nan_mask.sum().item()
            if not hasattr(self, '_total_nan_count'):
                self._total_nan_count = 0
            self._total_nan_count += nan_count

            nan_penalty = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)


        if "log" not in self.extras:
            self.extras["log"] = dict()
        
        for name, tensor in self._reward_components.items():
            if tensor is not None and tensor.numel() > 0:
                self.extras["log"][f"reward_{name}"] = tensor.mean()
        
        valid_clean = self._recent_success[:self._recent_filled]
        valid_total = self._recent_total[:self._recent_filled]
        self.extras["log"]["success_rate_recent"] = valid_clean.float().mean().item() if self._recent_filled > 0 else 0.0
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:


        # === NaN 自动检测和重置 ===
        hand_body_pos = self.franka.data.body_pos_w
        hand_nan_mask = torch.isnan(hand_body_pos).any(dim=2).any(dim=1)

        hand_body_vel = self.franka.data.body_lin_vel_w
        hand_vel_nan_mask = torch.isnan(hand_body_vel).any(dim=2).any(dim=1)

        drill_pos = self.drill.data.root_pos_w
        drill_nan_mask = torch.isnan(drill_pos).any(dim=1)

        drill_vel = self.drill.data.root_lin_vel_w
        drill_vel_nan_mask = torch.isnan(drill_vel).any(dim=1)

        joint_pos = self.franka.data.joint_pos
        joint_pos_nan_mask = torch.isnan(joint_pos).any(dim=1)

        joint_vel = self.franka.data.joint_vel
        joint_vel_nan_mask = torch.isnan(joint_vel).any(dim=1)

        joint_effort_nan_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if hasattr(self.franka.data, 'dof_effort'):
            joint_effort = self.franka.data.dof_effort
            joint_effort_nan_mask = torch.isnan(joint_effort).any(dim=1)

        proximal_joint_pos = joint_pos[:, self.finger_indices]
        proximal_lower = self.joint_lower_limits[self.num_arm_joints:]
        proximal_upper = self.joint_upper_limits[self.num_arm_joints:]
        eps = 2e-1
        proximal_out_of_limits = (proximal_joint_pos < proximal_lower - eps) | (proximal_joint_pos > proximal_upper + eps)
        proximal_limit_terminated = proximal_out_of_limits.any(dim=1)
        if proximal_limit_terminated.any():
            env_ids_out = proximal_limit_terminated.nonzero(as_tuple=True)[0]
            # print(f"[ProximalJointLimit] terminated envs: {env_ids_out.tolist()}")
            # print(f"  pos={proximal_joint_pos[env_ids_out].cpu().numpy()}")
            diff = (proximal_joint_pos[env_ids_out] - proximal_lower).clamp(max=0) + (proximal_joint_pos[env_ids_out] - proximal_upper).clamp(min=0)
            # print(f"  violation={diff.cpu().numpy()}")

        sensor_nan_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if hasattr(self, 'scene') and hasattr(self.scene, 'sensors'):
            for sensor_name, sensor in self.scene.sensors.items():
                for attr in ('force_matrix_w', 'net_forces_w', 'net_wrenches_w'):
                    if not hasattr(sensor.data, attr):
                        continue
                    data = sensor.data.__getattribute__(attr)
                    if data is None:
                        continue
                    # 统一处理：展平除 batch 维外的所有维度，检测是否有 NaN
                    flat = data.reshape(data.shape[0], -1)
                    sensor_nan_mask |= torch.isnan(flat).any(dim=1)

        total_nan_mask = (
            hand_nan_mask | hand_vel_nan_mask |
            drill_nan_mask | drill_vel_nan_mask |
            joint_pos_nan_mask | joint_vel_nan_mask | joint_effort_nan_mask |
            sensor_nan_mask
        )

        if not hasattr(self, '_pending_nan_mask'):
            self._pending_nan_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._pending_nan_mask = self._pending_nan_mask | total_nan_mask

        # if total_nan_mask.any():
        #     nan_envs = total_nan_mask.nonzero(as_tuple=True)[0]
        #     for i in nan_envs[:5]:  # only first 5 to avoid flooding
        #         reasons = []
        #         if hand_nan_mask[i]:      reasons.append("hand_pos")
        #         if hand_vel_nan_mask[i]:  reasons.append("hand_vel")
        #         if drill_nan_mask[i]:     reasons.append("drill_pos")
        #         if drill_vel_nan_mask[i]: reasons.append("drill_vel")
        #         if joint_pos_nan_mask[i]: reasons.append("joint_pos")
        #         if joint_vel_nan_mask[i]: reasons.append("joint_vel")
        #         if joint_effort_nan_mask[i]: reasons.append("joint_effort")
        #         if sensor_nan_mask[i]:    reasons.append("sensor")
        #         print(f"[NaN] env={i.item()} reasons={reasons}  "
        #               f"hand_pos={hand_body_pos[i].cpu().numpy()}  "
        #               f"drill_pos={drill_pos[i].cpu().numpy()}  "
        #               f"joint_pos={joint_pos[i].cpu().numpy()}")

        failure_terminated = self._check_failure()

        active_mask = ~failure_terminated & ~self._pending_nan_mask
        success_now = self._check_success() & active_mask
        idx = self._success_window_idx.long()  # [num_envs]
        self._success_window.scatter_(1, idx.unsqueeze(1), success_now.unsqueeze(1))
        self._success_window_idx = (self._success_window_idx + 1) % self._success_window_size
        self._cached_instant_success = success_now  

        lenient_success = self._success_window.sum(dim=1) >= 20
        time_outs = self.episode_length_buf >= self.max_episode_length
        terminated_nan_mask = self._pending_nan_mask
        terminated = failure_terminated | terminated_nan_mask | proximal_limit_terminated 
        truncated = time_outs 

        self._cached_failure_mask = failure_terminated
        self._cached_lenient_success = lenient_success
        self._normal_timeout_mask = time_outs & ~failure_terminated & ~terminated_nan_mask & ~proximal_limit_terminated

        return terminated, truncated

    def _setup_physics_scene(self):
        """为 Inspire Hand 的 link 添加物理 API，启用 contact sensor"""
        try:
            from pxr import Usd, UsdPhysics, UsdGeom
            import omni.usd
            
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print("[WARN] _setup_physics_scene: 无法获取 USD stage")
                return
            
            franka_asset = self.scene["franka"]
            franka_prim = franka_asset._prim  # 内部 _prim 属性
            if franka_prim is None:
                print("[WARN] _setup_physics_scene: franka._prim 为 None，尝试其他方式")
                # 尝试通过 body_names 获取路径
                body_pos = franka_asset.data.body_pos_w
                if body_pos is not None and body_pos.shape[0] > 0:
                    print("[DEBUG] _setup_physics_scene: 可以通过 body_pos 访问 franka")
                return
            
            franka_path = str(franka_prim.GetPath())
            print(f"[DEBUG] _setup_physics_scene: franka_path = {franka_path}")
            
            # 手部 link 名称列表
            hand_body_names = [
                "R_hand_base_link",
                "R_index_proximal", "R_index_intermediate",
                "R_middle_proximal", "R_middle_intermediate",
                "R_ring_proximal", "R_ring_intermediate",
                "R_pinky_proximal", "R_pinky_intermediate",
                "R_thumb_proximal_base", "R_thumb_proximal", "R_thumb_intermediate",
            ]
            
            # 只保留在 body_names 中存在的 link
            available_bodies = [name for name in hand_body_names if name in self.franka.body_names]
            
            # 为每个 link 添加物理 API
            for body_name in available_bodies:
                prim_path = f"{franka_path}/{body_name}"
                prim = stage.GetPrimAtPath(prim_path)
                
                if not prim.IsValid():
                    continue
                
                # 检查是否已有 RigidBodyAPI
                has_rb = prim.HasAPI(UsdPhysics.RigidBodyAPI)
                has_collider = prim.HasAPI(UsdPhysics.CollisionAPI)
                
                if not has_rb:
                    # 添加 RigidBodyAPI
                    rb_api = UsdPhysics.RigidBodyAPI.Apply(prim)
                    rb_api.CreateLinearVelocityAttr()
                    rb_api.CreateAngularVelocityAttr()
                
                if not has_collider:
                    collider_api = UsdPhysics.CollisionAPI.Apply(prim)
                # 检查并添加 ContactReporterAPI（contact sensor 需要这个）
                if not prim.HasAPI(UsdPhysics.ContactReporterAPI):
                    contact_api = UsdPhysics.ContactReporterAPI.Apply(prim)

            # 强制刷新 stage
            stage.Save()
            
        except ImportError as e:
            print(f"[WARN] _setup_hand_contact_sensors: 无法导入 pxr 模块: {e}")
        except Exception as e:
            import traceback
            traceback.print_exc()

    def _setup_scene(self):
        """设置场景 - DirectRLEnv 版本"""
        super()._setup_scene()

        try:
            from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Sdf
            import omni.usd
            
            stage = omni.usd.get_context().get_stage()
            
            # === 1. 配置手部关节 (Inspire Hand on fr3) ===
            inspire_hand_path = "/World/envs/env_0/fr3_inspire_hand"
            joints_path = f"{inspire_hand_path}/joints"
            
            # 需要禁用的中间关节 Mimic Joint
            intermediate_joints = [
                "R_index_intermediate_joint",
                "R_middle_intermediate_joint",
                "R_pinky_intermediate_joint",
                "R_ring_intermediate_joint",
                "R_thumb_intermediate_joint",
                "R_thumb_distal_joint",
            ]
            
            # 需要配置 Drive 的近端关节
            proximal_joints = [
                "R_index_proximal_joint",
                "R_middle_proximal_joint",
                "R_pinky_proximal_joint",
                "R_ring_proximal_joint",
                "R_thumb_proximal_yaw_joint",
                "R_thumb_proximal_pitch_joint",
            ]

            fr3_joints_path = "/World/envs/env_0/fr3_inspire_hand/joints"
            
            fr3_joint_names = [
                "fr3_joint1",
                "fr3_joint2", 
                "fr3_joint3",
                "fr3_joint4",
                "fr3_joint5",
                "fr3_joint6",
                "fr3_joint7",
            ]
            


        except Exception as e:
            import traceback
            traceback.print_exc()

    def _compute_link_body_dists(self) -> torch.Tensor:
        """Compute hand-link → drill-body surface distances for all envs.

        Returns:
            link_body_dists: [num_envs, num_obs_links] per-env per-link distance to nearest body surface.
        """
        hand_body_pos = self.franka.data.body_pos_w  # [num_envs, num_bodies, 3]
        hand_link_mask = self._hand_link_obs_mask   # [num_bodies] bool
        hand_pos_filtered = hand_body_pos[:, hand_link_mask, :]  # [num_envs, num_obs_links, 3]
        num_obs_links = hand_pos_filtered.shape[1]

        drill_pos_w = self.drill.data.root_pos_w
        drill_quat_norm = torch.nn.functional.normalize(self.drill.data.root_quat_w, dim=1)
        R_drill = math_utils.matrix_from_quat(drill_quat_norm)  # [num_envs, 3, 3]

        link_body_dists = torch.zeros(self.num_envs, num_obs_links, device=self.device)

        unique_vids = self._drill_variant_indices.unique().tolist()
        for vid in unique_vids:
            env_mask = (self._drill_variant_indices == vid)
            env_idx_list = env_mask.nonzero(as_tuple=True)[0]
            if len(env_idx_list) == 0:
                continue

            vdata = self._variant_attrs[vid]
            body_pts = vdata.get("body_surface_points", None)
            if body_pts is None or len(body_pts) == 0:
                continue

            scale = vdata["scale"]
            scaled_pts = body_pts * scale  # [N, 3]

            R_v = R_drill[env_idx_list]        # [n, 3, 3]
            pos_v = drill_pos_w[env_idx_list]  # [n, 3]

            scaled_pts_expanded = scaled_pts.T.unsqueeze(0).expand(len(env_idx_list), 3, -1)  # [n, 3, N]
            pts_w = torch.bmm(R_v, scaled_pts_expanded).transpose(1, 2) + pos_v.unsqueeze(1)  # [n, N, 3]

            hand_pts = hand_pos_filtered[env_idx_list]  # [n, num_links, 3]
            dists = torch.cdist(hand_pts, pts_w)  # [n, num_links, N]
            link_body_dists[env_idx_list] = dists.min(dim=2).values  # [n, num_obs_links]

        self._cached_link_body_dists = link_body_dists  # [num_envs, num_obs_links]
        return link_body_dists

    def _get_camera_observations(self) -> dict[str, torch.Tensor]:
        """Return camera observations using the official IsaacLab sensor access pattern."""
        camera_obs: dict[str, torch.Tensor] = {}
        if not hasattr(self, "scene") or not hasattr(self.scene, "sensors"):
            print("[CAM-TRACE][env] scene or scene.sensors missing", flush=True)
            return camera_obs

        for camera_name in ("cam1", "cam2"):
            try:
                sensor = self.scene.sensors.get(camera_name)
                if sensor is None:
                    continue
                data_types = getattr(sensor.cfg, "data_types", [])
                if not data_types:
                    continue
                data_type = "rgb" if "rgb" in data_types else data_types[0]
                camera_tensor = sensor.data.output[data_type].clone()
                if data_type in {"depth", "distance_to_image_plane", "distance_to_camera"}:
                    camera_tensor[torch.isinf(camera_tensor)] = 0
                    camera_tensor = torch.nan_to_num(camera_tensor, nan=0.0, posinf=0.0, neginf=0.0)
                camera_obs[camera_name] = camera_tensor
            except Exception as exc:
                import traceback
                print(f"[CAM-TRACE][env][ERROR] failed while reading {camera_name}: {exc}", flush=True)
                traceback.print_exc()
                raise
        return camera_obs

    def _get_observations(self) -> torch.Tensor:
        joint_pos = self.franka.data.joint_pos
        joint_vel = self.franka.data.joint_vel

        obs_list = []

        proximal_joint_indices = self.controlled_joint_indices.tolist()

        obs_list.append(joint_pos[:, proximal_joint_indices])
        obs_list.append(joint_vel[:, proximal_joint_indices])

        drill_pos_w = self.drill.data.root_pos_w  # 世界坐标（用于距离计算）
        drill_quat = self.drill.data.root_quat_w
        drill_lin_vel = self.drill.data.root_lin_vel_w
        drill_ang_vel = self.drill.data.root_ang_vel_w

        env_origins = self.scene.env_origins  # [num_envs, 3]
        drill_pos = drill_pos_w - env_origins  # [num_envs, 3]

        variant_ids = self._drill_variant_indices.long()  # [num_envs], values are global {0,1,2}
        variant_onehot = torch.zeros(self.num_envs, self.total_num_variants, device=self.device)
        variant_onehot.scatter_(1, variant_ids.unsqueeze(1), 1.0)  # [num_envs, total_num_variants]


        quat_mags = torch.norm(drill_quat, dim=1)
        if (quat_mags < 0.9).any() or (quat_mags > 1.1).any():
            print(f"[WARN] drill_quat invalid magnitudes: min={quat_mags.min():.4f}, max={quat_mags.max():.4f}")
        drill_quat_norm = torch.nn.functional.normalize(drill_quat, dim=1)
        quat_norm_check = torch.norm(drill_quat_norm, dim=1)
        if (torch.abs(quat_norm_check - 1.0) > 0.01).any():
            print("[WARN] drill_quat normalization issue, forcing re-normalization")
            drill_quat_norm = torch.nn.functional.normalize(drill_quat, dim=1)

        R_drill = math_utils.matrix_from_quat(drill_quat_norm)  # [num_envs, 3, 3]

        index_body_idx = self._body_name_to_idx.get("R_index_intermediate", -1)
        if index_body_idx >= 0:
            index_pos = self.franka.data.body_pos_w[:, index_body_idx, :]  # [num_envs, 3]
        else:
            index_pos = torch.zeros(self.num_envs, 3, device=self.device)
        trigger1_scaled = self._trigger1_offset * self._drill_scale  # [num_envs, 3]
        trigger1_world = torch.bmm(R_drill, trigger1_scaled.unsqueeze(-1)).squeeze(-1) + drill_pos_w  # [num_envs, 3]
        tip_trigger_dist = torch.norm(index_pos - trigger1_world, dim=1, keepdim=True)  # [num_envs, 1]

        thumb_body_idx = self._thumb_distal_body_idx
        if thumb_body_idx >= 0:
            thumb_pos = self.franka.data.body_pos_w[:, thumb_body_idx, :]  # [num_envs, 3]
        else:
            thumb_pos = torch.zeros(self.num_envs, 3, device=self.device)
        thumb_target_scaled = self._thumb_target_local * self._drill_scale  # [num_envs, 3]
        thumb_target_world = torch.bmm(R_drill, thumb_target_scaled.unsqueeze(-1)).squeeze(-1) + drill_pos_w
        thumb_target_dist = torch.norm(thumb_pos - thumb_target_world, dim=1, keepdim=True)  # [num_envs, 1]

        base_body_idx = self._body_name_to_idx["R_hand_base_link"]
        hand_base_pos_w = self.franka.data.body_pos_w[:, base_body_idx, :]
        hand_base_quat_w = self.franka.data.body_quat_w[:, base_body_idx, :]
        hand_base_pos_env = hand_base_pos_w - env_origins  # [num_envs, 3]
        hand_base_quat_env = torch.nn.functional.normalize(hand_base_quat_w, dim=1)  # [num_envs, 4]

        link_body_dists = self._compute_link_body_dists()

        num_surface_pts = 512
        num_envs = self.num_envs
        surface_pts_world = torch.zeros(
            num_envs, num_surface_pts, 3, dtype=torch.float32, device=self.device
        )
        unique_vids = variant_ids.unique().tolist()
        for vid in unique_vids:
            env_mask = variant_ids == vid
            env_idx_list = env_mask.nonzero(as_tuple=True)[0]
            if len(env_idx_list) == 0:
                continue
            vdata = self._variant_attrs[vid]
            raw_pts = vdata.get("mesh_points")
            if raw_pts is None or len(raw_pts) == 0:
                continue
            pts_local = self._sample_surface_points_uniform(raw_pts, num_surface_pts)
            scale_v = vdata["scale"]  # [3]
            pts_scaled = pts_local * scale_v
            R_v = R_drill[env_idx_list]      # [n, 3, 3]
            pos_v = drill_pos_w[env_idx_list]  # [n, 3]
            pts_exp = pts_scaled.T.unsqueeze(0).expand(len(env_idx_list), 3, -1)  # [n, 3, 512]
            pts_w = torch.bmm(R_v, pts_exp).permute(0, 2, 1) + pos_v.unsqueeze(1)  # [n, 512, 3]
            surface_pts_world[env_idx_list] = pts_w

        surface_pts_flat = (surface_pts_world - env_origins.unsqueeze(1)).reshape(num_envs, -1)

        if self.debug:
            print(f"\n[SURFACE PTS DEBUG step={getattr(self, '_surface_pts_debug_counter', 0)}]")
            for vid in unique_vids:
                env_of_vid = (variant_ids == vid).nonzero(as_tuple=True)[0]
                if len(env_of_vid) == 0:
                    continue
                e0 = env_of_vid[0].item()
                pts_w = surface_pts_world[e0]
                drill_w = drill_pos_w[e0]
                print(f"  VID{vid} env{e0}: drill_world=[{drill_w[0]:.3f},{drill_w[1]:.3f},{drill_w[2]:.3f}]")
                print(f"  VID{vid} pts_world: x=[{pts_w[:,0].min():.3f},{pts_w[:,0].max():.3f}] "
                      f"y=[{pts_w[:,1].min():.3f},{pts_w[:,1].max():.3f}] "
                      f"z=[{pts_w[:,2].min():.3f},{pts_w[:,2].max():.3f}]")
        if hasattr(self, '_surface_pts_debug_counter'):
            self._surface_pts_debug_counter += 1
        else:
            self._surface_pts_debug_counter = 1

        # ── Visualize surface points in IsaacSim ──
        # try:
        #     import isaacsim.util.debug_draw._debug_draw as omni_debug_draw
        #     import numpy as np
        #     draw = omni_debug_draw.acquire_debug_draw_interface()
        #     num_vis = min(3, num_envs)
        #     variant_vis_colors = [
        #         [1.0, 0.4, 0.4, 1.0],
        #         [0.4, 1.0, 0.4, 1.0],
        #         [1.0, 0.9, 0.2, 1.0],
        #     ]
        #     vis_pos, vis_colors, vis_sizes = [], [], []
        #     for eidx in range(num_vis):
        #         vid = variant_ids[eidx].item()
        #         color = variant_vis_colors[vid % len(variant_vis_colors)]
        #         pts_np = surface_pts_world[eidx].cpu().numpy()
        #         for pt in pts_np.tolist():
        #             vis_pos.append(pt)
        #             vis_colors.append(color)
        #             vis_sizes.append(5.0)
        #     if vis_pos:
        #         draw.draw_points(vis_pos, vis_colors, vis_sizes)
        # except Exception:
        #     pass  # Silently skip if debug draw unavailable

        drill_up_axis = torch.bmm(R_drill, self._up_axis.unsqueeze(-1)).squeeze(-1)  # R @ v_local = v_world

        contact_forces_raw = self._get_contact_forces_obs()
        contact_forces = torch.log1p(contact_forces_raw) / torch.log1p(torch.tensor(50.0, device=self.device))  # 归一化
        goal_quat = self._drill_goal_quat_per_env  # [num_envs, 4]

        obs_parts = [
            joint_pos[:, proximal_joint_indices],               # 关节位置
            joint_vel[:, proximal_joint_indices],
            drill_pos,                                          # drill 位置 (相对于 env_origin)
            drill_lin_vel,                                      # drill 线速度 (世界系)
            drill_ang_vel,                                     # drill 角速度 (世界系)
            drill_up_axis,                                     # drill up_axis 世界投影 (3维)
            tip_trigger_dist,                                  # 食指末端 → trigger
            thumb_target_dist,                                 # 拇指末端 → thumb target
            link_body_dists,                                   # 各 link → drill body 最近距离
            hand_base_pos_env,                                 # hand base 位置 (相对于 env_origin)
            hand_base_quat_env,                                # hand base 四元数 (wxyz) 归一化
            goal_quat,                                         # 每个 env 的目标四元数
            contact_forces,                                    # 各 contact sensor 的力大小 (归一化)
        ]
        observations = torch.cat(obs_parts, dim=1)

        nan_mask = torch.isnan(observations)
        if nan_mask.any():
            nan_count = nan_mask.sum().item()
            observations = torch.where(nan_mask, torch.zeros_like(observations), observations)
            nan_dims = nan_mask.any(dim=0).nonzero(as_tuple=True)[0]
            obs_nan_mask = nan_mask.any(dim=1)
            if not hasattr(self, '_nan_env_mask'):
                self._nan_env_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._nan_env_mask[:] = self._nan_env_mask | obs_nan_mask
            self._pending_nan_mask = self._nan_env_mask.clone()

        if not hasattr(self, "_obs_dim_logged"):
            print(f"[OBS DIM] Total observation dim = {observations.shape[1]}")
            self._obs_dim_logged = True

        self._obs_print_counter += 1
        variant_names = [v.name for v in self.drill_variants]
        # for i in range(min(10, self.num_envs)):
            # vid = variant_ids[i].item()
            # o = observations[i]
            # print(f"[env{i}] variant_id={vid} [{variant_names[vid] if vid < len(variant_names) else '?'}] obs_dim={o.shape[0]}")
            # print(f"  joint_pos[0:13]    = {o[:13].cpu().numpy()}")
            # print(f"  joint_vel[13:26]   = {o[13:26].cpu().numpy()}")
            # print(f"  drill_pos[26:29]   = {o[26:29].cpu().numpy()}")
            # print(f"  drill_lin_vel[29:32] = {o[29:32].cpu().numpy()}")
            # print(f"  drill_ang_vel[32:35] = {o[32:35].cpu().numpy()}")
            # print(f"  drill_up_axis[35:38] = {o[35:38].cpu().numpy()}")
            # print(f"  tip_trigger[38]     = {o[38]:.4f}")
            # print(f"  thumb_target[39]    = {o[39]:.4f}")
            # print(f"  link_dists[40:47]   = {o[40:47].cpu().numpy()}")
            # print(f"  hand_base_pos[47:50] = ({o[47]:.4f}, {o[48]:.4f}, {o[49]:.4f})")
            # print(f"  hand_base_quat[50:54] = ({o[50]:.4f}, {o[51]:.4f}, {o[52]:.4f}, {o[53]:.4f})")
            # print(f"  goal_quat[54:58]      = ({o[54]:.4f}, {o[55]:.4f}, {o[56]:.4f}, {o[57]:.4f})")
            # print(f"  contact_forces[58:71] = {o[58:71].cpu().numpy()}")

        return {"policy": observations}

    def _debug_action_manager(self):

        print(f"\n动作空间 (action_space): {self.cfg.action_space}")
        
        num_joints = len(self.franka.joint_names)


    def _get_contact_forces(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 5, device=self.device)

    def _get_all_contact_forces(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 21, device=self.device)

    def _get_total_contact_force(self) -> torch.Tensor:

        return self._get_contact_forces().sum(dim=1)

    def _get_contact_forces_obs(self) -> torch.Tensor:
        """获取所有 contact sensor 的力大小，用于观测。

        Returns:
            contact_forces: [num_envs, num_sensors] 各 sensor 检测到的力大小 (标量)
        """
        all_hand_sensors = [
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

        num_sensors = len(all_hand_sensors)
        contact_forces = torch.zeros(self.num_envs, num_sensors, device=self.device)

        try:
            if hasattr(self, 'scene') and hasattr(self.scene, 'sensors'):
                for i, sensor_name in enumerate(all_hand_sensors):
                    if sensor_name in self.scene.sensors:
                        sensor = self.scene.sensors[sensor_name]
                        force_matrix = sensor.data.force_matrix_w
                        if force_matrix is not None:
                            force_matrix = torch.nan_to_num(force_matrix, nan=0.0, posinf=0.0, neginf=0.0)
                            force_matrix = torch.clamp(force_matrix, -50.0, 50.0)
                            total_force = force_matrix.sum(dim=(1, 2))  # [num_envs, 3]
                            force_mag = torch.norm(total_force, dim=1)  # [num_envs]
                        else:
                            net_forces = sensor.data.net_forces_w
                            if net_forces is not None:
                                force_mag = torch.norm(net_forces, dim=1)
                            else:
                                force_mag = torch.zeros(self.num_envs, device=self.device)
                        contact_forces[:, i] = force_mag
        except Exception:
            pass

        return contact_forces  

    def _get_hand_palm_pos(self) -> torch.Tensor:
        return self.franka.data.root_pos_w

    def _get_wrist_pose_in_drill(self) -> tuple[torch.Tensor, torch.Tensor]:

        hand_pos_w = self.franka.data.root_pos_w
        hand_quat_w = self.franka.data.root_quat_w
        drill_pos_w = self.drill.data.root_pos_w
        drill_quat_w = self.drill.data.root_quat_w

        delta = hand_pos_w - drill_pos_w

        qw, qx, qy, qz = drill_quat_w[:, 0], drill_quat_w[:, 1], drill_quat_w[:, 2], drill_quat_w[:, 3]
        dx, dy, dz = delta[:, 0], delta[:, 1], delta[:, 2]

        tw = -qx * dx - qy * dy - qz * dz
        tx = qw * dx + qy * dz - qz * dy
        ty = qw * dy - qx * dz + qz * dx
        tz = qw * dz + qx * dy - qy * dx

        wx = tw * qx + tx * qw + ty * qz - tz * qy
        wy = tw * qy - tx * qz + ty * qw + tz * qx
        wz = tw * qz + tx * qy - ty * qx + tz * qw

        wrist_pos = torch.stack([wx, wy, wz], dim=1)

        wrist_quat = math_utils.quat_mul(math_utils.quat_conjugate(drill_quat_w), hand_quat_w)

        return wrist_pos, wrist_quat

    def _compute_success_hold_reward(self, hold_threshold: int, hold_reward: float) -> torch.Tensor:
        recent_success_counts = self._success_window.sum(dim=1) 
        return (recent_success_counts >= hold_threshold).float() * hold_reward

    def _check_success(self) -> torch.Tensor:
        drill_pos = self.drill.data.root_pos_w
        z_diff = drill_pos[:, 2] - self.initial_drill_pos[:, 2]
        height_success = z_diff > self.lift_z_threshold
        thumb_near_target = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        thumb_success_dist = 0.03  
        try:
            ti = self._thumb_distal_body_idx
            if ti >= 0:
                thumb_pos = self.franka.data.body_pos_w[:, ti, :]  

                thumb_target_scaled = self._thumb_target_local * self._drill_scale  
                drill_quat = self.drill.data.root_quat_w
                w, x, y, z = drill_quat[:, 0], drill_quat[:, 1], drill_quat[:, 2], drill_quat[:, 3]
                norm = torch.sqrt(w*w + x*x + y*y + z*z + 1e-8)
                w, x, y, z = w/norm, x/norm, y/norm, z/norm
                Rt = torch.zeros((self.num_envs, 3, 3), device=self.device)
                Rt[:, 0, 0] = 1 - 2*(y*y + z*z); Rt[:, 0, 1] = 2*(x*y - w*z); Rt[:, 0, 2] = 2*(x*z + w*y)
                Rt[:, 1, 0] = 2*(x*y + w*z); Rt[:, 1, 1] = 1 - 2*(x*x + z*z); Rt[:, 1, 2] = 2*(y*z - w*x)
                Rt[:, 2, 0] = 2*(x*z - w*y); Rt[:, 2, 1] = 2*(y*z + w*x); Rt[:, 2, 2] = 1 - 2*(x*x + y*y)

                thumb_target_world = torch.bmm(
                    Rt, thumb_target_scaled.unsqueeze(-1)
                ).squeeze(-1) + drill_pos  # [num_envs, 3]

                thumb_dist = torch.norm(thumb_pos - thumb_target_world, dim=1)
                thumb_near_target = thumb_dist < thumb_success_dist
        except Exception:
            thumb_near_target = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        finger_near_trigger = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        try:
            body_names = self.franka.data.body_names
            if "R_index_intermediate" in body_names:
                index_idx = body_names.index("R_index_intermediate")
                index_pos = self.franka.data.body_pos_w[:, index_idx, :]  # [num_envs, 3]

                scale_tensor = self._drill_scale  # [num_envs, 3]
                trigger_offset = self._trigger1_offset  # [num_envs, 3]
                trigger_offset = trigger_offset * scale_tensor  # broadcast [num_envs, 3]

                drill_quat = self.drill.data.root_quat_w
                w, x, y, z = drill_quat[:, 0], drill_quat[:, 1], drill_quat[:, 2], drill_quat[:, 3]
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
                    R, trigger_offset.unsqueeze(-1)
                ).squeeze(-1) + drill_pos  # [num_envs, 3]

                dist = torch.norm(index_pos - trigger_world, dim=1)  # [num_envs]
                finger_near_trigger = dist < 0.03

        except Exception as e:
            finger_near_trigger = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        contact_ok = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        multi_contact_ok = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        contact_force_threshold = 0.1  
        all_hand_sensors = [
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

        try:
            if hasattr(self, 'scene') and hasattr(self.scene, 'sensors'):
                contact_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
                for sensor_name in all_hand_sensors:
                    if sensor_name in self.scene.sensors:
                        sensor = self.scene.sensors[sensor_name]
                        force_matrix = sensor.data.force_matrix_w
                        if force_matrix is not None:
                            force_matrix = torch.nan_to_num(force_matrix, nan=0.0, posinf=0.0, neginf=0.0)
                            force_matrix = torch.clamp(force_matrix, -50.0, 50.0)
                            force = force_matrix.sum(dim=(1, 2))
                        else:
                            force = sensor.data.net_forces_w.sum(dim=1)
                        force_mag = torch.norm(force, dim=1)
                        force_mag = torch.nan_to_num(force_mag, nan=0.0)
                        has_contact = force_mag > contact_force_threshold
                        contact_count += has_contact.float()
                        if sensor_name == "contact_index_intermediate":
                            contact_ok = has_contact
                multi_contact_ok = contact_count >= 7
        except Exception as e:
            contact_ok = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            multi_contact_ok = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        success = finger_near_trigger & multi_contact_ok & thumb_near_target
        # Debug 输出
        if self.debug and hasattr(self, '_success_debug_counter'):
            self._success_debug_counter += 1
            if self._success_debug_counter % 10 == 0:
                print(
                    f"[SUCCESS CHECK] "
                    f"finger_near={finger_near_trigger.float().mean().item()*100:.1f}% | "
                    f"height={height_success.float().mean().item()*100:.1f}% | "
                    f"contact_ok={contact_ok.float().mean().item()*100:.1f}% | "
                    f"multi_contact={multi_contact_ok.float().mean().item()*100:.1f}% | "
                    f"final_success={success.float().mean().item()*100:.1f}%"
                )
        elif self.debug and not hasattr(self, '_success_debug_counter'):
            self._success_debug_counter = 0

        return success

    def _check_failure(self) -> torch.Tensor:
            drill_pos = self.drill.data.root_pos_w
            drill_quat = self.drill.data.root_quat_w  
            hand_body_pos = self.franka.data.body_pos_w
            body_names = self.franka.body_names

            qw, qx, qy, qz = drill_quat[:, 0], drill_quat[:, 1], drill_quat[:, 2], drill_quat[:, 3]
            r_x = 2.0 * (qx * qz - qy * qw)
            r_y = 2.0 * (qy * qz + qw * qx)
            r_z = 1.0 - 2.0 * (qx * qx + qy * qy)
            curr_up_z = (
                self._up_axis[:, 0] * r_x +
                self._up_axis[:, 1] * r_y +
                self._up_axis[:, 2] * r_z
            )

            drill_flipped = curr_up_z < 0.6

            if hasattr(self, 'initial_drill_pos'):
                drill_xy_diff = drill_pos[:, :2] - self.initial_drill_pos[:, :2]
                drill_xy_distance = torch.norm(drill_xy_diff, dim=1)
                drill_knocked_away = drill_xy_distance > 0.5
                if self.debug and drill_knocked_away.any():
                    print(f"[DEBUG drill_knocked_away] n={drill_knocked_away.sum()}, xy_dist={drill_xy_distance[drill_knocked_away][:5].cpu().numpy()}")
            else:
                drill_knocked_away = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

            failure = (
                # drill_flipped |
                drill_knocked_away
            )

            active_names = _get_active_failure_names()
            newly_active = set()
            if "drill_knocked_away" in active_names and drill_knocked_away.any():
                newly_active.add("drill_knocked_away")
            if "drill_flipped" in active_names and drill_flipped.any():
                newly_active.add("drill_flipped")

            if newly_active:
                if not hasattr(self, '_active_failure_reasons'):
                    self._active_failure_reasons = set()
                self._active_failure_reasons |= newly_active

            if "drill_knocked_away" in active_names:
                self._cached_knocked_away_mask = drill_knocked_away.clone()
            if "drill_flipped" in active_names:
                self._cached_drill_flipped_mask = drill_flipped.clone()

            return failure
    def _compute_reward_breakdown(self) -> dict:
        breakdown = {}

        if hasattr(self, '_reward_components'):
            for name, reward_tensor in self._reward_components.items():
                if reward_tensor is not None and reward_tensor.numel() > 0:
                    breakdown[name] = reward_tensor[0].item()
                    breakdown[f"{name}_mean"] = reward_tensor.mean().item()

        return breakdown

    def get_failure_stats(self) -> dict:
        total = self._failure_stats["total"]
        active = getattr(self, '_active_failure_reasons', set())

        result = {
            "total": total,
            "drill_knocked_away": self._failure_stats.get("drill_knocked_away", 0) if "drill_knocked_away" in active else None,
            "physics_nan": self._failure_stats["physics_nan"],
            "normal_total": self._failure_stats["normal_total"],
            "lenient_success_count": self._failure_stats["lenient_success_count"],
            "failure_count": total - self._failure_stats["normal_total"] - self._failure_stats["physics_nan"],
        }

        return result

    def reset_failure_stats(self):
        self._failure_stats = {
            "physics_nan": 0,
            "drill_knocked_away": 0,
            "normal_total": 0,
            "lenient_success_count": 0,
            "total": 0,
        }
        if hasattr(self, '_active_failure_reasons'):
            self._active_failure_reasons = set()

    def _attach_hand_to_franka(self):
        pass  


    def _write_all_drill_initial_poses(self):

        all_env_ids = torch.arange(self.num_envs, device=self.device)
        env_origins = self.scene.env_origins  # [num_envs, 3]

        for v in self.drill_variants:
            vid = v.variant_index  # global index
            v_pos = torch.tensor(v.initial_pos_list[0], dtype=torch.float32, device=self.device)
            v_rot = torch.tensor(v.initial_rot_list[0], dtype=torch.float32, device=self.device)
            vids = (self._drill_variant_indices == vid).nonzero(as_tuple=True)[0]
            if len(vids) == 0:
                continue
            pos_world = v_pos.unsqueeze(0).expand(len(vids), -1) + env_origins[vids]
            quat = v_rot.unsqueeze(0).expand(len(vids), -1)
            state = torch.cat([pos_world, quat, torch.zeros(len(vids), 6, device=self.device)], dim=1)
            self._drill.write_root_state_to_sim(state, env_ids=vids)

    def _reset_idx(self, env_ids: torch.Tensor):
        import random
        if len(env_ids) > 0:
            failure_reasons = getattr(
                self, '_cached_failure_mask',
                torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            )[env_ids].clone()
            nan_batch = self._pending_nan_mask[env_ids].clone()
            normal_timeout_batch = self._normal_timeout_mask[env_ids]

            cached_lenient_success = getattr(
                self, '_cached_lenient_success',
                torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            )[env_ids].clone()
            success_batch = cached_lenient_success

            n_batch = len(success_batch)
            indices = torch.arange(self._recent_idx, self._recent_idx + n_batch) % self._recent_success_buffer_size
            self._recent_success[indices] = success_batch.cpu()
            self._recent_total[indices] = True
            self._recent_idx += n_batch
            self._recent_filled = min(self._recent_filled + n_batch, self._recent_success_buffer_size)

            n = len(env_ids)
            n_failure = failure_reasons.sum().item()
            n_nan = nan_batch.sum().item()
            n_normal = normal_timeout_batch.sum().item()
            self._failure_stats["total"] += n
            self._failure_stats["physics_nan"] += n_nan
            self._failure_stats["normal_total"] += n_normal
            self._failure_stats["lenient_success_count"] += (success_batch & normal_timeout_batch).sum().item()

            reset_mask = failure_reasons | nan_batch
            knocked_away_cache = getattr(
                self, '_cached_knocked_away_mask',
                torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            )[env_ids]
            self._failure_stats["drill_knocked_away"] += (knocked_away_cache & reset_mask).sum().item()

        if hasattr(self, '_pending_nan_mask'):
            self._pending_nan_mask[:] = False

        if hasattr(self, 'contact_history_buffer'):
            self.contact_history_buffer[env_ids] = False

        if hasattr(self, 'joint_pos_history'):
            self.joint_pos_history[env_ids] = 0.0
        if hasattr(self, 'drill_pos_history'):
            self.drill_pos_history[env_ids] = 0.0
        if hasattr(self, 'finger_dist_history'):
            self.finger_dist_history[env_ids] = 0.0
        if hasattr(self, '_success_window_idx'):
            self._success_window_idx[env_ids] = 0
            self._success_window[env_ids] = False

        if self.num_drill_variants > 1:
            local_ids = env_ids % self.num_drill_variants
            variant_ids = self._active_indices_tensor[local_ids]
            self._drill_variant_indices[env_ids] = variant_ids

        super()._reset_idx(env_ids)

        _VA = self._variant_attrs
        n = len(env_ids)
        variant_ids = self._drill_variant_indices[env_ids]  

        _VA = self._variant_attrs
        nv = self.total_num_variants  

        vid_oh = torch.nn.functional.one_hot(variant_ids.long(), num_classes=nv).float()  # [n, nv]

        _flat_pos  = torch.zeros(nv * 3, device=self.device)
        _flat_rot  = torch.zeros(nv * 4, device=self.device)
        _flat_toff = torch.zeros(nv * 3, device=self.device)
        _flat_up   = torch.zeros(nv * 3, device=self.device)
        _flat_thumb = torch.zeros(nv * 3, device=self.device)
        _flat_bmin = torch.zeros(nv, device=self.device)
        _flat_bmax = torch.zeros(nv, device=self.device)
        _flat_scale = torch.zeros(nv * 3, device=self.device)

        for vid, vdata in _VA.items():
            _flat_toff[vid*3:(vid+1)*3] = vdata["trigger_offset"]
            _flat_up[vid*3:(vid+1)*3]   = vdata["up_axis"]
            _flat_thumb[vid*3:(vid+1)*3] = vdata["thumb_target_local"]
            _flat_bmin[vid] = vdata["body_mask_min"]
            _flat_bmax[vid] = vdata["body_mask_max"]
            _flat_scale[vid*3:(vid+1)*3] = vdata["scale"]
        self._trigger1_offset[env_ids] = vid_oh @ _flat_toff.view(nv, 3)
        self._up_axis[env_ids] = vid_oh @ _flat_up.view(nv, 3)
        self._thumb_target_local[env_ids] = vid_oh @ _flat_thumb.view(nv, 3)
        self._body_mask_y_min[env_ids] = vid_oh.float() @ _flat_bmin
        self._body_mask_y_max[env_ids] = vid_oh.float() @ _flat_bmax
        self._drill_scale[env_ids] = vid_oh @ _flat_scale.view(nv, 3)

        env_origins = self.scene.env_origins[env_ids]

        # --- overfit 测试钩子:外部(deploy)注入电钻初始位姿/goal 以回放 collect 记录的场景 ---
        # 设了 self._drill_pose_override_fn(env_ids, variant_ids)->(pos_local[n,3],quat[n,4],goal[n,4])
        # 就用注入值、跳过随机采样;否则走原随机分支(collect 用)。
        _override = getattr(self, "_drill_pose_override_fn", None)
        if _override is not None:
            _pl, _q, _g = _override(env_ids, variant_ids)
            drill_pos_world = _pl.to(self.device).float() + env_origins
            result_quats = _q.to(self.device).float()
            drill_initial_rot_tensor = result_quats
            self._drill_goal_quat_per_env[env_ids] = _g.to(self.device).float()
        else:
            drill_pos_rand = getattr(getattr(self.cfg, 'randomization', None), 'drill_pos_random_range', (0.0, 0.0, 0.0))
            drill_rot_rand = getattr(getattr(self.cfg, 'randomization', None), 'drill_rot_random_range', (0.0, 0.0, 0.0))

            drill_pos_x_delta = ((torch.rand(n, device=self.device) * 2.0 - 1.0) * drill_pos_rand[0])
            drill_pos_y_delta = ((torch.rand(n, device=self.device) * 2.0 - 1.0) * drill_pos_rand[1])
            drill_pos_z_delta = torch.zeros(n, device=self.device)

            from isaaclab.utils.math import quat_mul, quat_from_euler_xyz

            roll  = (torch.rand(n, device=self.device) * 2.0 - 1.0) * drill_rot_rand[0]
            pitch = (torch.rand(n, device=self.device) * 2.0 - 1.0) * drill_rot_rand[1]
            yaw   = (torch.rand(n, device=self.device) * 2.0 - 1.0) * drill_rot_rand[2]
            result_quats = quat_from_euler_xyz(roll, pitch, yaw)

            _VA = self._variant_attrs
            env_pos_list = []
            env_rot_list = []
            env_goal_quat_list = []  # per-env goal quat（固定在整个 episode 中）
            for i, eid in enumerate(env_ids.tolist()):
                vid = variant_ids[i].item()
                vdata = _VA[vid]
                pos_list = vdata["initial_pos_list"]   # [P, 3]
                num_poses = pos_list.shape[0]
                idx = torch.randint(0, num_poses, (1,), device=self.device).item()
                env_pos_list.append(pos_list[idx])
                env_rot_list.append(vdata["initial_rot_list"][idx])
                # 随机选择一个 goal，并固定在 env 级别
                goal_list = vdata["goal_rot_list"]  # [G, 4]
                goal_idx = torch.randint(0, goal_list.shape[0], (1,), device=self.device).item()
                env_goal_quat_list.append(goal_list[goal_idx])

            drill_initial_pos_tensor = torch.stack(env_pos_list, dim=0)  # [n, 3]
            drill_initial_rot_tensor = torch.stack(env_rot_list, dim=0)   # [n, 4]
            self._drill_goal_quat_per_env[env_ids] = torch.stack(env_goal_quat_list, dim=0)  # [n, 4]

            result_quats = quat_mul(result_quats, drill_initial_rot_tensor)

            drill_pos_world = drill_initial_pos_tensor + torch.stack([drill_pos_x_delta, drill_pos_y_delta, drill_pos_z_delta], dim=1)
            drill_pos_world += env_origins
        drill_root_state = torch.cat([drill_pos_world, result_quats, torch.zeros(n, 6, device=self.device)], dim=1)
        self._drill.write_root_state_to_sim(drill_root_state, env_ids=env_ids)

        if not hasattr(self, 'initial_drill_pos') or self.initial_drill_pos.shape[0] != self.num_envs:
            self.initial_drill_pos = torch.zeros(self.num_envs, 3, device=self.device)
            self.initial_drill_pos[:, 2] = 0.5
        self.initial_drill_pos[env_ids] = drill_root_state[:, :3].clone()

        if not hasattr(self, 'drill_initial_rot_tensor') or self.drill_initial_rot_tensor.shape[0] != self.num_envs:
            self.drill_initial_rot_tensor = torch.zeros(self.num_envs, 4, device=self.device)
            self.drill_initial_rot_tensor[:] = drill_initial_rot_tensor[0].unsqueeze(0)
        self.drill_initial_rot_tensor[env_ids] = result_quats.clone()

        hand_initial_pos_tensor = torch.as_tensor(self.cfg.hand_params.initial_pos, dtype=torch.float32, device=self.device)
        hand_initial_rot_tensor = torch.as_tensor(self.cfg.hand_params.initial_rot, dtype=torch.float32, device=self.device)
        hand_root_state = self.franka.data.default_root_state[env_ids].clone()
        hand_root_state[:, 0:3] = hand_initial_pos_tensor.unsqueeze(0).repeat(n, 1)
        hand_root_state[:, 3:7] = hand_initial_rot_tensor.unsqueeze(0).repeat(n, 1)
        hand_root_state[:, 0:3] += env_origins
        self.franka.write_root_pose_to_sim(hand_root_state[:, :7], env_ids=env_ids)
        self.franka.write_root_velocity_to_sim(hand_root_state[:, 7:], env_ids=env_ids)

        joint_pos = self.franka.data.default_joint_pos[env_ids].clone()
        joint_vel = self.franka.data.default_joint_vel[env_ids].clone()
        finger_initial_positions = {
            "R_index_proximal_joint": 0.0,
            "R_middle_proximal_joint": 0.0,
            "R_pinky_proximal_joint": 0.0,
            "R_ring_proximal_joint": 0.0,
            "R_thumb_proximal_yaw_joint": 7*np.pi/18,
            "R_thumb_proximal_pitch_joint": 0.0,
        }
        for joint_name, init_pos in finger_initial_positions.items():
            if joint_name in self.franka.joint_names:
                idx = self.franka.joint_names.index(joint_name)
                joint_pos[:, idx] = init_pos
        self.franka.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        if hasattr(self, 'cur_targets'):
            self.cur_targets[env_ids] = joint_pos[:, self.controlled_joint_indices].clone()

        drill_quat_reset = result_quats
        qw, qx, qy, qz = drill_quat_reset[:, 0], drill_quat_reset[:, 1], drill_quat_reset[:, 2], drill_quat_reset[:, 3]
        r_x = 2.0 * (qx * qz - qy * qw)
        r_y = 2.0 * (qy * qz + qw * qx)
        r_z = 1.0 - 2.0 * (qx * qx + qy * qy)
        initial_up = (
            self._up_axis[env_ids, 0] * r_x +
            self._up_axis[env_ids, 1] * r_y +
            self._up_axis[env_ids, 2] * r_z
        )
        self._initial_upright_score[env_ids] = initial_up.clamp(-1.0, 1.0).nan_to_num(nan=0.0)

        # 势函数塑形:把朝向 Φ_prev 重置为新一局初始姿态的 Φ,切断上一局残留,
        # 使新一局的朝向进展从 0 起步(配合 _get_rewards 的 r_upright/r_euler 势差)。
        self._upright_phi_prev[env_ids] = initial_up.clamp(0.0, 1.0).nan_to_num(nan=0.0)
        _gq0 = self._drill_goal_quat_per_env[env_ids]
        _dq0 = torch.nn.functional.normalize(result_quats, dim=1)
        _dot0 = (_dq0 * _gq0).sum(dim=1).abs().clamp(-1.0, 1.0)
        _angle0 = 2.0 * torch.acos(_dot0)
        self._euler_phi_prev[env_ids] = torch.exp(-_angle0 / 0.5).nan_to_num(nan=0.0)


def create_grasp_drill_env_cfg(
    num_envs: int = 4096,
    device: str = "cuda:0",
    headless: bool = False,
    hyperparameters=None,
    drill_config_path: str = None,
    drill_variants_path: str = None,
    debug: bool = False,
    enable_pc_viz: bool = False,
    img_height: int = 512,
    img_width: int = 512,
    enable_cameras: bool = True,
) -> GraspDrillEnvCfg:
    import os
    from isaaclab.managers import SceneEntityCfg
    
    if hyperparameters is None:
        from .config.hyperparameters import DEFAULT_HYPERPARAMETERS
        hyperparams = DEFAULT_HYPERPARAMETERS
    else:
        hyperparams = hyperparameters

    hyperparams.debug = debug

    if drill_config_path and os.path.exists(drill_config_path):
        try:
            with open(drill_config_path, "r") as f:
                config = yaml.safe_load(f)
            print(f"✓ 已从 {drill_config_path} 加载配置")
            
            if "scene" in config:
                scene_cfg = config["scene"]
                if "env_spacing" in scene_cfg:
                    hyperparams.scene.env_spacing = scene_cfg["env_spacing"]
                if "table_height" in scene_cfg:
                    hyperparams.drill.table_height = scene_cfg["table_height"]
            
            if "hand" in config:
                hand_cfg = config["hand"]
                if "stiffness" in hand_cfg:
                    hyperparams.hand.stiffness = hand_cfg["stiffness"]
                if "damping" in hand_cfg:
                    hyperparams.hand.damping = hand_cfg["damping"]
                if "effort_limit_sim" in hand_cfg:
                    hyperparams.hand.effort_limit_sim = hand_cfg["effort_limit_sim"]
                if "velocity_limit_sim" in hand_cfg:
                    hyperparams.hand.velocity_limit_sim = hand_cfg["velocity_limit_sim"]
                if "initial_pos" in hand_cfg:
                    pos = hand_cfg["initial_pos"]
                    if isinstance(pos, list):
                        pos = tuple(pos)
                    hyperparams.hand.initial_pos = pos
                if "initial_rot" in hand_cfg:
                    rot = hand_cfg["initial_rot"]
                    if isinstance(rot, list):
                        rot = tuple(rot)
                    hyperparams.hand.initial_rot = rot
            
            if "drill" in config:
                drill_cfg = config["drill"]
                if "scale" in drill_cfg:
                    scale = drill_cfg["scale"]
                    if isinstance(scale, list):
                        scale = tuple(scale)
                    hyperparams.drill.scale = scale
                if "mass" in drill_cfg:
                    hyperparams.drill.mass = drill_cfg["mass"]
                if "initial_pos" in drill_cfg:
                    pos = drill_cfg["initial_pos"]
                    if isinstance(pos, list):
                        pos = tuple(pos)
                    hyperparams.drill.initial_pos = pos
                if "initial_rot" in drill_cfg:
                    rot = drill_cfg["initial_rot"]
                    if isinstance(rot, list):
                        rot = tuple(rot)
                    hyperparams.drill.initial_rot = rot
                if "contact_offset" in drill_cfg:
                    hyperparams.drill.contact_offset = drill_cfg["contact_offset"]
                if "rest_offset" in drill_cfg:
                    hyperparams.drill.rest_offset = drill_cfg["rest_offset"]

            # 加载终止条件配置
            if "termination" in config:
                term_cfg = config["termination"]
                if "lift_z_threshold" in term_cfg:
                    hyperparams.termination.lift_z_threshold = term_cfg["lift_z_threshold"]
                if "fall_dist" in term_cfg:
                    hyperparams.termination.fall_dist = term_cfg["fall_dist"]
            
            # 加载奖励配置
            if "reward" in config:
                reward_cfg = config["reward"]
                if "distance_reward_weight" in reward_cfg:
                    hyperparams.reward.distance_reward_weight = reward_cfg["distance_reward_weight"]
                if "lift_reward_weight" in reward_cfg:
                    hyperparams.reward.lift_reward_weight = reward_cfg["lift_reward_weight"]
                if "stability_reward_weight" in reward_cfg:
                    hyperparams.reward.stability_reward_weight = reward_cfg["stability_reward_weight"]
                if "success_bonus_weight" in reward_cfg:
                    hyperparams.reward.success_bonus_weight = reward_cfg["success_bonus_weight"]
            
            # 加载动作配置
            if "action" in config:
                action_cfg = config["action"]
                if "wrist_pos_scale" in action_cfg:
                    hyperparams.action.wrist_pos_scale = action_cfg["wrist_pos_scale"]
                if "wrist_euler_scale" in action_cfg:
                    hyperparams.action.wrist_euler_scale = action_cfg["wrist_euler_scale"]
                if "joint_pos_scale" in action_cfg:
                    hyperparams.action.joint_pos_scale = action_cfg["joint_pos_scale"]
            
            
        except Exception as e:
            print(f"⚠ 加载配置文件失败: {e}，使用默认值")
    
    cfg = GraspDrillEnvCfg()

    # 基本参数
    cfg.num_envs = num_envs
    cfg.device = device
    cfg.headless = headless

    cfg.decimation = hyperparams.simulation.decimation
    cfg.episode_length_s = hyperparams.simulation.episode_length_s
    
    from .config.grasp_drill_env_cfg import (
        HandHyperparametersCfg,
        DrillHyperparametersCfg,
        SimulationHyperparametersCfg,
    )
    cfg.hand_params = HandHyperparametersCfg(
        initial_pos=hyperparams.hand.initial_pos,
        initial_rot=hyperparams.hand.initial_rot,
        effort_limit_sim=hyperparams.hand.effort_limit_sim,
        velocity_limit_sim=hyperparams.hand.velocity_limit_sim,
        stiffness=hyperparams.hand.stiffness,
        damping=hyperparams.hand.damping,
    )
    cfg.drill_params = DrillHyperparametersCfg(
        initial_pos=hyperparams.drill.initial_pos,
        initial_rot=hyperparams.drill.initial_rot,
        mass=hyperparams.drill.mass,
        scale=hyperparams.drill.scale,
        contact_offset=hyperparams.drill.contact_offset,
        rest_offset=hyperparams.drill.rest_offset,
        max_depenetration_velocity=hyperparams.drill.max_depenetration_velocity,
    )
    cfg.sim_params = SimulationHyperparametersCfg(
        hand_static_friction=hyperparams.simulation.hand_static_friction,
        hand_dynamic_friction=hyperparams.simulation.hand_dynamic_friction,
        hand_restitution=hyperparams.simulation.hand_restitution,
        drill_static_friction=hyperparams.simulation.drill_static_friction,
        drill_dynamic_friction=hyperparams.simulation.drill_dynamic_friction,
        drill_restitution=hyperparams.simulation.drill_restitution,
        table_static_friction=hyperparams.simulation.table_static_friction,
        table_dynamic_friction=hyperparams.simulation.table_dynamic_friction,
        table_restitution=hyperparams.simulation.table_restitution,
    )

    if drill_variants_path is None:
        drill_variants_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "config", "drill_variants.yaml"
        )
    drill_variants, total_num_variants = load_drill_variants_from_yaml(drill_variants_path)

    cfg.drill_variants = drill_variants
    cfg.total_num_variants = total_num_variants
    cfg._drill_variants_path = drill_variants_path
    cfg._assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")

    cfg.observation_space = 71  

    from isaaclab.sim.spawners.wrappers import MultiAssetSpawnerCfg

    _rigid_props_base = sim_utils.RigidBodyPropertiesCfg(
        rigid_body_enabled=True,
        solver_position_iteration_count=64,
        solver_velocity_iteration_count=1,
        max_angular_velocity=hyperparams.drill.max_angular_velocity,
        max_linear_velocity=hyperparams.drill.max_linear_velocity,
        disable_gravity=False,
        stabilization_threshold=0.0005,
    )
    _collision_props = sim_utils.CollisionPropertiesCfg(
        collision_enabled=True,
        contact_offset=hyperparams.drill.contact_offset,
        rest_offset=hyperparams.drill.rest_offset,
    )
    _mass_props = sim_utils.MassPropertiesCfg(mass=1) 
    _assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")

    variant_spawn_cfgs = []
    for v in drill_variants:
        spawn_cfg = sim_utils.UsdFileCfg(
            usd_path=os.path.join(_assets_dir, v.usd_path),
            scale=v.scale,
            copy_from_source=False,
        )
        spawn_cfg.init_pos = v.initial_pos
        spawn_cfg.init_rot = v.initial_rot
        variant_spawn_cfgs.append(spawn_cfg)

    drill_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Drill",
        spawn=MultiAssetSpawnerCfg(
            assets_cfg=variant_spawn_cfgs,
            rigid_props=_rigid_props_base,
            collision_props=_collision_props,
            mass_props=_mass_props,
            random_choice=False,  
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=drill_variants[0].initial_pos,
            rot=drill_variants[0].initial_rot,
        ),
    )

    table_height = -0.23  
    table_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.CuboidCfg(
            size=(1.5, 1.5, 0.5),  
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.3, 0.3, 0.35),  # 蓝灰色
                metallic=0.1,
                roughness=0.6,
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
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1, 0.2, table_height),  
        ),
    )

    franka_cfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/fr3_inspire_hand",
        spawn=sim_utils.UsdFileCfg(
            usd_path="assets/inspire_tac/fr3_inspire_hand_right/fr3_inspire_hand_right.usd",
            activate_contact_sensors=True,  # USD 中已有 PhysicsRigidBodyAPI，改用 PhysxRigidBodyAPI 会报错；使用显式 ContactSensorCfg
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,  
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,  
                rest_offset=0.001,    
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=1,
                stabilization_threshold=0.0005,
            ),
        ),

        init_state=ArticulationCfg.InitialStateCfg(
            pos=[0, 0.0, 0.0],  
            rot=[1.0, 0.0, 0.0, 0.0],
            joint_pos={
                # "fr3_joint1":  0.0120,
                # "fr3_joint2": -0.5700,
                # "fr3_joint3":  0.0,
                # "fr3_joint4": -2.8093,
                # "fr3_joint5":  0.0,
                # "fr3_joint6":  3.0369,
                # "fr3_joint7":  0.7410,
                "fr3_joint1":  0.7143,   #  35.2°
                "fr3_joint2":  0.2403,   #  36.7°
                "fr3_joint3": -0.6948,   # -39.8°
                "fr3_joint4": -2.0367,   # -116.7°
                "fr3_joint5":  2.5005,   #  149.0°
                "fr3_joint6":  2.2584,   #  129.4°
                "fr3_joint7": -0.7802,   # -44.7°
            }
         ),
        actuators={
            "fr3_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["fr3_joint[1-4]"],
                velocity_limit_sim=2.0,
                effort_limit_sim=87.0,
                stiffness=400.0,
                damping=80.0,
            ),
            "fr3_forearm": ImplicitActuatorCfg(
                joint_names_expr=["fr3_joint[5-7]"],
                effort_limit_sim=40.0,
                velocity_limit_sim=2.0,
                stiffness=80.0,
                damping=4.0,
            ),
            "inspire_hand": ImplicitActuatorCfg(
                joint_names_expr=[
                    "R_.*_proximal_joint",
                    "R_thumb_proximal_.*_joint",
                ],
            effort_limit_sim={
                "R_(index|middle|ring|pinky)_proximal_joint": 10,
                "R_thumb_proximal_yaw_joint": 10,
                "R_thumb_proximal_pitch_joint": 10,
            },

            stiffness={
                "R_(index|middle|ring|pinky)_proximal_joint": 3, 
                "R_thumb_proximal_yaw_joint": 3,
                "R_thumb_proximal_pitch_joint": 3,
            },

            damping={
                "R_(index|middle|ring|pinky)_proximal_joint": 0.1,  
                "R_thumb_proximal_yaw_joint": 0.1,
                "R_thumb_proximal_pitch_joint": 0.1,
            },
                velocity_limit_sim={
                    "R_(index|middle|ring|pinky)_proximal_joint": 10.0,  # 提高速度限制
                    "R_thumb_proximal_yaw_joint": 10.0,
                    "R_thumb_proximal_pitch_joint": 10.0,
                },
            ),
        },
    )


    env_spacing = hyperparams.scene.env_spacing
    replicate_physics = hyperparams.scene.replicate_physics
    clone_in_fabric = hyperparams.scene.clone_in_fabric
    if len(drill_variants) > 1:
        replicate_physics = False
        clone_in_fabric = False

    cfg.scene = GraspDrillSceneCfg(
        num_envs=num_envs,
        env_spacing=env_spacing,
        replicate_physics=replicate_physics,
        clone_in_fabric=clone_in_fabric,
        franka=franka_cfg,
        table=table_cfg,
        drill=drill_cfg,
    )
    
    def obs_hand_joint_pos(env):
        return env.scene["franka"].data.joint_pos
    def obs_hand_joint_vel(env):
        return env.scene["franka"].data.joint_vel

    @configclass
    class PolicyObsCfg(ObsGroup):
        # 关节状态 (13 proximal joints * 2 = 26维)
        hand_joint_pos = ObsTerm(func=obs_hand_joint_pos)
        hand_joint_vel = ObsTerm(func=obs_hand_joint_vel)
        

    cfg.observations = {"policy": PolicyObsCfg()}

    from .mdp import rewards_optimized as reward_funcs

    cfg.rewards = {
    }
    
    cfg.terminations = {
        "time_out": DoneTerm(
            func=lambda env: env.episode_length_buf >= env.max_episode_length,
            time_out=True,
        ),
    }
    
    cfg.events = {
        "reset_all": EventTerm(func=mdp.reset_scene_to_default, mode="reset"),
    }

    cfg.randomization = type('RandomizationCfg', (), {
        'drill_pos_random_range': hyperparams.randomization.drill_pos_random_range,
        'drill_rot_random_range': hyperparams.randomization.drill_rot_random_range,
    })()

    # Keep AA disabled to avoid SyntheticData / TiledCamera initialization failures.
    cfg.num_rerenders_on_reset = 3
    cfg.sim.render.antialiasing_mode = "Off"

    _cam_focal_length = 24.0
    _cam_horiz_aperture = 20.955
    _cam_clipping = (0.01, 3.0)

    _cam1_pos = (1.4, -0.6,1.1)
    _cam1_rot = (0.3522, -0.6717, -0.651, -0.0308)



    _cam2_pos = (1, 1.1, 1.3)
    _cam2_rot = (0.0, 0.0, 0.9659, -0.2588)
    # _cam2_pos = (0.8, 0.2, 2)
    # _cam2_rot = (0.0, 1.0, 0.0, 0.0)
    def _make_cam_cfg(name, pos, rot):
        return TiledCameraCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            height=img_height,
            width=img_width,
            offset=CameraCfg.OffsetCfg(pos=pos, rot=rot, convention="ros"),
            update_period=0,
            data_types=["distance_to_image_plane"],
            depth_clipping_behavior="none",
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=_cam_focal_length,
                horizontal_aperture=_cam_horiz_aperture,
                clipping_range=_cam_clipping,
            ),
        )

    if enable_cameras:
        cfg.scene.cam1 = _make_cam_cfg("Camera1", _cam1_pos, _cam1_rot)
        # cam1: 30cm 跟随框采 1024 点(电钻细节);cam2(正上方俯视,位置不变):不跟随,
        # 在固定工作区采 1024 点(全局上下文,像素充足几乎不塌)。collect/deploy 已同步。
        cfg.scene.cam2 = _make_cam_cfg("Camera2", _cam2_pos, _cam2_rot)

    return cfg