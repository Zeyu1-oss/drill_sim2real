"""
超参数配置文件
集中管理所有超参数，方便调整和实验
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class RewardHyperparameters:
    """奖励函数超参数 - 参考 RobustDexGrasp 设计"""


    finger_reward_scale: float = 3.0      # 手指奖励最大值
    finger_temperature: float = 0.05      # 手指温度参数（5cm 内奖励明显）

    palm_penalty_scale: float = 0.5       # 手掌惩罚系数（增大手掌阻挡惩罚）
    palm_temperature: float = 0.5       # 手掌惩罚温度（手掌可稍微远离 drill）

    # === 接触奖励（参考 RobustDexGrasp）===
    contact_reward_scale: float = 5.0     # 接触奖励权重
    contact_threshold: float = 0.03       # 接触判定距离（3cm）

    # === 对齐奖励 ===
    alignment_reward_scale: float = 0.5   # 对齐奖励权重

    # === 抬起奖励 ===
    lift_height: float = 0.05            # 抬起高度阈值（10cm）- 与 lift_z_threshold 对应
    lift_reward_weight: float = 1      # 抬起奖励权重（指数增长，需要小权重防止 NaN）
    success_reward_weight: float = 10.0   # 成功奖励权重

    # === 稳定性奖励（参考 RobustDexGrasp）===
    velocity_reward_weight: float = 0.1   # 线速度惩罚权重
    angular_velocity_reward_weight: float = 0.05  # 角速度惩罚权重

    object_lin_vel_penalty: float = -15.0
    object_ang_vel_penalty: float = -0.5
    object_max_lin_vel: float = 0.3        # 最大允许线速度 (m/s)
    object_max_ang_vel: float = 1.0        # 最大允许角速度 (rad/s)





@dataclass
class TerminationHyperparameters:
    """终止条件超参数"""

    lift_z_threshold: float = 0.2  # 10cm - 抬起超过10cm判定成功
    fall_dist: float = 0.10  # 下降超过10cm才判定失败，必须小于初始高度 0.15m



@dataclass
class ActionHyperparameters:
    """动作超参数"""

    # 动作缩放步长
    wrist_pos_scale: float = 0.005     # 手腕位置每次变化（米）
    wrist_euler_scale: float = 0.05    # 手腕姿态每次变化（弧度）
    joint_pos_scale: float = 0.85      # 关节位置每次变化（弧度，action -1~1 覆盖整个关节范围）

    action_penalty_weight: float = -0.0002  # 负权重 = 惩罚
    action_penalty_scale: float = 0.0002    # 惩罚缩放因子


@dataclass
class SimulationHyperparameters:
    """仿真超参数"""

    dt: float = 1/120
    decimation: int = 4 
    episode_length_s: float = 10.0  

    # PhysX 参数o
    bounce_threshold_velocity: float = 0.01
    gpu_found_lost_aggregate_pairs_capacity: int = 1024 * 1024 * 4
    gpu_total_aggregate_pairs_capacity: int = 16 * 1024
    friction_correlation_distance: float = 0.00625

    hand_static_friction: float = 0.8
    hand_dynamic_friction: float = 0.8
    hand_restitution: float = 0.0

    drill_static_friction: float = 0.4
    drill_dynamic_friction: float = 0.4
    drill_restitution: float = 0.0

    table_static_friction: float = 0.05
    table_dynamic_friction: float = 0.05
    table_restitution: float = 0.0


@dataclass
class HandHyperparameters:
    """灵巧手超参数"""

    effort_limit_sim: float = 10.0
    velocity_limit_sim: float = 10.0
    stiffness: float = 50.0
    damping: float = 8.0

    initial_pos: Tuple[float, float, float] = (0.0, 0.0, 0)
    initial_rot: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    activate_contact_sensors: bool = True


@dataclass
class DrillHyperparameters:
    """电钻超参数"""

    # 物理属性
    mass: float = 1
    scale: Tuple[float, float, float] = (1, 1, 1)

    initial_pos: Tuple[float, float, float] = (0.0, 0.0, 0.15)
    initial_rot: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)  #

    contact_offset: float = 0.005
    rest_offset: float = 0
    max_depenetration_velocity: float = 10.0  # 限制穿透解算瞬间甩出的速度（原 10 偏大）

    table_height: float = 0.6

    # 求解器参数
    solver_position_iteration_count: int = 64  #
    solver_velocity_iteration_count: int = 1  #
    max_angular_velocity: float = 50.0   # 原 1000 形同不限；钻头正常运动远低于此
    max_linear_velocity: float = 10.0     # 原 1000；被打飞也不该超过 ~10 m/s


@dataclass
class SceneHyperparameters:

    env_spacing: float = 3.0  # 
    replicate_physics: bool = True  # 
    clone_in_fabric: bool = False  # 


@dataclass
class RandomizationHyperparameters:

    enable_randomization: bool = True

    joint_pos_noise_std: float = 0.002  
    joint_vel_noise_std: float = 0.001 

    action_noise_std: float = 0.05  

    drill_pos_random_range: Tuple[float, float, float] = (0.05, 0.05, 0.00) 
    drill_rot_random_range: Tuple[float, float, float] = (0, 0, 1)  

    gravity_bias_range: Tuple[float, float, float] = (0.0, 0.0, 0.4)  

    friction_random_range: Tuple[float, float] = (0.5, 1.5)
    mass_random_range: Tuple[float, float] = (0.7, 1.3)


@dataclass
class PerceptionHyperparameters:
    workspace: Tuple[float, float, float, float, float, float] = (0.25, 1.75, -0.55, 0.95, 0.025, 2.0)
    ground_z: float = 0.02
    ground_points: int = 96  
    ground_xy_noise_std: float = 0.02
    camera_follow_drill: bool = False
    drill_crop_half: float = 0.2
    img_height: int = 256
    img_width: int = 256


@dataclass
class AllHyperparameters:

    reward: RewardHyperparameters = None
    termination: TerminationHyperparameters = None
    action: ActionHyperparameters = None
    simulation: SimulationHyperparameters = None
    hand: HandHyperparameters = None
    drill: DrillHyperparameters = None
    scene: SceneHyperparameters = None
    randomization: RandomizationHyperparameters = None
    perception: PerceptionHyperparameters = None
    debug: bool = True  # 调试模式开关

    def __post_init__(self):
        """初始化默认值"""
        if self.reward is None:
            self.reward = RewardHyperparameters()
        if self.termination is None:
            self.termination = TerminationHyperparameters()
        if self.action is None:
            self.action = ActionHyperparameters()
        if self.simulation is None:
            self.simulation = SimulationHyperparameters()
        if self.hand is None:
            self.hand = HandHyperparameters()
        if self.drill is None:
            self.drill = DrillHyperparameters()
        if self.scene is None:
            self.scene = SceneHyperparameters()
        if self.randomization is None:
            self.randomization = RandomizationHyperparameters()
        if self.perception is None:
            self.perception = PerceptionHyperparameters()


# 默认超参数实例
DEFAULT_HYPERPARAMETERS = AllHyperparameters()



def create_hyperparameters_from_config(config: dict) -> AllHyperparameters:
    hp = DEFAULT_HYPERPARAMETERS

    # 从配置更新（如果存在）
    if "scene" in config:
        scene_cfg = config["scene"]
        if "env_spacing" in scene_cfg:
            hp.scene.env_spacing = scene_cfg["env_spacing"]
        if "replicate_physics" in scene_cfg:
            hp.scene.replicate_physics = scene_cfg["replicate_physics"]

    if "hand" in config:
        hand_cfg = config["hand"]
        if "stiffness" in hand_cfg:
            hp.hand.stiffness = hand_cfg["stiffness"]
        if "damping" in hand_cfg:
            hp.hand.damping = hand_cfg["damping"]
        if "effort_limit_sim" in hand_cfg:
            hp.hand.effort_limit_sim = hand_cfg["effort_limit_sim"]
        if "velocity_limit_sim" in hand_cfg:
            hp.hand.velocity_limit_sim = hand_cfg["velocity_limit_sim"]
        if "initial_pos" in hand_cfg:
            hp.hand.initial_pos = tuple(hand_cfg["initial_pos"])
        if "activate_contact_sensors" in hand_cfg:
            hp.hand.activate_contact_sensors = hand_cfg["activate_contact_sensors"]

    if "drill" in config:
        drill_cfg = config["drill"]
        if "mass" in drill_cfg:
            hp.drill.mass = drill_cfg["mass"]
        if "contact_offset" in drill_cfg:
            hp.drill.contact_offset = drill_cfg["contact_offset"]
        if "rest_offset" in drill_cfg:
            hp.drill.rest_offset = drill_cfg["rest_offset"]
        if "scale" in drill_cfg:
            hp.drill.scale = tuple(drill_cfg["scale"])
        if "table_height" in drill_cfg:
            hp.drill.table_height = drill_cfg["table_height"]

    if "termination" in config:
        term_cfg = config["termination"]
        if "lift_z_threshold" in term_cfg:
            hp.termination.lift_z_threshold = term_cfg["lift_z_threshold"]
        if "fall_dist" in term_cfg:
            hp.termination.fall_dist = term_cfg["fall_dist"]

    if "reward" in config:
        reward_cfg = config["reward"]
        if "distance_reward_weight" in reward_cfg:
            hp.reward.distance_reward_weight = reward_cfg["distance_reward_weight"]
        if "lift_reward_weight" in reward_cfg:
            hp.reward.lift_reward_weight = reward_cfg["lift_reward_weight"]
        if "success_reward_weight" in reward_cfg:
            hp.reward.success_reward_weight = reward_cfg["success_reward_weight"]
        if "lift_height" in reward_cfg:
            hp.reward.lift_height = reward_cfg["lift_height"]
        if "dist_reward_scale" in reward_cfg:
            hp.reward.dist_reward_scale = reward_cfg["dist_reward_scale"]

    if "action" in config:
        action_cfg = config["action"]
        if "wrist_pos_scale" in action_cfg:
            hp.action.wrist_pos_scale = action_cfg["wrist_pos_scale"]
        if "wrist_euler_scale" in action_cfg:
            hp.action.wrist_euler_scale = action_cfg["wrist_euler_scale"]
        if "joint_pos_scale" in action_cfg:
            hp.action.joint_pos_scale = action_cfg["joint_pos_scale"]
        if "action_penalty_weight" in action_cfg:
            hp.action.action_penalty_weight = action_cfg["action_penalty_weight"]
        if "action_penalty_scale" in action_cfg:
            hp.action.action_penalty_scale = action_cfg["action_penalty_scale"]

    if "simulation" in config:
        sim_cfg = config["simulation"]
        if "dt" in sim_cfg:
            hp.simulation.dt = sim_cfg["dt"]
        if "decimation" in sim_cfg:
            hp.simulation.decimation = sim_cfg["decimation"]
        if "episode_length_s" in sim_cfg:
            hp.simulation.episode_length_s = sim_cfg["episode_length_s"]

        # 灵巧手摩擦
        if "hand_static_friction" in sim_cfg:
            hp.simulation.hand_static_friction = sim_cfg["hand_static_friction"]
        if "hand_dynamic_friction" in sim_cfg:
            hp.simulation.hand_dynamic_friction = sim_cfg["hand_dynamic_friction"]
        if "hand_restitution" in sim_cfg:
            hp.simulation.hand_restitution = sim_cfg["hand_restitution"]
        # 电钻摩擦
        if "drill_static_friction" in sim_cfg:
            hp.simulation.drill_static_friction = sim_cfg["drill_static_friction"]
        if "drill_dynamic_friction" in sim_cfg:
            hp.simulation.drill_dynamic_friction = sim_cfg["drill_dynamic_friction"]
        if "drill_restitution" in sim_cfg:
            hp.simulation.drill_restitution = sim_cfg["drill_restitution"]
        # 桌子摩擦
        if "table_static_friction" in sim_cfg:
            hp.simulation.table_static_friction = sim_cfg["table_static_friction"]
        if "table_dynamic_friction" in sim_cfg:
            hp.simulation.table_dynamic_friction = sim_cfg["table_dynamic_friction"]
        if "table_restitution" in sim_cfg:
            hp.simulation.table_restitution = sim_cfg["table_restitution"]

    if "randomization" in config:
        rand_cfg = config["randomization"]
        if "enable_randomization" in rand_cfg:
            hp.randomization.enable_randomization = rand_cfg["enable_randomization"]
        if "randomization_interval" in rand_cfg:
            hp.randomization.randomization_interval = rand_cfg["randomization_interval"]
        if "drill_pos_random_range" in rand_cfg:
            hp.randomization.drill_pos_random_range = tuple(rand_cfg["drill_pos_random_range"])
        if "drill_rot_random_range" in rand_cfg:
            hp.randomization.drill_rot_random_range = tuple(rand_cfg["drill_rot_random_range"])

    return hp