"""
抓取电钻环境配置
仅保留实际使用的配置类
"""

from dataclasses import MISSING

from isaaclab.utils import configclass


@configclass
class HandHyperparametersCfg:
    """灵巧手超参数"""

    # 执行器参数
    effort_limit_sim: float = 10.0
    velocity_limit_sim: float = 10.0
    stiffness: float = 40.0
    damping: float = 10.0

    # 初始位置
    initial_pos: tuple = (0.5, 0.0, 0.6)
    initial_rot: tuple = (1.0, 0.0, 0.0, 0.0)


@configclass
class DrillHyperparametersCfg:
    """电钻超参数"""

    # 物理属性
    mass: float = 0.5
    scale: tuple = (1, 1, 1)

    # 初始位置（相对于桌面的偏移）
    initial_pos: tuple = (0.0, 0.0, 0.08)
    initial_rot: tuple = (0.0, 0.0, 0.0, 1.0)

    # 碰撞参数
    contact_offset: float = 0.01
    rest_offset: float = 0.001
    max_depenetration_velocity: float = 15

    # 桌子高度（用于计算绝对位置）
    table_height: float = 0.6


@configclass
class SimulationHyperparametersCfg:
    """仿真全局超参数"""

    # 灵巧手摩擦参数
    hand_static_friction: float = 0.5
    hand_dynamic_friction: float = 0.5
    hand_restitution: float = 0.0

    # 电钻摩擦参数
    drill_static_friction: float = 0
    drill_dynamic_friction: float = 0
    drill_restitution: float = 0.0

    # 桌子摩擦参数
    table_static_friction: float = 0
    table_dynamic_friction: float = 0
    table_restitution: float = 0.0
