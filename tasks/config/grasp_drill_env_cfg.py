"""
Drill grasping env config.
Only the config classes actually in use are kept.
"""

from dataclasses import MISSING

from isaaclab.utils import configclass


@configclass
class HandHyperparametersCfg:
    """Dexterous hand hyperparameters."""

    # actuator params
    effort_limit_sim: float = 10.0
    velocity_limit_sim: float = 10.0
    stiffness: float = 40.0
    damping: float = 10.0

    # initial pose
    initial_pos: tuple = (0.5, 0.0, 0.6)
    initial_rot: tuple = (1.0, 0.0, 0.0, 0.0)


@configclass
class DrillHyperparametersCfg:
    """Drill hyperparameters."""

    # physics
    mass: float = 0.5
    scale: tuple = (1, 1, 1)

    # initial pose (offset relative to table top)
    initial_pos: tuple = (0.0, 0.0, 0.08)
    initial_rot: tuple = (0.0, 0.0, 0.0, 1.0)

    # collision params
    contact_offset: float = 0.01
    rest_offset: float = 0.001
    max_depenetration_velocity: float = 15

    # table height (for computing absolute position)
    table_height: float = 0.6


@configclass
class SimulationHyperparametersCfg:
    """Global simulation hyperparameters."""

    # hand friction
    hand_static_friction: float = 0.5
    hand_dynamic_friction: float = 0.5
    hand_restitution: float = 0.0

    # drill friction
    drill_static_friction: float = 0
    drill_dynamic_friction: float = 0
    drill_restitution: float = 0.0

    # table friction
    table_static_friction: float = 0
    table_dynamic_friction: float = 0
    table_restitution: float = 0.0
