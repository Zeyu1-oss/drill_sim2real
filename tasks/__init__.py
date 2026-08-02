"""
Task module.
Task implementations for dexterous-hand drill grasping.
"""

# Export RL env (lazy import so it doesn't fail without Isaac Lab).
__all__ = [
    "GraspDrillEnv",
    "GraspDrillEnvCfg",
    "create_grasp_drill_env_cfg",
]

try:
    from .grasp_drill_env import GraspDrillEnv, GraspDrillEnvCfg, create_grasp_drill_env_cfg
except ImportError:
    # No Isaac Lab: skip exporting the RL env.
    pass
