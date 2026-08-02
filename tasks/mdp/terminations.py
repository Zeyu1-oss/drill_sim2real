"""
Termination functions.
Based on relative height (relative to each episode's initial position).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def check_success(
    env: ManagerBasedRLEnv,
    lift_z_threshold: float = 0.1,  # lift height relative to initial position
    drill_cfg: SceneEntityCfg = SceneEntityCfg("drill"),
) -> torch.Tensor:
    """Success: drill z is higher than initial position by more than lift_z_threshold.

    e.g. lift_z_threshold=0.1 means success when the drill is 10cm above its initial z.
    """
    drill: RigidObject = env.scene[drill_cfg.name]
    drill_pos = drill.data.root_pos_w
    # use env.initial_drill_pos (full initial position recorded at each episode start)
    z_diff = drill_pos[:, 2] - env.initial_drill_pos[:, 2]
    return z_diff > lift_z_threshold


def check_failure(
    env: ManagerBasedRLEnv,
    fall_dist: float = 0.05,  # drop distance relative to initial position
    drill_cfg: SceneEntityCfg = SceneEntityCfg("drill"),
) -> torch.Tensor:
    """Failure: drill z is lower than initial position by more than fall_dist.

    e.g. fall_dist=0.05 means failure when the drill is 5cm below its initial z.
    """
    drill: RigidObject = env.scene[drill_cfg.name]
    drill_pos = drill.data.root_pos_w
    # use env.initial_drill_pos (full initial position recorded at each episode start)
    z_diff = env.initial_drill_pos[:, 2] - drill_pos[:, 2]
    return z_diff > fall_dist
