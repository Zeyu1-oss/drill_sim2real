"""
终止条件函数定义
基于相对高度（相对于每个 episode 的初始位置）
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
    lift_z_threshold: float = 0.1,  # 相对于初始位置的抬起高度
    drill_cfg: SceneEntityCfg = SceneEntityCfg("drill"),
) -> torch.Tensor:
    """成功条件：电钻 z 轴高于初始位置超过 lift_z_threshold
    
    例如：lift_z_threshold=0.1 表示电钻 z 轴高于初始位置 10cm 判定成功
    """
    drill: RigidObject = env.scene[drill_cfg.name]
    drill_pos = drill.data.root_pos_w
    # 使用 env.initial_drill_pos（每个 episode 开始时记录的完整初始位置）
    z_diff = drill_pos[:, 2] - env.initial_drill_pos[:, 2]
    return z_diff > lift_z_threshold


def check_failure(
    env: ManagerBasedRLEnv,
    fall_dist: float = 0.05,  # 相对于初始位置的下降距离
    drill_cfg: SceneEntityCfg = SceneEntityCfg("drill"),
) -> torch.Tensor:
    """失败条件：电钻 z 轴低于初始位置超过 fall_dist

    例如：fall_dist=0.05 表示电钻 z 轴低于初始位置 5cm 判定失败
    """
    drill: RigidObject = env.scene[drill_cfg.name]
    drill_pos = drill.data.root_pos_w
    # 使用 env.initial_drill_pos（每个 episode 开始时记录的完整初始位置）
    z_diff = env.initial_drill_pos[:, 2] - drill_pos[:, 2]
    return z_diff > fall_dist


