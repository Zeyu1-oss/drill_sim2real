"""
Stage2 奖励函数：电钻对齐任务
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """四元数 (N, 4) [w, x, y, z] -> 旋转矩阵 (N, 3, 3)"""
    q = torch.nn.functional.normalize(quat, dim=1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    N = quat.shape[0]
    R = torch.zeros((N, 3, 3), device=quat.device, dtype=quat.dtype)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


# ---------------------------------------------------------------------------
# R1: 对齐奖励（位置 + 旋转，稀疏）
# ---------------------------------------------------------------------------

def stage2_alignment_reward(
    env: ManagerBasedRLEnv,
    pos_thresh: float = 0.001,
    rot_thresh: float = 0.9999,
    pos_weight: float = 100.0,
    rot_weight: float = 100.0,
    pos_partial_start: float = 0.01,
    rot_partial_start: float = 0.999,
) -> torch.Tensor:
    """
    稀疏对齐奖励：
    - 位置误差 < pos_thresh (1mm) 时给 pos_weight
    - 旋转误差（四元数点积）> rot_thresh 时给 rot_weight
    - 中间区间给线性部分奖励

    目标姿态从 env.target_pos / env.target_quat 读取。
    """
    num_envs = env.num_envs
    device = env.device

    drill_pos = env.drill.data.root_pos_w
    drill_quat = env.drill.data.root_quat_w
    target_pos = env.target_stage2_alignment_rewardpos
    target_quat = env.target_quat

    # 位置误差
    pos_error = torch.norm(drill_pos - target_pos, dim=1)
    pos_success = pos_error < pos_thresh
    pos_partial = (pos_error < pos_partial_start) & ~pos_success
    pos_reward = torch.zeros(num_envs, device=device, dtype=torch.float32)
    pos_reward[pos_success] = pos_weight
    pos_reward[pos_partial] = pos_weight * 0.5 * (1 - pos_error[pos_partial] / pos_partial_start)

    # 旋转误差（四元数点积，绝对值避免符号歧义）
    # q1 · q2 = w1*w2 + x1*x2 + y1*y2 + z1*z2; 1=同向, 0=90°, -1=180°
    q_dot = (
        drill_quat[:, 0] * target_quat[:, 0] +
        drill_quat[:, 1] * target_quat[:, 1] +
        drill_quat[:, 2] * target_quat[:, 2] +
        drill_quat[:, 3] * target_quat[:, 3]
    ).abs()
    rot_success = q_dot > rot_thresh
    rot_partial = (q_dot > rot_partial_start) & ~rot_success
    rot_reward = torch.zeros(num_envs, device=device, dtype=torch.float32)
    rot_reward[rot_success] = rot_weight
    rot_reward[rot_partial] = rot_weight * 0.5 * ((q_dot[rot_partial] - rot_partial_start) / (rot_thresh - rot_partial_start))

    return pos_reward + rot_reward


# ---------------------------------------------------------------------------
# R2: 抓取质量维持奖励
# ---------------------------------------------------------------------------

def stage2_grasp_quality_reward(
    env: ManagerBasedRLEnv,
    finger_dist_thresh: float = 0.03,
    min_contacts: int = 6,
    force_thresh: float = 0.1,
    weight: float = 80.0,
) -> torch.Tensor:
    """
    维持 Stage1 成功时的抓取质量：
    - R_index_intermediate 在 trigger 附近（< 3cm）
    - 至少 min_contacts 个传感器接触力 > force_thresh
    """
    num_envs = env.num_envs
    device = env.device

    hand = env.scene["franka"]
    drill = env.drill

    # finger_near_trigger
    finger_near_trigger = torch.zeros(num_envs, dtype=torch.bool, device=device)
    try:
        body_names = hand.data.body_names
        if "R_index_intermediate" in body_names:
            idx = body_names.index("R_index_intermediate")
            index_pos = hand.data.body_pos_w[:, idx, :]
            trigger_offset = env._trigger1_offset * env._drill_scale
            drill_pos = drill.data.root_pos_w
            R = quat_to_rotmat(drill.data.root_quat_w)
            trigger_world = torch.bmm(R, trigger_offset.unsqueeze(-1)).squeeze(-1) + drill_pos
            dist = torch.norm(index_pos - trigger_world, dim=1)
            finger_near_trigger = dist < finger_dist_thresh
    except Exception:
        finger_near_trigger = torch.ones(num_envs, dtype=torch.bool, device=device)

    # multi_contact_ok
    multi_contact_ok = torch.zeros(num_envs, dtype=torch.bool, device=device)
    all_hand_sensors = [
        "contact_index_intermediate",
        "contact_thumb_distal",
        "contact_middle_intermal",
        "contact_middle_proximal",
        "contact_ring_proximal",
        "contact_pinky_proximal",
        "contact_hand_base",
    ]
    try:
        if hasattr(env.scene, "sensors"):
            contact_count = torch.zeros(num_envs, device=device, dtype=torch.float32)
            for sname in all_hand_sensors:
                if sname in env.scene.sensors:
                    s = env.scene.sensors[sname]
                    fm = s.data.force_matrix_w
                    if fm is not None:
                        fm = torch.nan_to_num(fm, nan=0.0, posinf=0.0, neginf=0.0)
                        fm = torch.clamp(fm, -50.0, 50.0)
                        fmag = torch.norm(fm.sum(dim=(1, 2)), dim=1)
                    else:
                        fmag = torch.norm(s.data.net_forces_w.sum(dim=1), dim=1)
                    fmag = torch.nan_to_num(fmag, nan=0.0)
                    contact_count += (fmag > force_thresh).float()
            multi_contact_ok = contact_count >= min_contacts
    except Exception:
        multi_contact_ok = torch.zeros(num_envs, dtype=torch.bool, device=device)

    # 权重：部分满足给部分奖励
    reward = torch.zeros(num_envs, device=device, dtype=torch.float32)
    both_ok = finger_near_trigger & multi_contact_ok
    reward[both_ok] = weight
    any_ok = finger_near_trigger | multi_contact_ok
    partial = any_ok & ~both_ok
    reward[partial] = weight * 0.3

    return reward


# ---------------------------------------------------------------------------
# R3: tip2tip 距离奖励
# ---------------------------------------------------------------------------

def stage2_tip_distance_reward(
    env: ManagerBasedRLEnv,
    temp: float = 0.03,
    weight: float = 15.0,
) -> torch.Tensor:
    """
    thumb_distal 和 R_index_intermediate 的 Gaussian 距离奖励：
    exp(-dist / temp)
    """
    num_envs = env.num_envs
    device = env.device

    hand = env.scene["franka"]
    try:
        body_names = hand.data.body_names
        thumb_candidates = [n for n in body_names if "thumb" in n and "distal" in n]
        index_candidates = [n for n in body_names if "index" in n and "intermediate" in n]
        if not thumb_candidates or not index_candidates:
            return torch.zeros(num_envs, device=device, dtype=torch.float32)

        thumb_idx = body_names.index(thumb_candidates[0])
        index_idx = body_names.index(index_candidates[0])

        thumb_pos = hand.data.body_pos_w[:, thumb_idx, :]
        index_pos = hand.data.body_pos_w[:, index_idx, :]
        dist = torch.norm(thumb_pos - index_pos, dim=1)
        dist = torch.nan_to_num(dist, nan=1.0, posinf=1.0)
        reward = weight * torch.exp(-dist / temp)
        return torch.clamp(reward, 0.0, weight)
    except Exception:
        return torch.zeros(num_envs, device=device, dtype=torch.float32)
