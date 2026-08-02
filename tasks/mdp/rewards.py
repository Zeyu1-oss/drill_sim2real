"""
Reward function definitions

1. approach_reward_improved (approach reward)
   body_approach_reward = Σ_i exp(-d_i / T) * 2
   
   where:
   - d_i = min(||hand_link_i - body_point_j||)  # distance from the i-th hand link to the nearest body point
   - T = finger_temp = 0.02  # temperature param
   - weight: 2.0
   
   tip_trigger_reward = exp(-d_tip / 0.02) * 15
   
   
   avoid_penalty = -Σ_j forbidden_j * 5
   
   - constraint: handle region y in [-0.03, 0.055] (drill local coordinates)
   
   total: approach_reward = body_approach + tip_trigger + avoid_penalty

2. lift_reward 
   reward = lift_reward_weight * (z_diff / lift_height) * not_knocked_away
   
   where:
   - z_diff = max(0, drill_z - initial_z)
   - z_diff = min(z_diff, lift_height)  # clamp to max lift_height
   - lift_height = 0.1 
   - lift_reward_weight = 1.0
   - not_knocked_away = (xy_distance < max_xy_distance) ? 1 : 0
   - xy_distance = sqrt((x - x0)² + (y - y0)²)

3. success_reward (success reward)
   reward = success_reward_weight * success
   
4. contact_reward_detailed (contact reward)
   reward = Σ sensors (contact_binary * weight + contact_binary * log(1 + force) * weight)
   clamp to 50N
   - contact_binary = (force > force_threshold) ? 1 : 0

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import numpy as np
import math

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

def quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to rotation matrix

    Args:
        quat: quaternion (N, 4), format (w, x, y, z)

    Returns:
        rotation matrix (N, 3, 3)
    """
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


def _points_to_world(local_points: torch.Tensor, drill_pos: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Transform local points to world coordinates (using precomputed R matrix)"""
    num_envs = drill_pos.shape[0]
    local_expanded = local_points.T.unsqueeze(0).expand(num_envs, -1, -1)
    world_transformed = torch.bmm(R, local_expanded)
    return world_transformed.permute(0, 2, 1) + drill_pos.unsqueeze(1)


def _point_to_world(single_local: torch.Tensor, drill_pos: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Transform a single local point to world coordinates (1 point, expand then bmm+squeeze)"""
    num_envs = drill_pos.shape[0]
    expanded = single_local.view(1, 3, 1).expand(num_envs, -1, -1)
    return torch.bmm(R, expanded).squeeze(-1) + drill_pos


# precomputed body_names indices (avoid repeated .index() every step)
# key: function name, value: precomputed link_name -> body_names index mapping
# filled on first call, reused after
_HAND_BODY_NAME_INDICES: dict = {}



if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_reward_step_counter = 0
REWARD_REGISTRY: dict = {}
def contact_reward_detailed(
    env: ManagerBasedRLEnv,
    fingertip_contact_bonus: float = 1,
    other_link_contact_bonus: float = 2,
    hand_base_contact_bonus: float = 50,
    index_contact_bonus: float = 1,
    force_threshold: float = 0.1,
    dist_temp: float = 0.05,
    drill_scale: tuple = (1, 1.2, 1),
) -> torch.Tensor:
    num_envs = env.num_envs
    device = env.device

    # these three thumb links only get contact binary, not multiplied by distance weight
    thumb_no_dist_links = {
        "R_thumb_distal",
        "R_thumb_intermediate",
        "R_thumb_proximal",
    }

    index_sensors = ["contact_index_intermediate"]

    fingertip_sensors = [
        "contact_thumb_distal",
        "contact_thumb_intermediate",
        "contact_middle_intermediate",
        "contact_ring_intermediate",
        "contact_pinky_intermediate",
    ]
    other_link_sensors = [
        "contact_thumb_proximal",
        "contact_thumb_proximal_base",
        "contact_index_proximal",
        "contact_middle_proximal",
        "contact_ring_proximal",
        "contact_pinky_proximal",
    ]
    base_sensors = ["contact_hand_base"]

    sensor_to_link = {
        "contact_index_intermediate":  "R_index_intermediate",
        "contact_thumb_distal":        "R_thumb_distal",
        "contact_thumb_intermediate":  "R_thumb_intermediate",
        "contact_thumb_proximal":      "R_thumb_proximal",
        "contact_thumb_proximal_base": "R_thumb_proximal_base",
        "contact_middle_intermediate": "R_middle_intermediate",
        "contact_ring_intermediate":   "R_ring_intermediate",
        "contact_pinky_intermediate":  "R_pinky_intermediate",
        "contact_index_proximal":      "R_index_proximal",
        "contact_middle_proximal":     "R_middle_proximal",
        "contact_ring_proximal":       "R_ring_proximal",
        "contact_pinky_proximal":      "R_pinky_proximal",
        "contact_hand_base":           "R_hand_base_link",
    }

    # === precompute distance weights ===
    try:
        drill = env.drill
        hand = env.scene["franka"]
        drill_pos = drill.data.root_pos_w
        drill_quat = drill.data.root_quat_w
        hand_body_names = hand.body_names
        hand_body_pos = hand.data.body_pos_w

        # precompute link_name -> body_names index (avoid 13 .index() calls every step)
        link_to_body_idx = {name: i for i, name in enumerate(hand_body_names)}

        link_to_body_idx = {name: i for i, name in enumerate(hand_body_names)}

        # Group envs by variant so each uses its correct mesh + body_mask
        variant_ids = env._drill_variant_indices  # [num_envs]
        unique_vids = variant_ids.unique().tolist()
        reward = torch.zeros(num_envs, device=device)

        for vid in unique_vids:
            mask = variant_ids == vid
            n_v = mask.sum().item()
            idx_v = mask.nonzero(as_tuple=True)[0]  # which env indices

            # Variant-specific mesh (body already filtered by body_mask_axis + body_mask_min/max)
            vdata = env._variant_attrs[vid]
            trigger_pts, body_pts_f, _, _ = env._build_variant_surface_points(vid)
            scale = vdata["scale"]
            body_pts_f = body_pts_f * scale
            trigger_pts_s = trigger_pts * scale
            body_pts_f = torch.nan_to_num(body_pts_f, nan=0.0, posinf=0.0, neginf=0.0)
            trigger_pts_s = torch.nan_to_num(trigger_pts_s, nan=0.0, posinf=0.0, neginf=0.0)

            trigger_center = trigger_pts_s.mean(dim=0)

            R_v = quat_to_rotmat(drill_quat[idx_v])
            body_w = torch.bmm(R_v, body_pts_f.T.unsqueeze(0).expand(n_v, -1, -1)).permute(0, 2, 1) + drill_pos[idx_v].unsqueeze(1)
            body_w = torch.nan_to_num(body_w, nan=0.0, posinf=0.0, neginf=0.0)
            trigger_c_w = _point_to_world(trigger_center, drill_pos[idx_v], R_v)
            trigger_c_w = torch.nan_to_num(trigger_c_w, nan=0.0, posinf=0.0, neginf=0.0)

            def get_weight_v(link_name, use_trigger=False):
                body_idx = link_to_body_idx.get(link_name)
                if body_idx is None:
                    return torch.ones(n_v, device=device) * 2.0
                lp = hand_body_pos[idx_v, body_idx, :]
                lp = torch.nan_to_num(lp, nan=0.0, posinf=0.0, neginf=0.0)
                if use_trigger:
                    dist = torch.norm(lp - trigger_c_w, dim=-1)
                else:
                    lp_exp = lp.unsqueeze(1)
                    dists = torch.norm(lp_exp - body_w, dim=-1)
                    dists = torch.nan_to_num(dists, nan=1.0, posinf=1.0, neginf=0.0)
                    dist = dists.min(dim=1)[0]
                dist = torch.nan_to_num(dist, nan=1.0, posinf=1.0, neginf=0.0)
                dist = torch.clamp(dist, min=0.0, max=10.0)
                w = 1.0 + torch.exp(-dist / dist_temp)
                w = torch.nan_to_num(w, nan=1.0, posinf=2.0, neginf=1.0)
                return torch.clamp(w, min=1.0, max=2.0)

            def sensor_contact(sensors, c_bonus, use_trig):
                total = torch.zeros(n_v, device=device)
                for sname in sensors:
                    if sname not in env.scene.sensors:
                        continue
                    s = env.scene.sensors[sname]
                    fm = s.data.force_matrix_w[idx_v]
                    fm = torch.nan_to_num(fm, nan=0.0, posinf=0.0, neginf=0.0)
                    fm = torch.clamp(fm, -50.0, 50.0)
                    fmag = torch.norm(fm.sum(dim=(1, 2)), dim=1)
                    fmag = torch.nan_to_num(fmag, nan=0.0)
                    binary = (fmag > force_threshold).float()
                    lname = sensor_to_link.get(sname, "")
                    if lname in thumb_no_dist_links:
                        contrib = binary * c_bonus
                    else:
                        contrib = binary * c_bonus * get_weight_v(lname, use_trig)
                    total += contrib
                    # === DEBUG: print per-sensor data for first 3 envs ===
                    # print(f"  [Sensor: {sname}] fmag={fmag[:3].cpu().numpy()} | binary={binary[:3].cpu().numpy()} | contrib={contrib[:3].cpu().numpy()}")
                return total

            reward[idx_v] = (
                sensor_contact(index_sensors, index_contact_bonus, True) +
                sensor_contact(fingertip_sensors, fingertip_contact_bonus, False) +
                sensor_contact(other_link_sensors, other_link_contact_bonus, False) +
                sensor_contact(base_sensors, hand_base_contact_bonus, False)
            )

        total_reward = reward

    except Exception:
        import traceback
        traceback.print_exc()
        total_reward = torch.zeros(num_envs, device=device)

    if not torch.all(torch.isfinite(total_reward)):
        print(f"[ERROR] contact_reward_detailed produced NaN/Inf, force-zeroed")
        total_reward = torch.nan_to_num(total_reward, nan=0.0)


    return total_reward.to(torch.float32)


def tip_trigger_reward(
    env,
    finger_scale: float = 1.0,
    finger_temp: float = 0.08,
    hand_cfg: SceneEntityCfg = SceneEntityCfg("franka"),
    drill_scale: tuple = (1, 1, 1),
    verbose: bool = False,
) -> torch.Tensor:
    hand: Articulation = env.scene[hand_cfg.name]
    drill = env.drill  # multi-variant proxy
    device = env.device
    num_envs = env.num_envs

    # === get trigger point (per-env, using env's cached per-env tensors) ===
    trigger_offset = env._trigger1_offset  # [num_envs, 3] per-env
    scale_per_env = env._drill_scale       # [num_envs, 3] per-env
    trigger_offset_scaled = trigger_offset * scale_per_env  # [num_envs, 3]

    # local coords -> world coords
    drill_pos  = drill.data.root_pos_w
    drill_quat = drill.data.root_quat_w
    R = quat_to_rotmat(drill_quat)  # [num_envs, 3, 3]
    trigger_center_world = torch.bmm(R, trigger_offset_scaled.unsqueeze(-1)).squeeze(-1) + drill_pos  # [num_envs, 3]

    # === get R_index_intermediate position ===
    body_names = hand.data.body_names
    if "R_index_intermediate" not in body_names:
        if verbose:
            print("[WARN] tip_trigger_reward: R_index_intermediate not found")
        return torch.zeros(num_envs, device=device, dtype=torch.float32)

    index_idx = body_names.index("R_index_intermediate")
    intermediate_pos_world = hand.data.body_pos_w[:, index_idx, :]  # [num_envs, 3]

    # === compute reward ===
    dist   = torch.norm(intermediate_pos_world - trigger_center_world, dim=-1)

    precise_bonus = torch.exp(-dist / 0.015) * 50

    reward = torch.exp(-dist / finger_temp)


    if verbose:
        print(f"[TIP_TRIGGER] dist:   mean={dist.mean().item():.4f}m  min={dist.min().item():.4f}m")
        print(f"[TIP_TRIGGER] reward: mean={reward.mean().item():.4f}  max={reward.max().item():.4f}")

    result = (reward * finger_scale).to(torch.float32) + precise_bonus
    nan_mask = torch.isnan(result) | torch.isinf(result)
    if nan_mask.any():
        result = torch.where(nan_mask, torch.zeros_like(result), result)

    # === Debug viz: index_intermediate -> trigger_center ===
    debug_mode = getattr(env, 'debug', False)
    if debug_mode:
        if not hasattr(env, '_tip_trigger_vis_counter'):
            env._tip_trigger_vis_counter = 0
        env._tip_trigger_vis_counter += 1
        
        if env._tip_trigger_vis_counter % 1 == 0:  # draw every 5 steps, change this number
            try:
                import isaacsim.util.debug_draw._debug_draw as omni_debug_draw
                import numpy as np
                draw_interface = omni_debug_draw.acquire_debug_draw_interface()
                num_vis_envs = min(3, num_envs)
                tip_lines_src, tip_lines_tgt       = [], []
                tip_points_pos, tip_points_colors, tip_points_sizes = [], [], []

                for env_idx in range(num_vis_envs):
                    tip_pos    = intermediate_pos_world[env_idx].cpu().numpy()
                    target_pos = trigger_center_world[env_idx].cpu().numpy()
                    dist_val   = dist[env_idx].item()

                    # print(f"[ENV {env_idx}] INDEX_INTERMEDIATE->TRIGGER_CENTER dist = {dist_val:.4f} m")

                    tip_lines_src.append(tip_pos.tolist())
                    tip_lines_tgt.append(target_pos.tolist())

                    # orange=index intermediate joint, white=trigger centroid
                    tip_points_pos   += [tip_pos.tolist(), target_pos.tolist()]
                    tip_points_colors += [[1.0, 0.5, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
                    tip_points_sizes  += [40.0, 30.0]

                if tip_lines_src:
                    draw_interface.draw_lines(
                        tip_lines_src, tip_lines_tgt,
                        [[1.0, 0.5, 0.0, 1.0]] * len(tip_lines_src),
                        [8.0] * len(tip_lines_src),
                    )
                if tip_points_pos:
                    draw_interface.draw_points(tip_points_pos, tip_points_colors, tip_points_sizes)

            except Exception:
                import traceback
                traceback.print_exc()

    return result

def thumb_approach_reward(
    env,
    finger_scale: float = 1.0,
    finger_temp: float = 0.08,
    hand_cfg: SceneEntityCfg = SceneEntityCfg("franka"),
    drill_scale: tuple = (1, 1, 1),
    verbose: bool = False,
) -> torch.Tensor:
    hand: Articulation = env.scene[hand_cfg.name]
    drill = env.drill  # multi-variant proxy
    device = env.device
    num_envs = env.num_envs

    # Read thumb target from env (per-env tensor set during _reset_idx)
    thumb_target_local = env._thumb_target_local  # [num_envs, 3]
    scale_per_env = env._drill_scale              # [num_envs, 3]
    thumb_target_local_scaled = thumb_target_local * scale_per_env  # match trigger_offset treatment

    # === local coords -> world coords ===
    drill_pos  = drill.data.root_pos_w
    drill_quat = drill.data.root_quat_w
    R = quat_to_rotmat(drill_quat)
    thumb_target_world = torch.bmm(R, thumb_target_local_scaled.unsqueeze(-1)).squeeze(-1) + drill_pos

    # === get thumb_distal position ===
    body_names = hand.body_names
    thumb_distal_candidates = [name for name in body_names if "thumb" in name and "distal" in name]
    if len(thumb_distal_candidates) == 0:
        if verbose:
            print("[WARN] thumb_approach_reward: thumb_distal link not found")
        return torch.zeros(num_envs, device=device, dtype=torch.float32)

    thumb_distal_name = thumb_distal_candidates[0]
    thumb_idx = body_names.index(thumb_distal_name)
    thumb_distal_pos_world = hand.data.body_pos_w[:, thumb_idx, :]  # [num_envs, 3]

    # === compute reward ===
    dist   = torch.norm(thumb_distal_pos_world - thumb_target_world, dim=-1)
    # print(f"[thumb_approach] dist env0={dist[0]:.4f} env1={dist[1]:.4f} env2={dist[2]:.4f} m")
    reward = torch.exp(-dist / finger_temp)


    result = (reward * finger_scale).to(torch.float32)
    nan_mask = torch.isnan(result) | torch.isinf(result)
    if nan_mask.any():
        result = torch.where(nan_mask, torch.zeros_like(result), result)

    # === Debug viz: thumb_distal -> target point ===
    debug_mode = getattr(env, 'debug', False)
    if debug_mode:
        try:
            import isaacsim.util.debug_draw._debug_draw as omni_debug_draw
            import numpy as np
            draw_interface = omni_debug_draw.acquire_debug_draw_interface()

            num_vis_envs = min(3, num_envs)
            tip_lines_src, tip_lines_tgt = [], []
            tip_points_pos, tip_points_colors, tip_points_sizes = [], [], []

            for env_idx in range(num_vis_envs):
                thumb_pos  = thumb_distal_pos_world[env_idx].cpu().numpy()
                target_pos = thumb_target_world[env_idx].cpu().numpy()
                dist_val   = dist[env_idx].item()

                

                tip_lines_src.append(thumb_pos.tolist())
                tip_lines_tgt.append(target_pos.tolist())

                # purple=thumb tip, white=target point
                tip_points_pos    += [thumb_pos.tolist(), target_pos.tolist()]
                tip_points_colors += [[0.8, 0.2, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
                tip_points_sizes  += [40.0, 30.0]

            if tip_lines_src:
                draw_interface.draw_lines(
                    tip_lines_src, tip_lines_tgt,
                    [[0.8, 0.2, 1.0, 1.0]] * len(tip_lines_src),
                    [8.0] * len(tip_lines_src),
                )
            if tip_points_pos:
                draw_interface.draw_points(tip_points_pos, tip_points_colors, tip_points_sizes)

        except Exception:
            import traceback
            traceback.print_exc()

    return result
def approach_reward_improved(
    env,
    finger_scale: float = 1.0,
    finger_temp: float = 0.02,
    hand_cfg: SceneEntityCfg = SceneEntityCfg("franka"),
    drill_scale: tuple = (1, 1, 1),
    verbose: bool = False,
) -> torch.Tensor:
    hand: Articulation = env.scene[hand_cfg.name]
    drill = env.drill  # multi-variant proxy
    device = env.device
    num_envs = env.num_envs

    # === get drill variant info ===
    use_split = False
    try:
        all_variant_ids = env._drill_variant_indices.tolist()
        use_split = True
    except Exception:
        all_variant_ids = []

    NUM_SAMPLE_POINTS_BODY_FILTERED = 600

    drill_pos = drill.data.root_pos_w
    drill_quat = drill.data.root_quat_w

    R_all = quat_to_rotmat(drill_quat)  # [num_envs, 3, 3]
    scales_tensor = env._drill_scale  # [num_envs, 3]

    if use_split:
        unique_vids = env._drill_variant_indices.unique().tolist()
        # pre-downsample: compute the post-sampling point count for each variant
        var_sampled_pts = {}
        for vid in unique_vids:
            vdata = env._variant_attrs[vid]
            bp = vdata.get("body_surface_points", None)
            if bp is None or len(bp) == 0:
                var_sampled_pts[vid] = None
            else:
                n = len(bp)
                if n > NUM_SAMPLE_POINTS_BODY_FILTERED:
                    bp = bp[::n // NUM_SAMPLE_POINTS_BODY_FILTERED][:NUM_SAMPLE_POINTS_BODY_FILTERED]
                var_sampled_pts[vid] = bp  # [N, 3] preprocessed at env init

        max_body_pts = max(
            pts.shape[0] for pts in var_sampled_pts.values() if pts is not None
        ) if var_sampled_pts else 0

        if max_body_pts == 0:
            return torch.zeros(num_envs, device=device, dtype=torch.float32)

        body_pts_padded = torch.zeros((num_envs, max_body_pts, 3), device=device)
        body_pts_mask = torch.zeros((num_envs, max_body_pts), dtype=torch.bool, device=device)

        for vid in unique_vids:
            env_mask = env._drill_variant_indices == vid
            env_idx_list = env_mask.nonzero(as_tuple=True)[0]
            if len(env_idx_list) == 0:
                continue

            bp = var_sampled_pts.get(vid)
            if bp is None or len(bp) == 0:
                continue

            scale = env._variant_attrs[vid]["scale"]  # [3], same for all envs of the same variant
            bp_scaled = bp * scale                     # [N, 3]
            n_pts = bp_scaled.shape[0]

            R_v = R_all[env_idx_list]                    # [n, 3, 3]
            pos_v = drill_pos[env_idx_list]                # [n, 3]

            # batch transform: [n, 3, N] -> [n, N, 3]
            bp_exp = bp_scaled.T.unsqueeze(0).expand(len(env_idx_list), 3, -1)  # [n, 3, N]
            pts_w = torch.bmm(R_v, bp_exp).transpose(1, 2) + pos_v.unsqueeze(1)  # [n, N, 3]

            body_pts_padded[env_idx_list, :n_pts] = pts_w
            body_pts_mask[env_idx_list, :n_pts] = True

    else:
        body_pts_padded = torch.zeros((num_envs, 1, 3), device=device)
        body_pts_mask = torch.zeros((num_envs, 1), dtype=torch.bool, device=device)
        max_body_pts = 1

    # link positions
    body_names = hand.body_names
    hand_link_mask = torch.tensor(
        [name.startswith("R_")
        and "index" not in name
        and ("thumb_proximal_base" in name or "thumb" not in name)
        and name != "R_hand_base_link"
        for name in body_names], device=device
    )
    hand_names = [name for name in body_names
                if name.startswith("R_")
                and "index" not in name
                and ("thumb_proximal_base" in name or "thumb" not in name)
                and name != "R_hand_base_link"
                ]
    num_hand = len(hand_names)
    hand_pos_all = hand.data.body_pos_w[:, hand_link_mask, :]

    # === all links use points from the body_mask region ===
    threshold_far = 1.0

    if max_body_pts > 0 and body_pts_mask.any():
        # batch distance: [n_env, n_links, 3] x [n_env, max_pts, 3] -> [n_env, n_links, max_pts]
        all_dists = torch.cdist(hand_pos_all, body_pts_padded)  # [n_env, n_links, max_pts]
        all_dists = all_dists.masked_fill(~body_pts_mask.unsqueeze(1), float('inf'))
        dists_min = all_dists.min(dim=2).values  # [n_env, n_links]
        dists_min = torch.where(torch.isinf(dists_min),
                               torch.tensor(threshold_far + 1.0, device=device),
                               dists_min)
    else:
        dists_min = torch.full((num_envs, num_hand), threshold_far + 1.0, device=device)

    # batch-compute reward
    valid_mask = dists_min < threshold_far
    reward_per_link = torch.where(valid_mask, torch.exp(-dists_min / finger_temp),
                                  torch.zeros_like(dists_min))
    body_approach_reward = reward_per_link.sum(dim=1)

    approach_reward = body_approach_reward
    result = (approach_reward * finger_scale).to(torch.float32)

    nan_mask = torch.isnan(result) | torch.isinf(result)
    if nan_mask.any():
        result = torch.where(nan_mask, torch.zeros_like(result), result)

    # === DEBUG viz: each env uses its own variant body_mask region ===
    debug_mode = getattr(env, 'debug', False)
    if debug_mode:
        if not hasattr(env, '_reward_debug_counter'):
            env._reward_debug_counter = 0
        env._reward_debug_counter += 1

        if env._reward_debug_counter % 50 == 0:
            print(f"[REWARD DEBUG Step {env._reward_debug_counter}]")
            print(f"  body_approach: mean={body_approach_reward.mean().item():.4f}, min={body_approach_reward.min().item():.4f}, max={body_approach_reward.max().item():.4f}")
            print(f"  total_approach: mean={result.mean().item():.4f}, min={result.min().item():.4f}, max={result.max().item():.4f}")

        try:
            import isaacsim.util.debug_draw._debug_draw as omni_debug_draw
            import numpy as np
            draw_interface = omni_debug_draw.acquire_debug_draw_interface()

            num_vis_envs = min(3, num_envs)

            hand_link_mask_np = [
                name.startswith("R_")
                and "index" not in name
                and ("thumb_proximal_base" in name or "thumb" not in name)
                and name != "R_hand_base_link"
                for name in hand.body_names
            ]

            # Per-variant distinct colors for body points
            variant_colors = [
                [1.0, 0.4, 0.4, 1.0],   # red  — drill2 (variant 0)
                [0.4, 1.0, 0.4, 1.0],   # green — drill_blue (variant 1)
                [1.0, 0.9, 0.2, 1.0],   # yellow — drill_yellow (variant 2)
            ]

            # Per-env world points
            all_body_filtered_world = []
            for env_idx in range(num_vis_envs):
                vid = all_variant_ids[env_idx] if use_split else 0
                if use_split:
                    mask_np = body_pts_mask[env_idx].cpu().numpy()
                    bf = body_pts_padded[env_idx].cpu().numpy()[mask_np]
                else:
                    bf = None
                all_body_filtered_world.append((vid, bf))
                variant_name = env._variant_attrs.get(vid, {}).get("name", "unknown")

            import numpy as np
            # === Detailed debug: print every step of coordinate transform for first env of each variant ===
            for vid_check in range(env.num_drill_variants):
                envs_of_vid = [i for i, v in enumerate(all_variant_ids) if v == vid_check]
                if not envs_of_vid:
                    continue
                env_idx = envs_of_vid[0]
                vid = all_variant_ids[env_idx]
                bf_torch = body_pts_padded[env_idx]
                mask_np = body_pts_mask[env_idx].cpu().numpy()
                bf_np = bf_torch.cpu().numpy()[mask_np] if bf_torch.numel() > 0 and mask_np.any() else None
                drill_root = drill.data.root_pos_w[env_idx].cpu().numpy()
                drill_quat_np = drill.data.root_quat_w[env_idx].cpu().numpy()
                scale_np = scales_tensor[env_idx].cpu().numpy()
                variant_name = env._variant_attrs.get(vid, {}).get("name", "unknown")
                # print(f"[DEBUG-VISUAL] VID{vid} ({variant_name}) env={env_idx}:")
                # print(f"  drill_root_pos (world): [{drill_root[0]:.4f}, {drill_root[1]:.4f}, {drill_root[2]:.4f}]")
                # print(f"  drill_quat (wxyz):     [{drill_quat_np[0]:.4f}, {drill_quat_np[1]:.4f}, {drill_quat_np[2]:.4f}, {drill_quat_np[3]:.4f}]")
                # print(f"  scale:                 [{scale_np[0]:.4f}, {scale_np[1]:.4f}, {scale_np[2]:.4f}]")
                # if bf_np is not None and len(bf_np) > 0:
                #     print(f"  body_filtered world:   {len(bf_np)} pts, x=[{bf_np[:,0].min():.4f},{bf_np[:,0].max():.4f}] y=[{bf_np[:,1].min():.4f},{bf_np[:,1].max():.4f}] z=[{bf_np[:,2].min():.4f},{bf_np[:,2].max():.4f}]")
                # print(f"  mesh_info types: {[info['type'] for info in env._variant_attrs[vid]['mesh_info']]}")

            source_pos, target_pos, lines_colors, line_thicknesses = [], [], [], []
            all_drill_points_pos, all_drill_points_colors, all_drill_points_sizes = [], [], []

            for env_idx in range(num_vis_envs):
                vid, body_w = all_body_filtered_world[env_idx]
                env_color = variant_colors[vid % len(variant_colors)]

                hand_pos_np = hand.data.body_pos_w[env_idx].cpu().numpy()
                hand_pos_filtered = hand_pos_np[hand_link_mask_np]

                # Lines: hand link -> nearest body_filtered point (real world positions)
                if body_w is not None and len(body_w) > 0:
                    for i in range(num_hand):
                        dists = np.linalg.norm(hand_pos_filtered[i] - body_w, axis=1)
                        min_idx = dists.argmin()
                        source_pos.append(hand_pos_filtered[i].tolist())
                        target_pos.append(body_w[min_idx].tolist())
                        lines_colors.append(env_color)
                        line_thicknesses.append(6.0)

                    # Draw filtered body points at REAL world positions (variant color)
                    for pt in body_w.tolist():
                        all_drill_points_pos.append(pt)
                        all_drill_points_colors.append(env_color)
                        all_drill_points_sizes.append(10.0)

            total_filtered = sum(bf.shape[0] for (_, bf) in all_body_filtered_world if bf is not None)
            if source_pos:
                draw_interface.draw_lines(source_pos, target_pos, lines_colors, line_thicknesses)
                # print(f"[DEBUG-VISUAL] Drew {len(source_pos)} lines")
            if all_drill_points_pos:
                import numpy as np
                pts_arr = np.array(all_drill_points_pos)
                # print(f"[DEBUG-VISUAL] Drew {len(all_drill_points_pos)} filtered body pts (total_filtered={total_filtered})")
                # print(f"[DEBUG-VISUAL] Point world range: x=[{pts_arr[:,0].min():.3f}, {pts_arr[:,0].max():.3f}], y=[{pts_arr[:,1].min():.3f}, {pts_arr[:,1].max():.3f}], z=[{pts_arr[:,2].min():.3f}, {pts_arr[:,2].max():.3f}]")
                for env_idx in range(min(num_vis_envs, 5)):
                    vid = all_variant_ids[env_idx] if use_split else 0
                    start = env_idx * NUM_SAMPLE_POINTS_BODY_FILTERED
                    end = start + NUM_SAMPLE_POINTS_BODY_FILTERED
                    env_pts = all_drill_points_pos[start:min(end, len(all_drill_points_pos))]
                    if len(env_pts) > 0:
                        arr = np.array(env_pts)
                        # print(f"[DEBUG-VISUAL] ENV{env_idx}(vid={vid}): {len(env_pts)} pts, x=[{arr[:,0].min():.3f},{arr[:,0].max():.3f}], y=[{arr[:,1].min():.3f},{arr[:,1].max():.3f}], z=[{arr[:,2].min():.3f},{arr[:,2].max():.3f}]")
                draw_interface.draw_points(all_drill_points_pos, all_drill_points_colors, all_drill_points_sizes)

        except Exception as e:
            import traceback
            traceback.print_exc()

    return result



def lift_reward(
    env: ManagerBasedRLEnv,
    lift_height: float = 0.1,
    lift_reward_weight: float = 1.0,
    max_xy_distance: float = 0.1,
    hand_cfg: SceneEntityCfg = SceneEntityCfg("franka"),
    tip_trigger_threshold: float = 0.03,
    debug_draw: bool = False,
) -> torch.Tensor:
    device = env.device
    num_envs = env.num_envs

    drill = env.drill
    drill_pos = drill.data.root_pos_w
    drill_quat = drill.data.root_quat_w

    initial_z  = env.initial_drill_pos[:, 2]
    initial_xy = env.initial_drill_pos[:, :2]

    xy_diff = drill_pos[:, :2] - initial_xy
    xy_distance = torch.sqrt(xy_diff[:, 0]**2 + xy_diff[:, 1]**2)
    not_knocked_away = xy_distance < max_xy_distance

    z_diff = drill_pos[:, 2] - initial_z
    z_diff = torch.where(z_diff < 0, torch.zeros_like(z_diff), z_diff)
    z_diff = torch.clamp(z_diff, min=0.0, max=lift_height)
    reward = lift_reward_weight * (z_diff / lift_height)

    # === trigger point world coords (using variant 0 as representative, acceptable error) ===
    try:
        vid0 = env._drill_variant_indices[0].item()
        trigger_pts, _, _, _ = env._build_variant_surface_points(vid0)
        scale_v = env._variant_attrs[vid0]["scale"]
        trigger_center = (trigger_pts * scale_v).mean(dim=0)  # [3]
    except Exception:
        trigger_center = torch.zeros(3, device=device, dtype=torch.float32)

    w_q, x_q, y_q, z_q = drill_quat[:, 0], drill_quat[:, 1], drill_quat[:, 2], drill_quat[:, 3]
    norm = torch.sqrt(w_q*w_q + x_q*x_q + y_q*y_q + z_q*z_q + 1e-8)
    w_q, x_q, y_q, z_q = w_q/norm, x_q/norm, y_q/norm, z_q/norm
    R = torch.zeros((num_envs, 3, 3), device=device, dtype=torch.float32)
    R[:, 0, 0] = 1 - 2*(y_q*y_q + z_q*z_q)
    R[:, 0, 1] = 2*(x_q*y_q - w_q*z_q)
    R[:, 0, 2] = 2*(x_q*z_q + w_q*y_q)
    R[:, 1, 0] = 2*(x_q*y_q + w_q*z_q)
    R[:, 1, 1] = 1 - 2*(x_q*x_q + z_q*z_q)
    R[:, 1, 2] = 2*(y_q*z_q - w_q*x_q)
    R[:, 2, 0] = 2*(x_q*z_q - w_q*y_q)
    R[:, 2, 1] = 2*(y_q*z_q + w_q*x_q)
    R[:, 2, 2] = 1 - 2*(x_q*x_q + y_q*y_q)
    trigger_world = torch.bmm(
        R, trigger_center.view(1, 3, 1).expand(num_envs, -1, -1)
    ).squeeze(-1) + drill_pos  # [num_envs, 3]

    # === index intermediate position (same as tip_trigger_reward, no offset) ===
    hand: Articulation = env.scene[hand_cfg.name]
    body_names_full = list(hand.body_names)
    if "R_index_intermediate" in body_names_full:
        idx_body = body_names_full.index("R_index_intermediate")
        intermediate_pos = hand.data.body_pos_w[:, idx_body, :]
        tip_trig_dist = torch.norm(intermediate_pos - trigger_world, dim=-1)
        tip_close_enough = tip_trig_dist < tip_trigger_threshold
        tip_dist_val = tip_trig_dist
    else:
        tip_close_enough = torch.zeros(num_envs, dtype=torch.bool, device=device)
        tip_dist_val = torch.zeros(num_envs, device=device)

    reward = reward  * not_knocked_away.float()

    debug_mode = getattr(env, 'debug', False) or debug_draw
    # if debug_mode:
    #     try:
    #         init = env.initial_drill_pos[0].cpu().numpy()
    #         cur  = drill_pos[0].cpu().numpy()
    #         dist_val = tip_dist_val[0].item()
    #         print(
    #             f"[LIFT DEBUG env0] "
    #             f"init=({init[0]:.3f}, {init[1]:.3f}, {init[2]:.3f}) | "
    #             f"cur=({cur[0]:.3f}, {cur[1]:.3f}, {cur[2]:.3f}) | "
    #             f"tip_trig_dist={dist_val:.4f} | allowed={tip_close_enough[0].item()}"
    #         )
    #     except Exception as e:
    #         import traceback
    #         traceback.print_exc()

    return reward.to(torch.float32)
def success_reward(
    env: ManagerBasedRLEnv,
    base_weight: float = 200.0,
    bonus_per_contact: float = 18.18,
    success_contact_force_threshold: float = 0.05,
    drill_scale: tuple = (1, 1.2, 1),
    hand_cfg: SceneEntityCfg = SceneEntityCfg("franka"),
) -> torch.Tensor:
    device = env.device
    num_envs = env.num_envs

    drill = env.drill
    hand: Articulation = env.scene[hand_cfg.name]

    drill_pos  = drill.data.root_pos_w
    drill_quat = drill.data.root_quat_w

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

    contact_count = torch.zeros(num_envs, device=device, dtype=torch.float32)
    thumb_intermediate_contact = torch.zeros(num_envs, device=device, dtype=torch.bool)
    thumb_proximal_contact = torch.zeros(num_envs, device=device, dtype=torch.bool)
    index_intermediate_contact = torch.zeros(num_envs, device=device, dtype=torch.bool)

    try:
        if hasattr(env, 'scene') and hasattr(env.scene, 'sensors'):
            for sensor_name in all_hand_sensors:
                if sensor_name in env.scene.sensors:
                    sensor = env.scene.sensors[sensor_name]
                    force_matrix = sensor.data.force_matrix_w
                    if force_matrix is not None:
                        force_matrix = torch.nan_to_num(force_matrix, nan=0.0, posinf=0.0, neginf=0.0)
                        force_matrix = torch.clamp(force_matrix, -50.0, 50.0)
                        force = force_matrix.sum(dim=(1, 2))
                    else:
                        force = sensor.data.net_forces_w.sum(dim=1)
                    force_mag = torch.norm(force, dim=1)
                    force_mag = torch.nan_to_num(force_mag, nan=0.0)
                    has_contact = force_mag > success_contact_force_threshold
                    contact_count += has_contact.float()
                    # separately track the thumb's two key contact points
                    if sensor_name == "contact_thumb_intermediate":
                        thumb_intermediate_contact = has_contact.clone()
                    elif sensor_name == "contact_thumb_proximal":
                        thumb_proximal_contact = has_contact.clone()
                    elif sensor_name == "contact_index_intermediate":
                        index_intermediate_contact = has_contact.clone()
    except Exception:
        pass

    trigger_offset = env._trigger1_offset  # [num_envs, 3] per-env
    scale_per_env = env._drill_scale      # [num_envs, 3] per-env
    trigger_offset_scaled = trigger_offset * scale_per_env  # [num_envs, 3]

    w, x, y, z = drill_quat[:, 0], drill_quat[:, 1], drill_quat[:, 2], drill_quat[:, 3]
    norm = torch.sqrt(w*w + x*x + y*y + z*z + 1e-8)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    R = torch.zeros((num_envs, 3, 3), device=device, dtype=torch.float32)
    R[:, 0, 0] = 1 - 2*(y*y + z*z);  R[:, 0, 1] = 2*(x*y - w*z);    R[:, 0, 2] = 2*(x*z + w*y)
    R[:, 1, 0] = 2*(x*y + w*z);      R[:, 1, 1] = 1 - 2*(x*x + z*z); R[:, 1, 2] = 2*(y*z - w*x)
    R[:, 2, 0] = 2*(x*z - w*y);      R[:, 2, 1] = 2*(y*z + w*x);    R[:, 2, 2] = 1 - 2*(x*x + y*y)

    trigger_center_world = torch.bmm(R, trigger_offset_scaled.unsqueeze(-1)).squeeze(-1) + drill_pos  # [num_envs, 3]

    # === index_intermediate to trigger distance ===
    body_names = list(hand.body_names)
    if "R_index_intermediate" not in body_names:
        return torch.zeros(num_envs, device=device, dtype=torch.float32)

    index_idx = body_names.index("R_index_intermediate")
    intermediate_pos_world = hand.data.body_pos_w[:, index_idx, :]

    dist = torch.norm(intermediate_pos_world - trigger_center_world, dim=-1)
    dist_ok = dist < 0.023

    critical_contact_ok = (thumb_intermediate_contact | thumb_proximal_contact) & index_intermediate_contact
    dist_ok_f = dist_ok.float()

    success_base = (critical_contact_ok & dist_ok).float() * base_weight
    extra_contacts = torch.clamp(contact_count - 2.0, min=0.0)
    extra_reward = extra_contacts * bonus_per_contact * dist_ok_f * critical_contact_ok.float()

    reward = success_base + extra_reward
    reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

    debug_mode = getattr(env, 'debug', False)
    if debug_mode:
        for env_idx in range(min(3, num_envs)):
            variant_id = getattr(env, '_drill_variant_indices', None)
            variant_name = "unknown"
            if variant_id is not None and env_idx < env.num_envs:
                vid = variant_id[env_idx].item()
                variant_name = env._variant_attrs.get(vid, {}).get("name", "unknown")
            q = drill_quat[env_idx].cpu().numpy()
            pos = drill_pos[env_idx].cpu().numpy()
            up_axis = getattr(env, '_up_axis', None)
            up_val = "?"
            if up_axis is not None:
                uv = up_axis[env_idx].cpu().numpy()
                qw, qx, qy, qz = q[0], q[1], q[2], q[3]
                r_x = 2.0 * (qx * qz - qy * qw)
                r_y = 2.0 * (qy * qz + qw * qx)
                r_z = 1.0 - 2.0 * (qx * qx + qy * qy)
                up_val = f"{(uv[0]*r_x + uv[1]*r_y + uv[2]*r_z):.4f}"
            # print(
            #     f"pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}) "
            #     f"quat(wxyz)={q} up_z={up_val} | "
            # )

    return reward.to(torch.float32)


class TriggerTargetCache:
    def __init__(self):
        self.target_offset = None
        self.initialized = False

    def initialize(self, drill: RigidObject, stage):
        if self.initialized:
            return
        try:
            from pxr import UsdGeom, Usd
            drill_prim_path = str(drill.cfg.prim_path).replace("env_.*", "env_0")
            drill_prim = stage.GetPrimAtPath(drill_prim_path)

            if drill_prim.IsValid():
                trigger_path = f"{drill_prim_path}/Trigger_1/Object_2"
                trigger_prim = stage.GetPrimAtPath(trigger_path)

                if trigger_prim.IsValid():
                    trigger_xform = UsdGeom.Xformable(trigger_prim)
                    trigger_world = trigger_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    trigger_pos = trigger_world.ExtractTranslation()

                    drill_xform = UsdGeom.Xformable(drill_prim)
                    drill_world = drill_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    drill_pos = drill_world.ExtractTranslation()

                    self.target_offset = np.array([
                        trigger_pos[0] - drill_pos[0],
                        trigger_pos[1] - drill_pos[1],
                        trigger_pos[2] - drill_pos[2]
                    ])
                    self.initialized = True
                    return

            self.target_offset = np.array([0.0, 0.0, 0.05])
            self.initialized = True
        except:
            self.target_offset = np.array([0.0, 0.0, 0.05])
            self.initialized = True

    def get_target_world_pos(self, drill_pos: torch.Tensor, drill_quat: torch.Tensor) -> torch.Tensor:
        if not self.initialized or self.target_offset is None:
            return drill_pos + torch.tensor([0.0, 0.0, 0.05], device=drill_pos.device)
        offset = torch.tensor(self.target_offset, dtype=torch.float32, device=drill_pos.device)
        offset_world = quat_apply(drill_quat, offset.unsqueeze(0).repeat(drill_pos.shape[0], 1))
        return drill_pos + offset_world


_trigger_target_cache = None


def get_trigger_target_cache() -> TriggerTargetCache:
    global _trigger_target_cache
    if _trigger_target_cache is None:
        _trigger_target_cache = TriggerTargetCache()
    return _trigger_target_cache





def drill_y_axis_tilt_penalty(
    env: ManagerBasedRLEnv,
    penalty: float = -0.02,
    tilt_threshold_deg: float = 15.0,
) -> torch.Tensor:
    device = env.device
    num_envs = env.num_envs

    drill = env.drill
    drill_quat = drill.data.root_quat_w  # [num_envs, 4], format (w, x, y, z)

    qw = drill_quat[:, 0]
    qx = drill_quat[:, 1]
    qy = drill_quat[:, 2]
    qz = drill_quat[:, 3]

    # rotation matrix elements: local axis · world +Z
    # R[2,0] = local+X · world+Z = 2*(qx*qz - qy*qw)
    # R[2,1] = local+Y · world+Z = 2*(qy*qz + qw*qx)
    # R[2,2] = local+Z · world+Z = 1 - 2*(qx*qx + qy*qy)
    r_x = 2.0 * (qx * qz - qy * qw)
    r_y = 2.0 * (qy * qz + qw * qx)
    r_z = 1.0 - 2.0 * (qx * qx + qy * qy)

    # Use per-env up_axis from _up_axis tensor
    up_z = (
        env._up_axis[:, 0] * r_x +
        env._up_axis[:, 1] * r_y +
        env._up_axis[:, 2] * r_z
    )

    # larger |up_z| = more vertical; acos(|up_z|) gives the tilt angle
    up_z_clamped = torch.clamp(torch.abs(up_z), -1.0, 1.0)
    tilt_angle_deg = torch.acos(up_z_clamped) * 180.0 / torch.pi

    # continuous penalty: beyond the threshold, penalty grows linearly with tilt
    # tilt = tilt_threshold_deg -> 0
    # tilt = 60deg -> penalty
    # tilt = 90deg -> 2 * penalty
    # formula: reward = penalty * (tilt - threshold) / (60 - threshold)
    reward = torch.zeros(num_envs, device=device, dtype=torch.float32)
    if tilt_threshold_deg < 60.0:
        slope = penalty / (60.0 - tilt_threshold_deg)
        excess = tilt_angle_deg - tilt_threshold_deg
        reward = slope * torch.clamp(excess, min=0.0)
    else:
        # fallback: full penalty only after tilt > 60deg
        reward = torch.where(
            tilt_angle_deg > tilt_threshold_deg,
            torch.full_like(tilt_angle_deg, penalty),
            torch.zeros_like(tilt_angle_deg),
        )

    # debug print
    # debug_mode = getattr(env, 'debug', False)
    # if debug_mode:
    #     if not hasattr(env, '_y_tilt_debug_counter'):
    #         env._y_tilt_debug_counter = 0
    #     env._y_tilt_debug_counter += 1
    #     if env._y_tilt_debug_counter % 50 == 0:
    #         print(
    #             f"[Y-TILT] tilt_angle: mean={tilt_angle_deg.mean().item():.1f}°  "
    #             f"max={tilt_angle_deg.max().item():.1f}°  "
    #             f"penalized={(tilt_angle_deg > tilt_threshold_deg).float().mean().item()*100:.1f}%  "
    #             f"penalty_mean={reward.mean().item():.4f}"
    #         )

    return reward.to(torch.float32)


# penalty for excessive sensor contact force (force > 15N)
# +1 penalty per 5N over: penalty = max(0, floor((fmag - 15) / 5))
def weak_contact_penalty(
    env,
    force_high: float = 15.0,
) -> torch.Tensor:
    num_envs = env.num_envs
    device = env.device

    all_sensors = [
        "contact_thumb_distal",
        "contact_thumb_intermediate",
        "contact_middle_intermediate",
        "contact_ring_intermediate",
        "contact_pinky_intermediate",
        "contact_index_intermediate",
        "contact_thumb_proximal",
        "contact_thumb_proximal_base",
        "contact_index_proximal",
        "contact_middle_proximal",
        "contact_ring_proximal",
        "contact_pinky_proximal",
        "contact_hand_base",
    ]

    total_penalty = torch.zeros(num_envs, device=device)

    for sname in all_sensors:
        if sname not in env.scene.sensors:
            continue

        s = env.scene.sensors[sname]
        fm = s.data.force_matrix_w
        fm = torch.nan_to_num(fm, nan=0.0, posinf=0.0, neginf=0.0)
        fm = torch.clamp(fm, -100.0, 100.0)
        fmag = torch.norm(fm.sum(dim=(1, 2)), dim=1)
        fmag = torch.nan_to_num(fmag, nan=0.0)
        excess = torch.clamp(fmag - force_high, min=0.0)
        penalty = excess  
        total_penalty += penalty

    return (-total_penalty).to(torch.float32)


# ---------------------------------------------------------------------------
def register_all_rewards():
    REWARD_REGISTRY.update({
        # existing rewards (kept)
        'lift': lift_reward,
        'approach_improved': approach_reward_improved,
        'tip_trigger': tip_trigger_reward,
        'success': success_reward,
        'contact_detailed': contact_reward_detailed,  # detailed contact reward (from the paper)
        'y_axis_tilt_penalty': drill_y_axis_tilt_penalty,  # drill Y-axis tilt penalty (>30deg gives -0.02)
        'thumb_distal_reward': thumb_approach_reward,  # thumb tip reward
        'weak_contact_penalty': weak_contact_penalty,  # excessive contact force penalty (force > 15N)
    })


register_all_rewards()


# ===========================================================================
# Stage2 rewards (drill alignment task) -- merged from the former stage2_rewards.py.
# These reuse quat_to_rotmat defined above. Currently unused by the live stage2
# path (stage2_env._get_rewards computes rewards inline); kept here as a library.
# ===========================================================================

# ---------------------------------------------------------------------------
# R1: alignment reward (position + rotation, sparse)
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
    Sparse alignment reward:
    - position error < pos_thresh (1mm) gives pos_weight
    - rotation error (quaternion dot) > rot_thresh gives rot_weight
    - linear partial reward in between

    Target pose read from env.target_pos / env.target_quat.
    """
    num_envs = env.num_envs
    device = env.device

    drill_pos = env.drill.data.root_pos_w
    drill_quat = env.drill.data.root_quat_w
    target_pos = env.target_stage2_alignment_rewardpos
    target_quat = env.target_quat

    # position error
    pos_error = torch.norm(drill_pos - target_pos, dim=1)
    pos_success = pos_error < pos_thresh
    pos_partial = (pos_error < pos_partial_start) & ~pos_success
    pos_reward = torch.zeros(num_envs, device=device, dtype=torch.float32)
    pos_reward[pos_success] = pos_weight
    pos_reward[pos_partial] = pos_weight * 0.5 * (1 - pos_error[pos_partial] / pos_partial_start)

    # rotation error (quaternion dot, abs to avoid sign ambiguity)
    # q1 . q2 = w1*w2 + x1*x2 + y1*y2 + z1*z2; 1=aligned, 0=90deg, -1=180deg
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
# R2: grasp-quality maintenance reward
# ---------------------------------------------------------------------------

def stage2_grasp_quality_reward(
    env: ManagerBasedRLEnv,
    finger_dist_thresh: float = 0.03,
    min_contacts: int = 6,
    force_thresh: float = 0.1,
    weight: float = 80.0,
) -> torch.Tensor:
    """
    Maintain the grasp quality from Stage1 success:
    - R_index_intermediate near the trigger (< 3cm)
    - at least min_contacts sensor contact forces > force_thresh
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

    # weighting: partial satisfaction gives partial reward
    reward = torch.zeros(num_envs, device=device, dtype=torch.float32)
    both_ok = finger_near_trigger & multi_contact_ok
    reward[both_ok] = weight
    any_ok = finger_near_trigger | multi_contact_ok
    partial = any_ok & ~both_ok
    reward[partial] = weight * 0.3

    return reward


# ---------------------------------------------------------------------------
# R3: tip2tip distance reward
# ---------------------------------------------------------------------------

def stage2_tip_distance_reward(
    env: ManagerBasedRLEnv,
    temp: float = 0.03,
    weight: float = 15.0,
) -> torch.Tensor:
    """
    Gaussian distance reward between thumb_distal and R_index_intermediate:
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
