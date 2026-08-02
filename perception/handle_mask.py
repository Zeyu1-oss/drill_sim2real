"""Per-point labels for "is this camera point on the drill handle" (is_handle),
used to train a point-cloud segmentation network.

Label definition (same geometric source as the teacher's body-approach reward):
  is_handle(p) = (1) p is close enough to the drill handle surface (<threshold)
             AND (2) p is closer to the drill than to the robot (hand/arm FK points)
                     (excludes finger points touching the handle during contact)

Handle surface = env._variant_attrs[vid]["body_surface_points"] (drill-local, already
filtered to the graspable body/handle region by body_mask), scaled and rigid-transformed
to world with the drill's ground-truth pose (reuses the same transform as
grasp_drill_env._compute_link_body_dists).

Sim-only (needs ground-truth drill pose); the produced labels supervise a small
"point-cloud -> is_handle" network, which outputs the mask at deploy time without GT.
"""
import torch
import torch.nn.functional as F


def _handle_points_local(env, env_idx):
    """Return handle-surface points (env-local) for the given env subset, grouped by variant.
    Returns list[(idx_tensor, pts_local (n, N, 3))]; N differs per variant, hence grouped."""
    from isaaclab.utils import math as math_utils
    drill_pos_w = env.drill.data.root_pos_w
    drill_quat = F.normalize(env.drill.data.root_quat_w, dim=1)
    R_drill = math_utils.matrix_from_quat(drill_quat)
    env_origins = env.scene.env_origins
    vids = env._drill_variant_indices

    groups = []
    for vid in vids[env_idx].unique().tolist():
        sub = env_idx[(vids[env_idx] == vid)]
        vdata = env._variant_attrs[vid]
        body_pts = vdata.get("body_surface_points", None)
        if body_pts is None or len(body_pts) == 0:
            continue
        scaled = body_pts * vdata["scale"]                                  # (N,3) local x scale
        R_v = R_drill[sub]; pos_v = drill_pos_w[sub]; eo = env_origins[sub]
        pts_w = (torch.bmm(R_v, scaled.T.unsqueeze(0).expand(len(sub), 3, -1))
                 .transpose(1, 2) + pos_v.unsqueeze(1))                     # (n,N,3) world
        groups.append((sub, pts_w - eo.unsqueeze(1)))                       # env-local
    return groups


def compute_handle_mask(query_pc_local, env, robot_pc_local=None, threshold=0.015):
    """query_pc_local: (B, Q, 3) env-local camera points (points to label).
    robot_pc_local:  (B, R, 3) env-local robot FK points, used to drop finger points on the
                     handle; None to skip.
    threshold: distance threshold (m) for "on the handle surface".
    Return is_handle: (B, Q) bool.
    """
    B, Q, _ = query_pc_local.shape
    dev = query_pc_local.device
    d_handle = torch.full((B, Q), float("inf"), device=dev)

    all_idx = torch.arange(B, device=dev)
    for sub, pts_local in _handle_points_local(env, all_idx):
        q = query_pc_local[sub]                                            # (n,Q,3)
        d_handle[sub] = torch.cdist(q, pts_local).min(dim=2).values        # (n,Q) nearest handle dist

    is_handle = d_handle < threshold
    if robot_pc_local is not None and robot_pc_local.shape[1] > 0:
        d_robot = torch.cdist(query_pc_local, robot_pc_local).min(dim=2).values  # (B,Q)
        is_handle = is_handle & (d_handle < d_robot)                       # drop points closer to hand
    return is_handle
