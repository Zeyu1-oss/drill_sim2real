

import sys
from typing import Optional, Tuple

import torch

_PYTORCH3D_FPS = None


def init_fps_kernel(pytorch3d_path: Optional[str] = None) -> bool:

    global _PYTORCH3D_FPS
    if _PYTORCH3D_FPS is not None:
        return True
    try:
        if pytorch3d_path is not None:
            sys.path.insert(0, pytorch3d_path)
        from pytorch3d.ops import sample_farthest_points as _fps
        _PYTORCH3D_FPS = _fps
        return True
    except Exception:
        _PYTORCH3D_FPS = None
        return False


def farthest_point_sample_torch(points: torch.Tensor, num_points: int) -> torch.Tensor:

    B, N, _ = points.shape
    if N <= num_points:
        return torch.arange(N, device=points.device).unsqueeze(0).expand(B, -1)

    if _PYTORCH3D_FPS is not None and points.is_cuda:
        _, idx = _PYTORCH3D_FPS(points.contiguous(), K=num_points, random_start_point=False)
        return idx

    device = points.device
    result = torch.zeros(B, num_points, dtype=torch.long, device=device)
    batch_idx = torch.arange(B, device=device)

    min_dists = torch.full((B, N), float("inf"), device=device)
    farthest = torch.zeros(B, dtype=torch.long, device=device)
    result[:, 0] = farthest

    for i in range(1, num_points):
        last_farthest_expanded = points[batch_idx, farthest]
        dists = torch.sum((points - last_farthest_expanded.unsqueeze(1)) ** 2, dim=-1)
        min_dists = torch.minimum(min_dists, dists)
        farthest = torch.argmax(min_dists, dim=1)
        result[:, i] = farthest

    return result


def depth_to_pointcloud_batch(
    depth: torch.Tensor,
    intrinsic: torch.Tensor,
    pos_w: torch.Tensor,
    quat_w: torch.Tensor,
    workspace,
    num_points: int,
    env_origins: Optional[torch.Tensor] = None,
    pool_size: int = 2048,
    diag_only: bool = False,
):

    B, H, W, _ = depth.shape
    device = depth.device
    HW = H * W

    j_coords = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, W)
    i_coords = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1)
    cx = intrinsic[:, 0, 2].view(B, 1, 1)
    cy = intrinsic[:, 1, 2].view(B, 1, 1)
    fx = intrinsic[:, 0, 0].view(B, 1, 1)
    fy = intrinsic[:, 1, 1].view(B, 1, 1)

    z = depth[..., 0].float()
    z = torch.where(torch.isfinite(z), z, torch.zeros_like(z))
    valid_depth = z > 0

    x = (j_coords - cx) * z / fx
    y = (i_coords - cy) * z / fy
    points_cam_flat = torch.stack([x, y, z], dim=-1).view(B, HW, 3)

    q = quat_w.float()
    q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)
    qw, qx, qy, qz = q.unbind(dim=-1)
    rot = torch.empty(B, 3, 3, device=device, dtype=torch.float32)
    rot[:, 0, 0] = 1 - 2*(qy*qy + qz*qz); rot[:, 0, 1] = 2*(qx*qy - qw*qz); rot[:, 0, 2] = 2*(qx*qz + qw*qy)
    rot[:, 1, 0] = 2*(qx*qy + qw*qz); rot[:, 1, 1] = 1 - 2*(qx*qx + qz*qz); rot[:, 1, 2] = 2*(qy*qz - qw*qx)
    rot[:, 2, 0] = 2*(qx*qz - qw*qy); rot[:, 2, 1] = 2*(qy*qz + qw*qx); rot[:, 2, 2] = 1 - 2*(qx*qx + qy*qy)

    points_world = torch.bmm(points_cam_flat, rot.transpose(1, 2)) + pos_w.float().unsqueeze(1)
    if env_origins is None:
        env_origins = torch.zeros(B, 3, device=device, dtype=torch.float32)
    points_local = points_world - env_origins.float().unsqueeze(1)          # (B, HW, 3)

    if diag_only:
        return points_local, valid_depth.view(B, HW)

    x_min, x_max, y_min, y_max, z_min, z_max = workspace
    mask = valid_depth.view(B, HW) & (
        (points_local[..., 0] > x_min) & (points_local[..., 0] < x_max) &
        (points_local[..., 1] > y_min) & (points_local[..., 1] < y_max) &
        (points_local[..., 2] > z_min) & (points_local[..., 2] < z_max)
    )                                                                       # (B, HW)
    zero_pc_mask = mask.sum(dim=1) == 0                                     # (B,)

    rand = torch.rand(B, HW, device=device).masked_fill(~mask, float("inf"))
    k = min(pool_size, HW)
    sel_vals, sel_idx = torch.topk(rand, k, dim=1, largest=False)           # (B, k)
    cand = torch.gather(points_local, 1, sel_idx.unsqueeze(-1).expand(-1, -1, 3))  # (B, k, 3)
    pool_valid = torch.isfinite(sel_vals)                                  # (B, k)
    first_valid = cand[:, :1, :]
    cand = torch.where(pool_valid.unsqueeze(-1), cand, first_valid)

    idx = farthest_point_sample_torch(cand, num_points)                    # (B, num_points)
    result = torch.gather(cand, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
    result[zero_pc_mask] = 0.0
    return result, zero_pc_mask


def _quat_mul_wxyz(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


def _quat_rotate_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    w = q[:, :1]
    u = q[:, 1:]
    return v + 2.0 * torch.cross(u, torch.cross(u, v, dim=-1) + w * v, dim=-1)


def wrist_cam_pose_w(robot, cam) -> Tuple[torch.Tensor, torch.Tensor]:
    conv = getattr(cam.cfg.offset, "convention", "ros")
    if conv != "ros":
        raise ValueError(f"wrist_cam_pose_w only supports offset convention='ros', got '{conv}'")
    link_name = cam.cfg.prim_path.split("/")[-2]
    body_names = list(robot.body_names)
    if link_name not in body_names:
        raise ValueError(
            f"camera parent link '{link_name}' not in robot.body_names (first few: {body_names[:8]}...)")
    idx = body_names.index(link_name)
    link_pos = robot.data.body_pos_w[:, idx].float()      # (B,3) world frame (includes env origin)
    link_quat = robot.data.body_quat_w[:, idx].float()    # (B,4) wxyz
    B = link_pos.shape[0]
    dev = link_pos.device
    off_pos = torch.tensor(cam.cfg.offset.pos, device=dev, dtype=torch.float32)
    off_rot = torch.tensor(cam.cfg.offset.rot, device=dev, dtype=torch.float32)
    pos_w = link_pos + _quat_rotate_wxyz(link_quat, off_pos.unsqueeze(0).expand(B, 3))
    quat_w_ros = _quat_mul_wxyz(link_quat, off_rot.unsqueeze(0).expand(B, 4))
    return pos_w, quat_w_ros


def camera_crop_bounds(drill_pos_w, env_origins, perc, workspace):

    from .robot_pointcloud import drill_crop_bounds  # shared primitive (same-package relative import)
    if getattr(perc, "camera_follow_drill", False):
        return drill_crop_bounds(drill_pos_w, env_origins, perc.drill_crop_half,
                                 z_floor=workspace[4])
    return workspace


def raise_z_floor(crop, z_floor):
    """Raise the crop box z lower bound to at least z_floor (the other 5 bounds unchanged).
    crop = (x_min,x_max,y_min,y_max,z_min,z_max); z_min may be a scalar (fixed workspace)
    or a (B,1) tensor (drill-follow box). Returns unchanged when z_floor=None.
    Use: raise the box bottom for the wrist camera (cam2) alone, independent of the shared workspace z_min."""
    if z_floor is None:
        return crop
    x0, x1, y0, y1, z0, z1 = crop
    if torch.is_tensor(z0):
        z0 = z0.clamp_min(float(z_floor))
    else:
        z0 = max(float(z0), float(z_floor))
    return (x0, x1, y0, y1, z0, z1)


def _read_depth(cam):
    return torch.nan_to_num(
        cam.data.output["distance_to_image_plane"], nan=0.0, posinf=0.0, neginf=0.0)


def camera_pc(cam, crop_bounds, num_points, env_origins, pool_size: int = 2048,
              pose_w: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
  
    depth = _read_depth(cam)
    pos_w, quat_w = pose_w if pose_w is not None else (cam.data.pos_w, cam.data.quat_w_ros)
    pc, zmask = depth_to_pointcloud_batch(
        depth, cam.data.intrinsic_matrices, pos_w,
        quat_w, crop_bounds, num_points, env_origins, pool_size)
    return pc, zmask, depth


def build_fused_camera_pc(
    cam1, cam2, drill_pos_w, env_origins, perc, pc_num_points, workspace,
    follow_cam1: bool = True, follow_cam2: bool = True,
    cam2_pose_w: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    half = pc_num_points // 2
    depth1 = _read_depth(cam1)
    depth2 = _read_depth(cam2)

    crop1 = camera_crop_bounds(drill_pos_w, env_origins, perc, workspace) if follow_cam1 else workspace
    crop2 = camera_crop_bounds(drill_pos_w, env_origins, perc, workspace) if follow_cam2 else workspace
    crop2 = raise_z_floor(crop2, getattr(perc, "wrist_cam_z_floor", None))   # wrist camera independent z lower bound

    pos2, quat2 = (cam2_pose_w if cam2_pose_w is not None
                   else (cam2.data.pos_w, cam2.data.quat_w_ros))
    pc1, zmask1 = depth_to_pointcloud_batch(
        depth1, cam1.data.intrinsic_matrices, cam1.data.pos_w,
        cam1.data.quat_w_ros, crop1, half, env_origins)
    pc2, zmask2 = depth_to_pointcloud_batch(
        depth2, cam2.data.intrinsic_matrices, pos2,
        quat2, crop2, half, env_origins)

    B = depth1.shape[0]
    pc_fused = torch.zeros(B, pc_num_points, 3, device=depth1.device, dtype=torch.float32)
    pc_fused[:, :half] = pc1
    pc_fused[:, half:] = pc2
    return pc_fused, zmask1, zmask2, depth1, depth2


def build_plate_cam_pc(cam3, plate_pos_w, env_origins, num_points,
                       crop_half: float = 0.25):

    B = env_origins.shape[0]
    center = plate_pos_w - env_origins           

    x_min = (center[:, 0].min() - crop_half).item()
    x_max = (center[:, 0].max() + crop_half).item()
    y_min = (center[:, 1].min() - crop_half).item()
    y_max = (center[:, 1].max() + crop_half).item()
    z_min = (center[:, 2].min() - crop_half).item()
    z_max = (center[:, 2].max() + crop_half).item()
    crop = (x_min, x_max, y_min, y_max, z_min, z_max)
    return camera_pc(cam3, crop, num_points, env_origins)
