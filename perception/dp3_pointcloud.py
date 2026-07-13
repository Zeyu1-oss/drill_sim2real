"""DP3 点云构建 —— collect_dp3_data.py 与 deploy_dp3_sim.py 的单一真源。

为什么有这个文件:
  采集(collect)和部署(deploy)必须生成**完全相同**的点云,否则训练/部署观测对不上。
  之前这套逻辑(深度反投影 + 裁剪 + FPS + 相机编排)在两个脚本里各写了一遍,任何一边
  改动(相机跟随、分辨率、拼接顺序……)都要手动同步,极易漂移。把它收进这里后,两个脚本
  都 `from perception.dp3_pointcloud import ...`,物理上杜绝不一致。

依赖:
  - torch
  - robot_pointcloud.drill_crop_bounds(跟随框,已是共享原语)
  - 可选 pytorch3d 的 sample_farthest_points(有就用融合 CUDA 内核,没有回退纯 torch 实现)

不依赖 IsaacLab:相机对象按鸭子类型读取 .data.intrinsic_matrices / .pos_w / .quat_w_ros /
  .output["distance_to_image_plane"],所以可单独 import、单测。
"""

import sys
from typing import Optional, Tuple

import torch

# ---- 可选 pytorch3d 融合 FPS 内核 ----
# 注意:不在模块顶层 import pytorch3d。collect 对 C-extension 的 dlopen 顺序敏感
# (pytorch3d 必须在 triton/zarr 之后再加载),所以改成惰性:脚本在正确时机调用
# init_fps_kernel(path) 来加载。未加载时 farthest_point_sample_torch 自动回退纯 torch。
_PYTORCH3D_FPS = None


def init_fps_kernel(pytorch3d_path: Optional[str] = None) -> bool:
    """在正确时机加载 pytorch3d 融合 FPS 内核(collect/deploy 共用)。
    pytorch3d_path: 若给定,先插入 sys.path(指向 pytorch3d_simplified)。
    返回是否加载成功;失败则保持 None(回退纯 torch),不抛异常。
    """
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
    """GPU 批量最远点采样。points: (B, N, 3) -> indices: (B, num_points)。

    优先走 pytorch3d 融合 CUDA 内核(单次启动);否则用纯 torch 贪心循环(从索引 0 起),
    两者算法等价。N <= num_points 时直接返回全部索引(上层会重复填充)。
    """
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
    """深度图 -> env-local 点云。向量化:对所有 env 做一次批量 FPS(无 per-env python 循环)。

    depth:     (B, H, W, 1) distance_to_image_plane
    intrinsic: (B, 3, 3)
    pos_w:     (B, 3)   相机世界位置
    quat_w:    (B, 4)   相机世界四元数 wxyz(传 cam.data.quat_w_ros)
    workspace: (x_min, x_max, y_min, y_max, z_min, z_max) env-local 裁剪框
    返回:      (result (B, num_points, 3), zero_pc_mask (B,))
               zero_pc_mask[b]=True 表示该 env 本帧裁剪框内无有效点(全 0 占位)。
    diag_only=True 时只反投影、不裁剪/不 FPS,返回 (points_local (B,HW,3), valid_depth (B,HW)) 供诊断。
    """
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

    # Stage 1: 每个 env 随机取至多 pool_size 个有效点(无循环、无同步)
    rand = torch.rand(B, HW, device=device).masked_fill(~mask, float("inf"))
    k = min(pool_size, HW)
    sel_vals, sel_idx = torch.topk(rand, k, dim=1, largest=False)           # (B, k)
    cand = torch.gather(points_local, 1, sel_idx.unsqueeze(-1).expand(-1, -1, 3))  # (B, k, 3)
    pool_valid = torch.isfinite(sel_vals)                                  # (B, k)
    # 有效点不足 k 的 env,空位用其首个有效点填充,避免 FPS 采到垃圾
    first_valid = cand[:, :1, :]
    cand = torch.where(pool_valid.unsqueeze(-1), cand, first_valid)

    # Stage 2: 对所有 B 个 env 做一次批量 FPS
    idx = farthest_point_sample_torch(cand, num_points)                    # (B, num_points)
    result = torch.gather(cand, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
    result[zero_pc_mask] = 0.0
    return result, zero_pc_mask


def camera_crop_bounds(drill_pos_w, env_origins, perc, workspace):
    """相机裁剪框的**单一决策点**:跟随电钻 -> 返回以电钻真值位姿为中心的立方体;
    否则 -> 返回固定 workspace。collect 和 deploy 都调它,跟随策略只在这里改一次。

    perc: PerceptionHyperparameters(需 .camera_follow_drill / .drill_crop_half)
    workspace: (x_min..z_max);跟随时框底夹到 workspace[4](桌面),不采地面。
    """
    from .robot_pointcloud import drill_crop_bounds  # 共享原语(同包相对导入)
    if getattr(perc, "camera_follow_drill", False):
        return drill_crop_bounds(drill_pos_w, env_origins, perc.drill_crop_half,
                                 z_floor=workspace[4])
    return workspace


def _read_depth(cam):
    """从相机对象读 distance_to_image_plane 并清洗 NaN/Inf。"""
    return torch.nan_to_num(
        cam.data.output["distance_to_image_plane"], nan=0.0, posinf=0.0, neginf=0.0)


def camera_pc(cam, crop_bounds, num_points, env_origins, pool_size: int = 2048):
    """单相机 env-local 点云(collect/deploy 共用的最小单元)。
    返回 (pc (B,num_points,3), zmask (B,), depth (B,H,W,1))。
    depth 一并返回,供调用方做诊断或空帧沿用判断。
    """
    depth = _read_depth(cam)
    pc, zmask = depth_to_pointcloud_batch(
        depth, cam.data.intrinsic_matrices, cam.data.pos_w,
        cam.data.quat_w_ros, crop_bounds, num_points, env_origins, pool_size)
    return pc, zmask, depth


def build_fused_camera_pc(
    cam1, cam2, drill_pos_w, env_origins, perc, pc_num_points, workspace,
    follow_cam1: bool = True, follow_cam2: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """cam1 + cam2 融合相机点云的**单一真源**(collect / deploy 共用)。

    两台相机各采 pc_num_points//2 个点,前半段=cam1、后半段=cam2,拼成 (B, pc_num_points, 3)。
    跟随与否分别由 follow_cam1 / follow_cam2 控制(当前两者都跟随电钻 30cm 立方体)。

    返回 (pc_fused, zmask1, zmask2, depth1, depth2)。
      - pc_fused: (B, pc_num_points, 3) env-local,只含相机点(robot/drill/ground 由调用方追加)。
      - zmaskN:   (B,) 该相机本帧空帧标记(collect 跳过、deploy 沿用上一帧,各自处理)。
      - depthN:   清洗后的深度图,留给调用方做诊断(DUP-CHECK / ZPC)。
    """
    half = pc_num_points // 2
    depth1 = _read_depth(cam1)
    depth2 = _read_depth(cam2)

    crop1 = camera_crop_bounds(drill_pos_w, env_origins, perc, workspace) if follow_cam1 else workspace
    crop2 = camera_crop_bounds(drill_pos_w, env_origins, perc, workspace) if follow_cam2 else workspace

    pc1, zmask1 = depth_to_pointcloud_batch(
        depth1, cam1.data.intrinsic_matrices, cam1.data.pos_w,
        cam1.data.quat_w_ros, crop1, half, env_origins)
    pc2, zmask2 = depth_to_pointcloud_batch(
        depth2, cam2.data.intrinsic_matrices, cam2.data.pos_w,
        cam2.data.quat_w_ros, crop2, half, env_origins)

    B = depth1.shape[0]
    pc_fused = torch.zeros(B, pc_num_points, 3, device=depth1.device, dtype=torch.float32)
    pc_fused[:, :half] = pc1
    pc_fused[:, half:] = pc2
    return pc_fused, zmask1, zmask2, depth1, depth2
