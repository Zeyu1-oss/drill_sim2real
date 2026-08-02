"""Runtime module: synthesize the drill point-cloud segment from drill mesh points
(USD vertex downsample) + the current ground-truth drill pose.

Depth-camera coverage of the drill gets occluded by fingers / self-occlusion, especially
during close-contact grasping, so the drill's visible points in point_cloud become sparse or
partly missing. This segment is an oracle supplement (same design philosophy as the robot
segment in robot_pointcloud.py: mesh points + rigid pose transform): it transforms the
pre-downsampled mesh points to the current pose using the drill's GT pose, so it is always
complete and occlusion-free. Collection (collect_dp3_data.py) and deployment
(deploy_dp3_sim.py) share this module, guaranteeing point-for-point identical sets with zero gap.

Mechanism: mesh points are in the drill RigidBody local frame (already aligned to it with
R_correction, unscaled, when loaded in tasks/grasp_drill_env.py); each variant is scaled by its
runtime scale then FPS-downsampled to a fixed length. Every frame they are rigid-transformed to
world with the drill's root_pos_w / root_quat_w, then env_origin is subtracted for env-local
points. Suggested segment order: [camera | plate(chained only) | robot | drill | ground].
"""
import torch


def _quat_to_mat(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """(..., 4) wxyz -> (..., 3, 3), matching IsaacLab root_quat_w (same as robot_pointcloud._quat_to_mat)."""
    q = quat_wxyz / (torch.norm(quat_wxyz, dim=-1, keepdim=True) + 1e-8)
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1)
    return R.reshape(*q.shape[:-1], 3, 3)


class DrillMeshPointCloudFK:
    """Pre-downsample drill mesh points per variant, transform to env-local each frame with GT pose."""

    def __init__(self, env_unwrapped, num_points: int = 512, device=None, verbose: bool = True):
        """
        Args:
            env_unwrapped: built env (__init__ has already loaded, per active variant,
                           _variant_attrs[vid]["mesh_points"] (local frame, unscaled) and "scale").
            num_points:    fixed point count per variant after downsample (FPS, fixed seed,
                           consistent across collect/deploy).
            device:        torch device (defaults to env's).
        """
        from .dp3_pointcloud import farthest_point_sample_torch

        self.device = device or env_unwrapped.device
        self.num_points = int(num_points)
        VA = env_unwrapped._variant_attrs
        active_vids = sorted(VA.keys())

        pts_per_variant = []
        for vid in active_vids:
            mesh_pts = VA[vid].get("mesh_points", None)
            if mesh_pts is None or mesh_pts.shape[0] == 0:
                raise RuntimeError(
                    f"variant {vid} has no mesh_points (USD vertex load failed?), "
                    f"cannot build drill mesh segment (set --drill_mesh_points 0 to disable it)")
            scale = VA[vid]["scale"].to(self.device).float()             # (3,)
            pts = mesh_pts.to(self.device).float() * scale.unsqueeze(0)  # match runtime rigid scaling

            if pts.shape[0] > self.num_points:
                idx = farthest_point_sample_torch(pts.unsqueeze(0), self.num_points)[0]
                pts = pts[idx]
            elif pts.shape[0] < self.num_points:
                reps = (self.num_points + pts.shape[0] - 1) // pts.shape[0]
                pts = pts.repeat(reps, 1)[: self.num_points]   # repeat to fixed length when too few
            pts_per_variant.append(pts)
            if verbose:
                print(f"[DrillMeshPC] variant {vid}: mesh {mesh_pts.shape[0]} pts "
                      f"-> {self.num_points}(FPS, scale={scale.tolist()})", flush=True)

        self.local_points = torch.stack(pts_per_variant, dim=0)   # (nv, num_points, 3)
        # variant id -> row in the stack above (ids may be non-contiguous / not start at 0; use a lookup tensor)
        vid_to_row = torch.zeros(max(active_vids) + 1, dtype=torch.long, device=self.device)
        for row, vid in enumerate(active_vids):
            vid_to_row[vid] = row
        self._vid_to_row = vid_to_row

    @torch.no_grad()
    def __call__(self, drill_pos_w: torch.Tensor, drill_quat_w: torch.Tensor,
                 drill_variant_indices: torch.Tensor, env_origins: torch.Tensor) -> torch.Tensor:
        """
        Args:
            drill_pos_w:  (B,3) drill GT world position (env.drill.data.root_pos_w)
            drill_quat_w: (B,4) drill GT world quaternion wxyz (env.drill.data.root_quat_w)
            drill_variant_indices: (B,) current drill variant id per env
            env_origins:  (B,3)
        Returns:
            (B, num_points, 3) env-local drill point cloud.
        """
        rows = self._vid_to_row[drill_variant_indices.long()]     # (B,)
        local = self.local_points[rows]                            # (B,num_points,3)
        R = _quat_to_mat(drill_quat_w)                              # (B,3,3)
        world = torch.einsum("bij,bnj->bni", R, local) + drill_pos_w.unsqueeze(1)
        return world - env_origins.unsqueeze(1)
