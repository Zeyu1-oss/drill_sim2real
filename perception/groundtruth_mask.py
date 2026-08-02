"""Ground-truth handle mask for the fused DP3 point cloud -- the ONE place collect and deploy share.

Both sides need exactly the same thing: given the fused point cloud, mark which of its points lie on
the drill handle. Before this module they each carried their own copy of the segment slicing and each
had its own --mask_threshold default (collect 0.005, deploy 0.02 -- a silent 4x mismatch that made the
deployed 4th channel a different label definition than the one the policy was trained on). Keeping a
single implementation plus a single default is the whole point of this file.

Layout assumption (matches how collect_dp3_data.py concatenates the segments):

    [ camera 0:C | plate C:C+P | robot C+P:C+P+R | drill mesh | ground ]

Only the CAMERA segment is ever labelled. The robot FK / plate / drill-mesh / ground segments are
always 0: they are either the robot's own surface or synthetic geometry, never the drill handle.

Ground truth = simulator only. `compute_handle_mask` needs the drill's true pose and the variant's
body_mask region, so this cannot run on hardware. For real-robot deployment, swap
`handle_mask_for_pc` for a segmentation network (perception/handle_seg_net.py) trained on the labels
this module produces.

Usage:
    collect:  m = handle_mask_for_pc(pc_fused, env, pc_num_points, plate_M, robot_pc_M)   # (B,T) uint8
    deploy:   pc4 = add_mask_channel(pc_now,  env, pc_num_points, plate_M, robot_pc_M)    # (B,T,4)
"""
import torch

from .handle_mask import compute_handle_mask

# Single source of truth for the is_handle radius. Collect and deploy MUST agree: the radius *is* the
# definition of "handle", so a mismatch feeds the policy a 4th channel it was never trained on.
DEFAULT_MASK_THRESHOLD = 0.015


_HANDLE_PTS_SEED = 20240601   # fixed: the same mesh vertices must be picked at collect and deploy
_handle_pts_cache = {}        # (id(env), vid, n) -> (n,3) drill-local, scaled


def handle_points_pc(env_unwrapped, n_points):
    """Ground-truth handle point cloud: (B, n_points, 3) env-local.

    Samples n_points from each variant's body_mask-filtered mesh (the same `body_surface_points`
    that defines is_handle in handle_mask.py), then rigidly transforms them with the drill's TRUE
    pose. Unlike the camera-derived handle points this is complete and fixed-size -- it contains
    the occluded side of the handle too, and never runs short (camera-visible handle points are
    typically ~55 per frame, far below any useful budget).

    The per-variant vertex subset is drawn once under a fixed seed and cached, so collect and
    deploy feed the network the same points for a given variant and budget; resampling per frame
    would make consecutive observations jitter for no reason.

    Sim-only: needs the drill's ground-truth pose. This is a stronger oracle than the is_handle
    channel -- on hardware it would have to come from pose estimation or a segmentation network.
    """
    dev = env_unwrapped.scene.env_origins.device
    B = env_unwrapped.scene.env_origins.shape[0]
    out = torch.zeros(B, n_points, 3, device=dev, dtype=torch.float32)

    from isaaclab.utils import math as math_utils
    drill_pos_w = env_unwrapped.drill.data.root_pos_w
    drill_quat = torch.nn.functional.normalize(env_unwrapped.drill.data.root_quat_w, dim=1)
    R = math_utils.matrix_from_quat(drill_quat)
    eo = env_unwrapped.scene.env_origins
    vids = env_unwrapped._drill_variant_indices

    for vid in vids.unique().tolist():
        sel = (vids == vid).nonzero(as_tuple=True)[0]
        key = (id(env_unwrapped), int(vid), int(n_points))
        pts = _handle_pts_cache.get(key)
        if pts is None:
            vdata = env_unwrapped._variant_attrs[int(vid)]
            body = vdata.get("body_surface_points", None)
            if body is None or len(body) == 0:
                continue
            g = torch.Generator(device="cpu").manual_seed(_HANDLE_PTS_SEED + int(vid))
            if len(body) >= n_points:
                idx = torch.randperm(len(body), generator=g)[:n_points]
            else:   # fewer mesh vertices than the budget: pad by sampling with replacement
                idx = torch.cat([torch.arange(len(body)),
                                 torch.randint(len(body), (n_points - len(body),), generator=g)])
            pts = (body[idx.to(body.device)] * vdata["scale"]).to(dev)        # (n,3) drill-local
            _handle_pts_cache[key] = pts
        out[sel] = (torch.bmm(R[sel], pts.T.unsqueeze(0).expand(len(sel), 3, -1)).transpose(1, 2)
                    + drill_pos_w[sel].unsqueeze(1) - eo[sel].unsqueeze(1))
    return out


def handle_mask_for_pc(pc, env_unwrapped, pc_num_points, plate_M=0, robot_pc_M=0,
                       threshold=DEFAULT_MASK_THRESHOLD, use_robot_seg=True, dtype=torch.uint8):
    """Per-point is_handle label for a fused point cloud.

    pc:            (B, total_pc, >=3) env-local fused cloud (only [..., :3] is read).
    pc_num_points: length of the camera segment; only points [0, pc_num_points) can be labelled.
    plate_M:       plate segment length (sits between camera and robot), 0 when absent.
    robot_pc_M:    robot FK segment length; used to drop finger points resting on the handle.
    use_robot_seg: False disables that robot-proximity test (mirrors collect's --pc_mode camera).
    Returns (B, total_pc) of `dtype`, nonzero only inside the camera segment.
    """
    B, total_pc = pc.shape[0], pc.shape[1]
    cam = pc[:, :pc_num_points, :3]
    rs = pc_num_points + plate_M                       # robot segment starts after the plate segment
    robot_seg = (pc[:, rs:rs + robot_pc_M, :3]
                 if (use_robot_seg and robot_pc_M > 0) else None)
    is_handle = compute_handle_mask(cam, env_unwrapped, robot_pc_local=robot_seg,
                                    threshold=threshold)          # (B, Cam) bool
    out = torch.zeros(B, total_pc, dtype=dtype, device=pc.device)
    out[:, :pc_num_points] = is_handle.to(dtype)
    return out


def add_mask_channel(pc, env_unwrapped, pc_num_points, plate_M=0, robot_pc_M=0,
                     threshold=DEFAULT_MASK_THRESHOLD, use_robot_seg=True):
    """(B, total_pc, 3) -> (B, total_pc, 4) = [xyz | is_handle], for a 4-channel checkpoint.

    The mask rides in the point cloud's own dtype so the tensor stays homogeneous; collect instead
    stores it as a separate uint8 `pc_mask` array (see handle_mask_for_pc), which is why the training
    zarr keeps point_cloud at 3 channels and the dataset re-concatenates at load time.
    """
    m = handle_mask_for_pc(pc, env_unwrapped, pc_num_points, plate_M, robot_pc_M,
                           threshold=threshold, use_robot_seg=use_robot_seg,
                           dtype=pc.dtype)                          # (B, total_pc)
    return torch.cat([pc, m.unsqueeze(-1)], dim=-1)
