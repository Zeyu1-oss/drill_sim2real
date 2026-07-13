"""运行时模块:根据机器人当前 link 位姿,用 FK 生成"完整机器人点云"。

采集(collect_dp3_data.py)和部署(deploy_dp3_sim.py / 真机)共用同一份 canonical 点
(由 tools/build_robot_pointcloud.py 从 URDF 离线生成),保证两边几何逐点一致、零 sim2real gap。

机制:canonical 点在各自 link 局部系;每帧把它们用对应 link 的世界位姿刚体变换到世界系,
再减 env_origin 得到 env-local 点云。位姿来源可插拔:
  - IsaacLab(采集/仿真部署):franka.data.body_pos_w / body_quat_w(已是 FK 结果)
  - 真机:用 URDF + pytorch_kinematics 从关节角算出 link 位姿后传入

只依赖 torch,采集/部署的 env 无需任何 mesh / URDF 库。
"""
import numpy as np
import torch


def _quat_to_mat(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """(..., 4) wxyz 四元数 -> (..., 3, 3) 旋转矩阵。与 IsaacLab body_quat_w 约定一致。"""
    q = quat_wxyz / (torch.norm(quat_wxyz, dim=-1, keepdim=True) + 1e-8)
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1)
    return R.reshape(*q.shape[:-1], 3, 3)


def make_ground_points(workspace, n: int, z: float, device) -> torch.Tensor:
    """合成地面/桌面点云:在 workspace 的 x/y 矩形内、高度 z(env-local m)生成恰好 n 个点。
    确定性网格 -> collect 和 deploy 逐点一致(只要 workspace/n/z 相同)。
    workspace=[x_min, x_max, y_min, y_max, ...](后两位 z 范围不用)。"""
    import math
    x_min, x_max, y_min, y_max = (float(workspace[0]), float(workspace[1]),
                                  float(workspace[2]), float(workspace[3]))
    aspect = (x_max - x_min) / max(y_max - y_min, 1e-6)
    ny = max(1, int(round(math.sqrt(n / max(aspect, 1e-6)))))
    nx = max(1, int(math.ceil(n / ny)))                       # nx*ny >= n
    xs = torch.linspace(x_min, x_max, nx, device=device)
    ys = torch.linspace(y_min, y_max, ny, device=device)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1),
                       torch.full((nx * ny,), float(z), device=device)], dim=-1)
    return pts[:n]                                            # 取前 n 个,确定性


def jitter_ground_xy(ground_batch: torch.Tensor, std: float,
                     x_min: float, x_max: float, y_min: float, y_max: float) -> torch.Tensor:
    """每帧给地面点 xy 加 N(0,std) 高斯噪声(z 不变),抖动后 clamp 回 [x_min,x_max]x[y_min,y_max]
    区域内。collect 和 deploy 调用同一函数、同一 std(hyperparameters 单一真源)-> 同分布,
    等价数据增强,打破网格规整。std<=0 时原样返回(关闭)。ground_batch:(B,N,3)。"""
    if std is None or std <= 0:
        return ground_batch
    B, N, _ = ground_batch.shape
    out = ground_batch.clone()
    out[..., 0] = (ground_batch[..., 0]
                   + torch.randn(B, N, device=ground_batch.device) * std).clamp(x_min, x_max)
    out[..., 1] = (ground_batch[..., 1]
                   + torch.randn(B, N, device=ground_batch.device) * std).clamp(y_min, y_max)
    return out


def drill_crop_bounds(drill_pos_w: torch.Tensor, env_origins: torch.Tensor, half: float,
                      z_floor=None):
    """以电钻位置为中心、边长 2*half 的立方体裁剪框。返回 6 个 (B,1) 张量
    (x_min,x_max,y_min,y_max,z_min,z_max),env-local;直接当 depth_to_pointcloud_batch
    的 workspace 传入即可(mask 比较会逐 env 广播)。oracle:drill_pos_w 用真值位姿。
    z_floor 给定时,框底 z_min 夹到 >= z_floor(= 桌面高度,保证相机不采到真实桌面/
    地面;地面由合成点云负责),避免电钻躺下时立方体探到桌面以下。"""
    c = drill_pos_w - env_origins                            # (B,3) env-local 电钻中心
    z_lo = c[:, 2:3] - half
    if z_floor is not None:
        z_lo = z_lo.clamp_min(float(z_floor))                # 框底不低于桌面
    return (c[:, 0:1] - half, c[:, 0:1] + half,
            c[:, 1:2] - half, c[:, 1:2] + half,
            z_lo, c[:, 2:3] + half)


class RobotPointCloudFK:
    """加载 canonical 点 + 每帧 FK 变换成完整机器人点云。"""

    def __init__(self, npz_path: str, body_names, device, verbose: bool = True,
                 max_points: int = None):
        """
        Args:
            npz_path:   tools/build_robot_pointcloud.py 产出的 .npz
            body_names: 机器人 articulation 的 body 名顺序(franka.body_names),
                        canonical link 据此映射到 body 索引。
            device:     torch device。
            max_points: 降采样到的点数上限(None=不降)。固定种子 + 按 link 分层,
                        collect/deploy 用同一 npz 和 max_points 时选出的点完全一致。
        """
        data = np.load(npz_path, allow_pickle=True)
        points = data["points"].astype(np.float32)        # (M0,3) 各自 link 系
        link_idx = data["link_idx"].astype(np.int64)      # (M0,) 指向 link_names
        link_names = [str(n) for n in data["link_names"]]

        name_to_body = {n: i for i, n in enumerate(body_names)}
        # canonical link -> articulation body 索引;link 在 articulation 里不存在(被固定关节
        # 合并掉的 tip 等)就丢弃这些点并告警。
        keep = np.ones(len(points), dtype=bool)
        pt_body = np.zeros(len(points), dtype=np.int64)
        missing = {}
        for li, lname in enumerate(link_names):
            mask = link_idx == li
            bi = name_to_body.get(lname, None)
            if bi is None:
                missing[lname] = int(mask.sum())
                keep[mask] = False
            else:
                pt_body[mask] = bi

        pts = points[keep]
        pb = pt_body[keep]

        # 确定性分层降采样:按 link 配额(每 link 至少 1 点),小 link(指尖)不会被抽空。
        # 固定种子 -> 同 npz + 同 max_points 时 collect/deploy 结果逐点一致。
        if max_points is not None and int(max_points) < len(pts):
            tgt = int(max_points)
            M0 = len(pts)
            rs = np.random.RandomState(20240601)
            sel_chunks = []
            for bi in np.unique(pb):
                idx_b = np.where(pb == bi)[0]
                quota = max(1, int(round(len(idx_b) * tgt / M0)))
                pick = rs.permutation(len(idx_b))[:quota]
                sel_chunks.append(idx_b[pick])
            sel = np.concatenate(sel_chunks)
            # 逐 link 取整后总数可能差几点:多退少补
            if len(sel) > tgt:
                sel = sel[rs.permutation(len(sel))[:tgt]]
            elif len(sel) < tgt:
                rest = np.setdiff1d(np.arange(M0), sel)
                extra = rest[rs.permutation(len(rest))[:tgt - len(sel)]]
                sel = np.concatenate([sel, extra])
            sel = np.sort(sel)
            pts, pb = pts[sel], pb[sel]

        self.points = torch.from_numpy(pts).to(device)      # (M,3)
        self.pt_body = torch.from_numpy(pb).to(device)      # (M,)
        self.num_points = int(self.points.shape[0])
        self.device = device

        if verbose:
            print(f"[RobotPointCloudFK] {self.num_points} pts kept "
                  f"({len(points) - self.num_points} dropped from {len(missing)} "
                  f"merged links)")
            if missing:
                print(f"    dropped links (not live bodies): {missing}")

    @torch.no_grad()
    def __call__(self, body_pos_w: torch.Tensor, body_quat_w: torch.Tensor,
                 env_origins: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            body_pos_w:  (B, L, 3) 各 body 世界位置(IsaacLab franka.data.body_pos_w)
            body_quat_w: (B, L, 4) 各 body 世界四元数 wxyz
            env_origins: (B, 3) env 原点;给了就转成 env-local(和相机点云同系)
        Returns:
            (B, M, 3) 完整机器人点云。
        """
        # 取每个 canonical 点对应 link 的位姿:(B, M, ...)
        pos = body_pos_w[:, self.pt_body, :]                         # (B,M,3)
        quat = body_quat_w[:, self.pt_body, :]                       # (B,M,4)
        R = _quat_to_mat(quat)                                       # (B,M,3,3)
        local = self.points.unsqueeze(0)                            # (1,M,3)
        world = torch.einsum("bmij,bmj->bmi", R, local.expand(pos.shape[0], -1, -1)) + pos
        if env_origins is not None:
            world = world - env_origins.unsqueeze(1)
        return world
