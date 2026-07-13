"""分析 collect --save_init_poses 产出的 init_poses.npz 的位姿覆盖分布。

用法: python tools/analyze_init_poses.py [init_poses.npz]
打印每个 variant 的 x / y / yaw 一维直方图(ASCII),用来肉眼看采集是否均匀、哪里有洞。
只依赖 numpy。
"""
import sys
import math
import numpy as np


def hist1d(vals, lo, hi, bins=24, width=46, label=""):
    h, edges = np.histogram(vals, bins=bins, range=(lo, hi))
    mx = max(int(h.max()), 1)
    mean = vals.mean()
    print(f"  {label}: n={len(vals)} range=[{vals.min():+.3f},{vals.max():+.3f}] "
          f"mean={mean:+.3f} std={vals.std():.3f}")
    # 均匀度:每 bin 期望数 = n/bins;打印偏离
    exp = len(vals) / bins
    empty = int((h == 0).sum())
    cv = h.std() / max(h.mean(), 1e-9)   # 变异系数,越小越均匀
    for i in range(bins):
        bar = "#" * int(round(width * h[i] / mx))
        print(f"    [{edges[i]:+.3f},{edges[i+1]:+.3f}) {h[i]:6d} |{bar}")
    print(f"    -> 空桶={empty}/{bins}  变异系数CV={cv:.2f}  (期望/桶≈{exp:.0f}; CV<~0.2 算均匀)")


def yaw_from_quat(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _qmul(a, b):
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=1)


def _qconj(q):
    out = q.copy(); out[:, 1:] *= -1.0; return out


def load_base_rots():
    """返回 {variant: [(z_height, quat_wxyz), ...]} 取自 drill_variants.yaml(用于恢复相对 yaw)。"""
    import os
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "drill_variants.yaml")
    try:
        import yaml
        with open(p) as f:
            data = yaml.safe_load(f)
    except Exception as e:
        print(f"[warn] 读取 drill_variants.yaml 失败({e}),yaw 用绝对值(会被 base 朝向干扰)")
        return None
    out = {}
    for v in data["drill_variants"]:
        vid = int(v.get("variant_index", 0))
        bases = []
        if "initial_pos" in v and "initial_rot" in v:
            bases.append((float(v["initial_pos"][2]), np.array(v["initial_rot"], dtype=np.float64)))
        if "initial_pos_1" in v and "initial_rot_1" in v:
            bases.append((float(v["initial_pos_1"][2]), np.array(v["initial_rot_1"], dtype=np.float64)))
        out[vid] = bases
    return out


def relative_dyaw(quat, pz, bases):
    """按 z 把每条轨迹分到最近 base,恢复施加的 world-yaw 偏移 dyaw∈[-2,2](去掉 base 朝向干扰)。"""
    dyaw = np.zeros(len(quat))
    bz = np.array([b[0] for b in bases])
    for i in range(len(quat)):
        bi = int(np.argmin(np.abs(pz[i] - bz)))
        base = bases[bi][1].reshape(1, 4)
        rel = _qmul(quat[i:i+1], _qconj(base))[0]
        if rel[0] < 0:
            rel = -rel
        dyaw[i] = 2.0 * np.arctan2(rel[3], rel[0])
    return dyaw


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/_sobol_smoke.zarr_init_poses/init_poses.npz"
    z = np.load(path)
    print(f"file: {path}")
    print(f"keys: {list(z.keys())}")
    variant = np.asarray(z["variant"]).reshape(-1)
    pos = np.asarray(z["pos_local"]).reshape(len(variant), -1)   # (N,3) env-local
    quat = np.asarray(z["quat"]).reshape(len(variant), -1)       # (N,4) wxyz
    base_rots = load_base_rots()

    print(f"\nTOTAL episodes: {len(variant)}")
    for v in np.unique(variant):
        m = variant == v
        px, py, pz = pos[m, 0], pos[m, 1], pos[m, 2]
        print(f"\n================ variant {int(v)}: {int(m.sum())} episodes ================")
        # z(高度)区分两个 base;打印各 base 的计数
        uz, cnt = np.unique(np.round(pz, 3), return_counts=True)
        print(f"  z 高度(=base): " + ", ".join(f"{zz:.3f}m×{cc}" for zz, cc in zip(uz, cnt)))
        hist1d(px, px.min() - 0.005, px.max() + 0.005, 20, 40, "x (m)")
        hist1d(py, py.min() - 0.005, py.max() + 0.005, 20, 40, "y (m)")
        if base_rots is not None and int(v) in base_rots and base_rots[int(v)]:
            dyaw = relative_dyaw(quat[m], pz, base_rots[int(v)])
            hist1d(dyaw, -2.2, 2.2, 22, 40, "yaw 偏移 dyaw (rad, base 相对)")
        else:
            hist1d(yaw_from_quat(quat[m]), -math.pi, math.pi, 24, 40, "yaw 绝对 (rad)")


if __name__ == "__main__":
    main()
