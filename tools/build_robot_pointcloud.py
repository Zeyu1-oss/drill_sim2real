#!/usr/bin/env python3
"""一次性离线 builder:从 URDF 采样每个 link 的表面点(link 局部坐标系),存成 .npz。

输出 {link_name: (Ki,3) float32},供运行时模块 robot_pointcloud.py 用 FK 变换成
完整机器人点云。采集(collect_dp3_data.py)和部署(deploy_dp3_sim.py / 真机)共用同一份
canonical 点 → sim 真机几何一致、零域 gap。

只在改了机器人 URDF/mesh 时才需要重跑。运行时不依赖 trimesh。

用法:
  python tools/build_robot_pointcloud.py \
      --urdf assets/inspire_tac/fr3_inspire_hand_right.urdf \
      --out  assets/inspire_tac/robot_canonical_points.npz \
      --total_points 4096
"""
import os
import argparse
import xml.etree.ElementTree as ET

import numpy as np
import trimesh


def rpy_to_mat(rpy):
    """URDF fixed-axis roll-pitch-yaw -> 3x3 旋转矩阵 (R = Rz·Ry·Rx)。"""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _floats(s, n, default):
    if s is None:
        return np.array(default, dtype=np.float64)
    vals = [float(x) for x in s.replace(",", " ").split()]
    return np.array(vals[:n], dtype=np.float64)


def load_visual_mesh(geom, urdf_dir):
    """把一个 <geometry> 转成 trimesh.Trimesh(mesh / box / cylinder / sphere)。"""
    mesh = geom.find("mesh")
    if mesh is not None:
        fn = mesh.get("filename").replace("package://", "")
        path = os.path.normpath(os.path.join(urdf_dir, fn))
        m = trimesh.load(path, force="mesh", process=False)
        if isinstance(m, trimesh.Scene):
            m = m.dump(concatenate=True)
        scale = _floats(mesh.get("scale"), 3, [1.0, 1.0, 1.0])
        if not np.allclose(scale, 1.0):
            m = m.copy()
            m.apply_scale(scale)
        return m
    box = geom.find("box")
    if box is not None:
        return trimesh.creation.box(extents=_floats(box.get("size"), 3, [0.01, 0.01, 0.01]))
    cyl = geom.find("cylinder")
    if cyl is not None:
        return trimesh.creation.cylinder(radius=float(cyl.get("radius")),
                                         height=float(cyl.get("length")))
    sph = geom.find("sphere")
    if sph is not None:
        return trimesh.creation.icosphere(radius=float(sph.get("radius")))
    return None


def _is_hand_link(name):
    """手部/末端 link(抓取与接触的关键):上权重,别被大块手臂淹没。"""
    n = name.lower()
    return n.startswith("r_") or "hand" in n or "flange" in n


def _alloc_group(entries, budget, min_per_visual, rng, name_to_idx, link_names):
    """从一组 entries(同属手/臂)采样并精确修整到 budget 个点;组内按面积比例分配。"""
    if budget <= 0 or not entries:
        return np.zeros((0, 3), np.float32), np.zeros((0,), np.int64)
    tot = sum(e[4] for e in entries) or 1.0
    pts_all, idx_all = [], []
    for lname, m, R, t, warea in entries:
        k = max(min_per_visual, int(round(budget * warea / tot)))
        pts, _ = trimesh.sample.sample_surface(m, k)
        pts = (pts @ R.T + t).astype(np.float32)              # mesh 系 -> link 系
        if lname not in name_to_idx:
            name_to_idx[lname] = len(link_names)
            link_names.append(lname)
        pts_all.append(pts)
        idx_all.append(np.full(len(pts), name_to_idx[lname], dtype=np.int64))
    P = np.concatenate(pts_all, axis=0)
    I = np.concatenate(idx_all, axis=0)
    sel = rng.choice(len(P), budget, replace=(len(P) < budget))   # 精确到 budget
    return P[sel], I[sel]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--total_points", type=int, default=512,
                    help="最终固定点数(运行时直接 FK 变换这些点,无需再采样)")
    ap.add_argument("--hand_weight", type=float, default=4.0,
                    help="手部 link 的点数权重(相对手臂),保证抓取/接触区域够密")
    ap.add_argument("--hand_points", type=int, default=None,
                    help="手部 link 的精确总点数;与 --arm_points 一起给时启用分组精确分配"
                         "(覆盖 total_points/hand_weight)")
    ap.add_argument("--arm_points", type=int, default=None,
                    help="手臂 link 的精确总点数(与 --hand_points 一起用)")
    ap.add_argument("--min_per_visual", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    urdf_dir = os.path.dirname(os.path.abspath(args.urdf))
    root = ET.parse(args.urdf).getroot()

    # ---- 第一遍:载入每个 visual 的 mesh,算 加权面积(= 面积 × 手部权重)----
    entries = []  # (link_name, mesh, R, t, weighted_area)
    for link in root.findall("link"):
        lname = link.get("name")
        w = args.hand_weight if _is_hand_link(lname) else 1.0
        for vis in link.findall("visual"):
            geom = vis.find("geometry")
            if geom is None:
                continue
            m = load_visual_mesh(geom, urdf_dir)
            if m is None or m.area <= 0:
                continue
            o = vis.find("origin")
            xyz = _floats(o.get("xyz") if o is not None else None, 3, [0, 0, 0])
            rpy = _floats(o.get("rpy") if o is not None else None, 3, [0, 0, 0])
            entries.append((lname, m, rpy_to_mat(rpy), xyz, float(m.area) * w))

    total_warea = sum(e[4] for e in entries) or 1.0

    # ---- 第二遍:采样 + 变换到 link 系 ----
    link_names = []                       # 唯一 link 顺序
    name_to_idx = {}
    explicit = args.hand_points is not None and args.arm_points is not None
    if explicit:
        # 手/臂分组精确分配:手 = hand_points,臂 = arm_points(各自组内按面积比例)
        hand_e = [e for e in entries if _is_hand_link(e[0])]
        arm_e = [e for e in entries if not _is_hand_link(e[0])]
        hp, hi = _alloc_group(hand_e, args.hand_points, args.min_per_visual, rng, name_to_idx, link_names)
        ap_, ai = _alloc_group(arm_e, args.arm_points, args.min_per_visual, rng, name_to_idx, link_names)
        points = np.concatenate([hp, ap_], axis=0)
        link_idx = np.concatenate([hi, ai], axis=0)
    else:
        # 旧路径:按加权面积分配固定 total_points
        pts_all, idx_all = [], []
        for lname, m, R, t, warea in entries:
            k = max(args.min_per_visual, int(round(args.total_points * warea / total_warea)))
            pts, _ = trimesh.sample.sample_surface(m, k)
            pts = (pts @ R.T + t).astype(np.float32)          # mesh 系 -> link 系
            if lname not in name_to_idx:
                name_to_idx[lname] = len(link_names)
                link_names.append(lname)
            pts_all.append(pts)
            idx_all.append(np.full(len(pts), name_to_idx[lname], dtype=np.int64))
        points = np.concatenate(pts_all, axis=0)
        link_idx = np.concatenate(idx_all, axis=0)
        M = args.total_points
        sel = rng.choice(len(points), M, replace=(len(points) < M))
        points, link_idx = points[sel], link_idx[sel]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez(args.out,
             points=points,                                   # (M,3) 各自 link 系
             link_idx=link_idx,                               # (M,) 指向 link_names
             link_names=np.array(link_names))                 # (L,)

    print(f"[OK] {len(link_names)} links, {len(points)} points -> {args.out}")
    counts = {ln: int((link_idx == i).sum()) for ln, i in name_to_idx.items()}
    hand = sum(c for ln, c in counts.items() if _is_hand_link(ln))
    print(f"    hand points={hand} / arm points={len(points)-hand}")
    for ln in sorted(counts):
        tag = "HAND" if _is_hand_link(ln) else "arm "
        print(f"    [{tag}] {ln:28s} {counts[ln]:4d}")


if __name__ == "__main__":
    main()
