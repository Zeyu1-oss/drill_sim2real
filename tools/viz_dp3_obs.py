"""Interactive 3D view of ONE DP3 observation point cloud, split by segment.

Unlike tools/viz.py (matplotlib, colored by height), this colors each point by which part of the
observation it came from -- camera vs forward-kinematics robot cloud vs (if the zarr has pc_mask)
the handle-labelled points -- which is what you actually need to check when debugging an
observation: whether the robot cloud lands on the hand, whether the crop ate the drill, whether
the mask sits on the handle.

Writes a self-contained rotatable HTML (plotly, no CDN).

  python tools/viz_dp3_obs.py --zarr data/1.zarr --episode 0
  python tools/viz_dp3_obs.py --zarr data/masktest.zarr --episode 0 --frac 0.6
  python tools/viz_dp3_obs.py --zarr data/1.zarr --episode 3 --cam_pts 2048 --robot_pts 512
"""
import argparse
import os

import numpy as np
import zarr
import plotly.graph_objects as go

# dataviz reference palette, light surface (slots 1-3 validate all-pairs in both modes)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
CAM = "#2a78d6"      # slot 1 -- camera-derived points
ROBOT = "#eb6834"    # slot 2 -- forward-kinematics robot cloud
HANDLE = "#1baf7a"   # slot 3 -- points labelled is_handle (only when the zarr has pc_mask)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zarr", type=str, default="data/1.zarr")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--frame", type=int, default=None,
                   help="absolute frame index inside the episode; overrides --frac")
    p.add_argument("--frac", type=float, default=0.6,
                   help="pick the frame at this fraction through the episode (default 0.6, "
                        "i.e. mid-grasp rather than the initial reach-out)")
    p.add_argument("--cam_pts", type=int, default=None,
                   help="length of the camera segment (default: inferred, total - robot_pts)")
    p.add_argument("--robot_pts", type=int, default=512)
    p.add_argument("--point_size", type=float, default=2.2)
    p.add_argument("-o", "--out", type=str, default=None)
    return p.parse_args()


def episode_bounds(root, episode):
    ends = np.asarray(root["meta"]["episode_ends"][:])
    if episode < 0 or episode >= len(ends):
        raise SystemExit(f"episode {episode} out of range (zarr has {len(ends)})")
    start = 0 if episode == 0 else int(ends[episode - 1])
    return start, int(ends[episode])


def main():
    args = parse_args()
    root = zarr.open_group(args.zarr, mode="r")
    data = root["data"]
    start, end = episode_bounds(root, args.episode)
    n_ep = end - start
    idx = (start + int(args.frame) if args.frame is not None
           else start + int(np.clip(args.frac, 0, 1) * (n_ep - 1)))
    idx = int(np.clip(idx, start, end - 1))

    pc = np.asarray(data["point_cloud"][idx]).astype(np.float32)    # (N, 3)
    total = pc.shape[0]
    robot_pts = int(args.robot_pts)
    cam_pts = int(args.cam_pts) if args.cam_pts is not None else total - robot_pts
    if cam_pts + robot_pts != total:
        print(f"[WARN] cam_pts({cam_pts}) + robot_pts({robot_pts}) != total({total}); "
              f"the split below is a guess -- pass --cam_pts/--robot_pts explicitly")

    mask = None
    if "pc_mask" in data:
        mask = np.asarray(data["pc_mask"][idx]).astype(bool)

    state = np.asarray(data["state"][idx]) if "state" in data else None

    groups = []
    cam = pc[:cam_pts]
    rob = pc[cam_pts:cam_pts + robot_pts]
    if mask is not None:
        mc = mask[:cam_pts]
        groups.append((f"相机点 · 非把手 ({int((~mc).sum())})", cam[~mc], CAM, args.point_size))
        groups.append((f"相机点 · is_handle ({int(mc.sum())})", cam[mc], HANDLE, args.point_size + 1.6))
    else:
        groups.append((f"相机点 cam ({len(cam)})", cam, CAM, args.point_size))
    groups.append((f"机器人 FK 点 ({len(rob)})", rob, ROBOT, args.point_size))

    fig = go.Figure()
    for name, pts, color, size in groups:
        if len(pts) == 0:
            continue
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers", name=name,
            marker=dict(size=size, color=color, opacity=0.85, line=dict(width=0)),
            hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra>" + name + "</extra>"))

    lo = pc.min(0)
    hi = pc.max(0)
    span = float((hi - lo).max())
    mid = (hi + lo) / 2.0
    rng = [[mid[i] - span / 2, mid[i] + span / 2] for i in range(3)]   # equal aspect: one cube

    sub = (f"{os.path.basename(args.zarr.rstrip('/'))} · episode {args.episode} · "
           f"frame {idx - start}/{n_ep - 1} (绝对 {idx}) · {total} 点")
    fig.update_layout(
        title=dict(text=f"DP3 观测点云<br><span style='font-size:13px;color:{INK2}'>{sub}</span>",
                   x=0.02, xanchor="left", font=dict(size=20, color=INK)),
        scene=dict(
            xaxis=dict(title="x (m, env-local)", range=rng[0], backgroundcolor=SURFACE,
                       gridcolor="#e1e0d9", zerolinecolor="#c3c2b7", color=MUTED),
            yaxis=dict(title="y (m, env-local)", range=rng[1], backgroundcolor=SURFACE,
                       gridcolor="#e1e0d9", zerolinecolor="#c3c2b7", color=MUTED),
            zaxis=dict(title="z (m, env-local)", range=rng[2], backgroundcolor=SURFACE,
                       gridcolor="#e1e0d9", zerolinecolor="#c3c2b7", color=MUTED),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0))),
        legend=dict(x=0.02, y=0.92, bgcolor="rgba(0,0,0,0)", font=dict(color=INK2, size=12)),
        paper_bgcolor=SURFACE, margin=dict(l=0, r=0, t=70, b=0), height=780)

    out = args.out or os.path.join(os.path.dirname(args.zarr.rstrip("/")) or ".",
                                   f"viz_obs_{os.path.basename(args.zarr.rstrip('/'))}"
                                   f"_ep{args.episode}_f{idx - start}.html")
    fig.write_html(out, include_plotlyjs="inline", full_html=True)
    print(f"saved -> {out}")

    # ---- numbers worth eyeballing alongside the render ----
    print(f"\nepisode {args.episode}: {n_ep} 帧,取第 {idx - start} 帧(绝对 {idx})")
    print(f"点云 {pc.shape}  分段: 相机 [0,{cam_pts})  机器人 FK [{cam_pts},{total})")
    for nm, seg in (("相机段", cam), ("机器人段", rob)):
        z0 = int((np.abs(seg).sum(1) == 0).sum())
        print(f"  {nm}: x[{seg[:,0].min():.3f},{seg[:,0].max():.3f}] "
              f"y[{seg[:,1].min():.3f},{seg[:,1].max():.3f}] "
              f"z[{seg[:,2].min():.3f},{seg[:,2].max():.3f}]  原点处的点 {z0}")
    if mask is not None:
        print(f"  is_handle: 相机段 {int(mask[:cam_pts].sum())}/{cam_pts} "
              f"({mask[:cam_pts].mean():.1%}),机器人段 {int(mask[cam_pts:].sum())}(应为 0)")
    if state is not None:
        print(f"state {state.shape}: 关节位置 |max|={np.abs(state[:13]).max():.3f} rad"
              + (f",力矩 |max|={np.abs(state[13:]).max():.3f} N·m" if state.shape[0] > 13 else ""))


if __name__ == "__main__":
    main()
