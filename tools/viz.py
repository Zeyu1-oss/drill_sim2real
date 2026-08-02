"""Point-cloud visualization: take one or more frames from a zarr
(data/point_cloud + meta/episode_ends), matplotlib 3D scatter, save as png.
Supports xyz (3-dim, colored by height) and xyz+rgb (6-dim, real color) point clouds.

No IsaacLab/torch dependency: only zarr+numpy+matplotlib, runs in a plain python env
for quick data inspection.

Usage:
  python tools/viz.py --zarr <path> --episode 0 --frame 0
  python tools/viz.py --zarr <path> --episode 0 --frames 0,10,20,30 --out grid.png   # multi-frame grid
"""
import argparse
import os

import numpy as np
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zarr", type=str, required=True)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--frames", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--elev", type=float, default=20.0)
    p.add_argument("--azim", type=float, default=45.0)
    p.add_argument("--point_size", type=float, default=3.0)
    p.add_argument("--max_points", type=int, default=None)
    return p.parse_args()


def _episode_slice(z, episode):
    ends = np.asarray(z["meta/episode_ends"][:])
    starts = np.concatenate([[0], ends[:-1]])
    if episode < 0 or episode >= len(ends):
        raise ValueError(f"episode {episode} out of range, this zarr has {len(ends)} episodes")
    return int(starts[episode]), int(ends[episode])


def _plot_one(ax, pc, elev, azim, point_size, max_points, title):
    if max_points is not None and pc.shape[0] > max_points:
        idx = np.random.default_rng(0).choice(pc.shape[0], max_points, replace=False)
        pc = pc[idx]
    xyz = pc[:, :3]
    if pc.shape[1] >= 6:
        colors = np.clip(pc[:, 3:6] / 255.0, 0.0, 1.0)
    else:
        # no rgb: color by height (z) as an intuitive depth cue
        zc = xyz[:, 2]
        zn = (zc - zc.min()) / max(zc.max() - zc.min(), 1e-6)
        colors = plt.cm.viridis(zn)[:, :3]
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=point_size, marker=".", linewidths=0)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=9)
    # equal-aspect axes so the cloud is not visually stretched
    mins = xyz.min(0); maxs = xyz.max(0)
    center = (mins + maxs) / 2
    half = max((maxs - mins).max() / 2, 1e-3)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)


def main():
    args = parse_args()
    z = zarr.open(args.zarr, mode="r")
    pc_arr = z["data/point_cloud"]
    s, e = _episode_slice(z, args.episode)
    ep_len = e - s

    if args.frames:
        frames = [int(x) for x in args.frames.split(",")]
    else:
        frames = [args.frame]
    for f in frames:
        if f < 0 or f >= ep_len:
            raise ValueError(f"frame {f} out of episode {args.episode} length {ep_len}")

    n = len(frames)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(4.5 * ncols, 4.2 * nrows), dpi=140)

    zarr_name = os.path.basename(args.zarr.rstrip("/"))
    pt_counts = []
    for i, f in enumerate(frames):
        pc = np.asarray(pc_arr[s + f]).astype(np.float32)
        pt_counts.append(pc.shape[0])
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        _plot_one(ax, pc, args.elev, args.azim, args.point_size, args.max_points,
                  title=f"{zarr_name}\nep{args.episode} frame{f} (N={pc.shape[0]})")

    fig.tight_layout()
    out = args.out
    if out is None:
        base = zarr_name.replace(".zarr", "")
        tag = "_".join(str(x) for x in frames)
        out_dir = os.path.dirname(args.zarr.rstrip("/")) or "."
        out = os.path.join(out_dir, f"viz_{base}_ep{args.episode}_f{tag}.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out)
    print(f"[VIZ] saved -> {out}  (points/frame: {pt_counts})")


if __name__ == "__main__":
    main()
