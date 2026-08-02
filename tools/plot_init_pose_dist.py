"""Plot the drill initial-position distribution of a --cover_sobol collection run.

Reads the two npz files a run with --save_init_poses --cover_sobol leaves next to its zarr:
  <output>_init_poses/init_poses.npz      -- the initial pose of every episode actually stored
  <output>_init_poses/exam_proposals.npz  -- every Sobol candidate + its success mark
and draws one panel per (drill variant, base pose).

Two views:
  --view saved    (default) the stored dataset itself: where the 1:1-with-zarr episodes actually
                  sit, x/y marginals against the uniform expectation, duplicates marked
  --view outcome  the exam: which candidates the teacher solved and which it failed

    python3 tools/plot_init_pose_dist.py data/1.zarr_init_poses [--view saved] [-o out.png]
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---- palette: dataviz reference instance, light surface (validated all-pairs: worst CVD dE 24.7) ----
SURFACE = "#fcfcfb"
PLANE = "#f9f9f7"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
OK = "#2a78d6"        # categorical slot 1 -- the stored dataset (the subject of --view saved)
OK_RING = "#184f95"   # same hue, darker step: ring on duplicated stores
FAIL = "#eb6834"      # categorical slot 2 -- candidate failed, nothing stored (--view outcome)

XLIM = (0.49, 0.91)
YLIM = (-0.012, 0.412)
NBINS = 12

# base pose z per variant, in config/drill_variants.yaml order (initial_pos, initial_pos_1)
VARIANTS = [(0, "drill2", 0.0482, 0.13),
            (1, "drill_blue", 0.0537, 0.13),
            (2, "drill_yellow", 0.0552, 0.03)]


def match_saved_to_candidates(exam, saved, tol=1e-4):
    """For every stored episode, the index of the exam candidate it was run from (-1 if none:
    a pose that is not on the Sobol grid at all, e.g. an old retry-jittered attempt)."""
    out = np.full(len(saved["variant"]), -1, dtype=np.int64)
    for vid in np.unique(saved["variant"]):
        cand = np.where(exam["variant"] == vid)[0]
        rows = np.where(saved["variant"] == vid)[0]
        d = np.linalg.norm(saved["pos_local"][rows][:, None, :2]
                           - exam["pos_local"][cand][None, :, :2], axis=-1)
        j = d.argmin(1)
        out[rows] = np.where(d[np.arange(len(rows)), j] < tol, cand[j], -1)
    return out


def dead_zone_edge(x, solved, edges=(0.55, 0.60, 0.65, 0.70)):
    """Widest x < edge strip in which the teacher solved <10% of the candidates (0 if none)."""
    best = 0.0
    for e in edges:
        m = x < e
        if m.sum() >= 10 and solved[m].mean() < 0.10:
            best = e
    return best


def ks_uniform(v, lo, hi):
    """Max |empirical CDF - uniform CDF| over [lo, hi]. 0 = perfectly even spread."""
    if len(v) == 0 or hi <= lo:
        return float("nan")
    u = np.sort((np.asarray(v, dtype=np.float64) - lo) / (hi - lo))
    n = len(u)
    return float(np.max(np.abs(u - (np.arange(1, n + 1) - 0.5) / n)))


def _style_main(ax, row):
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.8)
    ax.set_xlim(*XLIM)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])


def _marginal(ax, v, lo, hi, horizontal=False):
    """Thin bars + a hairline at the height a perfectly uniform spread would reach.

    Clears only the count axis: this axes shares its position axis with the main scatter, and
    set_[xy]ticks([]) on a shared axis wipes the main panel's ticks and gridlines too."""
    cnt, edges = np.histogram(v, bins=NBINS, range=(lo, hi))
    w = (edges[1] - edges[0]) * 0.78
    if horizontal:
        ax.barh(edges[:-1] + (edges[1] - edges[0]) / 2, cnt, height=w, color=OK, lw=0)
        ax.axvline(len(v) / NBINS, color=MUTED, lw=0.8)
        ax.set_xlim(0, max(cnt.max() * 1.12, 1))
        ax.set_xticks([])
        ax.tick_params(axis="y", left=False, labelleft=False)
    else:
        ax.bar(edges[:-1] + (edges[1] - edges[0]) / 2, cnt, width=w, color=OK, lw=0)
        ax.axhline(len(v) / NBINS, color=MUTED, lw=0.8)
        ax.set_ylim(0, max(cnt.max() * 1.12, 1))
        ax.set_yticks([])
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_visible(False)


def draw_saved(fig, exam, saved, n_store, solved, pose_dir):
    """The stored dataset as the subject: every episode in the zarr is one mark here."""
    outer = fig.add_gridspec(2, 3, left=0.055, right=0.975, top=0.838, bottom=0.088,
                             wspace=0.22, hspace=0.34)
    tbl = []
    for col, (vid, name, z0, z1) in enumerate(VARIANTS):
        for row, (bi, z) in enumerate(((0, z0), (1, z1))):
            cell = outer[row, col]
            inner = cell.subgridspec(2, 2, width_ratios=[4.3, 1], height_ratios=[1, 4.3],
                                     wspace=0.05, hspace=0.05)
            ax = fig.add_subplot(inner[1, 0])
            ax_top = fig.add_subplot(inner[0, 0], sharex=ax)
            ax_rgt = fig.add_subplot(inner[1, 1], sharey=ax)

            mc = (exam["variant"] == vid) & (np.abs(exam["pos_local"][:, 2] - z) < 1e-3)
            cx, cy = exam["pos_local"][mc, 0], exam["pos_local"][mc, 1]
            ns = n_store[mc]
            lo_x, hi_x = cx.min(), cx.max()
            lo_y, hi_y = cy.min(), cy.max()
            # the stored dataset: one row per episode, so a pose stored k times appears k times
            sx = np.repeat(cx, ns)
            sy = np.repeat(cy, ns)

            _style_main(ax, row)
            ax.set_ylim(*YLIM)
            ax.set_aspect("equal")
            # context, not a series: candidates that produced nothing -- the holes in the dataset
            ax.scatter(cx[ns == 0], cy[ns == 0], s=15, facecolors="none", edgecolors=AXIS,
                       linewidths=0.8, zorder=2)
            ax.scatter(cx[ns == 1], cy[ns == 1], s=26, color=OK, edgecolors=SURFACE,
                       linewidths=0.6, zorder=4)
            ax.scatter(cx[ns > 1], cy[ns > 1], s=62, color=OK, edgecolors=OK_RING,
                       linewidths=1.2, zorder=5)
            _marginal(ax_top, sx, lo_x, hi_x)
            _marginal(ax_rgt, sy, lo_y, hi_y, horizontal=True)

            n_ep, n_uni = int(ns.sum()), int((ns > 0).sum())
            dx, dy = ks_uniform(sx, lo_x, hi_x), ks_uniform(sy, lo_y, hi_y)
            pos = cell.get_position(fig)
            fig.text(pos.x0, pos.y1 + 0.032, f"{name} · 姿态{bi} (z={z:.3f})", ha="left",
                     va="baseline", fontsize=11.5, color=INK, fontweight="semibold")
            fig.text(pos.x0, pos.y1 + 0.013,
                     f"存入 {n_ep} 条 · 不同位姿 {n_uni} · 重复 {n_ep - n_uni} · "
                     f"均匀性偏离 x {dx:.2f} / y {dy:.2f}",
                     ha="left", va="baseline", fontsize=8.5, color=INK2)
            if row == 1:
                ax.set_xlabel("x (m, env-local)", fontsize=9)
            if col == 0:
                ax.set_ylabel("y (m, env-local)", fontsize=9)
            tbl.append((name, bi, z, n_ep, n_uni, n_ep - n_uni, dx, dy))

    fig.suptitle("存入数据集的 1500 条 episode 的初始位置分布", x=0.055, y=0.988,
                 ha="left", fontsize=16, color=INK, fontweight="semibold")
    fig.text(0.055, 0.936, f"{os.path.normpath(pose_dir)} — 每个蓝点是一条真的存盘 episode。"
             f"上/右为 x、y 边际直方图({NBINS} 格),灰线 = 完全均匀时每格应有的高度。",
             ha="left", fontsize=10, color=INK2)
    handles = [
        Line2D([], [], marker="o", ls="", ms=6.0, mfc=OK, mec=SURFACE, mew=0.6, label="存入 1 条"),
        Line2D([], [], marker="o", ls="", ms=9.0, mfc=OK, mec=OK_RING, mew=1.2, label="同一位姿存入 2~4 条"),
        Line2D([], [], marker="o", ls="", ms=5.5, mfc="none", mec=AXIS, mew=0.9, label="该位置数据集里没有"),
        Line2D([], [], color=MUTED, lw=0.8, label="均匀分布的期望高度"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.050, 0.916), ncol=4,
               frameon=False, fontsize=9.5, labelcolor=INK2, handletextpad=0.5,
               columnspacing=2.2, borderaxespad=0)
    fig.text(0.055, 0.022, "均匀性偏离 = 存盘位置经验分布与均匀分布的最大累积偏差(0 = 完全均匀,"
             "y 的 0.03 量级是采样噪声,x 的 0.2~0.35 是真实的偏斜)。",
             ha="left", fontsize=8.5, color=MUTED)
    hdr = ("电钻", "姿态", "z", "存入", "不同位姿", "重复", "x偏离", "y偏离")
    return tbl, hdr


def draw_outcome(fig, exam, saved, n_store, solved, pose_dir):
    """The exam as the subject: one mark per Sobol candidate, colored by whether it was solved."""
    axes = fig.subplots(2, 3)
    fig.subplots_adjust(left=0.055, right=0.975, top=0.850, bottom=0.098, wspace=0.16, hspace=0.26)
    tbl = []
    for col, (vid, name, z0, z1) in enumerate(VARIANTS):
        for row, (bi, z) in enumerate(((0, z0), (1, z1))):
            ax = axes[row, col]
            m = (exam["variant"] == vid) & (np.abs(exam["pos_local"][:, 2] - z) < 1e-3)
            px, py, sv, ns = exam["pos_local"][m, 0], exam["pos_local"][m, 1], solved[m], n_store[m]
            _style_main(ax, row)
            ax.set_ylim(-0.012, 0.458)   # label lane above y=0.4
            ax.set_aspect("equal")

            edge = dead_zone_edge(px, sv)
            if edge:
                strip = px < edge
                ax.axvspan(XLIM[0], edge, color=GRID, alpha=0.55, lw=0, zorder=1)
                ax.annotate(f"灰带 x<{edge:.2f}:{int(strip.sum())} 个候选只成功了 "
                            f"{int(sv[strip].sum())} 个", xy=(0.495, 0.432), ha="left",
                            va="center", color=INK2, fontsize=8.5, zorder=6)

            ax.scatter(px[~sv], py[~sv], s=26, facecolors="none", edgecolors=FAIL,
                       linewidths=0.9, zorder=3)
            m1 = sv & (ns <= 1)
            ax.scatter(px[m1], py[m1], s=26, color=OK, edgecolors=SURFACE, linewidths=0.6, zorder=4)
            ax.scatter(px[ns > 1], py[ns > 1], s=62, color=OK, edgecolors=OK_RING,
                       linewidths=1.2, zorder=5)

            n_cand, n_ok = int(m.sum()), int(sv.sum())
            n_ep, n_extra = int(ns.sum()), int(np.maximum(ns - 1, 0).sum())
            ax.set_title(f"{name} · 姿态{bi} (z={z:.3f})", fontsize=11.5, color=INK,
                         fontweight="semibold", pad=14, loc="left")
            ax.annotate(f"候选 {n_cand} · 成功 {n_ok} ({n_ok / n_cand:.0%}) · "
                        f"存入 {n_ep} 条,其中重复 {n_extra}",
                        xy=(0, 1.012), xycoords="axes fraction", fontsize=8.5, color=INK2)
            if row == 1:
                ax.set_xlabel("x (m, env-local)", fontsize=9)
            if col == 0:
                ax.set_ylabel("y (m, env-local)", fontsize=9)
            tbl.append((name, bi, z, n_cand, n_ok, n_ok / n_cand, n_ep, n_extra))

    fig.suptitle("电钻初始位置分布:Sobol 考卷 vs 实际存入数据集", x=0.055, y=0.988,
                 ha="left", fontsize=16, color=INK, fontweight="semibold")
    fig.text(0.055, 0.936, f"{os.path.normpath(pose_dir)} — 每个电钻两个基础姿态,每姿态 250 个 "
             f"Sobol 候选。灰带 = 老师通过率不到 10% 的区域,也就是数据集里的空洞。",
             ha="left", fontsize=10, color=INK2)
    handles = [
        Line2D([], [], marker="o", ls="", ms=6.0, mfc=OK, mec=SURFACE, mew=0.6, label="成功 → 存入数据集"),
        Line2D([], [], marker="o", ls="", ms=9.0, mfc=OK, mec=OK_RING, mew=1.2, label="同一位姿被重复存入 2~4 次"),
        Line2D([], [], marker="o", ls="", ms=6.0, mfc="none", mec=FAIL, mew=0.9, label="失败 → 该位置数据集里没有"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.050, 0.916), ncol=3,
               frameon=False, fontsize=9.5, labelcolor=INK2, handletextpad=0.5,
               columnspacing=2.4, borderaxespad=0)
    fig.text(0.055, 0.022, "成功 = 考卷标记为成功,或该候选实际产出过存盘 episode"
             "(标记会被后续重抽的尝试覆盖,单看标记会少算)。", ha="left", fontsize=8.5, color=MUTED)
    hdr = ("电钻", "姿态", "z", "候选", "成功", "通过率", "存入", "重复")
    return tbl, hdr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pose_dir", help="<zarr output>_init_poses directory")
    ap.add_argument("--view", choices=("saved", "outcome"), default="saved",
                    help="saved = the stored dataset itself (default); outcome = the exam's pass/fail map")
    ap.add_argument("-o", "--out", default=None,
                    help="output png (default <pose_dir>/pos_distribution_<view>.png)")
    args = ap.parse_args()
    out = args.out or os.path.join(args.pose_dir, f"pos_distribution_{args.view}.png")

    exam = np.load(os.path.join(args.pose_dir, "exam_proposals.npz"))
    saved = np.load(os.path.join(args.pose_dir, "init_poses.npz"))
    src = match_saved_to_candidates(exam, saved)
    if (src < 0).any():
        print(f"[WARN] {int((src < 0).sum())} stored poses are not on the Sobol grid "
              f"(old retry-jitter run?) and are left out of the per-candidate counts", flush=True)
    # stores per candidate: 0 = never stored, >1 = the same pose written to the dataset twice or more
    n_store = np.zeros(len(exam["variant"]), dtype=np.int64)
    np.add.at(n_store, src[src >= 0], 1)
    # a candidate counts as solved if it is marked 1 OR it produced a stored episode -- the mark
    # alone undercounts, since a re-drawn candidate's later failure overwrites its earlier success
    solved = (exam["success"] == 1) | (n_store > 0)

    plt.rcParams.update({
        "font.family": ["Noto Sans CJK JP", "Droid Sans Fallback", "Noto Sans", "DejaVu Sans"],
        "axes.facecolor": SURFACE, "figure.facecolor": PLANE,
        "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
        "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 9,
    })
    # the saved view nests a square scatter + two marginal strips per cell, so it needs a taller
    # canvas: at 13.4x9.4 the equal-aspect scatter shrinks and the y-marginal floats off its side
    fig = plt.figure(figsize=(13.4, 11.2) if args.view == "saved" else (13.4, 9.4), dpi=170)
    draw = draw_saved if args.view == "saved" else draw_outcome
    tbl, hdr = draw(fig, exam, saved, n_store, solved, args.pose_dir)
    fig.savefig(out, facecolor=PLANE)
    print(f"saved -> {out}")
    print()

    print(f"{hdr[0]:<14}{hdr[1]:>4}{hdr[2]:>8}" + "".join(f"{h:>9}" for h in hdr[3:]))
    for r in tbl:
        cells = "".join(f"{v:>9.2f}" if isinstance(v, float) and v < 1 else f"{v:>9}" for v in r[3:])
        print(f"{r[0]:<14}{r[1]:>4}{r[2]:>8.3f}{cells}")
    cols = np.array([[float(v) for v in r[3:]] for r in tbl])
    tot = cols.sum(0)
    if args.view == "saved":
        summary = [int(tot[0]), int(tot[1]), int(tot[2]), cols[:, 3].max(), cols[:, 4].max()]
        note = "  (末两列为各面板最大值)"
    else:
        summary = [int(tot[0]), int(tot[1]), tot[1] / tot[0], int(tot[3]), int(tot[4])]
        note = ""
    print(f"{'合计':<13}{'':>4}{'':>8}"
          + "".join(f"{v:>9.2f}" if isinstance(v, float) else f"{v:>9}" for v in summary) + note)


if __name__ == "__main__":
    sys.exit(main())
