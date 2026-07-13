"""离线重标注:把现有 zarr 的动作标签换成前向滑动平均(消 teacher 的 PWM 抖动)。

与 collect_dp3_data.py --label_smooth_k 用同一个函数(perception.target_drive.
smooth_labels_forward),所以"重标注旧数据"与"用新脚本重采"结果一致,省一次全量仿真。

安全机制:
  - 原始 action 先备份到 <zarr>.action_raw_backup.npz(含 episode_ends 对齐检查);
  - 备份文件已存在时拒绝再次执行(防止双重平滑),--restore 可随时还原;
  - 检测到 train.py 正在运行时拒绝执行(dataloader 磁盘惰性读,中途改会毒化训练)。

用法:
  python tools/relabel_smooth_actions.py <zarr_path> [--k 8]
  python tools/relabel_smooth_actions.py <zarr_path> --restore   # 还原原始标签
"""
import argparse
import os
import subprocess
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np

from perception.target_drive import smooth_labels_forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zarr_path")
    ap.add_argument("--k", type=int, default=8, help="前向滑动平均窗口(步)")
    ap.add_argument("--restore", action="store_true", help="从备份还原原始 action")
    ap.add_argument("--force", action="store_true", help="跳过 train.py 运行检测")
    args = ap.parse_args()

    if not args.force:
        r = subprocess.run(["pgrep", "-f", "train.py"], capture_output=True, text=True)
        if r.stdout.strip():
            sys.exit("[ABORT] 检测到 train.py 正在运行(PID: %s)。训练对 zarr 是磁盘惰性读,"
                     "中途改标签会毒化该轮训练。请先停训练,或确认无关后加 --force。"
                     % r.stdout.strip().replace("\n", ","))

    import zarr
    z = zarr.open_group(args.zarr_path, mode="r+")
    ends = z["meta/episode_ends"][:]
    act = z["data/action"]
    backup_path = args.zarr_path.rstrip("/") + ".action_raw_backup.npz"

    if args.restore:
        if not os.path.exists(backup_path):
            sys.exit(f"[ABORT] 无备份文件: {backup_path}")
        b = np.load(backup_path)
        assert np.array_equal(b["episode_ends"], ends), "备份与当前 zarr 的 episode_ends 不一致"
        act[:] = b["action"]
        os.remove(backup_path)
        print(f"[OK] 已还原原始 action({b['action'].shape}),备份文件已删除。")
        return

    if os.path.exists(backup_path):
        sys.exit(f"[ABORT] 备份已存在: {backup_path}\n"
                 "该 zarr 已重标注过,再跑会双重平滑。要重做请先 --restore。")

    A = act[:]                                  # (T,13) ~80MB,载入内存
    assert A.shape[0] == ends[-1], f"zarr 不一致: action {A.shape[0]} != episode_ends[-1] {ends[-1]}"
    np.savez(backup_path, action=A, episode_ends=ends)
    print(f"[BACKUP] 原始 action -> {backup_path}")

    starts = np.concatenate([[0], ends[:-1]])
    new = np.empty_like(A)
    for s, e in zip(starts, ends):              # 逐 episode 平滑(窗口不跨集)
        new[s:e] = smooth_labels_forward(A[s:e], args.k)
    act[:] = new
    print(f"[OK] {len(ends)} 集 / {A.shape[0]} 帧 已重标注 (k={args.k})")

    # ---- 验证:抖动应大降,直流偏置(握力/抗重力)应保留 ----
    rng = np.random.default_rng(0)
    eps = rng.choice(len(ends), min(30, len(ends)), replace=False)
    st = z["data/state"]
    dA_old, dA_new, gap_old, gap_new = [], [], [], []
    for e in eps:
        s0, s1 = starts[e], ends[e]
        o, n, q = A[s0:s1], new[s0:s1], st[s0:s1, :13]
        dA_old.append(np.abs(np.diff(o[:, :7], axis=0)).mean())
        dA_new.append(np.abs(np.diff(n[:, :7], axis=0)).mean())
        gap_old.append((o[-100:, 7:13] - q[-100:, 7:13]).mean())
        gap_new.append((n[-100:, 7:13] - q[-100:, 7:13]).mean())
    sat_old = np.mean([(np.abs(np.diff(A[starts[e]:ends[e], :7], axis=0)) > 0.0195).mean() for e in eps])
    sat_new = np.mean([(np.abs(np.diff(new[starts[e]:ends[e], :7], axis=0)) > 0.0195).mean() for e in eps])
    print(f"[VERIFY] 手臂逐步|Δ|: {np.mean(dA_old):.4f} -> {np.mean(dA_new):.4f} rad "
          f"| 贴限速比例: {sat_old*100:.0f}% -> {sat_new*100:.0f}%")
    print(f"[VERIFY] 手指 hold 压入偏置(应基本不变): {np.mean(gap_old):+.3f} -> {np.mean(gap_new):+.3f} rad")
    print("[NOTE] 训练端会自动用新标签重算 normalizer;旧 checkpoint 与新标签不兼容,需重训。")


if __name__ == "__main__":
    main()
