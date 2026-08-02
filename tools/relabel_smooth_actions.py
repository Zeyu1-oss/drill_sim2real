"""Offline relabel: replace an existing zarr's action labels with a forward moving average
(removes the teacher's PWM jitter).

Uses the same function as collect_dp3_data.py --label_smooth_k (perception.target_drive.
smooth_labels_forward), so "relabel old data" gives the same result as "re-collect with the new
script", saving a full re-simulation.

Safety:
  - original action is first backed up to <zarr>.action_raw_backup.npz (with episode_ends check);
  - refuses to run again if the backup already exists (prevents double smoothing); --restore reverts;
  - refuses to run if train.py is detected running (dataloader lazily reads disk; changing mid-run
    poisons that training run).

Usage:
  python tools/relabel_smooth_actions.py <zarr_path> [--k 8]
  python tools/relabel_smooth_actions.py <zarr_path> --restore   # restore original labels
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
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--restore", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not args.force:
        r = subprocess.run(["pgrep", "-f", "train.py"], capture_output=True, text=True)
        if r.stdout.strip():
            sys.exit("[ABORT] train.py detected running (PID: %s). Training reads the zarr lazily "
                     "from disk; changing labels mid-run poisons that run. Stop training first, or "
                     "add --force if confirmed unrelated."
                     % r.stdout.strip().replace("\n", ","))

    import zarr
    z = zarr.open_group(args.zarr_path, mode="r+")
    ends = z["meta/episode_ends"][:]
    act = z["data/action"]
    backup_path = args.zarr_path.rstrip("/") + ".action_raw_backup.npz"

    if args.restore:
        if not os.path.exists(backup_path):
            sys.exit(f"[ABORT] no backup file: {backup_path}")
        b = np.load(backup_path)
        assert np.array_equal(b["episode_ends"], ends), "backup and current zarr episode_ends mismatch"
        act[:] = b["action"]
        os.remove(backup_path)
        print(f"[OK] restored original action ({b['action'].shape}), backup file removed.")
        return

    if os.path.exists(backup_path):
        sys.exit(f"[ABORT] backup already exists: {backup_path}\n"
                 "this zarr was already relabeled; running again would double-smooth. Use --restore to redo.")

    A = act[:]                                  # (T,13) ~80MB, load into memory
    assert A.shape[0] == ends[-1], f"zarr inconsistent: action {A.shape[0]} != episode_ends[-1] {ends[-1]}"
    np.savez(backup_path, action=A, episode_ends=ends)
    print(f"[BACKUP] original action -> {backup_path}")

    starts = np.concatenate([[0], ends[:-1]])
    new = np.empty_like(A)
    for s, e in zip(starts, ends):              # smooth per episode (window does not cross episodes)
        new[s:e] = smooth_labels_forward(A[s:e], args.k)
    act[:] = new
    print(f"[OK] {len(ends)} episodes / {A.shape[0]} frames relabeled (k={args.k})")

    # ---- verify: jitter should drop a lot, DC bias (grip force / anti-gravity) should be kept ----
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
    print(f"[VERIFY] arm per-step |d|: {np.mean(dA_old):.4f} -> {np.mean(dA_new):.4f} rad "
          f"| rate-limit-saturated ratio: {sat_old*100:.0f}% -> {sat_new*100:.0f}%")
    print(f"[VERIFY] finger hold press-in bias (should barely change): {np.mean(gap_old):+.3f} -> {np.mean(gap_new):+.3f} rad")
    print("[NOTE] the training side recomputes the normalizer with new labels; old checkpoints are incompatible, retrain.")


if __name__ == "__main__":
    main()
