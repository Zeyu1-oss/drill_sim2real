"""Train the handle point-cloud segmentation network from collected pc_mask labels.

Learns "camera point -> is_handle" so that at real-robot deploy you can produce the mask from
the point cloud alone (no ground-truth drill pose). Supervises on the exact labels that
collect_dp3_data.py --save_mask wrote (drill body_mask region), so the network reproduces
compute_handle_mask's definition.

Runs in the dp3 conda env (torch + zarr + numpy); does NOT need Isaac Lab.

Usage:
  python scripts/train_handle_seg.py --zarr data/inspire_drill_grasp_mask.zarr \
      --out data/handle_seg.pt --max_frames 40000 --epochs 30
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from perception.handle_seg_net import HandleSegNet, preprocess


def load_data(zarr_path, cam_pts, max_frames, block=8192):
    """Subsampled (fewer, less-correlated frames) camera segment + labels loaded into memory."""
    import zarr
    z = zarr.open(zarr_path, mode="r")
    assert "pc_mask" in z["data"], f"{zarr_path} has no data/pc_mask (collect with --save_mask)"
    pc = z["data"]["point_cloud"]; mk = z["data"]["pc_mask"]
    N = pc.shape[0]
    stride = max(1, N // max(max_frames, 1))
    Xs, Ys = [], []
    for s in range(0, N, block):
        pcb = np.asarray(pc[s:s + block, :cam_pts, :3], dtype=np.float32)   # (b, cam, 3)
        mkb = np.asarray(mk[s:s + block, :cam_pts], dtype=np.float32)        # (b, cam)
        Xs.append(pcb[::stride]); Ys.append(mkb[::stride])
    X = np.concatenate(Xs, 0); Y = np.concatenate(Ys, 0)
    print(f"[data] {zarr_path}: {N} frames -> {len(X)} sampled (stride {stride}), "
          f"cam_pts={cam_pts}, handle frac={Y.mean()*100:.2f}%", flush=True)
    return X, Y


def metrics(logit, y, thresh=0.5):
    p = (torch.sigmoid(logit) > thresh)
    t = y > 0.5
    tp = (p & t).sum().item(); fp = (p & ~t).sum().item(); fn = (~p & t).sum().item()
    tn = (~p & ~t).sum().item()
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    iou = tp / max(tp + fp + fn, 1); acc = (tp + tn) / max(tp + fp + fn + tn, 1)
    return dict(precision=prec, recall=rec, iou=iou, acc=acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", type=str, default="data/inspire_drill_grasp_mask.zarr")
    ap.add_argument("--out", type=str, default="data/handle_seg.pt")
    ap.add_argument("--cam_pts", type=int, default=1024)   # camera segment length (= pc_num_points)
    ap.add_argument("--max_frames", type=int, default=40000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = args.device if torch.cuda.is_available() else "cpu"

    X, Y = load_data(args.zarr, args.cam_pts, args.max_frames)
    X = torch.from_numpy(X); Y = torch.from_numpy(Y)

    n = len(X); perm = torch.randperm(n)
    nval = int(n * args.val_ratio)
    vi, ti = perm[:nval], perm[nval:]
    Xtr, Ytr, Xva, Yva = X[ti], Y[ti], X[vi], Y[vi]

    # handle points are the minority -> weight the positive class in BCE
    pos = Ytr.mean().clamp(1e-4, 1 - 1e-4)
    pos_weight = torch.tensor([(1 - pos) / pos], device=dev)
    print(f"[train] {len(Xtr)} train / {len(Xva)} val | pos_weight={pos_weight.item():.1f}", flush=True)

    net = HandleSegNet(in_ch=3).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def run_epoch(Xd, Yd, train):
        net.train(train)
        idx = torch.randperm(len(Xd)) if train else torch.arange(len(Xd))
        tot_loss = 0.0; logits_all, ys_all = [], []
        for s in range(0, len(Xd), args.batch):
            b = idx[s:s + args.batch]
            pc = preprocess(Xd[b].to(dev)); y = Yd[b].to(dev)
            with torch.set_grad_enabled(train):
                logit = net(pc)
                loss = lossf(logit, y)
                if train:
                    opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item() * len(b)
            if not train:
                logits_all.append(logit.detach().flatten().cpu()); ys_all.append(y.flatten().cpu())
        if train:
            return tot_loss / len(Xd), None
        return tot_loss / len(Xd), metrics(torch.cat(logits_all), torch.cat(ys_all))

    best_iou = -1.0
    for ep in range(args.epochs):
        tr_loss, _ = run_epoch(Xtr, Ytr, True)
        va_loss, m = run_epoch(Xva, Yva, False)
        print(f"epoch {ep:3d} | train_loss={tr_loss:.4f} val_loss={va_loss:.4f} | "
              f"IoU={m['iou']*100:.1f}% prec={m['precision']*100:.1f}% rec={m['recall']*100:.1f}% "
              f"acc={m['acc']*100:.1f}%", flush=True)
        if m["iou"] > best_iou:
            best_iou = m["iou"]
            os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
            torch.save({"state_dict": net.state_dict(), "cam_pts": args.cam_pts,
                        "in_ch": 3, "val_metrics": m}, args.out)
    print(f"[done] best val IoU={best_iou*100:.1f}% -> saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
