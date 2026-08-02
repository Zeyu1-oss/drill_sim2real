"""Lightweight PointNet-style point-wise binary segmentation for the drill handle.

Input:  camera point cloud (B, N, 3) env-local xyz (the camera segment, first pc_num_points).
Output: per-point is_handle logit (B, N); sigmoid > 0.5 -> handle.

Trained on the pc_mask labels produced by collect_dp3_data.py --save_mask, so it learns the
exact same "handle" definition (drill body_mask region) as compute_handle_mask, but from point
cloud alone -- no ground-truth drill pose needed at deploy. Shared by scripts/train_handle_seg.py
(training) and deploy (replace compute_handle_mask with this network's output).

Preprocessing (preprocess()) MUST be applied identically at train and deploy time.
"""
import torch
import torch.nn as nn


def preprocess(pc: torch.Tensor) -> torch.Tensor:
    """Per-cloud centering: subtract the centroid so the net is translation invariant.
    pc: (B, N, 3) -> (B, N, 3). Apply the same at deploy before feeding the net."""
    return pc - pc.mean(dim=1, keepdim=True)


class HandleSegNet(nn.Module):
    """PointNet segmentation: per-point MLP -> global max-pool -> concat back -> per-point MLP -> logit.
    Small and fast (real-time), point-cloud native, no external ops."""

    def __init__(self, in_ch: int = 3, feat=(64, 128, 256)):
        super().__init__()
        self.in_ch = in_ch
        self.point_mlp = nn.Sequential(
            nn.Linear(in_ch, feat[0]), nn.LayerNorm(feat[0]), nn.ReLU(),
            nn.Linear(feat[0], feat[1]), nn.LayerNorm(feat[1]), nn.ReLU(),
            nn.Linear(feat[1], feat[2]), nn.LayerNorm(feat[2]), nn.ReLU(),
        )
        self.seg_mlp = nn.Sequential(
            nn.Linear(feat[2] * 2, feat[1]), nn.LayerNorm(feat[1]), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feat[1], feat[0]), nn.LayerNorm(feat[0]), nn.ReLU(),
            nn.Linear(feat[0], 1),
        )

    def forward(self, pc: torch.Tensor) -> torch.Tensor:
        # pc: (B, N, in_ch)  ->  (B, N) per-point logit
        f = self.point_mlp(pc)                            # (B, N, C)
        g = f.max(dim=1, keepdim=True).values             # (B, 1, C) global feature
        h = torch.cat([f, g.expand(-1, f.shape[1], -1)], dim=-1)   # (B, N, 2C)
        return self.seg_mlp(h).squeeze(-1)                # (B, N)

    @torch.no_grad()
    def predict_mask(self, pc: torch.Tensor, thresh: float = 0.5) -> torch.Tensor:
        """pc: (B, N, 3) raw env-local xyz -> (B, N) bool is_handle. Handles preprocessing internally."""
        self.eval()
        logit = self.forward(preprocess(pc))
        return torch.sigmoid(logit) > thresh
