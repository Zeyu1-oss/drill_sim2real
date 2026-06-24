import numpy as np
import torch

from diffusion_policy_3d.dataset.realdex_dataset import RealDexDataset
from diffusion_policy_3d.model.common.normalizer import (
    LinearNormalizer, SingleFieldLinearNormalizer)


class InspireDrillDataset(RealDexDataset):
    """Dataset loader for the InspireDrill task.

    Data format (produced by collect_dp3_data.py):
      - state:   (T, 13)  13 controlled joint positions (7 arm + 6 hand)
      - action:  (T, 13)  7 arm joints + 6 hand joints
      - point_cloud: (T, total_pc, 3)  fused env-local point cloud
                     (camera + robot-FK + drill-oracle + ground)

    Zarr layout:
      /data/state         (N_steps, 13) float32
      /data/action        (N_steps, 13) float32
      /data/point_cloud   (N_steps, total_pc, 3) float32
      /meta/episode_ends  (N_episodes,) int64  cumulative lengths

    内存:数据集解压后远大于本机 RAM(~76-86 GB vs 15 GB),所以
      - replay_buffer 用 create_from_path(mode='r') 留在磁盘惰性读切片;
      - get_normalizer 对 point_cloud 分块流式统计 min/max,不整块进内存。
    """

    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.05,
        max_train_episodes=None,
        task_name=None,
    ):
        # Skip RealDexDataset.__init__ to avoid hardcoded 'img' key requirement.
        # Replicate the needed setup directly.
        from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
        from diffusion_policy_3d.common.sampler import (
            SequenceSampler, get_val_mask, downsample_mask)

        self.task_name = task_name
        # 盘上惰性读取:不把整个 point_cloud 读进内存(数据集 >> RAM)。
        # 每个样本只从磁盘解压读取 horizon 帧。
        self.replay_buffer = ReplayBuffer.create_from_path(zarr_path, mode='r')
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    @staticmethod
    def _limits_normalizer_from_stats(input_min, input_max, input_mean, input_std,
                                      output_min=-1.0, output_max=1.0, range_eps=1e-4):
        """复刻 normalizer._fit 的 mode='limits', fit_offset=True 数学,
        但 min/max/mean/std 由外部流式统计传入(避免整块 [:] 读入内存)。"""
        input_min = input_min.clone().float()
        input_max = input_max.clone().float()
        input_range = input_max - input_min
        ignore_dim = input_range < range_eps
        input_range[ignore_dim] = output_max - output_min
        scale = (output_max - output_min) / input_range
        offset = output_min - scale * input_min
        offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
        input_stats = {
            'min': input_min, 'max': input_max,
            'mean': input_mean.float(), 'std': input_std.float(),
        }
        return SingleFieldLinearNormalizer.create_manual(scale, offset, input_stats)

    def _streaming_pc_normalizer(self, block_frames=4096):
        """分块扫一遍磁盘上的 point_cloud,逐 xyz 维统计 min/max/mean/std。
        一次只持有 block_frames 帧(~0.2-0.4 GB),不会爆内存。"""
        pc = self.replay_buffer['point_cloud']          # zarr (T, P, 3),磁盘
        T = int(pc.shape[0])
        D = int(pc.shape[-1])                           # 3
        run_min = run_max = None
        s = torch.zeros(D, dtype=torch.float64)
        ss = torch.zeros(D, dtype=torch.float64)
        n = 0
        for start in range(0, T, block_frames):
            arr = pc[start:start + block_frames]        # numpy (b, P, 3),仅这一块解压
            t = torch.from_numpy(np.asarray(arr, dtype=np.float32)).reshape(-1, D)
            bmin = t.min(dim=0).values
            bmax = t.max(dim=0).values
            run_min = bmin if run_min is None else torch.minimum(run_min, bmin)
            run_max = bmax if run_max is None else torch.maximum(run_max, bmax)
            td = t.double()
            s += td.sum(dim=0)
            ss += (td * td).sum(dim=0)
            n += t.shape[0]
        mean = (s / max(n, 1)).float()
        var = (ss / max(n, 1)).float() - mean * mean
        std = var.clamp_min(0).sqrt()
        return self._limits_normalizer_from_stats(run_min, run_max, mean, std)

    def get_normalizer(self, mode='limits', **kwargs):
        assert mode == 'limits', f"InspireDrillDataset 只实现了 limits(收到 {mode})"
        normalizer = LinearNormalizer()
        # action / state 很小(~0.1 GB),正常 fit;它们是磁盘 zarr,_fit 内部 [:] 即可。
        normalizer['action'] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer['action'], mode=mode, **kwargs)
        normalizer['agent_pos'] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer['state'], mode=mode, **kwargs)
        # point_cloud 太大(76-86 GB),流式统计。
        normalizer['point_cloud'] = self._streaming_pc_normalizer()
        return normalizer
