# DP3 自定义任务配置指南

## 核心逻辑（M-N 不匹配完全 OK）

你的任务只需要关心三件事：
1. **数据格式** — zarr，包含 `state`（维度 M）和 `action`（维度 N）
2. **配置文件** — 一个 yaml 文件，声明 M 和 N 的值
3. **Dataset 类** — 如果你的 zarr key 跟现有的一致（`state`, `action`, `point_cloud`, `img`），则**完全不需要写新代码**

---

## 你需要做的 5 件事

### Step 1: 把数据转成 zarr 格式

如果你的数据已经是 numpy 数组，创建一个转换脚本：

```python
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer

# 你的数据：
# states[i]  shape: (T_i, M)  — M = state 维度
# actions[i] shape: (T_i, N)  — N = action 维度
# point_clouds[i] shape: (T_i, P, 3) 或 (T_i, P, 6)

buffer = ReplayBuffer.create_empty_zarr()
for i in range(num_episodes):
    buffer.add_episode({
        'state': states[i].astype(np.float32),
        'action': actions[i].astype(np.float32),
        'point_cloud': point_clouds[i].astype(np.float32),
    })
buffer.save_to_path('data/your_task_expert.zarr')
```

如果你的数据 key 跟上面一样（`state`, `action`, `point_cloud`），**跳过 Step 2**。

---

### Step 2: 写 Dataset 类（仅当 zarr key 不同才需要）

创建文件 `diffusion_policy_3d/dataset/your_task_dataset.py`：

```python
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.common.pytorch_util import dict_apply
import torch, numpy as np

class YourTaskDataset(BaseDataset):
    def __init__(self, zarr_path, horizon=16, pad_before=1, pad_after=7,
                 seed=42, val_ratio=0.02, max_train_episodes=None, task_name=None):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['state', 'action', 'point_cloud', 'img'])
        val_mask = get_val_mask(self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = downsample_mask(~val_mask, max_n=max_train_episodes, seed=seed)
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, sequence_length=horizon,
            pad_before=pad_before, pad_after=pad_after, episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],        # (T, N)
            'agent_pos': self.replay_buffer['state'],      # (T, M)
            'point_cloud': self.replay_buffer['point_cloud'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self): return len(self.sampler)

    def _sample_to_data(self, sample):
        return {
            'obs': {
                'point_cloud': sample['point_cloud'].astype(np.float32),
                'agent_pos': sample['state'].astype(np.float32),   # 输出 M 维
            },
            'action': sample['action'].astype(np.float32)           # 输出 N 维
        }

    def __getitem__(self, idx):
        return dict_apply(self._sample_to_data(self.sampler.sample_sequence(idx)), torch.from_numpy)
```

然后在 `diffusion_policy_3d/dataset/__init__.py` 中添加导出。

---

### Step 3: 创建 Task Config

创建文件 `diffusion_policy_3d/config/task/your_task.yaml`：

```yaml
name: your_task
task_name: your_env_name

shape_meta: &shape_meta
  obs:
    point_cloud:
      shape: [512, 3]   # ← 你的点云维度 (P, 3) 或 (P, 6)
      type: point_cloud
    agent_pos:
      shape: [M]        # ← 替换为你的 state 维度
      type: low_dim
  action:
    shape: [N]          # ← 替换为你的 action 维度

# === 仿真任务 ===
env_runner:
  _target_: diffusion_policy_3d.env_runner.adroit_runner.AdroitRunner
  eval_episodes: 20
  max_steps: 300
  n_obs_steps: ${n_obs_steps}
  n_action_steps: ${n_action_steps}
  fps: 10
  task_name: your_env_name
  render_size: 84
  use_point_crop: ${policy.use_point_crop}

# === 真机任务（无仿真环境）===
# env_runner: null

dataset:
  _target_: diffusion_policy_3d.dataset.adroit_dataset.AdroitDataset  # 如果 zarr key 一致用这个
  # _target_: diffusion_policy_3d.dataset.your_task_dataset.YourTaskDataset  # 如果你写了新 Dataset
  zarr_path: data/your_task_expert.zarr
  horizon: ${horizon}
  pad_before: ${eval:'${n_obs_steps}-1'}
  pad_after: ${eval:'${n_action_steps}-1'}
  seed: 42
  val_ratio: 0.02
  max_train_episodes: 90
```

> **唯一需要填的就是 `[M]`、`[N]` 和 `[P, 3]`/`[P, 6]` 这三个数字。**

---

### Step 4: Env Runner（仅仿真任务需要，且仅当环境不是 Adroit/MetaWorld 才需要写新代码）

参考 `diffusion_policy_3d/env_runner/adroit_runner.py`，继承 `BaseRunner`，核心是一个 `run(policy)` 方法。

如果你的仿真环境是 MuJoCo + point cloud，可以复用现有 runner，只需要在 `AdroitEnv` 中注册你的环境名。

---

### Step 5: 启动训练

```bash
cd 3D-Diffusion-Policy

# 训练
python train.py --config-name=dp3.yaml task=your_task \
  hydra.run.dir=data/outputs/your_task-dp3-seed0 \
  training.device="cuda:0" training.seed=0 \
  checkpoint.save_ckpt=True

# 或用脚本
bash scripts/train_policy.sh dp3 your_task 0601 0 0
```

---

## 常见问题

**Q: 我的 state 维度 M 和 action 维度 N 不一样，DP3 能处理吗？**
A: 完全没问题。DP3 的 MLP 预测头会根据 `shape_meta.action.shape` 自动调整，M 和 N 不需要相等。

**Q: 我的 zarr 里用的 key 不是 `state`，是 `robot_state` 怎么办？**
A: 在 Dataset 的 `_sample_to_data` 里改成你的 key 即可。参考 Step 2 的模板。

**Q: 我的数据只有点云没有 RGB，点云是 (T, P, 3) 而不是 (T, P, 6)，能跑吗？**
A: 能。`shape_meta.obs.point_cloud.shape: [P, 3]` 即可，policy 配置中 `in_channels: 3`（不是 6）。

**Q: 我不知道数据的维度怎么办？**
A: 用 Python 快速查看：
```python
import zarr
z = zarr.open('your_data.zarr')
print(z['data/point_cloud'].shape)  # 点云维度
print(z['data/state'].shape)         # state 维度
print(z['data/action'].shape)        # action 维度
print(z['meta/episode_ends'][:])      # episode 数量
```
