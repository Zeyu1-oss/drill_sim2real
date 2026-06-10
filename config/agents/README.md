# RL 训练配置文件

## 配置文件说明

### rl_games_ppo_cfg.yaml
**默认配置** - 适合中等规模训练（num_envs ~ 1024-2048）

### rl_games_ppo_cfg_small.yaml
**小规模配置** - 适合快速测试和调试（num_envs < 512）
- 较小的网络（128, 64）
- 较少的训练轮数（200）
- 更高的熵系数（更多探索）
- 较小的批次大小

### rl_games_ppo_cfg_large.yaml
**大规模配置** - 适合最终训练（num_envs >= 2048）
- 更大的网络（512, 256, 128）
- 更多的训练轮数（1000）
- 启用归一化
- 更大的批次和 horizon

## 主要超参数说明

### 网络架构
- `units`: MLP 隐藏层单元数，例如 `[256, 128, 64]`
- `activation`: 激活函数，可选 `elu`, `relu`, `tanh`
- `separate`: Actor 和 Critic 是否使用独立网络

### PPO 核心参数
- `gamma`: 折扣因子（0.99 常用）
- `tau`: GAE lambda 参数（0.95 常用）
- `learning_rate`: 学习率（3e-4 到 5e-4）
- `e_clip`: PPO 裁剪范围（0.2 常用）

### 训练配置
- `horizon_length`: 每次更新收集的步数
- `minibatch_size`: 小批量大小
- `mini_epochs`: 每次更新的小轮数
- `max_epochs`: 最大训练轮数

### 探索 vs 利用
- `entropy_coef`: 熵系数，越大越鼓励探索
- `kl_threshold`: KL 散度阈值，用于自适应学习率

## 使用方法

### 在训练脚本中使用

```python
from omni.isaac.lab_tasks.utils import load_cfg_from_registry, parse_env_cfg

# 加载 RL 配置
rl_cfg = load_cfg_from_registry("inspire_hand_grasp_drill", "rl_games_ppo_cfg")
# 或指定配置文件路径
rl_cfg = load_cfg_from_registry("inspire_hand_grasp_drill", "rl_games_ppo_cfg_small")
```

### 直接加载 YAML

```python
import yaml

with open("config/agents/rl_games_ppo_cfg.yaml", "r") as f:
    rl_params = yaml.safe_load(f)
```

## 调整建议

### 训练不稳定
- 降低 `learning_rate`
- 增加 `grad_norm`
- 启用 `normalize_input` 和 `normalize_value`

### 收敛太慢
- 增加 `learning_rate`
- 增加 `entropy_coef`（更多探索）
- 增加网络大小

### 过拟合
- 降低网络大小
- 增加 `entropy_coef`
- 增加 `mini_epochs`

### 样本效率低
- 增加 `horizon_length`
- 增加 `minibatch_size`
- 调整 `gamma` 和 `tau`
