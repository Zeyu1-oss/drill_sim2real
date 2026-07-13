#!/usr/bin/env python3
"""
Stage2 训练脚本：从 Stage1 成功的抓取姿态出发，训练将电钻对齐到目标坐标系。
数据来源：collect_success_data.py 生成的 success_data.pkl
"""

import argparse
import os
import sys
import yaml
from datetime import datetime
from typing import Tuple

import numpy as np
import torch


# ============================================================
# Dataset + Stage2 Env (imported from stage2_env.py after AppLauncher)
# ============================================================
_success_dataset_cls = None
_stage2_env_cls = None


def get_stage2_env_class():
    """Build Stage2Env (inherits GraspDrillEnv) after AppLauncher is running."""
    global _stage2_env_cls, _success_dataset_cls
    if _stage2_env_cls is not None:
        return _stage2_env_cls
    from tasks.stage2_env import get_stage2_env_class as _make, SuccessDataDataset
    _stage2_env_cls = _make()
    _success_dataset_cls = SuccessDataDataset
    return _stage2_env_cls


def get_success_dataset(pkl_path, device="cuda:0"):
    """Create a dataset from the pkl file."""
    global _success_dataset_cls
    if _success_dataset_cls is None:
        from tasks.stage2_env import SuccessDataDataset
        _success_dataset_cls = SuccessDataDataset
    ds = _success_dataset_cls(pkl_path)
    ds.device = device
    return ds


def create_stage2_env_cfg(
    num_envs: int = 256,
    device: str = "cuda:0",
    headless: bool = False,
    debug: bool = False,
    target_pos: Tuple[float, float, float] = (0.0, 0.0, 0.85),
    target_quat: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
):
    from tasks.stage2_env import create_stage2_env_cfg as _make_cfg
    return _make_cfg(
        num_envs=num_envs,
        device=device,
        headless=headless,
        debug=debug,
        target_pos=target_pos,
        target_quat=target_quat,
    )

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (scripts/ -> ../..)
    sys.path.insert(0, project_root)

    parser = argparse.ArgumentParser(description="Stage2 训练 - 从 success_data.pkl 加载姿态，对齐电钻到目标坐标系")
    parser.add_argument("--headless", action="store_true", help="无头模式")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--num_envs", type=int, default=4096, help="并行环境数量")
    parser.add_argument("--dataset", type=str, default="collected_data/success_data.pkl", help="success_data.pkl 路径")
    parser.add_argument("--rl_config", type=str, default="config/agents/rl_games_ppo_cfg.yaml", help="RL配置")
    parser.add_argument("--checkpoint", type=str, default=None, help="加载检查点（Stage1 权重）")
    parser.add_argument("--run_name", type=str, default=None, help="run名称")
    parser.add_argument("--target_pos", type=float, nargs=3, default=[0.0, 0.0, 0.85],
                        help="目标位置 (x y z)，单位米")
    parser.add_argument("--target_quat", type=float, nargs=4, default=[1.0, 0.0, 0.0, 0.0],
                        help="目标旋转 (w x y z)，单位四元数")
    parser.add_argument("--debug", action="store_true", help="开启 debug 可视化")
    args = parser.parse_args()

    # 先导入 AppLauncher，再导入 isaaclab 相关模块
    try:
        from isaaclab.app import AppLauncher
    except ImportError:
        from omni.isaac.lab.app import AppLauncher

    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app

    # 初始化数据集
    success_dataset = get_success_dataset(args.dataset, args.device)

    # 获取 Stage2Env 类（在 AppLauncher 之后才可导入）
    Stage2Env = get_stage2_env_class()

    print(f"[Stage2] 目标位置: {args.target_pos}  目标旋转: {args.target_quat}")

    cfg = create_stage2_env_cfg(
        num_envs=args.num_envs,
        device=args.device,
        headless=args.headless,
        target_pos=tuple(args.target_pos),
        target_quat=tuple(args.target_quat),
    )
    env = Stage2Env(
        cfg=cfg,
        success_dataset=success_dataset,
        target_pos=tuple(args.target_pos),
        target_quat=tuple(args.target_quat),
        debug=args.debug,
    )

    # 删除模板 prim（避免视觉干扰）
    from isaaclab.sim.utils import delete_prim
    try:
        delete_prim("/World/Template")
        print("[INFO] 已删除 /World/Template")
    except Exception as e:
        print(f"[WARN] 删除 /World/Template 失败: {e}")

    from rl_games.common import env_configurations, vecenv
    from rl_games.common.algo_observer import IsaacAlgoObserver
    from rl_games.torch_runner import Runner
    from rl_games.common import a2c_common
    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper as _RlGamesVecEnvWrapper

    def _fixed_discount_values(self, fdones, last_extrinsic_values,
                               mb_fdones, mb_extrinsic_values, mb_rewards):
        gamma = self.gamma
        tau = self.tau
        mb_advs = torch.zeros_like(mb_rewards)
        lastgaelam = torch.zeros_like(mb_rewards[0])

        for t in reversed(range(self.horizon_length)):
            if t == self.horizon_length - 1:
                nextnonterminal = 1.0 - fdones
                nextvalues = last_extrinsic_values
            else:
                nextnonterminal = 1.0 - mb_fdones[t + 1]
                nextvalues = mb_extrinsic_values[t + 1]

            if isinstance(nextvalues, torch.Tensor):
                nextvalues = nextvalues.float()
            if isinstance(nextnonterminal, torch.Tensor):
                nextnonterminal = nextnonterminal.float()

            if isinstance(nextvalues, torch.Tensor) and nextvalues.shape != mb_rewards[t].shape:
                nextvalues = nextvalues.view_as(mb_rewards[t])
            if isinstance(nextnonterminal, torch.Tensor) and nextnonterminal.shape != mb_rewards[t].shape:
                nextnonterminal = nextnonterminal.view_as(mb_rewards[t])

            delta = mb_rewards[t] + gamma * nextvalues * nextnonterminal - mb_extrinsic_values[t]
            lastgaelam = delta + gamma * tau * nextnonterminal * lastgaelam
            mb_advs[t] = lastgaelam

        return mb_advs

    a2c_common.A2CBase.discount_values = _fixed_discount_values

    class RlGamesVecEnvWrapper(_RlGamesVecEnvWrapper):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._pending_episode_rewards = {}

        def reset(self):
            obs_dict, extras = self.env.reset()

            if "log" in extras:
                episode_data = extras["log"]
                reward_keys = [k for k in episode_data.keys() if k.startswith("reward_") or k == "success_rate"]
                if reward_keys:
                    for key in reward_keys:
                        value = episode_data[key]
                        if hasattr(value, 'item'):
                            value = value.item()
                        self._pending_episode_rewards[key] = value

            return self._process_obs(obs_dict)

        def step(self, actions):
            actions = actions.detach().clone().to(device=self._sim_device)
            actions = torch.clamp(actions, -self._clip_actions, self._clip_actions)
            obs_dict, rew, terminated, truncated, extras = self.env.step(actions)

            if hasattr(rew, 'numel'):
                rew_nan = torch.isnan(rew).sum().item()
                if rew_nan > 0:
                    print(f"[WARN] 奖励包含 {rew_nan} NaN values!")
                    rew = torch.where(torch.isnan(rew), torch.zeros_like(rew), rew)

            if isinstance(terminated, torch.Tensor):
                terminated_bool = terminated.bool()
            else:
                terminated_bool = torch.as_tensor(terminated, dtype=torch.bool,
                                                  device=self._sim_device)

            if isinstance(truncated, torch.Tensor):
                truncated_bool = truncated.bool()
            else:
                truncated_bool = torch.as_tensor(truncated, dtype=torch.bool,
                                                 device=self._sim_device)

            if not self.unwrapped.cfg.is_finite_horizon:
                # time_outs = 超时截断的那一步(与官方 wrapper / stage1 写法一致)。
                # 之前误写成 ~truncated:value_bootstrap 在几乎每一步都加 γ·V,
                # V 指数爆炸 → 参数 NaN → "normal expects all elements of std >= 0"。
                time_outs = truncated_bool.to(dtype=torch.float32)
                if time_outs.dim() > 1:
                    time_outs = time_outs.squeeze(-1)
                extras["time_outs"] = time_outs.to(device=self._rl_device)

            cleaned_obs_dict = {}
            for key, obs in obs_dict.items():
                if hasattr(obs, 'numel'):
                    obs = torch.where(
                        torch.isnan(obs) | torch.isinf(obs),
                        torch.zeros_like(obs),
                        obs
                    )
                cleaned_obs_dict[key] = obs
            obs_and_states = self._process_obs(cleaned_obs_dict)

            rew = rew.to(device=self._rl_device)

            dones = torch.logical_or(terminated_bool, truncated_bool).to(dtype=torch.float32)
            if dones.dim() > 1:
                dones = dones.squeeze(-1)
            dones = dones.to(device=self._rl_device)

            processed_extras = {}
            for k, v in extras.items():
                if hasattr(v, "to"):
                    if isinstance(v, torch.Tensor):
                        if v.dim() > 1:
                            v = v.squeeze(-1)
                        v = v.to(device=self._rl_device, dtype=torch.float32)
                    processed_extras[k] = v
                else:
                    processed_extras[k] = v

            if "log" in processed_extras:
                processed_extras["episode"] = processed_extras.pop("log")

            if "episode" not in processed_extras:
                processed_extras["episode"] = {}

            if hasattr(self, '_pending_episode_rewards') and self._pending_episode_rewards:
                for k, v in self._pending_episode_rewards.items():
                    if k not in processed_extras["episode"]:
                        processed_extras["episode"][k] = v
                self._pending_episode_rewards = {}

            return obs_and_states, rew, dones, processed_extras

    class DebugIsaacAlgoObserver:

        def __init__(self):
            self.algo = None
            self.writer = None
            self.ep_infos = []
            self.direct_info = {}
            self.frame_count = 0
            self._cached_env = None

        def before_init(self, config, *args, **kwargs):
            pass

        def after_init(self, algo):
            self.algo = algo
            self.writer = algo.writer
            self.ep_infos = []
            self.direct_info = {}
            self.frame_count = 0

            self._cached_env = None
            try:
                vec_env = getattr(algo, 'vec_env', None)
                if vec_env is not None:
                    candidate = getattr(vec_env, 'env', None)
                    if candidate is not None:
                        inner = getattr(candidate, 'unwrapped', None)
                        if inner is not None and hasattr(inner, '_get_rewards'):
                            self._cached_env = inner
                        else:
                            inner2 = getattr(candidate, 'env', None)
                            if inner2 is not None:
                                self._cached_env = inner2
                            else:
                                self._cached_env = candidate
                if self._cached_env is not None:
                    print(f"  [Stage2] _cached_env = {type(self._cached_env).__name__}")
                else:
                    print(f"  [Stage2] WARN: 无法找到有效的 _cached_env")
                    self._cached_env = getattr(vec_env, 'env', vec_env)
            except Exception as e:
                import traceback
                print(f"  [Stage2] _cached_env 设置失败: {e}")
                traceback.print_exc()
            # ↑ 不要在这里写 self._cached_env = None！

        def process_infos(self, infos, done_indices):
            if not isinstance(infos, dict):
                classname = self.__class__.__name__
                raise ValueError(f"{classname} expected 'infos' as dict. Received: {type(infos)}")

            if "episode" in infos:
                episode_data = {k: (v.item() if hasattr(v, 'item') else v) for k, v in infos["episode"].items()}
                self.ep_infos.append(episode_data)
            if len(infos) > 0 and isinstance(infos, dict):
                for k, v in infos.items():
                    if (isinstance(v, float) or isinstance(v, int) or
                            (isinstance(v, torch.Tensor) and len(v.shape) == 0)):
                        self.direct_info[k] = v

        def after_clear_stats(self):
            self.ep_infos.clear()

        def after_steps(self, *args, **kwargs):
            pass

        def after_print_stats(self, frame, epoch_num, total_time):

            if hasattr(self.algo, 'game_rewards') and self.algo.game_rewards.current_size > 0:
                mean_reward = self.algo.game_rewards.get_mean()[0].item()
                mean_shaped = self.algo.game_shaped_rewards.get_mean()[0].item()
                mean_length = self.algo.game_lengths.get_mean().item()
                completed_eps = self.algo.game_rewards.current_size
                print(f"[Epoch {epoch_num}]")
                print(f"  mean_episode_reward = {mean_reward:>10.4f}")
                print(f"  mean_episode_length = {mean_length:>10.1f}")
                print(f"  mean                = {mean_reward/mean_length:>10.4f}")
                print(f"{'='*60}")
                if self.writer is not None:
                    self.writer.add_scalar("Episode/true_reward", mean_reward, epoch_num)
                    self.writer.add_scalar("Episode/true_shaped_reward", mean_shaped, epoch_num)
                    self.writer.add_scalar("Episode/true_length", mean_length, epoch_num)
            else:
                if hasattr(self.algo, 'current_rewards') and self.algo.current_rewards.numel() > 0:
                    step_reward_mean = self.algo.current_rewards.mean().item()
                    step_reward_std = self.algo.current_rewards.std().item()
                    step_reward_min = self.algo.current_rewards.min().item()
                    step_reward_max = self.algo.current_rewards.max().item()
                    print(f"\n{'='*60}")
                    print(f"[Epoch {epoch_num}] Horizon 累计 Reward")
                    print(f"  mean = {step_reward_mean:>10.4f}  std = {step_reward_std:>10.4f}")
                    print(f"  min  = {step_reward_min:>10.4f}  max = {step_reward_max:>10.4f}")
                    print(f"{'='*60}\n")

            if self.ep_infos and self.writer is not None:
                for key in list(self.ep_infos[0].keys()):
                    info_tensor = torch.tensor([], device=self.algo.device)
                    for ep_info in self.ep_infos:
                        if key not in ep_info:
                            continue
                        val = ep_info[key]
                        if not isinstance(val, torch.Tensor):
                            val = torch.tensor([val])
                        if val.dim() == 0:
                            val = val.unsqueeze(0)
                        info_tensor = torch.cat((info_tensor, val.to(self.algo.device)))
                    value = torch.mean(info_tensor)
                    tb_key = "Episode/" + key
                    self.writer.add_scalar(tb_key, value, epoch_num)
                    print(f"  [{key}] = {value.item():.6f}")
                self.ep_infos.clear()

            # ── 直接从 env 拿当前 reward 分量均值写 TensorBoard ──
            if self.writer is not None and self._cached_env is not None:
                env_obj = self._cached_env
                log_data = getattr(env_obj, 'extras', {}).get('log', {})
                for key, val in log_data.items():
                    if hasattr(val, 'item'):
                        val = val.item()
                    self.writer.add_scalar(f"Step/{key}", val, frame)
                    print(f"  [Step/{key}] = {val:.6f}")

            if self.writer is not None:
                for k, v in self.direct_info.items():
                    self.writer.add_scalar(f"{k}/frame", v, frame)
                    self.writer.add_scalar(f"{k}/iter", v, epoch_num)
                    self.writer.add_scalar(f"{k}/time", v, total_time)

            try:
                env_obj = self._cached_env
                if env_obj is None:
                    vec_env = getattr(self.algo, 'vec_env', None)
                    if vec_env is not None:
                        candidate = getattr(vec_env, 'env', None)
                        if candidate is not None:
                            inner = getattr(candidate, 'unwrapped', None)
                            if inner is not None and hasattr(inner, '_get_rewards'):
                                env_obj = inner
                            else:
                                env_obj = getattr(candidate, 'env', None)

                if env_obj is None:
                    print(f"\n  [Failure Stats] WARN: _cached_env is None (env 引用未正确缓存)")
                elif not hasattr(env_obj, '_get_rewards'):
                    print(f"\n  [Failure Stats] WARN: {type(env_obj).__name__} 没有 _get_rewards 方法")
                else:
                    failure_stats = env_obj.get_failure_stats()
                    total = failure_stats.get("total", 0)
                    print(f"\n  [Failure Stats] (n={total}) [{type(env_obj).__name__}]")
                    for reason, count in failure_stats.items():
                        if reason in ("total", "failure_count", "physics_nan",
                                      "normal_total", "lenient_success_count"):
                            continue
                        if count is None:
                            continue
                        print(f"    {reason:<28s}: {int(count):>6}")

                    print(f"    {'total':<28s}: {int(total):>6}")
                    print(f"    {'failure_count':<28s}: {int(failure_stats.get('failure_count', 0)):>6}")
                    print(f"    {'physics_nan':<28s}: {int(failure_stats.get('physics_nan', 0)):>6}")
                    print(f"    {'normal_total':<28s}: {int(failure_stats.get('normal_total', 0)):>6}")
                    print(f"    {'lenient_success_count':<28s}: {int(failure_stats.get('lenient_success_count', 0)):>6}")
                    if hasattr(env_obj, 'reset_failure_stats'):
                        env_obj.reset_failure_stats()
            except Exception as e:
                print(f"\n  [Failure Stats] ERROR: {e}")
                import traceback; traceback.print_exc()

    algo_observer = DebugIsaacAlgoObserver()

    with open(args.rl_config, "r") as f:
        rl_config = yaml.safe_load(f)

    clip_obs = rl_config["params"]["env"].get("clip_observations", 5.0)
    clip_actions = rl_config["params"]["env"].get("clip_actions", 1.0)

    rl_config["params"]["config"]["num_actors"] = args.num_envs
    rl_config["params"]["config"]["device"] = args.device
    rl_config["params"]["config"]["device_name"] = args.device
    # batch_size = num_actors * horizon_length; minibatch_size must divide it evenly
    horizon = rl_config["params"]["config"].get("horizon_length", 96)
    batch_size = args.num_envs * horizon
    original_minibatch_size = rl_config["params"]["config"].get("minibatch_size", 4096)

    if batch_size % original_minibatch_size != 0:
        best_minibatch_size = None
        min_diff = float('inf')

        search_range = min(original_minibatch_size, batch_size // 2)
        for candidate in range(original_minibatch_size, max(64, original_minibatch_size - search_range), -1):
            if batch_size % candidate == 0:
                diff = abs(candidate - original_minibatch_size)
                if diff < min_diff:
                    min_diff = diff
                    best_minibatch_size = candidate

        if best_minibatch_size is None:
            for candidate in range(original_minibatch_size + 1,
                                   min(batch_size, original_minibatch_size + search_range) + 1):
                if batch_size % candidate == 0:
                    diff = abs(candidate - original_minibatch_size)
                    if diff < min_diff:
                        min_diff = diff
                        best_minibatch_size = candidate

        if best_minibatch_size is None:
            for divisor in [batch_size // 2, batch_size // 3, batch_size // 4,
                            batch_size // 6, batch_size // 8]:
                if divisor >= 64 and batch_size % divisor == 0:
                    best_minibatch_size = divisor
                    break
            if best_minibatch_size is None:
                best_minibatch_size = max(64, batch_size // 8)

        rl_config["params"]["config"]["minibatch_size"] = best_minibatch_size
        print(f"[Stage2] minibatch_size {original_minibatch_size} does not divide batch_size {batch_size}, "
              f"adjusting to {best_minibatch_size}")

    run_name = args.run_name if args.run_name else f"stage2_{datetime.now().strftime('%y-%m-%d-%H-%M')}"
    train_dir = os.path.join(project_root, "runs", run_name)
    rl_config["params"]["config"]["train_dir"] = train_dir
    rl_config["params"]["config"]["full_experiment_name"] = run_name
    rl_config["params"]["config"]["name"] = run_name

    env = RlGamesVecEnvWrapper(env, args.device, clip_obs, clip_actions)

    vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    runner = Runner()

    load_path = rl_config["params"].get("load_path", None)
    load_checkpoint = rl_config["params"].get("load_checkpoint", False)

    if args.checkpoint:
        load_path = args.checkpoint
        load_checkpoint = True

    # ── Stage1 checkpoint 观测维度扩展 ──────────────────────────────
    # stage2 观测 = stage1 观测 + plate 位姿 7 维(pos3+quat4)。stage1 权重的输入层按新增列
    # 零填充(零权重 → 初始行为与 stage1 完全一致,plate 维度从 0 学起);
    # 优化器状态(exp_avg 等)中同形状张量一并填充,避免首次 update 形状不匹配。
    cleaned_checkpoint_path = None
    if load_checkpoint and load_path and os.path.exists(load_path):
        try:
            print(f"\n正在预处理检查点: {load_path}")
            new_dim = int(cfg.observation_space)   # Stage2Env.__init__ 已把 cfg 改为 +6
            checkpoint = torch.load(load_path, map_location='cpu', weights_only=False)

            old_dim = None
            for k, v in checkpoint.get('model', {}).items():
                if k.endswith('actor_mlp.0.weight') and torch.is_tensor(v) and v.dim() == 2:
                    old_dim = int(v.shape[1])
                    break

            if old_dim is not None and old_dim < new_dim:
                pad = new_dim - old_dim

                def _pad_tree(node):
                    """递归零填充 checkpoint 树中所有 [*, old_dim] 的 2 维张量。"""
                    n = 0
                    if isinstance(node, dict):
                        for kk in list(node.keys()):
                            vv = node[kk]
                            if torch.is_tensor(vv) and vv.dim() == 2 and vv.shape[1] == old_dim:
                                node[kk] = torch.cat(
                                    [vv, torch.zeros(vv.shape[0], pad, dtype=vv.dtype)], dim=1)
                                n += 1
                            else:
                                n += _pad_tree(vv)
                    elif isinstance(node, list):
                        for vv in node:
                            n += _pad_tree(vv)
                    return n

                n_padded = _pad_tree(checkpoint)
                cleaned_checkpoint_path = load_path + f".obs{new_dim}.tmp.pth"
                torch.save(checkpoint, cleaned_checkpoint_path)
                load_path = cleaned_checkpoint_path
                print(f"  [CKPT] 观测 {old_dim} -> {new_dim}: 零填充 {n_padded} 个张量"
                      f"(输入层权重 + 优化器动量),临时文件: {cleaned_checkpoint_path}")
            elif old_dim == new_dim:
                print(f"  [CKPT] 观测维度一致({new_dim}),直接加载")
            else:
                print(f"  [CKPT] 未识别输入层维度(old={old_dim}),按原样加载")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  [CKPT] 预处理失败,按原样加载: {e}")

    train_params = {
        "train": True,
        "play": False,
        "checkpoint": load_path,
        "sigma": None,
    }

    if load_checkpoint and load_path:
        print(f"\n从检查点加载: {load_path}")
    elif load_checkpoint:
        print("\nload_checkpoint=True 但 load_path 为空，将从头训练")

    rl_config["params"]["config"]["print_stats_frequency"] = 1
    print(f"\n每个 Epoch 结束时打印训练统计")

    print(f"\n[Stage2] 开始训练")
    print(f"  - 样本数量: {len(success_dataset)}")
    print(f"  - 环境数量: {args.num_envs}")
    print(f"  - 目标位置: {args.target_pos}")
    print(f"  - 目标旋转: {args.target_quat}")
    print(f"  - Run名称: {run_name}")
    if args.checkpoint:
        print(f"  - 加载检查点: {args.checkpoint}")

    try:
        # ⚠️ 必须在 runner.load() 之前设置 algo_observer，否则会被 DefaultAlgoObserver 覆盖
        runner.algo_observer = algo_observer
        runner.load(rl_config)

        # Patch restore() and set_weights() to avoid loading old checkpoint structure
        try:
            import rl_games.common.a2c_common as a2c_common_module
            _orig_restore = a2c_common_module.A2CBase.restore
            def _patched_restore(self, checkpoint, *args, **kwargs):
                if checkpoint and str(checkpoint).strip() not in ('', 'None', 'null'):
                    print(f"  [CKPT] restore() called with: {checkpoint}")
                    _orig_restore(self, checkpoint, *args, **kwargs)
                else:
                    print("  [CKPT] restore() skipped (no checkpoint)")
            a2c_common_module.A2CBase.restore = _patched_restore
            print("  [CKPT] patched A2CBase.restore() to skip empty checkpoints")

            def _make_patched_set_weights(orig_method):
                def _patched(self, weights, *args, **kwargs):
                    if 'model' not in weights:
                        print("  [CKPT] set_weights: no 'model' key, skipping model load")
                        if weights:
                            self.set_stats_weights(weights)
                        return
                    orig_method(self, weights, *args, **kwargs)
                return _patched

            for cls in (a2c_common_module.A2CBase,
                        getattr(a2c_common_module, 'ContinuousA2CBase', None)):
                if cls is not None and hasattr(cls, 'set_weights'):
                    cls.set_weights = _make_patched_set_weights(cls.set_weights)
                    print(f"  [CKPT] patched {cls.__name__}.set_weights() to skip missing model key")
        except Exception as e:
            print(f"  [CKPT] patch failed: {e}")

        runner.run(train_params)
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

    if cleaned_checkpoint_path and os.path.exists(cleaned_checkpoint_path):
        try:
            os.remove(cleaned_checkpoint_path)
            print(f"\n临时检查点已清理: {cleaned_checkpoint_path}")
        except:
            pass

    env.close()
    simulation_app.close()
    print("完成！")


if __name__ == "__main__":
    main()
