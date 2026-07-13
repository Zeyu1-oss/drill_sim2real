#!/usr/bin/env python3
import argparse
import os
import sys
from datetime import datetime

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (scripts/ -> ../..)
sys.path.insert(0, project_root)


def get_run_name():
    now = datetime.now()
    return f"inspire_hand_grasp_drill_{now.strftime('%y-%m-%d-%H-%M')}"


def main():
    parser = argparse.ArgumentParser(description="使用RL Games训练抓取电钻任务")
    parser.add_argument("--headless", action="store_true", help="无头模式（关闭GUI）")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--num_envs", type=int, default=256, help="并行环境数量")
    parser.add_argument(
        "--rl_config",
        type=str,
        default="config/agents/rl_games_ppo_cfg.yaml",
        help="RL训练配置文件路径（默认: config/agents/rl_games_ppo_cfg.yaml）"
    )
    parser.add_argument(
        "--config_preset",
        type=str,
        choices=["default", "small", "large"],
        default="default",
        help="使用预设配置: default, small, large"
    )
    parser.add_argument(
        "--drill_config",
        type=str,
        default=None,
        help="电钻配置文件路径（YAML格式）"
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="加载检查点路径")
    parser.add_argument("--play", action="store_true", help="仅播放模式（不训练）")
    parser.add_argument("--run_name", type=str, default=None, help="自定义run名称（用于保存检查点）")
    parser.add_argument("--debug", action="store_true", help="调试模式：只输出成功率，关闭其他日志")
    parser.add_argument("--debug_interval", type=int, default=10, help="调试输出间隔（多少个episode）")
    parser.add_argument("--joint_debug", action="store_true", help="关节调试模式：输出所有关节角度和动作详情")
    parser.add_argument("--viz", action="store_true", help="点云可视化模式：在渲染器中显示采样的点云（需要 GUI）")
    args = parser.parse_args()

    # 确定RL配置文件路径
    if args.rl_config:
        rl_config_path = args.rl_config
    else:
        config_name_map = {
            "default": "rl_games_ppo_cfg.yaml",
        }
        config_filename = config_name_map[args.config_preset]
        rl_config_path = os.path.join(project_root, "config/agents", config_filename)

    if not os.path.exists(rl_config_path):
        print(f"错误: 配置文件不存在: {rl_config_path}")
        sys.exit(1)

    # 检查Isaac Lab导入
    try:
        from isaaclab.app import AppLauncher
    except ImportError:
        try:
            from omni.isaac.lab.app import AppLauncher
        except ImportError:
            sys.exit(1)

    # 初始化应用
    app_launcher = AppLauncher(headless=args.headless, enable_cameras=False)
    simulation_app = app_launcher.app

    import omni.usd
    omni.usd.get_context().new_stage()

    # 导入环境
    from tasks.grasp_drill_env import GraspDrillEnv, create_grasp_drill_env_cfg

    # 创建环境配置
    cfg = create_grasp_drill_env_cfg(
        num_envs=args.num_envs,
        device=args.device,
        headless=args.headless,
        drill_config_path=args.drill_config,
        debug=args.debug,
        enable_cameras=False,
    )

    # 创建环境
    if args.debug:
        print("[DEBUG] 调试模式：减少日志输出，仅显示成功率")
    env = GraspDrillEnv(cfg=cfg, debug=args.debug)

    # 删除 /World/Template（在 env 创建之后删除）
    from isaaclab.sim.utils import delete_prim
    try:
        delete_prim("/World/Template")
        print("[INFO] 已删除 /World/Template")
    except Exception as e:
        print(f"[WARN] 删除 /World/Template 失败: {e}")

    if args.viz:
        env.enable_pc_viz = True

    try:
        from rl_games.common import env_configurations, vecenv
        from rl_games.common.algo_observer import IsaacAlgoObserver
        from rl_games.torch_runner import Runner
        from rl_games.common import a2c_common
        from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper as _RlGamesVecEnvWrapper
        import torch

        # ======================================================================
        # Monkey patch: 修复 GAE 计算中的 broadcast 问题
        # 原始代码用 unsqueeze(1) 会导致 [N,1] broadcast 成 [T,N,N]
        # 修复：用 view_as 保证 shape 一致，不做多余 unsqueeze
        # ======================================================================
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

        if hasattr(a2c_common.A2CBase, 'discount_values_masks'):
            def _fixed_discount_values_masks(self, fdones, last_extrinsic_values,
                                             mb_fdones, mb_extrinsic_values, mb_rewards, mb_masks):
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

                    masks_t = mb_masks[t].float()
                    if masks_t.shape != mb_rewards[t].shape:
                        masks_t = masks_t.view_as(mb_rewards[t])

                    delta = mb_rewards[t] + gamma * nextvalues * nextnonterminal - mb_extrinsic_values[t]
                    lastgaelam = (delta + gamma * tau * nextnonterminal * lastgaelam) * masks_t
                    mb_advs[t] = lastgaelam

                return mb_advs

            a2c_common.A2CBase.discount_values_masks = _fixed_discount_values_masks

        # ======================================================================

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
                        print(f"[DEBUG] Episode奖励项: {list(self._pending_episode_rewards.keys())}")

                return self._process_obs(obs_dict)

            def step(self, actions):
                actions = actions.detach().clone().to(device=self._sim_device)
                actions = torch.clamp(actions, -self._clip_actions, self._clip_actions)
                obs_dict, rew, terminated, truncated, extras = self.env.step(actions)

                for key, obs in obs_dict.items():
                    if hasattr(obs, 'numel'):
                        pass

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

                # 真正的 epoch 级别 reward 打印由 DebugIsaacAlgoObserver.after_print_stats 处理
                # wrapper 只做数据中转，不在这里打印 reward
                return obs_and_states, rew, dones, processed_extras

    except ImportError as e:
        sys.exit(1)
    except ModuleNotFoundError as e:
        sys.exit(1)

    # 加载 RL Games 配置
    import yaml
    with open(rl_config_path, "r") as f:
        rl_config = yaml.safe_load(f)

    clip_obs = rl_config["params"]["env"].get("clip_observations", 5.0)
    clip_actions = rl_config["params"]["env"].get("clip_actions", 1.0)

    rl_config["params"]["config"]["num_actors"] = args.num_envs
    rl_config["params"]["config"]["device"] = args.device
    rl_config["params"]["config"]["device_name"] = args.device

    run_name = args.run_name if args.run_name else get_run_name()
    train_dir = os.path.join(project_root, "runs", run_name)
    rl_config["params"]["config"]["train_dir"] = train_dir
    rl_config["params"]["config"]["full_experiment_name"] = run_name
    rl_config["params"]["config"]["name"] = run_name

    horizon_length = rl_config["params"]["config"].get("horizon_length", 96)
    batch_size = args.num_envs * horizon_length
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
    else:
        best_minibatch_size = original_minibatch_size

    env = RlGamesVecEnvWrapper(env, args.device, clip_obs, clip_actions)

    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    runner = Runner()

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

            # normalize_input=False + model 已被 patch 跳过，跳过 NORM-RESET
            pass

            # 调试：打印 algo 所有属性，找到 env 路径
            algo_attrs = [a for a in dir(algo) if not a.startswith('__')]
            print(f"\n  [DEBUG after_init] algo type: {type(algo).__name__}, attrs: {algo_attrs}")

            # ============================================================
            # 修复：找到实际的 GraspDrillEnv 引用
            # 层级链：algo.vec_env (RlGamesGpuEnv) → .env (RlGamesVecEnvWrapper) → .env (GraspDrillEnv)
            # ============================================================
            self._cached_env = None
            try:
                vec_env = getattr(algo, 'vec_env', None)
                if vec_env is not None:
                    # RlGamesGpuEnv → RlGamesVecEnvWrapper
                    candidate = getattr(vec_env, 'env', None)
                    if candidate is not None:
                        # RlGamesVecEnvWrapper → GraspDrillEnv (via .env or .unwrapped)
                        inner = getattr(candidate, 'unwrapped', None)
                        if inner is not None and hasattr(inner, 'get_failure_stats'):
                            self._cached_env = inner
                        else:
                            inner2 = getattr(candidate, 'env', None)
                            if inner2 is not None and hasattr(inner2, 'get_failure_stats'):
                                self._cached_env = inner2
                            elif inner2 is not None:
                                # 再深入一层
                                inner3 = getattr(inner2, 'unwrapped', None)
                                if inner3 is not None and hasattr(inner3, 'get_failure_stats'):
                                    self._cached_env = inner3
                                else:
                                    self._cached_env = inner2
                            else:
                                self._cached_env = candidate

                if self._cached_env is not None:
                    print(f"  [DEBUG after_init] _cached_env = {type(self._cached_env).__name__} ✓")
                else:
                    print(f"  [DEBUG after_init] WARN: 无法找到有效的 _cached_env")
                    # 最后的兜底：直接用 vec_env.env
                    self._cached_env = getattr(vec_env, 'env', vec_env)
            except Exception as e:
                import traceback
                print(f"  [DEBUG after_init] 设置 _cached_env 失败: {e}")
                traceback.print_exc()
            self._cached_env = None

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
            # rl_games 每个 step 后调用，这里不需要处理
            pass

        def after_print_stats(self, frame, epoch_num, total_time):

            if hasattr(self.algo, 'game_rewards') and self.algo.game_rewards.current_size > 0:
                # 有真实完成的 episode，打印累计 reward
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
                # 没有完整 episode，打印 horizon 级别的累计 reward
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

            # 打印 episode extras（如 reward_approach, success_rate 等）
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

            # 打印其他 direct info
            if self.writer is not None:
                for k, v in self.direct_info.items():
                    self.writer.add_scalar(f"{k}/frame", v, frame)
                    self.writer.add_scalar(f"{k}/iter", v, epoch_num)
                    self.writer.add_scalar(f"{k}/time", v, total_time)

            # === 打印失败原因统计 ===
            try:
                # 使用在 after_init 中缓存的 env 引用
                env_obj = self._cached_env
                if env_obj is None:
                    # 兜底：重新尝试遍历
                    vec_env = getattr(self.algo, 'vec_env', None)
                    if vec_env is not None:
                        candidate = getattr(vec_env, 'env', None)
                        if candidate is not None:
                            inner = getattr(candidate, 'unwrapped', None)
                            if inner is not None and hasattr(inner, 'get_failure_stats'):
                                env_obj = inner
                            else:
                                env_obj = getattr(candidate, 'env', None)

                if env_obj is None:
                    print(f"\n  [Failure Stats] WARN: _cached_env is None (env 引用未正确缓存)")
                elif not hasattr(env_obj, 'get_failure_stats'):
                    print(f"\n  [Failure Stats] WARN: {type(env_obj).__name__} 没有 get_failure_stats 方法")
                else:
                        failure_stats = env_obj.get_failure_stats()
                        total = failure_stats.get("total", 0)
                        print(f"\n  [Failure Stats] (n={total}) [{type(env_obj).__name__}]")
                        for reason, count in failure_stats.items():
                            # 汇总行单独格式化打印，不走循环
                            if reason in ("total", "failure_count", "physics_nan",
                                          "normal_total", "lenient_success_count"):
                                continue
                            # 自适应：值为 None 表示该 failure 原因从未触发（被注释掉），不打印
                            if count is None:
                                continue
                            print(f"    {reason:<28s}: {int(count):>6}")

                        # 汇总行
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

    try:
        algo_observer = DebugIsaacAlgoObserver()
        # Runner 没有 set_algo_observer 方法，直接赋值属性
        runner.algo_observer = algo_observer
    except Exception as e:
        pass
    load_path = rl_config["params"].get("load_path", None)
    load_checkpoint = rl_config["params"].get("load_checkpoint", False)

    if args.checkpoint:
        load_path = args.checkpoint
        load_checkpoint = True

    cleaned_checkpoint_path = None
    if load_checkpoint and load_path and os.path.exists(load_path):
        import torch
        try:
            print(f"\n正在预处理检查点: {load_path}")
            checkpoint = torch.load(load_path, map_location='cpu', weights_only=False)



        except Exception as e:
            pass
    train_params = {
        "train": not args.play,
        "play": args.play,
        "checkpoint": load_path,
        "sigma": None,
    }

    if load_checkpoint and load_path:
        print(f"\n从检查点加载: {load_path}")
    elif load_checkpoint:
        print("\nload_checkpoint=True 但 load_path 为空，将从头训练")

    print_frequency = 1
    rl_config["params"]["config"]["print_stats_frequency"] = print_frequency
    print(f"\n每个 Epoch 结束时打印训练统计")

    if args.play:
        print("\n开始播放模式...")
    else:
        if args.debug:
            print("\n开始训练... (调试模式)")
        else:
            print("开始训练...")
        print("=" * 60)

    try:
        runner.load(rl_config)

        # --- Patch restore() and set_weights() to avoid loading old checkpoint ---
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
        print("\n\n用户中断，停止训练...")
    except Exception as e:
        print(f"\n错误: {e}")
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
    print("\n完成！")


if __name__ == "__main__":
    main()
