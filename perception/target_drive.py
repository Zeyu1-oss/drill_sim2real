"""统一的 target 直接驱动 —— collect / deploy / play 共用,保证三处动力学逐字节一致。

env._apply_action 天生只懂 raw([-1,1] 绝对映射 + EMA + 限速 + 饱和)。本模块把它
monkeypatch 成"直接下发关节目标角 q*",从而:
  - deploy:DP3 直接输出 q* → 直接下发,不再走 target→raw 逆映射(逆映射会放大手臂动作、易乱动)。
  - collect/play:teacher/MLP 出 raw → raw_to_target 换成 q* → 直接下发。

三处用同一份代码,采集↔部署动力学完全一致。

API:
  p = install_direct_drive(env_unwrapped)            # monkeypatch,返回 DriveParams
  env_unwrapped._direct_target = q_star              # deploy:DP3 的 q*
  env_unwrapped._direct_target = raw_to_target(a, cur0, p)   # collect/play:raw→q*
"""
import types

import torch


class DriveParams:
    """从 env 一次性读出驱动所需常量(与 env._apply_action 同一真源)。"""

    def __init__(self, env_unwrapped):
        self.s = float(env_unwrapped.action_smoothing)
        self.n_sub = int(getattr(env_unwrapped.cfg, "decimation", 2))
        self.max_joint_delta = float(env_unwrapped.max_joint_delta)
        self.max_finger_delta = float(env_unwrapped.max_finger_delta)
        self.n_arm = int(env_unwrapped.num_arm_joints)
        self.jl = env_unwrapped.joint_lower_limits
        self.ju = env_unwrapped.joint_upper_limits
        self.rng_arm = (self.ju - self.jl)[:self.n_arm]

    def __repr__(self):
        return (f"DriveParams(s={self.s}, n_sub={self.n_sub}, "
                f"max_joint_delta={self.max_joint_delta}, "
                f"max_finger_delta={self.max_finger_delta}, n_arm={self.n_arm})")


def raw_to_target(a, cur0, p):
    """复刻 env._apply_action 前向(EMA+限速+饱和,n_sub 子步):raw 动作 a → 关节目标角 q*。

    a:    (B, 13) raw 动作 [-1,1];前 n_arm 维为手臂绝对位置,余为手指增量。
    cur0: (B, 13) step 前的 cur_targets。
    返回: (B, 13) 这个 a 本来会让 cur_targets 到达的目标角 q*。
    """
    na, s, mjd, mfd = p.n_arm, p.s, p.max_joint_delta, p.max_finger_delta
    cur = cur0.clone()
    arm_raw = p.jl[:na] + 0.5 * (a[:, :na] + 1.0) * p.rng_arm
    for _ in range(p.n_sub):
        smoothed = s * arm_raw + (1.0 - s) * cur[:, :na]
        if mjd > 0:
            d = torch.clamp(smoothed - cur[:, :na], -mjd, mjd)
            cur[:, :na] = cur[:, :na] + d
        else:
            cur[:, :na] = smoothed
        cur[:, na:] = cur[:, na:] + a[:, na:] * mfd
        cur = torch.max(torch.min(cur, p.ju.unsqueeze(0)), p.jl.unsqueeze(0))
    return cur


def smooth_labels_forward(actions, k: int):
    """动作标签前向滑动平均: new_a[t] = mean(a[t : t+k])(窗口在 episode 尾部自动收缩)。

    为什么需要:teacher(RL)的 cur_targets 是贴限速上限的 PWM 抖动
    (每步 ±max_joint_delta、方向近随机翻转,实测 95% 步贴 0.02 上限、翻转率 ~43%)。
    对 BC 这是不可预测的标签噪声:误差地板 ≈ 抖动幅度,学习信号被稀释 ~7 倍。
    前向均值把零均值抖动平均掉(k=8 -> 残余 ~0.02/√8 ≈ 0.007 rad),同时:
      - 握力/抗重力的直流偏置(target 比实测位置深的系统差)完整保留;
      - 平滑后逐步增量 |Δ| ≤ 原限速包络(均值的差 ≤ 差的均值),部署直接可执行;
      - 前向窗口自带 ~k/2 步超前,顺便抵消"标签晚一步"的时序约定。

    单一真源:collect_dp3_data.py(采集时平滑)、deploy 的 DeployCollector、
    tools/relabel_smooth_actions.py(离线重标注)都用本函数,保证三处逐字节一致。

    actions: (T, D) numpy 数组;k<=1 时原样返回。
    """
    import numpy as np
    actions = np.asarray(actions)
    T = len(actions)
    if k <= 1 or T == 0:
        return actions.astype(np.float32, copy=False)
    cs = np.concatenate([np.zeros((1, actions.shape[1]), dtype=np.float64),
                         np.cumsum(actions.astype(np.float64), axis=0)], axis=0)
    hi = np.minimum(np.arange(T) + k, T)
    lo = np.arange(T)
    out = (cs[hi] - cs[lo]) / (hi - lo)[:, None]
    return out.astype(np.float32)


def install_direct_drive(env_unwrapped):
    """monkeypatch env._apply_action,使每子步直接下发 env._direct_target(跳过 raw 解释/逆映射)。

    调用后,每个控制步在 step 前设置 env_unwrapped._direct_target 即可:
        deploy:       env._direct_target = q_star(DP3 直接输出)
        collect/play:  env._direct_target = raw_to_target(a, cur0, p)
    返回 DriveParams(collect/play 算 q* 用;deploy 不需要)。
    """
    p = DriveParams(env_unwrapped)
    jl, ju = p.jl, p.ju

    def _apply_action_direct(self):
        tgt = torch.max(torch.min(self._direct_target, ju.unsqueeze(0)), jl.unsqueeze(0))
        self.cur_targets = tgt
        self.franka.set_joint_position_target(tgt, joint_ids=self.controlled_joint_indices)

    env_unwrapped._apply_action = types.MethodType(_apply_action_direct, env_unwrapped)
    env_unwrapped._direct_target = env_unwrapped.cur_targets.clone()   # 占位,首 step 前先给合法值
    return p
