"""统一构建 DP3 student 的 agent_pos —— collect / deploy 共用,保证两端逐维一致。

两种模式:
  with_force=False -> 13 维:仅 13 个受控关节位置(旧行为)。
  with_force=True  -> 26 维:[关节位置(13) | 接触力(13,log1p 归一)]。(关节速度已移除)

接触力复用 env._get_contact_forces_obs()(13 个手部 link 传感器)+ 老师同款 log1p 压缩,
使 student 的触觉 obs 与 teacher 的 contact_forces 逐位对齐(见 grasp_drill_env._get_observations)。

注:位置/接触力两种量纲,DP3 的 LinearNormalizer 是逐维 min/max,会各自归一,混在一起没问题。
接触力先 log1p(f)/log1p(50) 压缩,避免接触尖峰主导 normalizer 的 min/max。
"""
import torch

POS_DIM = 13
FORCE_DIM = 26   # [关节位置(13) | 接触力(13)];关节速度已移除(原 39)
_CONTACT_LOG_DENOM = 50.0   # 同 teacher:log1p(f)/log1p(50)


def agent_pos_dim(with_force):
    """该模式下 agent_pos 的维度(collect 建 zarr / deploy 对齐用)。"""
    return FORCE_DIM if with_force else POS_DIM


def build_agent_pos(env_unwrapped, with_force):
    """返回 (B, 13) 或 (B, 26) 的 agent_pos。应在 env.step() 之后调用(与现有时序一致)。"""
    ci = env_unwrapped.controlled_joint_indices.tolist()
    pos = env_unwrapped.franka.data.joint_pos[:, ci]
    if not with_force:
        return pos
    cf = env_unwrapped._get_contact_forces_obs()                 # (B, 13) 力大小
    denom = torch.log1p(torch.tensor(_CONTACT_LOG_DENOM, device=cf.device))
    cf = torch.log1p(cf.clamp(min=0.0)) / denom                 # (B, 13) ~[0,1]
    return torch.cat([pos, cf], dim=1)
