
import torch

POS_DIM = 13
FORCE_DIM = 26   # [joint pos(13) | joint torque(13)]; joint velocity removed (was 39)


def agent_pos_dim(with_force):
    """agent_pos dim for this mode (collect builds zarr / deploy aligns to it)."""
    return FORCE_DIM if with_force else POS_DIM


def build_agent_pos(env_unwrapped, with_force):
    """Return (B, 13) or (B, 26) agent_pos. Call after env.step() (matches current timing)."""
    ci = env_unwrapped.controlled_joint_indices.tolist()
    pos = env_unwrapped.franka.data.joint_pos[:, ci]
    if not with_force:
        return pos
    # actual applied joint torque (post-clipping, what the sim really exerted), one per
    # controlled joint -- raw Nm, left unnormalized like `pos`; the training-time
    # LinearNormalizer (limits mode, fit from the collected data) rescales both to [-1,1].
    torque = env_unwrapped.franka.data.applied_torque[:, ci]     # (B, 13)
    return torch.cat([pos, torque], dim=1)
