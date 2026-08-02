"""TD3 for residual RL on top of a frozen DP3 policy. Mirrors the tuned recipe in Amazon's ResFiT
(residual-offpolicy-rl, arXiv 2509.19301): zero-initialized actor last layer (residual starts at
exactly 0 -> training begins at pure-DP3 behavior), tanh-squashed output scaled by a small
`action_scale`, twin critics with delayed policy updates and target-policy smoothing (standard
TD3), and a critic-only warmup phase (handled by the caller passing critic_only=True to update()).
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim, out_dim, hidden):
    layers = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class ResidualActor(nn.Module):
    """obs -> residual action, tanh-squashed to [-action_scale, action_scale].
    Last layer zero-initialized so the residual is exactly 0 at the start of training."""

    def __init__(self, obs_dim, action_dim=13, hidden=(256, 256), action_scale=0.1):
        super().__init__()
        self.net = _mlp(obs_dim, action_dim, hidden)
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.action_scale = action_scale

    def forward(self, obs):
        return torch.tanh(self.net(obs)) * self.action_scale


class TwinCritic(nn.Module):
    def __init__(self, obs_dim, action_dim=13, hidden=(256, 256)):
        super().__init__()
        self.q1 = _mlp(obs_dim + action_dim, 1, hidden)
        self.q2 = _mlp(obs_dim + action_dim, 1, hidden)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_value(self, obs, action):
        return self.q1(torch.cat([obs, action], dim=-1))


class ReplayBuffer:
    """CPU-resident circular buffer (avoids competing with Isaac Sim for GPU memory); sampled
    minibatches are moved to `device` on demand."""

    def __init__(self, capacity, obs_dim, action_dim, device):
        self.capacity = capacity
        self.device = device
        self.obs = torch.zeros(capacity, obs_dim)
        self.action = torch.zeros(capacity, action_dim)
        self.reward = torch.zeros(capacity)
        self.next_obs = torch.zeros(capacity, obs_dim)
        self.done = torch.zeros(capacity)
        self.ptr = 0
        self.size = 0

    def add_batch(self, obs, action, reward, next_obs, done):
        n = obs.shape[0]
        idx = (torch.arange(n) + self.ptr) % self.capacity
        self.obs[idx] = obs.detach().cpu()
        self.action[idx] = action.detach().cpu()
        self.reward[idx] = reward.detach().cpu().float()
        self.next_obs[idx] = next_obs.detach().cpu()
        self.done[idx] = done.detach().cpu().float()
        self.ptr = int((self.ptr + n) % self.capacity)
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,))
        return (
            self.obs[idx].to(self.device),
            self.action[idx].to(self.device),
            self.reward[idx].to(self.device),
            self.next_obs[idx].to(self.device),
            self.done[idx].to(self.device),
        )


class TD3ResidualAgent:
    def __init__(
        self, obs_dim, action_dim=13, device="cuda",
        action_scale=0.1, hidden=(256, 256),
        actor_lr=1e-6, critic_lr=1e-4,
        gamma=0.995, tau=0.005,
        policy_noise=0.05, noise_clip=0.1, policy_delay=2,
        explore_noise=0.05,
    ):
        self.device = device
        self.action_scale = action_scale
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.explore_noise = explore_noise

        self.actor = ResidualActor(obs_dim, action_dim, hidden, action_scale).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = TwinCritic(obs_dim, action_dim, hidden).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.actor_target.parameters():
            p.requires_grad_(False)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.total_updates = 0

    @torch.no_grad()
    def select_action(self, obs, explore=True):
        a = self.actor(obs)
        if explore:
            a = (a + torch.randn_like(a) * self.explore_noise).clamp(-self.action_scale, self.action_scale)
        return a

    def update(self, buffer: ReplayBuffer, batch_size=256, critic_only=False):
        obs, action, reward, next_obs, done = buffer.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_obs) + noise).clamp(-self.action_scale, self.action_scale)
            tq1, tq2 = self.critic_target(next_obs, next_action)
            target_q = reward.unsqueeze(-1) + self.gamma * (1.0 - done.unsqueeze(-1)) * torch.min(tq1, tq2)

        q1, q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        self.total_updates += 1
        actor_loss_val = None
        if (not critic_only) and (self.total_updates % self.policy_delay == 0):
            actor_loss = -self.critic.q1_value(obs, self.actor(obs)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            actor_loss_val = actor_loss.item()

            with torch.no_grad():
                for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                    tp.mul_(1.0 - self.tau).add_(self.tau * p)
                for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                    tp.mul_(1.0 - self.tau).add_(self.tau * p)

        return critic_loss.item(), actor_loss_val

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
        }, path)

    def load(self, path, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
