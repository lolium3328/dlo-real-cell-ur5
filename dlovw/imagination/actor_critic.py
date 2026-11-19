"""Actor and critic trained in the world model's imagination (docs/notes/imagination_ac_design_20261022.md, section 6).

The actor maximises lambda-returns of imagined rollouts; gradients pass through the reward, the rope
decoder and the prior dynamics (world-model and critic parameters frozen, activations not). The critic
regresses lambda-returns computed with a slowly updated target copy.
"""
from __future__ import annotations

import contextlib
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .episodes import ACT, GOAL, K, N_PTS, POS, PROTECTED_M, ROPE
from .feasible import feasible_torch
from .world_model import mlp

GAMMA, LAMBDA = 0.98, 0.95
CONT_SCALE_M, CRASH_PENALTY = 0.003, 10.0   # soft "still alive" sigmoid(d / 3 mm); penalty for ending in a violation


ORACLE_OFFS = (5, 10, 15)              # decisions ahead (0.4, 0.8, 1.2 s) given to the privileged-forecast actor


def oracle_input(future, i):
    """True obstacle positions ORACLE_OFFS decisions after imagined step i; future[:, j] is decision j + 1."""
    return torch.cat([future[:, i + k - 1] for k in ORACLE_OFFS], -1)


class Actor(nn.Module):
    def __init__(self, feat_dim, hid=256, extra=0):
        super().__init__()
        self.extra = extra                 # > 0: privileged forecast input (upper-bound experiment only)
        self.net = mlp(feat_dim + GOAL + extra, hid, ACT, 3)

    def forward(self, feat, goal, extra=None):
        """Raw command in cap units; feasible.feasible_torch / feasible_np turn it into what is executed."""
        x = [feat, goal] if extra is None else [feat, goal, extra]
        return torch.tanh(self.net(torch.cat(x, -1)))


class Critic(nn.Module):
    def __init__(self, feat_dim, hid=256):
        super().__init__()
        self.net = mlp(feat_dim + GOAL, hid, 1, 2)

    def forward(self, feat, goal):
        return self.net(torch.cat([feat, goal], -1)).squeeze(-1)


@contextlib.contextmanager
def frozen(*modules):
    params = [p for m in modules for p in m.parameters()]
    flags = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, f in zip(params, flags):
            p.requires_grad_(f)


PROGRESS = "relu"                      # r1-r8; "exp" from r9 (set by the trainer, stored in the checkpoint)


def progress(rope, goal):
    """Non-negative progress terms (up to 2.5 at the target) and the shape error (m).
    "relu": relu(1 - e/0.15) + 0.5 relu(1 - e_end/0.15) + exp(-e/0.02), flat (0) beyond 15 cm: a rope that
    drifted further away was never pulled back (r8: 9/134 and 34/300 timeouts ended 25-30 cm away).
    "exp": exp(-e/0.15) + 0.5 exp(-e_end/0.15) + exp(-e/0.02), positive and sloped everywhere."""
    pts = rope[:, :3 * N_PTS].reshape(-1, N_PTS, 3) / POS
    tgt = goal[:, :3 * N_PTS].reshape(-1, N_PTS, 3) / POS
    e_shape = ((pts - tgt) ** 2).sum(-1).mean(-1).clamp_min(1e-10).sqrt()
    grip = rope[:, 3 * N_PTS:ROPE].reshape(-1, 2, 3) / POS
    e_end = ((grip - goal[:, 3 * N_PTS:].reshape(-1, 2, 3) / POS) ** 2).sum(-1).mean(-1).clamp_min(1e-10).sqrt()
    if PROGRESS == "exp":
        far = torch.exp(-e_shape / 0.15) + 0.5 * torch.exp(-e_end / 0.15)
    else:
        far = F.relu(1.0 - e_shape / 0.15) + 0.5 * F.relu(1.0 - e_end / 0.15)
    return far + torch.exp(-e_shape / 0.02), e_shape


def action_cost(a, a_prev):
    return 0.02 * (a ** 2).mean(-1) + 0.05 * ((a - a_prev) ** 2).mean(-1)


def reward(rope, obstacle, goal, a, a_prev):
    """Per-decision reward from decoded rope/grippers (B, ROPE), the true obstacle position (B, 3) and the
    goal (B, GOAL), all in the scaled episode frame. Also returns shape error and clearance (m).

    The progress terms are non-negative: an episode that ends in a crash loses all the reward it would
    still have collected. With the earlier, mostly negative shaping (r2/r3: about -2.3 per decision),
    ending early was worth more than going on, so the actor learnt that crashing pays
    (docs/notes/imagination_ac_design_20261022.md, section 11)."""
    prog, e_shape = progress(rope, goal)
    pts = rope[:, :3 * N_PTS].reshape(-1, N_PTS, 3) / POS
    d = (pts - obstacle[:, None] / POS).norm(dim=-1).min(-1).values - PROTECTED_M
    r = prog - 4.0 * torch.sigmoid((0.010 - d) / 0.004) - 20.0 * F.relu(-d) - action_cost(a, a_prev)
    return r, e_shape, d


def imagine(wm, actor, state, goal, future, future_meas, a_prev, H, origin, lim):
    """Roll the prior for H decisions with the actor from `state` (h, z, o). Returns feats (H+1, B, F), rewards (H, B),
    continuation (H, B), shape errors (H, B) and clearances (H, B).

    A real episode ends at the first clearance violation, so an imagined one must too: each step has a
    soft probability of still being alive, c = sigmoid(d / 3 mm), the reward carries the expected crash
    penalty -CRASH_PENALTY (1 - c), and everything after the step is discounted by c. Both are
    differentiable in the clearance, so the actor also learns that a crash forfeits the future."""
    feats, rews, cont, errs, clear = [wm.feat(state)], [], [], [], []
    rope, off = wm.rope_out(state), origin.repeat(1, 2)
    for i in range(H):
        extra = oracle_input(future, i) if actor.extra else None
        a = feasible_torch(actor(feats[-1], goal, extra), rope[:, 3 * N_PTS:ROPE] / POS + off, lim, steps=K)
        state = wm.img_step(state, a, future_meas[:, i])
        f = wm.feat(state)
        rope = wm.rope_out(state)
        r, e, d = reward(rope, future[:, i], goal, a, a_prev)
        c = torch.sigmoid(d / CONT_SCALE_M)
        feats.append(f), rews.append(r - CRASH_PENALTY * (1 - c)), cont.append(c), errs.append(e), clear.append(d)
        a_prev = a
    return torch.stack(feats), torch.stack(rews), torch.stack(cont), torch.stack(errs), torch.stack(clear)


def lambda_return(rews, values, cont, gamma=GAMMA, lam=LAMBDA):
    G, out = values[-1], []
    for i in reversed(range(len(rews))):
        G = rews[i] + gamma * cont[i] * ((1 - lam) * values[i + 1] + lam * G)
        out.append(G)
    return torch.stack(out[::-1])


class ActorCritic:
    def __init__(self, feat_dim, device, lr_actor=1e-4, lr_critic=3e-4, tau=0.02, actor_extra=0):
        self.actor, self.critic = Actor(feat_dim, extra=actor_extra).to(device), Critic(feat_dim).to(device)
        self.target = copy.deepcopy(self.critic).requires_grad_(False)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.tau, self.scale = tau, 1.0

    def bc_loss(self, feat, goal, act_next, mask, extra=None):
        """Imitate the executed action after each posterior state (mask: states that have one)."""
        pred = self.actor(feat.detach(), goal[:, None].expand(-1, feat.shape[1], -1), extra)
        return (((pred - act_next) ** 2).mean(-1) * mask).sum() / mask.sum().clamp_min(1.0)

    def bc_step(self, feat, goal, act_next, mask, extra=None):
        loss = self.bc_loss(feat, goal, act_next, mask, extra)
        self.opt_a.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.opt_a.step()
        return float(loss.detach())

    def step(self, wm, state, goal, future, future_meas, a_prev, origin, H, lim, bc=None, w_bc=0.0):
        """One actor + critic update from start states (tuple, each (B, .)). `bc` = (feat, goal, act_next, mask)."""
        with frozen(wm, self.critic):
            feats, rews, cont, errs, clear = imagine(wm, self.actor, tuple(x.detach() for x in state), goal, future, future_meas, a_prev, H,
                                                     origin, lim)
            g = goal[None].expand(H + 1, -1, -1)
            G = lambda_return(rews, self.critic(feats, g), cont)
        # weight of step i: discount and probability of being alive before it (no gradient through the weight)
        alive = torch.cat([torch.ones_like(cont[:1]), torch.cumprod(cont[:-1], 0)]).detach()
        weight = (GAMMA ** torch.arange(H, device=G.device, dtype=G.dtype))[:, None] * alive
        with torch.no_grad():                    # return normalisation (scale of the 5-95% range, EMA)
            q = torch.quantile(G.flatten(), torch.tensor([0.05, 0.95], device=G.device))
            self.scale = 0.99 * self.scale + 0.01 * max(1.0, float(q[1] - q[0]))
        actor_loss = -(weight * G).mean() / self.scale
        if bc is not None and w_bc > 0:
            actor_loss = actor_loss + w_bc * self.bc_loss(*bc)
        self.opt_a.zero_grad()
        actor_loss.backward()
        gn = float(nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0))
        self.opt_a.step()
        with torch.no_grad():
            Gt = lambda_return(rews.detach(), self.target(feats.detach(), g), cont.detach())
        critic_loss = 0.5 * (weight * (self.critic(feats.detach()[:-1], g[:-1]) - Gt) ** 2).mean()
        self.opt_c.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.opt_c.step()
        with torch.no_grad():
            for p, q in zip(self.critic.parameters(), self.target.parameters()):
                q.mul_(1 - self.tau).add_(self.tau * p)
        return dict(actor_loss=float(actor_loss.detach()), critic_loss=float(critic_loss.detach()),
                    imag_return=float(G[0].mean().detach()), imag_reward=float(rews.mean().detach()),
                    imag_shape_mm=float(errs[-1].mean().detach() * 1000), imag_min_clear_mm=float(clear.min(0).values.mean().detach() * 1000),
                    imag_violation=float((clear.min(0).values < 0).float().mean().detach()),
                    imag_alive_end=float(torch.cumprod(cont, 0)[-1].mean().detach()), actor_grad=gn, ret_scale=self.scale)
