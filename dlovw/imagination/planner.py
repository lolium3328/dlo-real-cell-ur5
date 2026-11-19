"""Decision-level candidate scoring with the learned world model and value head
(docs/notes/imagination_ac_design_20261022.md, section 14). Nothing here is trained.

Every decision (K control steps) the planner imagines a few discrete candidates and executes the best:
- "actor": the trained actor at every imagined step;
- "wait": grippers still;
- "up", "down", "lat+", "lat-": both grippers at full speed, lateral = horizontal and perpendicular to
  the rope's end-to-end axis (the escape set of the stage-0 dodge rule);
- "target": each gripper at full speed straight towards its target position.
A fixed candidate is held for HOLD decisions, then the actor takes over until HORIZON decisions
(1.2 s). The obstacle is predicted blind by the obstacle filter (no future readings exist at execution
time), and the crash risk of an imagined state is the mixture-weighted probability that the rope is
inside the protected radius, with each component's scale inflated by KAPPA: the more uncertain the
obstacle, the more room a candidate has to leave. Score = discounted progress reward minus the expected
margin and crash penalties (same terms as in training) + discounted critic value at the end, both
weighted by the probability of having survived so far. A switch away from the previous choice needs a
score gain of at least SWITCH.
"""
from __future__ import annotations

import numpy as np
import torch

from .actor_critic import CONT_SCALE_M, CRASH_PENALTY, GAMMA, action_cost, progress
from .episodes import K, N_PTS, POS, PROTECTED_M, ROPE
from .feasible import feasible_torch

NAMES = ("actor", "wait", "up", "down", "lat+", "lat-", "target")


def _l1(d):
    return np.asarray(d, float) / max(np.abs(d).sum(), 1e-9)


def candidate_commands(grip, target_grip):
    """Fixed candidates (all but "actor") as per-step commands in cap units, (len(NAMES) - 1, 6)."""
    axis = grip[3:] - grip[:3]
    axis[2] = 0.0
    axis /= max(np.linalg.norm(axis), 1e-9)
    lat = np.cross([0.0, 0.0, 1.0], axis)
    both = lambda d: np.r_[_l1(d), _l1(d)]  # noqa: E731
    to = target_grip - grip
    out = [np.zeros(6), both([0, 0, 1.0]), both([0, 0, -1.0]), both(lat), both(-lat), np.r_[_l1(to[:3]), _l1(to[3:])]]
    return np.stack(out)


def particle_risk(rope, parts):
    """Expected crash probability and margin penalty with the obstacle as particles (M, 3), scaled frame."""
    pts = rope[:, :3 * N_PTS].reshape(-1, N_PTS, 3) / POS
    d = (pts[:, None] - (parts / POS)[None, :, None]).norm(dim=-1).min(-1).values - PROTECTED_M   # (B, M)
    return torch.sigmoid(-d / CONT_SCALE_M).mean(-1), (4.0 * torch.sigmoid((0.010 - d) / 0.004)).mean(-1)


_EPS = {}


def _normal_samples(n):
    """Fixed antithetic standard-normal samples (n, 3): common random numbers for every candidate and decision."""
    if n not in _EPS:
        e = torch.randn(n // 2, 3, generator=torch.Generator().manual_seed(20261027))
        _EPS[n] = torch.cat([e, -e])
    return _EPS[n]


def sampled_risk(rope, w, mu, sd, n=64, scale=1.0):
    """Expected crash probability and margin penalty under the obstacle mixture itself (docs/notes/
    risk_sampled_prereg_20261027.md): n samples per component at mu + scale * sd * eps, each scored like a
    particle. A wide component then spreads a small risk over every candidate instead of acting as a point at
    its mean with an inflated safety distance (crash_risk)."""
    x = mu[:, :, None] + scale * sd[:, :, None] * _normal_samples(n)                     # (B, 2, n, 3), scaled
    pts = rope[:, :3 * N_PTS].reshape(-1, N_PTS, 3) / POS
    d = ((x / POS)[:, :, :, None] - pts[:, None, None]).norm(dim=-1).min(-1).values - PROTECTED_M   # (B, 2, n)
    p = torch.sigmoid(-d / CONT_SCALE_M).mean(-1)
    m = (4.0 * torch.sigmoid((0.010 - d) / 0.004)).mean(-1)
    return (w * p).sum(-1), (w * m).sum(-1)


def crash_risk(rope, w, mu, sd, kappa):
    """Expected crash probability and margin penalty of imagined states under the obstacle mixture."""
    pts = rope[:, :3 * N_PTS].reshape(-1, N_PTS, 3) / POS
    dist = (pts[:, None] - (mu / POS)[:, :, None]).norm(dim=-1).min(-1).values          # (B, 2)
    d = dist - PROTECTED_M - kappa * (sd / POS).norm(dim=-1)
    p_crash = (w * torch.sigmoid(-d / CONT_SCALE_M)).sum(-1)
    margin = (w * 4.0 * torch.sigmoid((0.010 - d) / 0.004)).sum(-1)
    return p_crash, margin


class Planner:
    def __init__(self, wm, actor, critic, lim, hold=6, horizon=15, kappa=1.5, switch=0.3, point=False, far=0.0, oracle=False,
                 corridor=False, risk="mean", samples=64, sd_scale=1.0, act_samples=0, act_sigma=0.5, crash_scale=1.0):
        """point=True (ablation): the obstacle is the mixture mean only, no spread, no second mode.
        far > 0 adds far * exp(-e / 0.15 m) per step: the training progress terms are flat beyond 15 cm
        (relu(1 - e / 0.15) and exp(-e / 0.02)), so nothing pulls a rope back that has drifted further away."""
        self.wm, self.actor, self.critic, self.lim = wm, actor, critic, lim
        self.hold, self.horizon, self.kappa, self.switch, self.point, self.far = hold, horizon, kappa, switch, point, far
        self.oracle = oracle           # diagnosis only: the caller passes the true future obstacle positions
        self.corridor = corridor       # the caller passes corridor-prior particles (corridor.CorridorForecast)
        self.risk, self.samples, self.sd_scale = risk, samples, sd_scale   # "mean": crash_risk with kappa; "sampled"
        self.crash_penalty = CRASH_PENALTY * crash_scale     # docs/notes/caution_prereg_20261104.md
        self.last_audit = None
        # extra candidates (docs/notes/multi_obstacle_prereg_20261030.md, section 9): the actor plus a fixed perturbation
        # eps_k ~ N(0, act_sigma^2) per command dimension, held for HOLD decisions like the fixed candidates; the same
        # act_samples perturbations at every decision, so the switch margin compares like with like
        self.act_eps = (torch.randn(act_samples, 6, generator=torch.Generator().manual_seed(20261101)) * act_sigma
                        if act_samples else torch.zeros(0, 6))
        self.prev = None
        self.obst_model = None         # optional obstacle_model.ObstacleForecaster: the caller passes its blind forecast

    @torch.no_grad()
    def choose(self, state, goal, grip, target_grip, origin, a_prev, future=None, particles=None, obst=None, audit=None):
        """state: posterior (h, z, o) with batch 1; goal (1, GOAL); grip, target_grip (6,) and origin (3,)
        in metres. `future` (horizon, 3), scaled episode frame: true obstacle positions (oracle diagnosis);
        else the obstacle filter predicts blind. Returns (name, command in cap units, scores).
        Several obstacles (docs/notes/multi_obstacle_prereg_20261030.md): `obst` a list per obstacle of the
        per-step predictions, `future` (n_obstacles, horizon, 3); the crash probability of a step is then
        1 - prod_j (1 - p_j) and the margin penalties add up.
        `audit` (n_obstacles, horizon, 3), the true obstacle future: on the same imagined rope rollouts, also the crash
        probability every candidate would have against the true obstacles; stored in self.last_audit together with the
        predicted one (cumulative over the horizon). Does not change the choice."""
        wm, B = self.wm, len(NAMES) + len(self.act_eps)
        obsts = None if obst is None else (obst if isinstance(obst[0], list) else [obst])
        futs = None if future is None else (future if future.dim() == 3 else future[None])
        n_obs = len(futs) if futs is not None else len(obsts) if obsts is not None else 1
        st = tuple(x.expand(B, -1).clone() for x in state)
        g = goal.expand(B, -1)
        off = torch.as_tensor(np.r_[origin, origin], dtype=torch.float32)
        fixed = torch.as_tensor(candidate_commands(np.asarray(grip, float), np.asarray(target_grip, float)), dtype=torch.float32)
        first = self.actor(wm.feat(state), goal)[0]
        rope, ap = wm.rope_out(st), a_prev.expand(B, -1)
        total, alive, disc = torch.zeros(B), torch.ones(B), 1.0
        alive_true = torch.ones(B)
        auds = None if audit is None else (audit if audit.dim() == 3 else audit[None])
        for i in range(self.horizon):
            a = self.actor(wm.feat(st), g)
            if i < self.hold:
                a = torch.cat([a[:1], fixed, (a[len(NAMES):] + self.act_eps).clamp(-1.0, 1.0)], 0)
            a = feasible_torch(a, rope[:, 3 * N_PTS:ROPE] / POS + off, self.lim, steps=K)
            st = wm.img_step(st, a, None, sample=False)
            rope = wm.rope_out(st)
            prog, e_shape = progress(rope, g)
            if self.far:
                prog = prog + self.far * torch.exp(-e_shape / 0.15)
            if particles is not None:
                p_crash, margin = particle_risk(rope, particles[i])
                total += disc * alive * (prog - margin - CRASH_PENALTY * p_crash - action_cost(a, ap))
                alive, disc, ap = alive * (1.0 - p_crash), disc * GAMMA, a
                continue
            per = []
            for j in range(n_obs):
                if obsts is not None:             # standalone forecaster: (w, mu, sd) for this step, batch 1
                    w, mu, sd = (x.expand(B, *x.shape[1:]) for x in obsts[j][i])
                else:
                    w, mu, sd, _, _ = wm.obstacle_dist(st[2])
                if futs is not None:
                    w = torch.tensor([1.0, 0.0]).expand(B, 2)
                    mu = futs[j][i].reshape(1, 1, 3).expand(B, 2, 3)
                    sd = torch.zeros(B, 2, 3)
                elif self.point:
                    mu = (w[..., None] * mu).sum(-2, keepdim=True).expand_as(mu)
                    sd = torch.zeros_like(sd)
                if self.risk == "sampled":
                    per.append(sampled_risk(rope, w, mu, sd, self.samples, self.sd_scale))
                else:
                    per.append(crash_risk(rope, w, mu, sd, self.kappa))
            if n_obs == 1:
                p_crash, margin = per[0]
            else:
                p_crash = 1.0 - torch.stack([1.0 - p for p, _ in per]).prod(0)
                margin = torch.stack([m for _, m in per]).sum(0)
            total += disc * alive * (prog - margin - self.crash_penalty * p_crash - action_cost(a, ap))
            if auds is not None:
                zero, one = torch.zeros(B, 2, 3), torch.tensor([1.0, 0.0]).expand(B, 2)
                keep = torch.ones(B)
                for j in range(len(auds)):
                    keep = keep * (1.0 - sampled_risk(rope, one, auds[j][i].reshape(1, 1, 3).expand(B, 2, 3), zero, 2)[0])
                alive_true = alive_true * keep
            alive, disc, ap = alive * (1.0 - p_crash), disc * GAMMA, a
        if auds is not None:
            self.last_audit = dict(pred=(1.0 - alive).numpy(), true=(1.0 - alive_true).numpy())
        total += disc * alive * self.critic(wm.feat(st), g)
        pick = int(total.argmax())
        if self.prev is not None and float(total[self.prev]) >= float(total[pick]) - self.switch:
            pick = self.prev
        self.prev = pick
        if pick >= len(NAMES):
            k = pick - len(NAMES)
            return f"sample{k}", (first + self.act_eps[k]).clamp(-1.0, 1.0).numpy().astype(float), total.numpy()
        cmd = first if pick == 0 else fixed[pick - 1]
        return NAMES[pick], cmd.numpy().astype(float), total.numpy()


def load_policy(ckpt):
    """World model, actor and critic of a checkpoint (CPU, eval mode)."""
    from .actor_critic import Critic
    from .workers import _nets
    wm, actor = _nets(ckpt)
    critic = Critic(wm.feat_dim)
    critic.load_state_dict(torch.load(ckpt, map_location="cpu")["critic"])
    return wm, actor, critic.eval()
