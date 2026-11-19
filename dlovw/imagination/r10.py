"""r10: actor and critic over a set of obstacles, with several proposal heads (docs/notes/r10_prereg_20261104.md).

- The obstacle belief everywhere (imagination, collection, execution) is the frozen standalone forecaster
  (obstacle_model.ObstacleForecaster, one state per obstacle); the world model only supplies the rope (h, z).
- Each obstacle is encoded from its forecaster state, the predicted mixture and where its predicted mean sits
  relative to the nearest rope point; the encodings are pooled (sum and max) over the obstacles present.
- The actor has HEADS outputs. In imagination every head drives its own copy of each start for the whole
  horizon; the head with the best return gets the full gradient, the others a small weight (winner takes
  all), and the critic learns the value along the best head's trajectory. At execution the heads are the
  planner's candidates (HeadsPlanner), each imagined as the policy of its own rollout.
- Imagination adds 0-2 synthetic obstacles to each start (extra_obstacles): obstacles do not touch the rope,
  so a real rope episode can be combined with any independent obstacle track.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .actor_critic import CONT_SCALE_M, CRASH_PENALTY, GAMMA, action_cost, frozen, lambda_return, progress
from .episodes import ACT, CAP, GOAL, K, N_PTS, POS, PROTECTED_M, ROPE, VEL, encode_goal, encode_meas, encode_obs, origin_of
from .feasible import feasible_np, feasible_torch, limits_of
from .world_model import mlp

J_MAX, HEADS = 3, 8
O_IN = 128 + 2 + 6 + 6 + 3 + 1                 # forecaster state, mixture (w, mu, sd), offset from the rope, distance
POOL = 2 * 64 + 1


# ---------------------------------------------------------------------------------------------- synthetic obstacles
def extra_obstacles(ep, rng, n=2):
    """n independent obstacles for an episode, at decision resolution like ep["track"] / ep["track_meas"]:
    tracks (n, M, 3) and readings (n, M, 7), scaled episode frame. Each crosses the rope's path 15-25 cm from the
    episode's own obstacle along the target rope (one on each side), moving as the synthetic training data."""
    from ..belief import Sensor
    from ..motion import track as motion_track, velocities
    from ..scene_families import _end_occluder, _near_occluder, _perception
    from .obstacle_data import sample_motion
    tr0 = np.asarray(ep["track"], float) / POS                                     # metres, relative to the origin
    M = len(tr0)
    c0 = tr0.mean(0)
    dirs = np.linalg.svd(tr0 - c0, full_matrices=False)[2]
    n0 = dirs[0] / np.linalg.norm(dirs[0])
    tgt = np.asarray(ep["goal"][:3 * N_PTS], float).reshape(N_PTS, 3) / POS
    u = tgt[-1] - tgt[0]
    u[2] = 0.0
    u = u / max(np.linalg.norm(u), 1e-9)
    tracks, meas = np.zeros((n, M, 3), np.float32), np.zeros((n, M, 7), np.float32)
    sides = rng.permutation([-1.0, 1.0] * ((n + 1) // 2))[:n]
    for j in range(n):
        p = c0 + sides[j] * rng.uniform(0.15, 0.25) * u
        motion, amp = sample_motion(rng, p, n0)
        r = rng.uniform()
        occ = None if r < 0.2 else (_end_occluder(rng, p, n0, amp)[1] if r < 0.6 else _near_occluder(rng, p, n0)[1])
        sensor = Sensor(_perception(rng), occ, rng)
        tr = motion_track(motion, M * K + 1)
        vel = velocities(tr)
        for m in range(M):
            k = m * K
            rd = sensor.read(tr[:k + 1], vel[:k + 1])
            meas[j, m] = encode_meas(None if rd is None else (rd[1], rd[2]), np.zeros(3))
        tracks[j] = tr[0::K][:M] * POS
    return tracks, meas


# ---------------------------------------------------------------------------------------------- networks
def obstacle_inputs(fc, o, rope):
    """o (..., J, 128) forecaster states; rope (..., ROPE) scaled -> (..., J, O_IN)."""
    w, mu, sd = fc.dist(o)
    mean = (w[..., None] * mu).sum(-2)                                              # (..., J, 3)
    pts = rope[..., :3 * N_PTS].reshape(*rope.shape[:-1], N_PTS, 3)
    diff = mean[..., :, None, :] - pts[..., None, :, :]                             # (..., J, N, 3)
    dist = diff.norm(dim=-1)
    idx = dist.argmin(-1, keepdim=True)
    vec = torch.gather(diff, -2, idx[..., None].expand(*idx.shape, 3)).squeeze(-2)
    return torch.cat([o, w, mu.flatten(-2), sd.flatten(-2), vec, dist.gather(-1, idx)], -1)


class SetEncoder(nn.Module):
    def __init__(self, hid=128, out=64, o_in=O_IN):
        super().__init__()
        self.phi = mlp(o_in, hid, out, 2)

    def forward(self, x, mask):
        h = self.phi(x) * mask[..., None]
        m = torch.where(mask[..., None] > 0, h, torch.full_like(h, -1e4)).max(-2).values
        m = torch.where(mask.sum(-1, keepdim=True) > 0, m, torch.zeros_like(m))
        return torch.cat([h.sum(-2), m, mask.sum(-1, keepdim=True) / J_MAX], -1)


class SetActor(nn.Module):
    def __init__(self, rope_dim, heads=HEADS, hid=256, o_in=O_IN):
        super().__init__()
        self.heads = heads
        self.enc = SetEncoder(o_in=o_in)
        self.net = mlp(rope_dim + GOAL + POOL, hid, heads * ACT, 3)

    def forward(self, rf, goal, ox, mask):
        """-> (..., heads, 6) raw commands in cap units."""
        y = self.net(torch.cat([rf, goal, self.enc(ox, mask)], -1))
        return torch.tanh(y).reshape(*y.shape[:-1], self.heads, ACT)


def parse_groups(spec):
    """"free:3,shape:3,move:2" -> (("free", 3), ("shape", 3), ("move", 2))."""
    return tuple((k, int(n)) for k, n in (x.split(":") for x in spec.split(",")))


class GroupedSetActor(SetActor):
    """r11 (docs/notes/r11_prereg_20261106.md): heads in groups. "free" heads as SetActor; "shape" and "move" heads
    output a common part c and a differential part d and command [c + d, c - d], with c ("shape") or d ("move")
    limited to `cap` of the speed limit and the other part to 1 - cap."""

    def __init__(self, rope_dim, groups, cap=0.2, hid=256, o_in=O_IN):
        super().__init__(rope_dim, sum(n for _, n in groups), hid, o_in)
        self.groups, self.cap = tuple((str(k), int(n)) for k, n in groups), float(cap)
        kinds = [k for k, n in self.groups for _ in range(n)]
        if not set(kinds) <= {"free", "shape", "move"}:
            raise ValueError(f"unknown head group in {self.groups}")
        scale = [[1.0, 1.0] if k == "free" else ([cap, 1.0 - cap] if k == "shape" else [1.0 - cap, cap]) for k in kinds]
        self.register_buffer("scale_cd", torch.tensor(scale))                         # (heads, 2): scale of c and d
        self.register_buffer("is_free", torch.tensor([k == "free" for k in kinds]))

    def slices(self):
        out, s = [], 0
        for k, n in self.groups:
            out.append((k, s, s + n))
            s += n
        return out

    def forward(self, rf, goal, ox, mask):
        y = torch.tanh(self.net(torch.cat([rf, goal, self.enc(ox, mask)], -1)))
        y = y.reshape(*y.shape[:-1], self.heads, ACT)
        c, d = y[..., :3] * self.scale_cd[:, :1], y[..., 3:] * self.scale_cd[:, 1:]
        return torch.where(self.is_free[:, None], y, torch.cat([c + d, c - d], -1))


class SetCritic(nn.Module):
    def __init__(self, rope_dim, hid=256, o_in=O_IN):
        super().__init__()
        self.enc = SetEncoder(o_in=o_in)
        self.net = mlp(rope_dim + GOAL + POOL, hid, 1, 2)

    def forward(self, rf, goal, ox, mask):
        return self.net(torch.cat([rf, goal, self.enc(ox, mask)], -1)).squeeze(-1)


# ---------------------------------------------------------------------------------------------- imagination
def reward_multi(rope, obst, mask, goal, a, a_prev):
    """As actor_critic.reward with several obstacles: obst (B, J, 3) true positions, mask (B, J). Penalties add up
    over the obstacles; the clearance returned is the smallest."""
    prog, e_shape = progress(rope, goal)
    pts = rope[:, :3 * N_PTS].reshape(-1, N_PTS, 3) / POS
    d = (pts[:, None] - obst[:, :, None] / POS).norm(dim=-1).min(-1).values - PROTECTED_M
    d = torch.where(mask > 0, d, torch.ones_like(d))
    pen = (4.0 * torch.sigmoid((0.010 - d) / 0.004) + 20.0 * F.relu(-d)).sum(-1)
    return prog - pen - action_cost(a, a_prev), e_shape, d.min(-1).values


class _ClipRowGrad(torch.autograd.Function):
    """Identity forward; backward rescales each row's gradient to norm <= c (the gradient direction is kept)."""
    @staticmethod
    def forward(ctx, x, c):
        ctx.c = c
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        n = g.norm(dim=-1, keepdim=True)
        return g * torch.clamp(ctx.c / (n + 1e-12), max=1.0), None


def imagine_multi(wm, fc, actor, state, o, mask, goal, future, future_meas, a_prev, H, origin, lim, risk=True, reward_fn=None,
                  cont_scale=CONT_SCALE_M, truncate=0, state_clip=0.0):
    """state (h, z, o_wm) each (B, .); o (B, J, 128); mask (B, J); future (B, H, J, 3); future_meas (B, H, J, 7).
    Every start becomes one row per head (row b * heads + k), head k acting at every step of its row.
    reward_fn (cell cost v2, dlovw/cell/cost.py): (rope, obst, mask, goal, a, a_prev, ctx) -> (r, e, d); None: reward_multi.
    cont_scale: soft-survival sigmoid(d / cont_scale) width (m). truncate > 0: detach the imagined state, previous
    action and carried clearance every `truncate` steps (truncated backprop through the learned dynamics; values and
    rewards are unchanged, only how far a reward's gradient reaches back). state_clip > 0: after every imagined
    step, clip each row's gradient flowing back through each state component to that norm (full-horizon credit
    kept, only exploding magnitudes capped). Defaults reproduce r10 exactly."""
    Kh, B, J = actor.heads, len(o), o.shape[1]
    rep = lambda x: x.repeat_interleave(Kh, 0)                       # noqa: E731
    st, o, mask, goal, future, future_meas, a_prev, origin = (tuple(rep(x) for x in state), rep(o), rep(mask), rep(goal),
                                                              rep(future), rep(future_meas), rep(a_prev), rep(origin))
    rows = torch.arange(B * Kh, device=o.device)
    head = torch.arange(Kh, device=o.device).repeat(B)
    rope, off = wm.rope_out(st), origin.repeat(1, 2)
    inputs, rews, cont, errs, clear, ctx = [], [], [], [], [], {}
    for i in range(H):
        if truncate and i and i % truncate == 0:
            st, rope, a_prev = tuple(x.detach() for x in st), rope.detach(), a_prev.detach()
            if "d" in ctx:
                ctx["d"] = ctx["d"].detach()
        rf = torch.cat(st[:2], -1)
        ox = obstacle_inputs(fc, o, rope)
        inputs.append((rf, ox))
        a = actor(rf, goal, ox, mask)[rows, head]
        a = feasible_torch(a, rope[:, 3 * N_PTS:ROPE] / POS + off, lim, steps=K)
        if hasattr(lim, 'limit_change_torch'):
            a = lim.limit_change_torch(a, a_prev, K)
        st = wm.img_step(st, a, None)
        if state_clip:
            st = tuple(_ClipRowGrad.apply(x, state_clip) for x in st)
        o = torch.stack([fc.step(o[:, j], future_meas[:, i, j], None) for j in range(J)], 1)
        rope = wm.rope_out(st)
        r, e, d = (reward_multi(rope, future[:, i], mask, goal, a, a_prev) if reward_fn is None
                   else reward_fn(rope, future[:, i], mask, goal, a, a_prev, ctx))
        c = torch.sigmoid(d / cont_scale)
        if not risk:                                  # ablation d: no obstacle term in the reward, never "dead"
            r, c = progress(rope, goal)[0] - action_cost(a, a_prev), torch.ones_like(c)
        rews.append(r - CRASH_PENALTY * (1 - c)), cont.append(c), errs.append(e), clear.append(d)
        a_prev = a
    inputs.append((torch.cat(st[:2], -1), obstacle_inputs(fc, o, rope)))
    return inputs, torch.stack(rews), torch.stack(cont), torch.stack(errs), torch.stack(clear), goal, mask


class SetActorCritic:
    def __init__(self, rope_dim, device, heads=HEADS, lr_actor=1e-4, lr_critic=3e-4, tau=0.02, w_other=0.05, groups=None, cap=0.2,
                 o_in=O_IN):
        """groups (r11): winner takes all within each group of heads (GroupedSetActor); None: as r10. o_in: per-obstacle
        input size (G-B: gb.O_IN_G)."""
        self.groups, self.risk, self.reward_fn = groups, True, None
        actor = GroupedSetActor(rope_dim, groups, cap, o_in=o_in) if groups else SetActor(rope_dim, heads, o_in=o_in)
        self.actor, self.critic = actor.to(device), SetCritic(rope_dim, o_in=o_in).to(device)
        self.target = copy.deepcopy(self.critic).requires_grad_(False)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.tau, self.scale, self.w_other = tau, 1.0, w_other
        self.cont_scale, self.truncate, self.state_clip = CONT_SCALE_M, 0, 0.0      # imagine_multi defaults (r10)

    def bc_loss(self, rf, goal, ox, mask, act_next, valid):
        """Imitation on the closest head (all heads keep a small weight so none is left untrained)."""
        pred = self.actor(rf.detach(), goal, ox.detach(), mask)                   # (B, T, heads, 6)
        err = ((pred - act_next[..., None, :]) ** 2).mean(-1)
        e = err.min(-1).values + 0.1 * err.mean(-1)
        return (e * valid).sum() / valid.sum().clamp_min(1.0)

    def bc_step(self, *bc):
        loss = self.bc_loss(*bc)
        self.opt_a.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.opt_a.step()
        return float(loss.detach())

    def step(self, wm, fc, state, o, mask, goal, future, future_meas, a_prev, origin, H, lim, bc=None, w_bc=0.0):
        Kh, B = self.actor.heads, len(o)
        with frozen(wm, fc, self.critic):
            inputs, rews, cont, errs, clear, g, m = imagine_multi(wm, fc, self.actor, tuple(x.detach() for x in state), o.detach(), mask,
                                                                  goal, future, future_meas, a_prev, H, origin, lim, risk=self.risk,
                                                                  reward_fn=self.reward_fn, cont_scale=self.cont_scale,
                                                                  truncate=self.truncate, state_clip=self.state_clip)
            V = torch.stack([self.critic(rf, g, ox, m) for rf, ox in inputs])
            G = lambda_return(rews, V, cont)                                       # (H, B * heads)
        alive = torch.cat([torch.ones_like(cont[:1]), torch.cumprod(cont[:-1], 0)]).detach()
        weight = (GAMMA ** torch.arange(H, device=G.device, dtype=G.dtype))[:, None] * alive
        with torch.no_grad():
            q = torch.quantile(G.flatten(), torch.tensor([0.05, 0.95], device=G.device))
            self.scale = 0.99 * self.scale + 0.01 * max(1.0, float(q[1] - q[0]))
            best = G[0].reshape(B, Kh).argmax(-1)
            w_head = torch.full((B, Kh), self.w_other, device=G.device)
            if self.groups is None:
                w_head[torch.arange(B, device=G.device), best] = 1.0
            else:
                G0, extra = G[0].reshape(B, Kh), {}
                for k, s, e in self.actor.slices():
                    w_head[torch.arange(B, device=G.device), s + G0[:, s:e].argmax(-1)] = 1.0
                    extra[f"best_in_{k}"] = float(((best >= s) & (best < e)).float().mean())
                    extra[f"imag_return_{k}"] = float(G0[:, s:e].max(-1).values.mean())
        obj = (weight * G).mean(0).reshape(B, Kh)
        actor_loss = -(obj * w_head).sum(-1).mean() / w_head.sum(-1).mean() / self.scale
        if bc is not None and w_bc > 0:
            actor_loss = actor_loss + w_bc * self.bc_loss(*bc)
        self.opt_a.zero_grad()
        actor_loss.backward()
        gn = float(nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0))
        self.opt_a.step()
        sel = torch.arange(B, device=G.device) * Kh + best                         # the best head's rows
        with torch.no_grad():
            Vt = torch.stack([self.target(rf[sel].detach(), g[sel], ox[sel].detach(), m[sel]) for rf, ox in inputs])
            Gt = lambda_return(rews[:, sel].detach(), Vt, cont[:, sel].detach())
        Vc = torch.stack([self.critic(rf[sel].detach(), g[sel], ox[sel].detach(), m[sel]) for rf, ox in inputs[:-1]])
        critic_loss = 0.5 * (weight[:, sel] * (Vc - Gt) ** 2).mean()
        self.opt_c.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.opt_c.step()
        with torch.no_grad():
            for p, q_ in zip(self.critic.parameters(), self.target.parameters()):
                q_.mul_(1 - self.tau).add_(self.tau * p)
            usage = torch.bincount(best, minlength=Kh).float() / B
        return dict(actor_loss=float(actor_loss.detach()), critic_loss=float(critic_loss.detach()),
                    imag_return=float(G[0].detach().reshape(B, Kh).max(-1).values.mean()), imag_return_mean_head=float(G[0].detach().mean()),
                    imag_shape_mm=float(errs[-1].mean() * 1000), imag_violation=float((clear.min(0).values < 0).float().mean()),
                    imag_alive_end=float(torch.cumprod(cont, 0)[-1].mean()), head_usage_max=float(usage.max()),
                    head_usage_entropy=float(-(usage.clamp_min(1e-9) * usage.clamp_min(1e-9).log()).sum()), actor_grad=gn,
                    ret_scale=self.scale, **(extra if self.groups is not None else {}))


# ---------------------------------------------------------------------------------------------- checkpoints
def load_r10(path):
    """(wm, forecaster, actor, critic) of an r10 or r11 checkpoint, CPU, eval mode."""
    from . import actor_critic as ac_mod
    from .obstacle_model import MemorylessForecaster, ObstacleForecaster
    from .world_model import FactoredWM
    st = torch.load(path, map_location="cpu")
    ac_mod.PROGRESS = st.get("progress", "exp")
    if st.get("wm_kind") == "step":                 # ablation a2
        from .step_wm import StepWM
        wm = StepWM(**st["wm_kw"])
    else:
        wm = FactoredWM(**st["wm_kw"])
    wm.load_state_dict(st["wm"])
    if st.get("ac_input") == "rope":                # ablation a2', arm R-D
        from .rope_input import RopeInputWM
        wm = RopeInputWM(wm)
    kind = st.get("forecaster_kind", "gru")
    if kind == "gru":
        fc = ObstacleForecaster(feats=False)
        fc.load_state_dict(st["forecaster"])
    else:                                          # ablations a1 / b
        f = st["forecaster"]
        fc = MemorylessForecaster(f["coef"], f["sd"], f["prior_mu"], f["prior_sd"], use_velocity=kind == "cv")
    rope_dim = wm.h_dim + wm.z_dim
    o_in = st.get("o_in", O_IN)
    actor = (GroupedSetActor(rope_dim, st["groups"], st["cap"], o_in=o_in) if st.get("groups")
             else SetActor(rope_dim, st["heads"], o_in=o_in))
    critic = SetCritic(rope_dim, o_in=o_in)
    actor.load_state_dict(st["actor"])
    critic.load_state_dict(st["critic"])
    return wm.eval(), fc.eval(), actor.eval(), critic.eval()


# ---------------------------------------------------------------------------------------------- collection
def run_episode_r10(rec, wm, fc, actor, *, noise=0.0, seed=0, max_time_s=7.2, target=None, head=None):
    """As rollout.run_episode with the r10 actor (one obstacle, the training scenes): one head per episode (random
    unless given) plus Gaussian noise. Returns the decision-level episode."""
    from .episodes import build_episode
    from .rollout import open_actor_session
    s = open_actor_session(rec, None, max_time_s=max_time_s, target=target)
    rng = np.random.default_rng(seed)
    head = int(rng.integers(actor.heads)) if head is None else head
    seq = dict(center=[], left=[], right=[], meas=[], op=[], ov=[])
    try:
        origin = origin_of(s.target)
        goal = torch.as_tensor(encode_goal(s.target, origin))[None]
        drv = s.dynamic.drivers[0]
        lim = limits_of(s)

        def record():
            f = s.frames[-1]
            for k_, v in (("center", f.centerline), ("left", f.left_pose[:3]), ("right", f.right_pose[:3]),
                          ("op", drv.position), ("ov", drv.velocity)):
                seq[k_].append(np.array(v, float))
            seq["meas"].append(s.measurement(0))

        record()
        state, o = wm.initial(1, "cpu"), fc.initial(1)
        a_prev, n, interventions = torch.zeros(1, 6), 0, 0
        mask = torch.ones(1, 1)
        with torch.no_grad():
            while s.progress.terminal == "running":
                if n % K == 0:
                    if n:
                        a_prev = torch.as_tensor((np.r_[seq["left"][-1], seq["right"][-1]] - start) / (K * CAP), dtype=torch.float32)[None]
                    start = np.r_[seq["left"][-1], seq["right"][-1]]
                    ob = encode_obs(seq["center"][-1], seq["left"][-1], seq["right"][-1], seq["meas"][-1], origin)
                    state = wm.obs_step(state, a_prev, torch.as_tensor(ob)[None])[0]
                    o = fc.step(o, torch.as_tensor(encode_meas(seq["meas"][-1], origin))[None], None)
                    rf, rope = torch.cat(state[:2], -1), wm.rope_out(state)
                    a = actor(rf, goal, obstacle_inputs(fc, o[:, None], rope), mask)[0, head].numpy().astype(float)
                    if noise:
                        a = np.clip(a + noise * rng.normal(size=6), -1.0, 1.0)
                tr, _ = s.step(feasible_np(a, np.r_[seq["left"][-1], seq["right"][-1]], lim) * CAP)
                interventions += float(tr["cbf_intervention"]) > 1e-6
                n += 1
                record()
        info = dict(id=rec["id"], terminal=s.progress.terminal, steps=int(s.progress.steps),
                    cbf_active_frac=interventions / max(n, 1), noise=float(noise), head=head)
        track = np.asarray(drv.track)
    finally:
        s.close()
    ep = build_episode(seq["center"], seq["left"], seq["right"], seq["meas"], seq["op"], seq["ov"], track, 0, s.target, info, rec, seed)
    ep["target"] = {k_: np.array(v) for k_, v in s.target.items()}
    return ep


# Execution (HeadsPlanner, episode_r10) lives in r10_exec.py.
