"""G-B (= r12; docs/notes/geom_tracker_prereg_20261106.md, section 10): r11's grouped heads trained in imagination under
the tracker's belief, with geometric obstacle inputs. No GRU forecaster anywhere and no true future in training.

- Belief: the frozen tracker (g_system.TRACKER_CONFIG) run along an episode's readings; at every decision M_TRAIN
  whole-trajectory particles, PH decisions long (particle index i = i + 1 decisions after the decision), float16.
- Inputs: per obstacle, at particle indices i, i + 3, i + 6 of an imagined step i, the clearance between the rope
  and each particle's shape (sdf - rope radius - clearance; 10/50/90% over the particles), the mean unit direction
  from the nearest rope point to the obstacle, the nearest point's arc position (mean, sd), and at index i the
  particles' position sd: 3 x 8 + 3 = O_IN_G numbers, pooled over the obstacles by the set encoder.
- Imagination: reward and continuation from geom.TrajectoryRisk on every row's own particles (with the
  actor-critic reward's penetration term); the winner of each group, and the critic's trajectory, are picked by
  this belief return, the score the planner can see at execution.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .actor_critic import CRASH_PENALTY, GAMMA, action_cost, frozen, lambda_return, progress
from .episodes import K, N_PTS, POS, ROPE, VEL
from .feasible import feasible_torch
from .g_system import SPHERE, TRACKER_CONFIG
from .geom import CLEARANCE_M, ROPE_RADIUS_M, TrajectoryRisk, sdf
from .r10 import SetActorCritic
from .tracker import MultiModelTracker

O_IN_G = 27
LOOK = (0, 3, 6)
M_TRAIN, PH = 16, 39
PRIOR_BOX = ((-0.35, -0.35, -0.25), (0.35, 0.35, 0.25))     # episode frame, before the first reading
QS = (0.1, 0.5, 0.9)


# ---------------------------------------------------------------------------------------------- features
def geo_features(pts, parts, i, shape=SPHERE):
    """pts (R, N, 3) rope points, metres, episode frame; parts (R or 1, M, P, 3) -> (R, O_IN_G)."""
    R, N = pts.shape[:2]
    M, P = parts.shape[1], parts.shape[2]
    qi = [int(round(q * (M - 1))) for q in QS]
    out = []
    for lk in LOOK:
        c = parts[:, :, min(i + lk, P - 1)]                                   # (R', M, 3)
        s = sdf(shape, pts[:, None] - c[:, :, None]) - ROPE_RADIUS_M - CLEARANCE_M   # (R, M, N)
        dmin, k = s.min(-1)
        q = dmin.sort(-1).values[:, qi]                                      # (R, 3)
        near = torch.gather(pts[:, None].expand(-1, M, -1, -1), 2, k[..., None, None].expand(-1, -1, 1, 3))[:, :, 0]
        u = c.expand(R, -1, -1) - near
        u = (u / u.norm(dim=-1, keepdim=True).clamp_min(1e-6)).mean(1)
        arc = k.to(pts.dtype) / (N - 1)
        out += [10.0 * q, u, arc.mean(1, keepdim=True), arc.std(1, keepdim=True)]
    out.append(10.0 * parts[:, :, min(i, P - 1)].std(1).expand(R, -1))
    return torch.cat(out, -1)


@torch.no_grad()
def obstacle_feats(pts, parts_list, i, mask=None, shapes=None):
    """-> (R, J, O_IN_G); obstacles masked out (mask (R, J) == 0) are zeros. No gradient: the actor's gradient reaches
    the rope through the dynamics and the rewards, not through its own inputs (memory)."""
    f = torch.stack([geo_features(pts, P, i, SPHERE if shapes is None else shapes[j]) for j, P in enumerate(parts_list)], 1)
    return f if mask is None else f * mask[..., None]


# ---------------------------------------------------------------------------------------------- belief particles
def decode(enc):
    """Encoded reading (7,) -> (position, velocity) in metres / m/s, episode frame, or None while hidden."""
    enc = np.asarray(enc, float)
    return None if enc[6] <= 0 else (enc[:3] / POS, enc[3:6] / VEL)


def track_parts(readings, seed, T=None, M=M_TRAIN, ph=PH):
    """Readings (>= T, 7) at decision rate -> (T, M, ph, 3) float16: at each decision, the tracker's particles for the
    ph decisions after it."""
    T = len(readings) if T is None else T
    trk = MultiModelTracker(seed=seed, prior_box=PRIOR_BOX, config=TRACKER_CONFIG)
    out = np.empty((T, M, ph, 3), np.float16)
    for t in range(T):
        trk.update(t * K, decode(readings[t]))
        out[t] = trk.sample(t * K, ph, K, M)
    return out


def add_parts(ep, seed):
    """ep["g_parts"] (3, T, M, PH, 3): the episode's own obstacle, then its two synthetic ones (ep["extra_meas"])."""
    T = len(ep["obs"])
    own = track_parts(ep["obs"][:, ROPE:], seed, T)
    extras = [track_parts(np.concatenate([m, np.repeat(m[-1:], max(T - len(m), 0), 0)]), seed + 1 + j, T) for j, m in enumerate(ep["extra_meas"])]
    ep["g_parts"] = np.stack([own] + extras)
    return ep


# ---------------------------------------------------------------------------------------------- imagination
def imagine_gb(wm, actor, state, parts, mask, goal, a_prev, H, origin, lim):
    """state (h, z, o_wm) each (B, .); parts (B, J, M, PH, 3) metres; mask (B, J). Every start becomes one row per head
    (row b * heads + k). Returns the critic inputs at every step, rewards, continuations, shape errors, survival."""
    Kh, B, J = actor.heads, len(parts), parts.shape[1]
    rep = lambda x: x.repeat_interleave(Kh, 0)                                # noqa: E731
    st, parts, mask, goal, a_prev, origin = tuple(rep(x) for x in state), rep(parts), rep(mask), rep(goal), rep(a_prev), rep(origin)
    R = B * Kh
    rows, head = torch.arange(R, device=parts.device), torch.arange(Kh, device=parts.device).repeat(B)
    plist = [parts[:, j] for j in range(J)]
    risk = TrajectoryRisk(plist, [SPHERE] * J, R, mask=mask, penetration=20.0, k=1)       # rope points as they are (memory)
    off = origin.repeat(1, 2)
    pts_of = lambda r: r[:, :3 * N_PTS].reshape(R, N_PTS, 3) / POS           # noqa: E731
    rope = wm.rope_out(st)
    inputs, rews, cont, errs = [], [], [], []
    for i in range(H):
        rf, ox = torch.cat(st[:2], -1), obstacle_feats(pts_of(rope), plist, i, mask)
        inputs.append((rf, ox))
        a = feasible_torch(actor(rf, goal, ox, mask)[rows, head], rope[:, 3 * N_PTS:ROPE] / POS + off, lim, steps=K)
        st = wm.img_step(st, a, None)
        rope = wm.rope_out(st)
        prog, e = progress(rope, goal)
        p, margin = risk.step(i, pts_of(rope))
        rews.append(prog - margin - action_cost(a, a_prev) - CRASH_PENALTY * p)
        cont.append(1.0 - p)
        errs.append(e)
        a_prev = a
    inputs.append((torch.cat(st[:2], -1), obstacle_feats(pts_of(rope), plist, H, mask)))
    return inputs, torch.stack(rews), torch.stack(cont), torch.stack(errs), risk.survival(), goal, mask


class GBActorCritic(SetActorCritic):
    """SetActorCritic (grouped heads, winner takes all per group) with imagine_gb: belief rewards, belief winners."""

    def step(self, wm, state, parts, mask, goal, a_prev, origin, H, lim, bc=None, w_bc=0.0):
        Kh, B = self.actor.heads, len(parts)
        with frozen(wm, self.critic):
            inputs, rews, cont, errs, surv, g, m = imagine_gb(wm, self.actor, tuple(x.detach() for x in state), parts, mask, goal,
                                                              a_prev, H, origin, lim)
            V = torch.stack([self.critic(rf, g, ox, m) for rf, ox in inputs])
            G = lambda_return(rews, V, cont)
        alive = torch.cat([torch.ones_like(cont[:1]), torch.cumprod(cont[:-1], 0)]).detach()
        weight = (GAMMA ** torch.arange(H, device=G.device, dtype=G.dtype))[:, None] * alive
        with torch.no_grad():
            q = torch.quantile(G.flatten(), torch.tensor([0.05, 0.95], device=G.device))
            self.scale = 0.99 * self.scale + 0.01 * max(1.0, float(q[1] - q[0]))
            G0 = G[0].reshape(B, Kh)
            best = G0.argmax(-1)
            w_head = torch.full((B, Kh), self.w_other, device=G.device)
            extra = {}
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
        sel = torch.arange(B, device=G.device) * Kh + best
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
                    imag_return=float(G0.max(-1).values.mean()), imag_return_mean_head=float(G[0].detach().mean()),
                    imag_shape_mm=float(errs[-1].mean() * 1000), imag_crash=float(1.0 - surv.detach().mean()),
                    imag_crash_best=float(1.0 - surv.detach()[sel].mean()), head_usage_max=float(usage.max()),
                    head_usage_entropy=float(-(usage.clamp_min(1e-9) * usage.clamp_min(1e-9).log()).sum()), actor_grad=gn,
                    ret_scale=self.scale, **extra)
