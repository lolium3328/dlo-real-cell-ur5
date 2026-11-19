"""Cost v2 for the cell's single-obstacle policy (docs/notes/cell_policy_20261113.md, section 7).

The r10 reward (actor_critic.reward / r10.reward_multi), per imagined decision, with d the clearance of the rope
centerline to the sphere beyond the protected radius (55 mm centre distance):

    r = prog(e) - 4 sigmoid((10 mm - d) / 4 mm) - 20 relu(-d) - action_cost - 10 (1 - sigmoid(d / 3 mm))

v2 changes, each aimed at a failure seen on the cell (section 6-7):
  1. buffer: the soft barrier sits `buffer_m` further out (20 mm), so the rope starts to give way earlier (failed
     dodges were too late, 14 mm at the violation);
  2. closing: the clearance shrinking while the sphere is near costs w_close per close_ref of approach per
     decision, faded out beyond close_range: the actor pays for the approach itself, not only for being close;
  3. gate: near the sphere the shape progress is switched off, g(d) = sigmoid((d - gate_m) / gate_tau): in hold
     scenes, clinging to the target shape while the sphere sweeps through it was worth more than dodging;
  4. smoother: the command-change term drops from 0.05 to w_smooth (a dodge has to start abruptly).
The crash continuation (sigmoid(d / 3 mm), CRASH_PENALTY on the expected crash) is unchanged.

reward_v2: training imagination (true future sphere positions), r10.imagine_multi(reward_fn=...).
expected_terms: the same terms averaged over the sphere forecast's samples (CellPlanner).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from ..imagination.actor_critic import CONT_SCALE_M, progress
from ..imagination.episodes import N_PTS, POS, PROTECTED_M


@dataclass
class CostCfg:
    protected_m: float = 0.055
    segment_distance: bool = False
    buffer_m: float = 0.020
    w_barrier: float = 4.0
    tau_barrier_m: float = 0.006
    w_viol: float = 20.0
    w_close: float = 1.5
    close_ref_m: float = 0.010
    close_range_m: float = 0.040
    gate_m: float = 0.030
    gate_tau_m: float = 0.010
    w_act: float = 0.02
    w_smooth: float = 0.02

    def dict(self):
        return asdict(self)


def clearance(rope, x, cfg=None):
    """rope (B, ROPE) scaled; x (B, ..., 3) scaled sphere centres -> (B, ...) clearance beyond PROTECTED_M (m)."""
    pts = rope[:, :3 * N_PTS].reshape(-1, N_PTS, 3) / POS
    shape = x.shape[1:-1]
    xs = (x / POS).reshape(len(x), -1, 1, 3)
    if cfg is not None and cfg.segment_distance:
        from .calibrated.kernels import segment_distances
        d = segment_distances(pts, xs.squeeze(-2), cfg.protected_m)
    else:
        d = (xs - pts[:, None]).norm(dim=-1).min(-1).values - (PROTECTED_M if cfg is None else cfg.protected_m)
    return d.reshape(len(x), *shape)


def terms(d, d_prev, cfg):
    """Per-sample obstacle terms: (gate, penalty, crash probability), same shape as d."""
    gate = torch.sigmoid((d - cfg.gate_m) / cfg.gate_tau_m)
    barrier = cfg.w_barrier * torch.sigmoid((cfg.buffer_m - d) / cfg.tau_barrier_m) + cfg.w_viol * F.relu(-d)
    closing = (cfg.w_close * F.relu(d_prev - d) / cfg.close_ref_m * torch.exp(-F.relu(d) / cfg.close_range_m)
               if d_prev is not None else torch.zeros_like(d))
    return gate, barrier + closing, torch.sigmoid(-d / CONT_SCALE_M)


def action_cost_v2(a, a_prev, cfg):
    return cfg.w_act * (a ** 2).mean(-1) + cfg.w_smooth * ((a - a_prev) ** 2).mean(-1)


def reward_v2(cfg):
    """reward_fn for r10.imagine_multi: (rope, obst (B, J, 3), mask, goal, a, a_prev, ctx) -> (r, e_shape, d_min).
    ctx carries the previous clearance between calls of one rollout."""
    def fn(rope, obst, mask, goal, a, a_prev, ctx):
        prog, e_shape = progress(rope, goal)
        d = clearance(rope, obst, cfg)                                              # (B, J)
        d = torch.where(mask > 0, d, torch.ones_like(d))
        gate, pen, _ = terms(d, ctx.get("d"), cfg)
        ctx["d"] = d
        r = prog * gate.prod(-1) - pen.sum(-1) - action_cost_v2(a, a_prev, cfg)
        return r, e_shape, d.min(-1).values
    return fn


def expected_terms(rope, w, mu, sd, eps, d_prev, cfg, scale=1.0):
    """Sphere forecast as a mixture (w (B, 2), mu / sd (B, 2, 3), scaled) sampled at mu + scale sd eps (fixed eps,
    the same samples every decision so d_prev follows each sample). Returns expected (gate, penalty, crash
    probability) per row and the per-sample clearance (B, 2, n) for the next call."""
    x = mu[:, :, None] + scale * sd[:, :, None] * eps
    d = clearance(rope, x, cfg)
    gate, pen, p = terms(d, d_prev, cfg)
    ex = lambda v: (w * v.mean(-1)).sum(-1)                         # noqa: E731
    return ex(gate), ex(pen), ex(p), d
