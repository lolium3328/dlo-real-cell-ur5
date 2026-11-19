"""Decision rule v2 for the cell (docs/notes/cell_policy_20261113.md, section 7): r10's HeadsPlanner with the cost v2
terms (cost.py) and two selection rules.

Each head is imagined as its own policy for `horizon` decisions under the sphere forecast (blind after now, as
r10); a candidate's score is

    sum_i gamma^i S_i [ prog_i E[g_i] - E[pen_i] - CRASH_PENALTY p_i - action_cost_v2 ] + gamma^H S_H V,   S_{i+1} = S_i (1 - p_i)

with the expectations over the forecast's samples (cost.expected_terms; cost None: r10's terms, so v1 can be
re-run through this class). S_H is the candidate's predicted survival.

Selection (r10: the best score, kept unless another beats it by 0.3):
  - commitment: leave the current head only if another beats it by commit_abs + commit_rel |score|, or survives
    with commit_risk more probability. Diagnosis on the 14 failures of v1 (section 6): in 10 of them some head
    alone succeeds; in cell_hold_0044 every head alone succeeds, the planner kept head 0 until 1.9 s and then
    switched among heads 3 / 4 / 1 by score differences of 1-3 out of ~40, mid-dodge, and crashed;
  - fallback: when no candidate survives with fallback_p, take the one most likely to survive (score ignored):
    nothing gets the shape anyway, the clearance is all that is left to save.
"""
from __future__ import annotations

import numpy as np
import torch

from ..imagination.actor_critic import CRASH_PENALTY, GAMMA, action_cost, progress
from ..imagination.episodes import K, N_PTS, POS, ROPE
from ..imagination.feasible import feasible_torch
from ..imagination.planner import _normal_samples, sampled_risk
from ..imagination.r10 import obstacle_inputs
from ..imagination.r10_exec import HeadsPlanner
from .cost import action_cost_v2, expected_terms


class CellPlanner(HeadsPlanner):
    def __init__(self, wm, fc, actor, critic, lim=None, cost=None, commit=None, fallback_p=0.0, **kw):
        """cost: cost.CostCfg or None (r10 score); commit: dict(abs=, rel=, risk=) or None (r10: switch 0.3)."""
        super().__init__(wm, fc, actor, critic, lim, hand=False, **kw)
        self.cost, self.commit, self.fallback_p = cost, commit, fallback_p
        self.last = None

    def choose(self, state, o_list, goal, grip, target_grip, origin, a_prev, audit=None):
        wm, fc, B = self.wm, self.fc, self.actor.heads
        st = tuple(x.expand(B, -1).clone() for x in state)
        g = goal.expand(B, -1)
        o_im = torch.cat(o_list, 0)[None]                                            # (1, 1, 128): one sphere
        mask = torch.ones(B, 1)
        off = torch.as_tensor(np.r_[origin, origin], dtype=torch.float32)
        rows, head = torch.arange(B), torch.arange(B)
        rope, ap = wm.rope_out(st), a_prev.expand(B, -1)
        eps = _normal_samples(self.samples)
        total, alive, disc, first, d_prev = torch.zeros(B), torch.ones(B), 1.0, None, None
        for i in range(self.horizon):
            rf = torch.cat(st[:2], -1)
            a = self.actor(rf, g, obstacle_inputs(fc, o_im.expand(B, -1, -1), rope), mask)[rows, head]
            if first is None:
                first = a.clone()
            a = feasible_torch(a, rope[:, 3 * N_PTS:ROPE] / POS + off, self.lim, steps=K)
            if hasattr(self.lim, 'limit_change_torch'):
                a = self.lim.limit_change_torch(a, ap, K)
            st = wm.img_step(st, a, None, sample=False)
            o_im = fc.step(o_im[:, 0], torch.zeros(1, 7), None)[:, None]
            rope = wm.rope_out(st)
            prog, _ = progress(rope, g)
            w, mu, sd = fc.dist(o_im[:, 0])
            w, mu, sd = w.expand(B, -1), mu.expand(B, -1, -1), sd.expand(B, -1, -1)
            if self.cost is None:
                p, pen = sampled_risk(rope, w, mu, sd, self.samples, self.sd_scale)
                step = prog - pen - CRASH_PENALTY * p - action_cost(a, ap)
            else:
                gate, pen, p, d_prev = expected_terms(rope, w, mu, sd, eps, d_prev, self.cost, self.sd_scale)
                step = prog * gate - pen - CRASH_PENALTY * p - action_cost_v2(a, ap, self.cost)
            total += disc * alive * step
            alive, disc, ap = alive * (1.0 - p), disc * GAMMA, a
        total += disc * alive * self.critic(torch.cat(st[:2], -1), g, obstacle_inputs(fc, o_im.expand(B, -1, -1), rope), mask)
        pick = self._select(total, alive)
        self.prev, self.last = pick, dict(total=total.numpy(), survival=alive.numpy())
        return self.names()[pick], first[pick].numpy().astype(float), total.numpy()

    def _select(self, total, alive):
        pick = int(total.argmax())
        if self.fallback_p and float(alive.max()) < self.fallback_p:
            return int(alive.argmax())
        prev = self.prev
        if prev is None or pick == prev:
            return pick
        gain = float(total[pick] - total[prev])
        if self.commit is None:
            return prev if gain <= self.switch else pick
        need = self.commit["abs"] + self.commit["rel"] * abs(float(total[prev]))
        safer = float(alive[pick] - alive[prev]) >= self.commit["risk"]
        return pick if (gain >= need or safer) else prev
