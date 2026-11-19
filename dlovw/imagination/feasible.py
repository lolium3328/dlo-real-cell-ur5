"""Make the actor's action satisfy the safety filter's input constraints before anything else sees it
(docs/notes/imagination_ac_design_20261022.md, section 11).

The legacy CBF-QP (control/cbf_qp.py) has, besides the obstacle rows, three input constraints:
- workspace: each gripper coordinate stays inside its box;
- stretch: the gripper distance stays below stretch_ratio x cable length;
- step: per gripper |dx| + |dy| + |dz| <= max_translation_step_m (an L1 ball, inside the L2 cap).
In r5 the actor's raw command broke one of them on most steps (L1 step 44%, workspace 28%, stretch 14%)
and every large CBF change was one of these, far from any obstacle. Imagination had none of them, so the
world model was asked about motions that never happen. The same differentiable map is now applied in
imagination (torch, on decoded gripper positions) and in execution (numpy, on measured ones); the CBF
is left with the obstacles. Actions are in units of the step cap; grippers in metres (world frame).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import torch


@dataclasses.dataclass(frozen=True)
class Limits:
    ws_low: tuple          # 6: left xyz, right xyz (m)
    ws_high: tuple
    stretch_max: float     # m
    cap: float             # m per control step (L1 per gripper)
    fixed_right: bool = False   # the right end cannot move (cell 2026-11-12: a clamp): the left end takes all the stretch


def limits_of(session):
    c = session.safety.filter.config
    return Limits(tuple(float(x) for x in c.workspace_low), tuple(float(x) for x in c.workspace_high),
                  float(c.stretch_ratio) * float(session.state().L), float(c.max_translation_step_m))


def feasible_torch(a, grip, lim, steps=1, margin=1e-4):
    """a (B, 6) in cap units (per control step), grip (B, 6) metres -> feasible a. Workspace clamp and
    stretch projection of the displacement over `steps` control steps (a whole decision in imagination,
    one step in execution), then per-gripper L1 scaling of the per-step move (it only shrinks moves)."""
    if hasattr(lim, 'project_torch'):
        return lim.project_torch(a, grip, steps, margin)
    lo = torch.as_tensor(lim.ws_low, dtype=a.dtype, device=a.device) + margin
    hi = torch.as_tensor(lim.ws_high, dtype=a.dtype, device=a.device) - margin
    u = a * lim.cap * steps
    u = torch.minimum(torch.maximum(grip + u, lo), hi) - grip
    d = (grip[:, :3] + u[:, :3]) - (grip[:, 3:] + u[:, 3:])
    n = d.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    excess = torch.relu(n - (lim.stretch_max - margin)) * d / n
    if lim.fixed_right:
        u = torch.cat([u[:, :3] - excess, torch.zeros_like(u[:, 3:])], -1) / steps
    else:
        u = torch.cat([u[:, :3] - excess / 2, u[:, 3:] + excess / 2], -1) / steps
    u = u.reshape(-1, 2, 3)
    l1 = u.abs().sum(-1, keepdim=True)
    u = u * torch.clamp(lim.cap / l1.clamp_min(1e-12), max=1.0)
    return u.reshape(-1, 6) / lim.cap


def feasible_np(a, grip, lim, steps=1):
    t = feasible_torch(torch.as_tensor(np.asarray(a, float)[None]), torch.as_tensor(np.asarray(grip, float)[None]), lim, steps)
    return t[0].numpy()
