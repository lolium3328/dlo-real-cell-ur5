"""Ablation a2' arm R-D (docs/notes/a2prime_prereg_20261108.md section 3): the RSSM drives imagination and execution
unchanged, but the actor and critic read the rope and grippers decoded from (h, z) (123 numbers, as StepWM's state)
instead of (h, z) itself.

RopeInputWM wraps a FactoredWM. Its states are flat tuples (rope, empty, h, z, o) with rope = dec_rope(h, z), so every
place that builds the actor's rope input as torch.cat(state[:2], -1) (r10.py imagination and collection, r10_exec.py
planner, train_r10.py demo imitation) gets the decoded rope with no change there. Every step calls the wrapped
model's own step (same random draws), so imagination and execution are those of the RSSM. The world model is trained
and saved through `.core` (the plain FactoredWM): losses, optimiser and checkpoint weights are those of arm R-L.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .episodes import ROPE


class RopeInputWM(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core
        self.h_dim, self.z_dim = ROPE, 0

    def wrap(self, s):
        """(h, z, o) -> (rope, empty, h, z, o); works for (B, .) and (B, T, .) states."""
        h, z, o = s
        rope = self.core.dec_rope(torch.cat([h, z], -1))
        return (rope, rope.new_zeros(*rope.shape[:-1], 0), h, z, o)

    @staticmethod
    def unwrap(s):
        return s[2:]

    def initial(self, B, device="cpu"):
        return self.wrap(self.core.initial(B, device))

    def obs_step(self, state, a, obs):
        s, p, q = self.core.obs_step(self.unwrap(state), a, obs)
        return self.wrap(s), p, q

    def img_step(self, state, a, meas=None, sample=True):
        return self.wrap(self.core.img_step(self.unwrap(state), a, meas, sample=sample))

    def rope_out(self, state):
        return state[0]

    def observe(self, obs, act_prev):
        s, p, q = self.core.observe(obs, act_prev)
        return self.wrap(s), p, q


def wrap_loss(loss_fn):
    """A world-model loss on the core model whose returned posterior states are wrapped for the actor."""
    def f(wm, b, **kw):
        loss, m, states = loss_fn(wm.core, b, **kw)
        return loss, m, wm.wrap(states)
    return f
