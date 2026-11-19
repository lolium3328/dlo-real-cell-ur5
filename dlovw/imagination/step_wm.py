"""Ablation a2 (docs/notes/ablation_prereg_20261106.md): a rope world model without latent and without memory. The
state is the observed rope and grippers (the first ROPE numbers of an observation, scaled episode frame); one decision
ahead is x + f(x, a) with an MLP f. It has FactoredWM's interface where r10 uses it (state tuples (x, z, o) with an
empty z and a dummy obstacle slot, rope_out, obs_step, img_step, observe), so r10's training and execution run on it
unchanged. step_wm_loss: the open-loop error over ms_steps decisions with the recorded actions, from every step.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .episodes import ACT, N_PTS, POS, ROPE
from .world_model import mlp


class StepWM(nn.Module):
    def __init__(self, hid=512, layers=3):
        super().__init__()
        self.kw = dict(hid=hid, layers=layers)
        self.f = mlp(ROPE + ACT, hid, ROPE, layers)
        self.h_dim, self.z_dim = ROPE, 0

    def initial(self, B, device="cpu"):
        return (torch.zeros(B, ROPE, device=device), torch.zeros(B, 0, device=device), torch.zeros(B, 1, device=device))

    def _state(self, x):
        lead = x.shape[:-1]
        return (x, x.new_zeros(*lead, 0), x.new_zeros(*lead, 1))

    def obs_step(self, state, a_prev, obs):
        """The observation replaces the state (no memory); a_prev is not used."""
        return (self._state(obs[..., :ROPE]),)

    def img_step(self, state, a, feats=None, sample=True):
        x = state[0]
        return self._state(x + self.f(torch.cat([x, a], -1)))

    def rope_out(self, state):
        return state[0]

    def observe(self, obs, act_prev):
        return (self._state(obs[..., :ROPE]),)


def step_wm_loss(wm, b, ms_steps=8, rec_scale=1.0):
    """(loss, metrics, states) as world_model.wm_loss: mean squared open-loop error of the rope and grippers over
    ms_steps decisions (recorded actions) from every step of every episode."""
    obs, act_prev, mask = b["obs"], b["act_prev"], b["mask"]
    states = wm.observe(obs, act_prev)[0]
    T, M = obs.shape[1], ms_steps
    loss, first, last = torch.zeros((), device=obs.device), None, None
    if T > M:
        x = obs[:, :T - M, :ROPE].reshape(-1, ROPE)
        tot, cnt = torch.zeros((), device=obs.device), torch.zeros((), device=obs.device)
        for i in range(1, M + 1):
            x = x + wm.f(torch.cat([x, act_prev[:, i:T - M + i].reshape(-1, ACT)], -1))
            m = mask[:, i:T - M + i].reshape(-1)
            err = ((x - obs[:, i:T - M + i, :ROPE].reshape(-1, ROPE)) ** 2).mean(-1)
            tot, cnt = tot + (err * m).sum(), cnt + m.sum()
            pts = (x - obs[:, i:T - M + i, :ROPE].reshape(-1, ROPE))[:, :3 * N_PTS].reshape(-1, N_PTS, 3).norm(dim=-1).mean(-1) / POS * 1000
            mm = float((pts * m).sum() / m.sum().clamp_min(1.0))
            first = mm if i == 1 else first
            last = mm
        loss = rec_scale * tot / cnt.clamp_min(1.0)
    return loss, dict(wm_loss=float(loss.detach()), rope_mm=first, ms_last_rope_mm=last), states
