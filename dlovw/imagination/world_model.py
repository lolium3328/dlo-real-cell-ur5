"""Factored world model (docs/notes/imagination_ac_design_20261022.md, sections 5 and 10).

Rope part: a state RSSM (deterministic GRU h + Gaussian z) that sees only the rope and the grippers and
is driven by our actions; the posterior sees the current rope, the prior does not.
Obstacle part: a GRU filter whose only input is the obstacle measurement (zeros while hidden). It
cannot see which scene it is in, so it has to track the measurement and learn the motion from its own
history; in imagination it gets "hidden" inputs, exactly like an occluded stretch. (Run r1 used one
joint RSSM for both: it memorised the training scenes' obstacle motions, 24 mm decoding error on
training scenes vs 81 mm on test scenes while the obstacle was visible.)
The obstacle filter outputs a distribution, not a point: a two-component Gaussian mixture over the
position (an obstacle that may or may not have turned behind the occluder is bimodal) and a Gaussian
over the velocity, trained by likelihood on the present and on blind forecasts. Knowing that it does
not know where the obstacle is, is itself information for the decision.
A state is the tuple (h, z, o); `feat` concatenates it with the obstacle distribution for the actor and critic.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .episodes import ACT, OBS, POS, ROPE, TRUTH

MEAS = OBS - ROPE                      # obstacle measurement block of obs (position, velocity, visible)
OBST_OUT = 2 + 2 * 3 + 2 * 3 + 3 + 3   # mixture logits, means, scales; velocity mean, scale
STD_MIN = 0.01                         # 1 mm in the scaled frame


def mlp(i, hid, o, n=2):
    layers, d = [], i
    for _ in range(n):
        layers += [nn.Linear(d, hid), nn.LayerNorm(hid), nn.ELU()]
        d = hid
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


def kl_normal(qm, qs, pm, ps):
    return torch.log(ps / qs) + (qs ** 2 + (qm - pm) ** 2) / (2 * ps ** 2) - 0.5


class FactoredWM(nn.Module):
    def __init__(self, h=256, z=32, hid=256, ho=128, expose_sd=True):
        super().__init__()
        self.h_dim, self.z_dim, self.o_dim, self.expose_sd = h, z, ho, expose_sd
        self.enc = mlp(ROPE, hid, hid)
        self.inp = nn.Sequential(nn.Linear(z + ACT, hid), nn.ELU())
        self.gru = nn.GRUCell(hid, h)
        self.prior = mlp(h, hid, 2 * z, 1)
        self.post = mlp(h + hid, hid, 2 * z, 1)
        self.dec_rope = mlp(h + z, 512, ROPE)
        self.o_inp = nn.Sequential(nn.Linear(MEAS, 64), nn.ELU())
        self.o_gru = nn.GRUCell(64, ho)
        self.dec_obst = mlp(ho, 128, OBST_OUT)

    @property
    def feat_dim(self):
        return self.h_dim + self.z_dim + self.o_dim + OBST_OUT

    def initial(self, B, device):
        return (torch.zeros(B, self.h_dim, device=device), torch.zeros(B, self.z_dim, device=device),
                torch.zeros(B, self.o_dim, device=device))

    def obstacle_dist(self, o):
        """weights (.., 2), means (.., 2, 3), scales (.., 2, 3), velocity mean (.., 3) and scale (.., 3)."""
        x = self.dec_obst(o)
        lead = x.shape[:-1]
        w = torch.softmax(x[..., :2], -1)
        mu = x[..., 2:8].reshape(*lead, 2, 3)
        sd = F.softplus(x[..., 8:14]).reshape(*lead, 2, 3) + STD_MIN
        return w, mu, sd, x[..., 14:17], F.softplus(x[..., 17:20]) + STD_MIN

    def obstacle_params(self, o):
        """The obstacle distribution as actor/critic input. With expose_sd=False (ablation) only the point
        estimate: both means set to the mixture mean, weights and scales zeroed (same width)."""
        w, mu, sd, vm, vs = self.obstacle_dist(o)
        lead = w.shape[:-1]
        if not self.expose_sd:
            mean = (w[..., None] * mu).sum(-2)
            z = torch.zeros(*lead, 6, device=o.device)
            return torch.cat([torch.zeros_like(w), mean, mean, z, vm, torch.zeros_like(vm)], -1)
        return torch.cat([w, mu.reshape(*lead, 6), torch.log(sd).reshape(*lead, 6), vm, torch.log(vs)], -1)

    def feat(self, state):
        return torch.cat([*state, self.obstacle_params(state[2])], -1)

    def rope_out(self, state):
        return self.dec_rope(torch.cat(state[:2], -1))

    def obstacle_out(self, state):
        """Point estimate (mixture mean position, velocity mean) and the position spread (rms, scaled)."""
        w, mu, sd, vm, _ = self.obstacle_dist(state[2])
        mean = (w[..., None] * mu).sum(-2)
        var = (w[..., None] * (sd ** 2 + mu ** 2)).sum(-2) - mean ** 2
        return torch.cat([mean, vm], -1), var.clamp_min(0).mean(-1).sqrt()

    @staticmethod
    def _stats(x):
        m, s = x.chunk(2, -1)
        return m, F.softplus(s) + 0.1

    def _obstacle_step(self, o, meas):
        return self.o_gru(self.o_inp(meas), o)

    def img_step(self, state, a, meas=None, sample=True):
        """Prior step. `meas` (B, MEAS): the obstacle sensor reading to feed the filter (the obstacle moves
        on its own, so its future readings can be replayed); None = nothing seen. sample=False uses the
        prior mean (deterministic scores for planning)."""
        h, z, o = state
        h = self.gru(self.inp(torch.cat([z, a], -1)), h)
        m, s = self._stats(self.prior(h))
        meas = torch.zeros(len(o), MEAS, device=o.device) if meas is None else meas
        return (h, m + s * torch.randn_like(s) if sample else m, self._obstacle_step(o, meas))

    def obs_step(self, state, a, obs):
        h, z, o = state
        h = self.gru(self.inp(torch.cat([z, a], -1)), h)
        pm, ps = self._stats(self.prior(h))
        qm, qs = self._stats(self.post(torch.cat([h, self.enc(obs[:, :ROPE])], -1)))
        return (h, qm + qs * torch.randn_like(qs), self._obstacle_step(o, obs[:, ROPE:])), (pm, ps), (qm, qs)

    def observe(self, obs, act_prev):
        """Posterior states over (B, T): ((h, z, o) each (B, T, .), prior stats, posterior stats)."""
        B, T, _ = obs.shape
        state, out = self.initial(B, obs.device), []
        for t in range(T):
            state, p, q = self.obs_step(state, act_prev[:, t], obs[:, t])
            out.append((state, p, q))
        stack = lambda f: torch.stack([f(x) for x in out], 1)  # noqa: E731
        states = tuple(stack(lambda x, i=i: x[0][i]) for i in range(3))
        return states, (stack(lambda x: x[1][0]), stack(lambda x: x[1][1])), (stack(lambda x: x[2][0]), stack(lambda x: x[2][1]))


def mixture_nll(w, mu, sd, x):
    """-log p(x) under the diagonal two-component mixture; x (.., 3)."""
    comp = (-0.5 * ((x[..., None, :] - mu) / sd) ** 2 - torch.log(sd)).sum(-1)
    return -torch.logsumexp(torch.log(w + 1e-8) + comp, -1)


def gauss_nll(m, s, x):
    return (0.5 * ((x - m) / s) ** 2 + torch.log(s)).sum(-1)


def wm_loss(wm, b, ms_steps=8, obst_steps=32, free_nats=1.0, w_kl=1.0, w_ms=1.0, w_obst=0.1, rec_scale=1.0):
    """b: dict of tensors from episodes.batch. Returns (loss, metrics, posterior states).
    rec_scale multiplies the rope reconstruction and prediction errors (they are means over 123
    coordinates, about 100x smaller than the KL term otherwise). Episodes without an obstacle
    (b["has_obst"] = 0, e.g. the old motion-diverse data) do not enter the obstacle losses. With b["rope_truth"]
    (B, T, ROPE) the rope terms are scored against it instead of obs (the cell's perceived rope is noisy)."""
    obs, act_prev, truth, mask, future = b["obs"], b["act_prev"], b["truth"], b["mask"], b["future"]
    omask = mask * b["has_obst"][:, None] if "has_obst" in b else mask
    (hs, zs, os_), (pm, ps), (qm, qs) = wm.observe(obs, act_prev)
    n = mask.sum().clamp_min(1.0)
    msum = lambda x, m=mask: (x * m).sum() / m.sum().clamp_min(1.0)  # noqa: E731
    # cell 2026-11-12: obs holds the perceived rope, b["rope_truth"] the true one (the reconstruction target)
    rope_t = b["rope_truth"] if "rope_truth" in b else obs[..., :ROPE]
    rec = ((wm.dec_rope(torch.cat([hs, zs], -1)) - rope_t) ** 2).mean(-1)
    kl_prior = kl_normal(qm.detach(), qs.detach(), pm, ps).sum(-1)
    kl_post = kl_normal(qm, qs, pm.detach(), ps.detach()).sum(-1)
    kl = 0.8 * kl_prior.clamp_min(free_nats) + 0.2 * kl_post.clamp_min(free_nats)
    w_o, mu_o, sd_o, vm_o, vs_o = wm.obstacle_dist(os_)
    filt = mixture_nll(w_o, mu_o, sd_o, truth[..., :3]) + gauss_nll(vm_o, vs_o, truth[..., 3:])
    point, spread = wm.obstacle_out((None, None, os_))
    filt_pos = ((point[..., :3] - truth[..., :3]) ** 2).sum(-1).detach()
    inside = (((point[..., :3] - truth[..., :3]) ** 2).sum(-1).sqrt() < 2 * spread * 3 ** 0.5).float().detach()
    omsum = lambda x: (x * omask).sum() / omask.sum().clamp_min(1.0)  # noqa: E731
    # rope: open-loop prior rollouts with the recorded actions, from every posterior state
    T, M = obs.shape[1], ms_steps
    ms_rope = last_rope = torch.zeros((), device=obs.device)
    if T > M:
        h, z = hs[:, :T - M].reshape(-1, wm.h_dim), zs[:, :T - M].reshape(-1, wm.z_dim)
        msk = torch.zeros((), device=obs.device)
        for i in range(1, M + 1):
            h = wm.gru(wm.inp(torch.cat([z, act_prev[:, i:T - M + i].reshape(-1, ACT)], -1)), h)
            mu, sd = wm._stats(wm.prior(h))
            z = mu + sd * torch.randn_like(sd)
            m = mask[:, i:T - M + i].reshape(-1)
            err = ((wm.dec_rope(torch.cat([h, z], -1)) - rope_t[:, i:T - M + i].reshape(-1, ROPE)) ** 2).mean(-1)
            ms_rope, msk = ms_rope + (err * m).sum(), msk + m.sum()
        ms_rope, last_rope = ms_rope / msk.clamp_min(1.0), (err * m).sum() / m.sum().clamp_min(1.0)
    # obstacle: from every filtered state, predict blind (hidden inputs) against the true future track
    o = os_.reshape(-1, wm.o_dim)
    fut = future.reshape(-1, future.shape[2], 3)
    m_o = omask.reshape(-1)
    pred, pred_pos, pred_sd = torch.zeros((), device=obs.device), [], []
    for i in range(min(obst_steps, fut.shape[1])):
        o = wm._obstacle_step(o, torch.zeros(len(o), MEAS, device=o.device))
        w_i, mu_i, sd_i, _, _ = wm.obstacle_dist(o)
        pred = pred + (mixture_nll(w_i, mu_i, sd_i, fut[:, i]) * m_o).sum()
        with torch.no_grad():
            pt, sp = wm.obstacle_out((None, None, o))
            pred_pos.append((((pt[:, :3] - fut[:, i]) ** 2).sum(-1) * m_o).sum() / m_o.sum().clamp_min(1.0))
            pred_sd.append((sp * m_o).sum() / m_o.sum().clamp_min(1.0))
    pred = pred / (m_o.sum().clamp_min(1.0) * max(len(pred_pos), 1))
    loss = msum(rec_scale * rec + w_kl * kl) + w_ms * rec_scale * ms_rope + w_obst * (omsum(filt) + pred)
    rms = lambda x: float(torch.sqrt(x.detach()) / POS * 1000)  # noqa: E731  (mm)
    metrics = dict(wm_loss=float(loss.detach()), rope_mm=rms(msum(rec)), kl=float(msum(kl_prior).detach()),
                   ms_rope_mm=rms(ms_rope), ms_last_rope_mm=rms(last_rope), obstacle_filter_mm=rms(omsum(filt_pos)),
                   obstacle_spread_mm=float(omsum(spread.detach()) * 3 ** 0.5 / POS * 1000), obstacle_within_2sd=float(omsum(inside)),
                   obstacle_nll=float(omsum(filt).detach()),
                   obstacle_pred_8_mm=rms(pred_pos[7]) if len(pred_pos) > 7 else None,
                   obstacle_pred_8_spread_mm=float(pred_sd[7] * 3 ** 0.5 / POS * 1000) if len(pred_sd) > 7 else None,
                   obstacle_pred_32_mm=rms(pred_pos[-1]) if pred_pos else None,
                   obstacle_pred_32_spread_mm=float(pred_sd[-1] * 3 ** 0.5 / POS * 1000) if pred_sd else None)
    return loss, metrics, (hs.detach(), zs.detach(), os_.detach())
