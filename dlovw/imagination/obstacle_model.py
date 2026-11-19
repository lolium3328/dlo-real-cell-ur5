"""Standalone obstacle forecaster trained on synthetic obstacle tracks (docs/notes/imagination_ac_design_20261022.md,
section 21). Same output as the world model's obstacle part (a two-component Gaussian mixture over the
position, scaled episode frame), so the planner can use either. Input per decision: the sensor reading (7)
and, optionally, the corridor features (16, obstacle_data.corridor_features). Blind forecasts feed "not
seen" readings and keep the corridor features of the last decision.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .episodes import K, POS, VEL
from .obstacle_contract import FEATS
from .world_model import STD_MIN, mixture_nll, mlp

MEAS = 7
DT = 0.008


class ObstacleForecaster(nn.Module):
    def __init__(self, feats=True, hid=128):
        super().__init__()
        self.use_feats, self.hid = feats, hid
        self.inp = nn.Sequential(nn.Linear(MEAS + (FEATS if feats else 0), 128), nn.ELU())
        self.gru = nn.GRUCell(128, hid)
        self.head = mlp(hid, 128, 2 + 6 + 6)

    def initial(self, B, device="cpu"):
        return torch.zeros(B, self.hid, device=device)

    def step(self, o, meas, feats):
        x = torch.cat([meas, feats], -1) if self.use_feats else meas
        return self.gru(self.inp(x), o)

    def dist(self, o):
        x = self.head(o)
        lead = x.shape[:-1]
        return (torch.softmax(x[..., :2], -1), x[..., 2:8].reshape(*lead, 2, 3),
                F.softplus(x[..., 8:14]).reshape(*lead, 2, 3) + STD_MIN)

    def blind(self, o, feats, H):
        """Mixture parameters for 1..H decisions ahead with nothing seen; feats frozen."""
        zero, out = torch.zeros(len(o), MEAS, device=o.device), []
        for _ in range(H):
            o = self.step(o, zero, feats)
            out.append(self.dist(o))
        return out


def forecaster_loss(model, b, H=32):
    """b: meas (B,T,7), feats (B,T,FEATS), truth (B,T,3), track (B,T+H,3), mask (B,T)."""
    B, T, _ = b["meas"].shape
    o, states = model.initial(B, b["meas"].device), []
    for t in range(T):
        o = model.step(o, b["meas"][:, t], b["feats"][:, t])
        states.append(o)
    S = torch.stack(states, 1)                                  # (B, T, hid)
    m = b["mask"]
    w, mu, sd = model.dist(S)
    filt = (mixture_nll(w, mu, sd, b["truth"]) * m).sum() / m.sum()
    o, f = S.reshape(B * T, -1), b["feats"].reshape(B * T, -1)
    zero = torch.zeros(B * T, MEAS, device=o.device)
    idx = torch.arange(T, device=o.device)
    blind, errs = 0.0, []
    for i in range(1, H + 1):
        o = model.step(o, zero, f)
        w, mu, sd = model.dist(o)
        target = b["track"][:, idx + i].reshape(B * T, 3)
        blind = blind + (mixture_nll(w, mu, sd, target) * m.reshape(-1)).sum() / m.sum()
        if i in (5, 10, 15):
            with torch.no_grad():
                mean = (w[..., None] * mu).sum(-2)
                errs.append(((mean - target).norm(dim=-1) / 10.0 * 1000).reshape(B, T)[m > 0].median())
    return filt + blind / H, dict(filt_nll=float(filt.detach()), blind_nll=float(blind.detach() / H),
                                  err_04_mm=float(errs[0]), err_08_mm=float(errs[1]), err_12_mm=float(errs[2]))


class PriorSwitch:
    """Diagnosis (docs/notes/perception_robustness_20261023.md, section 14): forecasts come from `prior` until the
    first decision with an obstacle reading and from `main` afterwards; both filters see every input."""

    def __init__(self, main, prior):
        self.main, self.prior = main, prior

    def initial(self, B, device="cpu"):
        return self.main.initial(B, device), self.prior.initial(B, device), torch.zeros(B, dtype=torch.bool, device=device)

    def step(self, o, meas, feats):
        om, op, seen = o
        return self.main.step(om, meas, feats), self.prior.step(op, meas, feats), seen | (meas[:, -1] > 0)

    def blind(self, o, feats, H):
        om, op, seen = o
        return self.main.blind(om, feats, H) if bool(seen.all()) else self.prior.blind(op, feats, H)


class ConstantVelocityForecaster:
    """Memoryless baseline (docs/notes/memory_vs_cv_prereg_20261027.md): from the last reading only, the centre
    moves on at the reading's velocity; the per-axis spread depends only on the time since that reading and is
    fitted on the same synthetic training sequences as ObstacleForecaster (scripts/imagination/fit_cv.py).
    Before any reading: the mean and spread of the obstacle position over the training sequences. Same interface
    as ObstacleForecaster (batch 1); one Gaussian (the second mixture component duplicates it with weight 0)."""

    def __init__(self, sd, prior_mu, prior_sd, tau=float("inf"), coef=None):
        """tau (decisions): the velocity's influence decays, the centre moves v * tau * (1 - exp(-L / tau)) after
        L decisions; inf = constant velocity, small = stays where it was last seen. coef (L, 3 axes, 3): instead,
        the best affine prediction from the last reading per lag L and axis, a * p + c * v + b (fit_cv.py)."""
        self.sd = torch.as_tensor(sd, dtype=torch.float32)                      # (L, 3), scaled frame
        self.prior_mu = torch.as_tensor(prior_mu, dtype=torch.float32)
        self.prior_sd = torch.as_tensor(prior_sd, dtype=torch.float32)
        self.tau = float(tau)
        self.coef = None if coef is None else torch.as_tensor(coef, dtype=torch.float32)

    @classmethod
    def load(cls, path):
        st = torch.load(path, map_location="cpu")
        return cls(st["sd"], st["prior_mu"], st["prior_sd"], float(st.get("tau", float("inf"))), st.get("coef"))

    def gain(self, L):
        return L if self.tau == float("inf") else self.tau * (1.0 - np.exp(-L / self.tau))

    def initial(self, B, device="cpu"):
        return ()                      # no reading yet; (p, v, age) after one. Not None: diagnose treats None as "no forecaster"

    def step(self, o, meas, feats):
        m = meas[0]
        if m[-1] > 0:
            return m[:3].clone(), m[3:6] / VEL * POS * (K * DT), 0               # velocity in scaled units per decision
        return o if not o else (o[0], o[1], o[2] + 1)

    def blind(self, o, feats, H):
        out, w = [], torch.tensor([[1.0, 0.0]])
        for i in range(1, H + 1):
            if not o:
                mu, sd = self.prior_mu, self.prior_sd
            else:
                L = min(o[2] + i, len(self.sd) - 1)
                if self.coef is not None:
                    c = self.coef[L]
                    mu = c[:, 0] * o[0] + c[:, 1] * o[1] + c[:, 2]
                else:
                    mu = o[0] + o[1] * self.gain(o[2] + i)
                sd = self.sd[L]
            out.append((w, mu.expand(1, 2, 3).clone(), sd.expand(1, 2, 3).clone()))
        return out


class MemorylessForecaster(nn.Module):
    """Ablations a1 / b (docs/notes/ablation_prereg_20261106.md): a batched forecaster with ObstacleForecaster's
    interface but no history. The state is the last reading only, (position, velocity, decisions since it, seen):
    8 numbers, scaled episode frame. dist() is the best affine prediction from that reading at the state's lag
    (fit_cv.py: per lag and axis, a_p p + a_v v + b, with the residual spread), or the prior before any reading.
    use_velocity=False (ablation b): the velocity is dropped from the state and the fit has a_v = 0 (pos.pt)."""

    STATE = 8

    def __init__(self, coef, sd, prior_mu, prior_sd, use_velocity=True):
        super().__init__()
        self.use_velocity = bool(use_velocity)
        self.register_buffer("coef", torch.as_tensor(coef, dtype=torch.float32))      # (L + 1, 3, 3)
        self.register_buffer("sd", torch.as_tensor(sd, dtype=torch.float32))          # (L + 1, 3)
        self.register_buffer("prior_mu", torch.as_tensor(prior_mu, dtype=torch.float32))
        self.register_buffer("prior_sd", torch.as_tensor(prior_sd, dtype=torch.float32))

    @classmethod
    def load(cls, path, use_velocity=True):
        st = torch.load(path, map_location="cpu")
        return cls(st["coef"], st["sd"], st["prior_mu"], st["prior_sd"], use_velocity)

    def initial(self, B, device="cpu"):
        return torch.zeros(B, self.STATE, device=device)

    def step(self, o, meas, feats=None):
        seen = meas[..., 6:7] > 0
        v = meas[..., 3:6] / VEL * POS * (K * DT) if self.use_velocity else torch.zeros_like(meas[..., 3:6])
        new = torch.cat([meas[..., :3], v, torch.zeros_like(seen, dtype=o.dtype), torch.ones_like(seen, dtype=o.dtype)], -1)
        old = torch.cat([o[..., :6], o[..., 6:7] + o[..., 7:8], o[..., 7:8]], -1)     # lag counts only once seen
        return torch.where(seen, new, old)

    def dist(self, o):
        L = o[..., 6].round().long().clamp(0, len(self.coef) - 1)
        c, sd = self.coef[L], self.sd[L]                                             # (..., 3, 3), (..., 3)
        mu = c[..., 0] * o[..., :3] + c[..., 1] * o[..., 3:6] + c[..., 2]
        seen = o[..., 7:8] > 0
        mu = torch.where(seen, mu, self.prior_mu.expand_as(mu))
        sd = torch.where(seen, sd, self.prior_sd.expand_as(sd))
        lead = o.shape[:-1]
        w = torch.tensor([1.0, 0.0], device=o.device).expand(*lead, 2)
        return w, mu[..., None, :].expand(*lead, 2, 3), sd[..., None, :].expand(*lead, 2, 3)

    def blind(self, o, feats, H):
        zero, out = torch.zeros(*o.shape[:-1], MEAS, device=o.device), []
        for _ in range(H):
            o = self.step(o, zero)
            out.append(self.dist(o))
        return out
