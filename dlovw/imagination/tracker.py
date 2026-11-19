"""Multi-model obstacle tracker for the G system (docs/notes/geom_tracker_prereg_20261106.md, section 2). Nothing
is learnt.

- Motion models fitted online to the visible readings (the readings are the explicit memory): constant velocity,
  constant acceleration, periodic along an axis, back-and-forth patrol (corridor.CorridorForecast), and a wide
  constant-velocity fallback with a large acceleration noise.
- Model weights: every model is fitted on the history up to HOLD_S ago and scored by the likelihood of the
  readings since then under its own particles (tempered, as the readings' errors are correlated). A model that
  cannot be fitted yet (too little history, no visible extent) gets no weight. The fallback's likelihood is flat,
  so it takes over when every other model predicts badly. These are likelihood weights, not an IMM with Markov
  switching between models.
- Uncertainty: sample() draws whole future trajectories: a model per particle (stratified by weight), then that
  model's parameters, then a rollout with its process noise. Hidden obstacles are simply not updated; the models
  are functions of absolute time, so the prediction runs on through the occlusion.
- Before any reading: positions uniform in a prior box (the caller's workspace), velocities uniform within
  +-PRIOR_V, wide noise.
Time is in control steps (DT = 8 ms) for the interface and seconds inside; positions in metres, any fixed frame.
"""
from __future__ import annotations

import numpy as np

from .corridor import CorridorForecast

DT = 0.008
HOLD_S, HISTORY_S = 0.8, 12.0
MEAS_SD, VEL_SD = 0.004, 0.015          # reading noise per axis (m, m/s)
Q_CV, Q_CA, Q_WIDE = 0.005, 0.002, 0.5  # white acceleration noise density, m^2/s^3 (velocity sd after 1 s: 7, 4.5, 700 mm/s)
CV_WINDOW_S, CA_WINDOW_S = 0.4, 1.2
PERIODS_S = np.geomspace(0.6, 12.0, 80)
PHASE_DIFF = 0.05                       # phase random walk of the periodic model, rad^2/s
PRIOR_V = 0.4
MODELS = ("cv", "ca", "periodic", "patrol", "wide")
PRIOR_W = dict(cv=0.25, ca=0.15, periodic=0.25, patrol=0.25, wide=0.10)
TEMPER = 4.0                            # the hold-out readings count as this many independent ones
SUB_S = 0.04                            # integration step of the noisy rollouts
# Settings that the validation-set tuning may change (geom_tracker_prereg section 4); these defaults are the
# step-1 tracker. cv_tau: the constant-velocity models' velocity decays with this time constant (s);
# floor: least weight of each fitted bounded model (periodic, patrol), before renormalising.
# extent: if set, every particle stays within the box of the positions seen so far grown by this much (m).
DEFAULTS = dict(temper=TEMPER, q_cv=Q_CV, q_ca=Q_CA, q_wide=Q_WIDE, prior_w=PRIOR_W, hold_s=HOLD_S, cv_tau=float("inf"), floor=0.0,
                extent=None)


def _grid(t0, times):
    """Fine grid from t0 covering times, and the index of each time on it."""
    n = int(np.ceil(max(times.max() - t0, 0.0) / SUB_S)) + 1
    g = t0 + np.arange(n + 1) * SUB_S
    return g, np.clip(np.searchsorted(g, times - 1e-9), 0, n)


def _noisy_rollout(rng, n, p0, v0, times, t0, q, acc=None, tau=float("inf")):
    """n trajectories (n, len(times), 3): from p0, v0 (n, 3) at t0 with white acceleration noise q (and a fixed
    acceleration acc (n, 3) if given; the velocity decays with time constant tau), read at times (absolute s)."""
    g, idx = _grid(t0, times)
    p, v = p0.copy(), v0.copy()
    out = np.empty((n, len(g), 3))
    out[:, 0] = p
    for i in range(1, len(g)):
        h = g[i] - g[i - 1]
        a = (0.0 if acc is None else acc) + rng.normal(size=(n, 3)) * np.sqrt(q / h)
        p = p + v * h + 0.5 * a * h * h
        v = (v if tau == float("inf") else v * np.exp(-h / tau)) + a * h
        out[:, i] = p
    return out[:, idx]


def _axis(P):
    c = P.mean(0)
    if np.ptp(P, axis=0).max() < 0.03:
        return c, None
    return c, np.linalg.svd(P - c, full_matrices=False)[2][0]


# ---------------------------------------------------------------------------------------------- models
def fit_cv(T, P, V):
    m = T >= T[-1] - CV_WINDOW_S
    proj = P[m] + V[m] * (T[-1] - T[m])[:, None]          # each reading carried to the last reading's time
    n = m.sum()
    return dict(t0=T[-1], p=proj.mean(0), v=V[m].mean(0), sd_p=MEAS_SD / np.sqrt(n) + 0.002, sd_v=VEL_SD / np.sqrt(n) + 0.01)


def sample_cv(f, rng, n, times, q=Q_CV, tau=float("inf")):
    p0 = f["p"] + rng.normal(size=(n, 3)) * f["sd_p"]
    v0 = f["v"] + rng.normal(size=(n, 3)) * f["sd_v"]
    return _noisy_rollout(rng, n, p0, v0, times, f["t0"], q, tau=tau)


def fit_ca(T, P, V):
    m = T >= T[-1] - CA_WINDOW_S
    if m.sum() < 6 or np.ptp(T[m]) < 0.5:
        return None
    tau = T[m] - T[-1]
    A = np.r_[np.c_[np.ones_like(tau), tau, 0.5 * tau ** 2] / MEAS_SD, np.c_[np.zeros_like(tau), np.ones_like(tau), tau] / VEL_SD]
    B = np.r_[P[m] / MEAS_SD, V[m] / VEL_SD]
    x, *_ = np.linalg.lstsq(A, B, rcond=None)
    cov = np.linalg.pinv(A.T @ A)
    sd = np.sqrt(np.diag(cov))[:, None] * np.ones((1, 3))
    return dict(t0=T[-1], p=x[0], v=x[1], a=x[2], sd=sd)


def sample_ca(f, rng, n, times, q=Q_CA):
    p0 = f["p"] + rng.normal(size=(n, 3)) * f["sd"][0]
    v0 = f["v"] + rng.normal(size=(n, 3)) * f["sd"][1]
    acc = f["a"] + rng.normal(size=(n, 3)) * f["sd"][2]
    return _noisy_rollout(rng, n, p0, v0, times, f["t0"], q, acc)


def fit_periodic(T, P, V):
    c, ax = _axis(P)
    if ax is None or np.ptp(T) < 1.0:
        return None
    s, vs = (P - c) @ ax, V @ ax
    best = None
    for per in PERIODS_S:
        if np.ptp(T) < 0.75 * per:
            break
        w = 2 * np.pi / per
        A = np.r_[np.c_[np.ones_like(T), np.sin(w * T), np.cos(w * T)] / MEAS_SD,
                  np.c_[np.zeros_like(T), w * np.cos(w * T), -w * np.sin(w * T)] / VEL_SD]
        B = np.r_[s / MEAS_SD, vs / VEL_SD]
        x, *_ = np.linalg.lstsq(A, B, rcond=None)
        r = float(((A @ x - B) ** 2).mean())
        if best is None or r < best[0]:
            best = (r, w, x, A)
    if best is None:
        return None
    r, w, x, A = best
    cov = np.linalg.pinv(A.T @ A) * max(r, 1.0)
    perp = P - c - np.outer(s, ax)
    return dict(c=c + x[0] * ax, ax=ax, w=w, ab=x[1:], cov=cov[1:, 1:], sd_c=np.sqrt(cov[0, 0]), rms=np.sqrt(r),
                perp=perp.mean(0), sd_perp=perp.std(0) + 0.002, t0=T[-1])


def sample_periodic(f, rng, n, times):
    ab = f["ab"] + rng.multivariate_normal(np.zeros(2), f["cov"], size=n)
    w = f["w"] * (1.0 + rng.normal(size=n) * (0.03 + 0.02 * min(f["rms"], 3.0)))
    c = f["c"] + rng.normal(size=(n, 1)) * f["sd_c"] * f["ax"]
    g, idx = _grid(f["t0"], times)
    phase = np.cumsum(np.c_[np.zeros(n), rng.normal(size=(n, len(g) - 1)) * np.sqrt(PHASE_DIFF * SUB_S)], 1)[:, idx]
    arg = f["w"] * f["t0"] + w[:, None] * (times[None] - f["t0"]) + phase       # the frequency error acts from the last reading on
    s = ab[:, :1] * np.sin(arg) + ab[:, 1:] * np.cos(arg)
    perp = f["perp"] + rng.normal(size=(n, 1, 3)) * f["sd_perp"]
    return c[:, None] + s[..., None] * f["ax"] + perp


def fit_patrol(steps, P, V):
    cf = CorridorForecast(seed=0)
    for k, p, v in zip(steps, P, V):
        cf.update(int(k), (p, v))
    if not cf.ready() or _axis(P)[1] is None:
        return None
    return dict(cf=cf)


def sample_patrol(f, rng, n, steps):
    """steps: absolute control steps (integers) of the wanted times."""
    cf = f["cf"]
    cf.M, cf.rng = n, rng
    last = cf.hist[-1][0]
    span = int(max(steps.max() - last, 1))
    out = cf.forecast(last, span, 1)                       # (span, n, 3) at last + 1 .. last + span
    idx = np.clip(steps - last - 1, 0, span - 1)
    return out[idx].transpose(1, 0, 2)


# ---------------------------------------------------------------------------------------------- tracker
class MultiModelTracker:
    def __init__(self, seed=0, prior_box=None, config=None):
        self.cfg = dict(DEFAULTS, **(config or {}))
        self.rng = np.random.default_rng(seed)
        self.prior_box = None if prior_box is None else (np.asarray(prior_box[0], float), np.asarray(prior_box[1], float))
        self.steps, self.P, self.V = [], [], []
        self.weights = dict(self.cfg["prior_w"])
        self.fits, self.last_ll = {}, {}

    def update(self, step, meas):
        """meas: (position, velocity) or None while hidden."""
        if meas is None:
            return
        self.steps.append(int(step))
        self.P.append(np.asarray(meas[0], float))
        self.V.append(np.asarray(meas[1], float))
        keep = np.asarray(self.steps) >= step - int(HISTORY_S / DT)
        self.steps = list(np.asarray(self.steps)[keep])
        self.P, self.V = list(np.asarray(self.P)[keep]), list(np.asarray(self.V)[keep])
        self._refit()

    def _fit_all(self, m):
        S, P, V = np.asarray(self.steps)[m], np.asarray(self.P)[m], np.asarray(self.V)[m]
        T = S * DT
        return dict(cv=fit_cv(T, P, V), ca=fit_ca(T, P, V), periodic=fit_periodic(T, P, V), patrol=fit_patrol(S, P, V),
                    wide=fit_cv(T, P, V))

    def _sample_model(self, k, f, n, steps):
        times = np.asarray(steps, float) * DT
        c = self.cfg
        if k == "cv":
            return sample_cv(f, self.rng, n, times, c["q_cv"], c["cv_tau"])
        if k == "wide":
            return sample_cv(f, self.rng, n, times, c["q_wide"], c["cv_tau"])
        if k == "ca":
            return sample_ca(f, self.rng, n, times, c["q_ca"])
        if k == "periodic":
            return sample_periodic(f, self.rng, n, times)
        return sample_patrol(f, self.rng, n, np.asarray(steps))

    def _refit(self):
        S = np.asarray(self.steps)
        c = self.cfg
        old = S <= S[-1] - int(c["hold_s"] / DT)
        new = ~old
        if old.sum() >= 3 and new.sum() >= 2:
            fits = self._fit_all(old)
            ll = {}
            for k in MODELS:
                if fits[k] is None:
                    continue
                X = self._sample_model(k, fits[k], 32, S[new])                 # (32, n_new, 3)
                mu, var = X.mean(0), X.var(0) + MEAS_SD ** 2
                e = np.asarray(self.P)[new] - mu
                ll[k] = float((-0.5 * (e ** 2 / var + np.log(2 * np.pi * var))).sum()) * c["temper"] / new.sum()
            if ll:
                self.last_ll = ll
                top = max(ll.values())
                w = {k: c["prior_w"][k] * np.exp(ll[k] - top) for k in ll}
                tot = sum(w.values())
                self.weights = {k: w.get(k, 0.0) / tot for k in MODELS}
        self.fits = self._fit_all(np.ones(len(S), bool))
        valid = {k: self.weights.get(k, 0.0) for k in MODELS if self.fits[k] is not None}
        if sum(valid.values()) <= 0:
            valid = {k: c["prior_w"][k] for k in MODELS if self.fits[k] is not None}
        tot = sum(valid.values())
        if c["floor"] > 0:
            valid = {k: max(v / tot, c["floor"]) if k in ("periodic", "patrol") else v / tot for k, v in valid.items()}
            tot = sum(valid.values())
        self.active = {k: v / tot for k, v in valid.items()}

    def sample(self, now, horizon, K, M=64):
        """(M, horizon, 3) positions at control steps now + K, ..., now + horizon * K: whole trajectories."""
        steps = now + K * np.arange(1, horizon + 1)
        if not self.steps:
            return self._prior(M, steps)
        names = list(self.active)
        w = np.array([self.active[k] for k in names])
        counts = np.floor(w * M).astype(int)
        rest = M - counts.sum()
        if rest:
            counts += self.rng.multinomial(rest, (w * M - counts) / max((w * M - counts).sum(), 1e-12))
        out = np.concatenate([self._sample_model(k, self.fits[k], c, steps) for k, c in zip(names, counts) if c > 0], 0)
        if self.cfg["extent"] is not None:
            P = np.asarray(self.P)
            out = np.clip(out, P.min(0) - self.cfg["extent"], P.max(0) + self.cfg["extent"])
        return out

    def _prior(self, M, steps):
        if self.prior_box is None:
            raise ValueError("no reading yet and no prior box")
        lo, hi = self.prior_box
        p0 = self.rng.uniform(lo, hi, size=(M, 3))
        v0 = self.rng.uniform(-PRIOR_V, PRIOR_V, size=(M, 3))
        return _noisy_rollout(self.rng, M, p0, v0, np.asarray(steps, float) * DT, (steps[0] - 1) * DT, self.cfg["q_wide"])
