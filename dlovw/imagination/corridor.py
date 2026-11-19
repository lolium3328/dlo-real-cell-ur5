"""Corridor prior for the obstacle forecast (docs/notes/imagination_ac_design_20261022.md, section 20).

Physics prior: the obstacle moves back and forth along a fixed line segment (its "corridor") at bounded
speed and turns round, possibly after a pause, at the segment ends. From the visible measurement history
the forecaster estimates the axis (principal direction), the centre, the observed extent, the turning
points seen so far and the typical peak speed. What is not known is carried by particles:
- where an end that has not been seen yet lies (somewhere beyond the observed extent);
- the cruise speed (a fraction of the observed peak speed) and the pause at the ends;
- while the obstacle is hidden, where along the corridor it is now (the particles keep moving).
Nothing is learnt. The priors are deliberately wider than the scene generator's ranges: speed <= 0.5 m/s,
pause 0-1.5 s, total extent <= 0.4 m.
"""
from __future__ import annotations

import numpy as np

DT = 0.008                      # control period (s)
V_MAX, DWELL_MAX, EXTENT_MAX = 0.5, 1.5, 0.40
SIM_DT = 0.02                   # particle integration step (s)
TURN_SPEED = 0.03               # along-axis speed above which the direction counts as known (m/s)


class CorridorForecast:
    def __init__(self, n_particles=48, seed=0, min_visible=5):
        self.M, self.min_visible = n_particles, min_visible
        self.rng = np.random.default_rng(seed)
        self.hist = []          # (control step, position, velocity) of the visible measurements

    def update(self, step, meas):
        if meas is not None:
            self.hist.append((int(step), np.asarray(meas[0], float), np.asarray(meas[1], float)))

    def ready(self):
        return len(self.hist) >= self.min_visible

    def _frame(self):
        P = np.array([h[1] for h in self.hist])
        V = np.array([h[2] for h in self.hist])
        c = P.mean(0)
        if np.ptp(P, axis=0).max() > 0.02:
            axis = np.linalg.svd(P - c, full_matrices=False)[2][0]
        else:                    # barely moved yet: the velocity tells the direction
            v = V[-1] if np.linalg.norm(V[-1]) > 1e-6 else np.array([1.0, 0.0, 0.0])
            axis = v / np.linalg.norm(v)
        s, vs = (P - c) @ axis, V @ axis
        return c, axis, s, vs

    def _ends(self, s, vs):
        """Observed extent and whether each end has been seen as a turning point."""
        lo, hi = float(s.min()), float(s.max())
        steps = np.array([h[0] for h in self.hist])
        seen_lo = seen_hi = False
        sign = np.sign(vs) * (np.abs(vs) > TURN_SPEED)
        moving = np.flatnonzero(sign != 0)
        for a, b in zip(moving[:-1], moving[1:]):
            if sign[a] != sign[b] and (steps[b] - steps[a]) * DT < 2.0:
                turn = s[a:b + 1].max() if sign[a] > 0 else s[a:b + 1].min()
                seen_hi |= sign[a] > 0 and turn > hi - 0.02
                seen_lo |= sign[a] < 0 and turn < lo + 0.02
        return lo, hi, seen_lo, seen_hi

    def forecast(self, now, horizon, K):
        """(horizon, M, 3) particle positions (world frame) at now + K*i control steps, i = 1..horizon."""
        c, axis, s, vs = self._frame()
        lo, hi, seen_lo, seen_hi = self._ends(s, vs)
        M, rng = self.M, self.rng
        room = max(EXTENT_MAX - (hi - lo), 0.0)
        open_ends = (not seen_lo) + (not seen_hi)
        ext = rng.uniform(0.0, room / max(open_ends, 1), size=(2, M))
        end_lo = np.where(seen_lo, lo + rng.normal(0, 0.005, M), lo - ext[0])
        end_hi = np.where(seen_hi, hi + rng.normal(0, 0.005, M), hi + ext[1])
        recent = np.abs(vs[[h[0] >= self.hist[-1][0] - int(3.0 / DT) for h in self.hist]])
        v_peak = max(float(np.percentile(recent, 90)), 0.05)
        speed = np.clip(v_peak * rng.uniform(0.5, 1.1, M), 0.03, V_MAX)
        dwell = rng.uniform(0.0, DWELL_MAX, M)
        pos = np.clip(s[-1] + rng.normal(0, 0.005, M), end_lo, end_hi)
        d_last = np.sign(vs[-1]) if abs(vs[-1]) > TURN_SPEED else 0.0
        direc = np.full(M, d_last) if d_last else rng.choice([-1.0, 1.0], M)
        wait = np.where(d_last == 0.0, rng.uniform(0.0, 1.0, M) * dwell, 0.0)   # slow at the last reading: maybe pausing
        times = [(now - self.hist[-1][0]) * DT + K * DT * i for i in range(1, horizon + 1)]
        out, t = [], 0.0
        for target in times:
            while t < target - 1e-9:
                h = min(SIM_DT, target - t)
                moving = wait <= 0
                wait = np.where(moving, wait, wait - h)
                pos = pos + np.where(moving, direc * speed * h, 0.0)
                over_hi, over_lo = pos > end_hi, pos < end_lo
                pos = np.where(over_hi, end_hi, np.where(over_lo, end_lo, pos))
                direc = np.where(over_hi, -1.0, np.where(over_lo, 1.0, direc))
                wait = np.where(over_hi | over_lo, dwell, wait)
                t += h
            out.append(c[None] + pos[:, None] * axis[None])
        return np.stack(out)
