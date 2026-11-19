"""Obstacle beliefs for stage 0 (docs/notes/stage0_prereg_20261022.md, section 3).

`BeliefSession` is the round-2 dodge rule (B2) whose view of each obstacle comes from a belief instead
of the legacy estimator: the planner and CBF get the belief's current centre, velocity and an inflated
radius (its extra uncertainty), and the dodge rule uses the belief's predicted path. Scoring still uses
the true obstacle positions.

Arms: H0 Kalman-style constant velocity from the last measurement; H1 worst case (still centre, radius
growing at V_MAX) while hidden; H2 period replay from the visible history; P privileged true track.
Measurements come from `Sensor`: the true state `latency` steps ago, plus bounded noise and a fixed
unknown bias, or nothing while the obstacle is inside its occluder.
"""
from __future__ import annotations

import copy
import dataclasses

import numpy as np

from . import legacy  # noqa: F401  (sets sys.path)
from control.obstacles import ObstacleSet  # noqa: E402
from .motion import hidden
from .ttc_dodge import PROTECTED_M, TTCDodgeSession

DT = 0.008
V_MAX = 0.45                          # obstacle speed bound used by the worst-case arm (m/s)
REPLAY_GROWTH_M = 0.015               # H2: extra radius per period replayed back
GATE_S = 2.5                          # go/wait gate horizon cap (s)
GATE_AFTER_ARRIVAL_S = 0.25           # the gate checks the way to the target plus this long after arrival
GATE_MARGIN_M = 0.01                  # gate margin on top of the protected radius (the dodge rule keeps 20 mm)
GATE_HYSTERESIS_M = 0.01              # extra clearance needed to start moving again after waiting
NOMINAL = dict(level="nominal", delta_c_m=0.005, delta_v_mps=0.02, latency_steps=0, bias_m=[0.0, 0.0, 0.0])


def _bounded(rng, bound):
    if bound <= 0:
        return np.zeros(3)
    d = rng.normal(size=3)
    return d / np.linalg.norm(d) * rng.uniform(0.0, bound)


class Sensor:
    def __init__(self, perception, occluder, rng):
        p = perception or NOMINAL
        self.dc, self.dv, self.latency = float(p["delta_c_m"]), float(p["delta_v_mps"]), int(p["latency_steps"])
        self.bias, self.occluder, self.rng = np.asarray(p["bias_m"], float), occluder, rng

    def read(self, positions, velocities):
        """(step, position, velocity) of the measurement taken now, or None while hidden."""
        k = max(len(positions) - 1 - self.latency, 0)
        if hidden(positions[k], self.occluder):
            return None
        return k, positions[k] + _bounded(self.rng, self.dc) + self.bias, velocities[k] + _bounded(self.rng, self.dv)


class KalmanBelief:
    """H0: constant velocity from the last measurement; radius grows at delta_v since then."""

    def __init__(self, delta_v):
        self.delta_v, self.last, self.seen_now = delta_v, None, False

    def update(self, k, meas):
        self.seen_now = meas is not None
        if meas is not None:
            self.last = meas

    def _age(self, k):
        return (k - self.last[0]) * DT

    def now(self, k):
        _, p, v = self.last
        age = self._age(k)
        return p + v * age, v, self.delta_v * age

    def path(self, k, ts):
        _, p, v = self.last
        age = self._age(k)
        return p[None] + (age + ts)[:, None] * v[None], self.delta_v * (age + ts)


class WorstCaseBelief(KalmanBelief):
    """H1: as H0 while visible; while hidden the centre stays where it was last seen and the radius
    grows at V_MAX."""

    def now(self, k):
        if self.seen_now:
            return super().now(k)
        _, p, _ = self.last
        return p.copy(), np.zeros(3), V_MAX * self._age(k)

    def path(self, k, ts):
        if self.seen_now:
            return super().path(k, ts)
        _, p, _ = self.last
        return np.repeat(p[None], len(ts), axis=0), V_MAX * (self._age(k) + ts)


class PeriodicBelief(WorstCaseBelief):
    """H2: estimate the period from the visible history (lag with the smallest mean squared
    difference along the principal axis, 1-4 s, >= 30 pairs, below 25% of the variance) and predict
    by replaying the position one or more periods back; fall back to H1 without a period."""

    def __init__(self, delta_v):
        super().__init__(delta_v)
        self.hist, self.lag, self._last_fit = {}, None, -10

    def update(self, k, meas):
        super().update(k, meas)
        if meas is not None:
            self.hist[meas[0]] = meas[1]
        if k - self._last_fit >= 10:
            self._last_fit = k
            self.lag = self._fit(k)

    def _fit(self, k):
        if len(self.hist) < 60:
            return None
        series = np.full((k + 1, 3), np.nan)
        for i, p in self.hist.items():
            series[i] = p
        ok = ~np.isnan(series[:, 0])
        centred = series[ok] - series[ok].mean(0)
        axis = np.linalg.svd(centred, full_matrices=False)[2][0]
        s = np.full(k + 1, np.nan)
        s[ok] = centred @ axis
        var = float(np.nanvar(s))
        best = None
        for lag in range(int(1.0 / DT), int(4.0 / DT) + 1, 2):
            if lag >= k - 30:
                break
            a, b = s[lag:], s[:-lag]
            m = ~(np.isnan(a) | np.isnan(b))
            if m.sum() < 30:
                continue
            msd = float(np.mean((a[m] - b[m]) ** 2)) / max(var, 1e-12)
            if best is None or msd < best[0]:
                best = (msd, lag)
        return best[1] if best is not None and best[0] < 0.25 else None

    def _replay(self, q, k):
        for m in range(1, 7):
            src = int(round(q - m * self.lag))
            if src > k:
                continue
            for off in (0, -1, 1, -2, 2):
                if src + off in self.hist:
                    return self.hist[src + off], m
            if src < 0:
                break
        return None

    def now(self, k):
        if self.seen_now or self.lag is None:
            return super().now(k)
        here, before, after = self._replay(k, k), self._replay(k - 1, k), self._replay(k + 1, k)
        if here is None:
            return super().now(k)
        v = (after[0] - before[0]) / (2 * DT) if before and after else np.zeros(3)
        return here[0], v, REPLAY_GROWTH_M * here[1]

    def path(self, k, ts):
        if self.lag is None:
            return super().path(k, ts)
        fallback_c, fallback_g = super().path(k, ts)
        centres, grow = fallback_c.copy(), fallback_g.copy()
        for i, t in enumerate(ts):
            r = self._replay(k + t / DT, k)
            if r is not None:
                centres[i], grow[i] = r[0], REPLAY_GROWTH_M * r[1]
        return centres, grow


def true_track(driver, steps=1300):
    """The true future positions: the scripted track, or a deterministic replay of a legacy driver."""
    if hasattr(driver, "track"):
        return driver.track
    d = copy.deepcopy(driver)
    out = [d.position.copy()]
    for _ in range(steps):
        out.append(d.advance()[0].copy())
    return np.asarray(out)


class BeliefSession(TTCDodgeSession):
    """With `trace_enabled`, `trace` keeps per obstacle-estimate step k (k = 0 at the start, then one
    per control step, aligned with the obstacle track index) the belief centre, velocity and extra
    radius each obstacle is given, whether it is inside its occluder, and the age of the belief's last
    measurement (steps); and per control step the rule's mode (0 planner, 1 dodge, 2 hold, 3 gate wait)."""
    arm = "H0"
    trigger_s, reinit_after_dodge, choose_direction = 0.6, True, True
    trace_enabled = False
    rule = True                       # False: no dodge rule and no gate, the planner and CBF only see the belief

    def _belief_setup(self):
        if hasattr(self, "_beliefs"):
            return
        extra = self.cfg.get("dlovw") or {}
        perception = extra.get("perception") or NOMINAL
        occluders = extra.get("occluders") or []
        drivers = self.dynamic.drivers
        rng = np.random.default_rng(int(self.seed) + 7919)
        cls = {"H0": KalmanBelief, "H1": WorstCaseBelief, "H2": PeriodicBelief, "P": None}[self.arm]
        self._k = -1
        self._hist = [([], []) for _ in drivers]
        self._sensors = [Sensor(perception, occluders[j] if j < len(occluders) else None, rng) for j in range(len(drivers))]
        self._beliefs = [cls(float(perception["delta_v_mps"])) if cls else None for _ in drivers]
        self._true = [true_track(d) for d in drivers] if self.arm == "P" else None
        self.belief_log = dict(hidden_steps=0, steps=0)
        self.trace = dict(centre=[], velocity=[], grow=[], hidden=[], age=[], mode=[])

    def _estimate_obstacles(self):
        if self.dynamic is None:
            return super()._estimate_obstacles()
        self._belief_setup()
        self._k += 1
        items, row = [], []
        for j, (o, d) in enumerate(zip(self.safety.obstacles, self.dynamic.drivers)):
            positions, velocities = self._hist[j]
            positions.append(d.position.copy())
            velocities.append(d.velocity.copy())
            is_hidden = hidden(d.position, self._sensors[j].occluder)
            self.belief_log["steps"] += 1
            self.belief_log["hidden_steps"] += is_hidden
            if self.arm == "P":
                track = self._true[j]
                k = min(self._k, len(track) - 1)
                c, v, grow = track[k], (track[min(k + 1, len(track) - 1)] - track[max(k - 1, 0)]) / (2 * DT), 0.0
                age = 0
            else:
                b = self._beliefs[j]
                b.update(self._k, self._sensors[j].read(positions, velocities))
                if b.last is None:                    # hidden from the start: treat as at its true start
                    b.last = (0, positions[0].copy(), np.zeros(3))
                c, v, grow = b.now(self._k)
                age = self._k - b.last[0]
            row.append((np.asarray(c, float), np.asarray(v, float), float(grow), bool(is_hidden), int(age)))
            ob = o.moved(c).with_velocity(v)
            if grow > 0:
                size = np.array(ob.size, float)
                size[0] += grow
                ob = dataclasses.replace(ob, size=size)
            items.append(ob)
        if self.trace_enabled:
            for name, i in (("centre", 0), ("velocity", 1), ("grow", 2), ("hidden", 3), ("age", 4)):
                self.trace[name].append([r[i] for r in row])
        self.current_obstacles = ObstacleSet.from_iterable(items)
        self._set_obstacle_context()

    def step(self, nominal=None, **kw):
        self._ttc_init()
        if not self.rule:                # skip TTCDodgeSession.step: plain legacy planner + CBF on the belief
            return super(TTCDodgeSession, self).step(nominal, **kw)
        log = self.dodge_log
        before = (log["steps"], log["hold_steps"], log["gate_steps"])
        out = super().step(nominal, **kw)
        if self.trace_enabled:
            self.trace["mode"].append(3 if log["gate_steps"] > before[2] else 2 if log["hold_steps"] > before[1]
                                      else 1 if log["steps"] > before[0] else 0)
        return out

    def _gate_blocked(self, x):
        """SIPP-style go/wait gate shared by all arms (pre-registration amendment, section 7): predict
        the rope moving straight to its target shape at the dodge speed and wait while any obstacle,
        as this arm predicts it, would come within the protected radius on the way."""
        target = np.asarray(self.target["centerline"], float)
        y = np.asarray(self.state().y_k, float)
        goal = np.r_[np.asarray(self.target["left_pose"])[:3], np.asarray(self.target["right_pose"])[:3]]
        dist = max(np.linalg.norm(goal[:3] - y[:3]), np.linalg.norm(goal[3:] - y[3:]))
        speed = self.speed_frac * float(self.sim_cfg["max_translation_step_m"]) / self.protocol.dt
        ts = np.arange(0.0, min(GATE_S, dist / speed + GATE_AFTER_ARRIVAL_S) + 1e-9, DT * 4)
        frac = np.minimum(1.0, ts * speed / max(dist, 1e-6))
        rope = x[None] + frac[:, None, None] * (target - x)[None]                  # (T, N, 3)
        need = GATE_HYSTERESIS_M if self._holding else 0.0
        for j, o in enumerate(self.current_obstacles.obstacles):
            path, grow = self._path(j, o, ts)
            d = np.linalg.norm(rope - path[:, None, :], axis=-1).min(1)
            if np.any(d - (PROTECTED_M + GATE_MARGIN_M - 0.040 + float(o.size[0]) + grow) < need):
                return True
        return False

    def _path(self, j, o, ts):
        if self.arm == "P":
            track = self._true[j]
            idx = np.minimum(self._k + np.round(ts / DT).astype(int), len(track) - 1)
            return track[idx], np.zeros(len(ts))
        centres, grow = self._beliefs[j].path(self._k, ts)
        return centres, grow - (float(o.size[0]) - 0.040)
