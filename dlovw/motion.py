"""Scripted obstacle motions and occluders for stage 0 (docs/notes/stage0_prereg_20261022.md, section 2).

The legacy driver (envs/dynamic_obstacle.py) has neither jittered periodic motion nor dwells, so these
motions are generated as position tracks and replayed by `ScriptedDriver`, which mimics the legacy
driver's interface. A motion is a JSON-able dict stored in the scene file; `track()` is deterministic.

- quasi_periodic: c + n A sin(phi(t)); phi advances by pi over each half period, and half periods
  change only at the extremes (where the velocity is zero), so the velocity stays continuous.
- patrol: back and forth between c - A n and c + A n at a cruise speed with trapezoidal ramps,
  dwelling at each end for its own (jittered) time; the track starts `start_offset_s` into the cycle.
- varspeed (generalisation set, geom_tracker_prereg section 9): as patrol, but every leg has its own cruise
  speed (`leg_speeds_mps`, used in turn).
- random_walk (same): c + x1 a1 + x2 a2, each x a damped random oscillator (natural period, damping ratio,
  stationary sd) driven by white noise from the motion's own seed; prefix-consistent for any track length.
"""
from __future__ import annotations

import numpy as np

DT = 0.008
FINE = 0.001                       # integration step for the patrol state machine


def _quasi_periodic_offset(m, t):
    amp, halves, phi0 = float(m["amplitude_m"]), np.asarray(m["half_periods_s"], float), float(m["phase0"])
    first = np.pi / 2 + np.pi * np.floor((phi0 - np.pi / 2) / np.pi + 1.0)      # next extreme after phi0
    times, phases = [0.0], [phi0]
    times.append(halves[0] * (first - phi0) / np.pi)
    phases.append(first)
    for h in halves[1:]:
        times.append(times[-1] + h)
        phases.append(phases[-1] + np.pi)
    if times[-1] < t[-1]:
        raise ValueError("quasi_periodic motion shorter than the requested track")
    return amp * np.sin(np.interp(t, times, phases))


def _patrol_offset(m, t):
    amp, speed, acc = float(m["amplitude_m"]), float(m["speed_mps"]), float(m["accel_mps2"])
    dwells = list(np.asarray(m["dwells_s"], float))
    t_end = float(m["start_offset_s"]) + t[-1] + FINE
    out_t, out_s = [], []
    s, v, side, now, mode, left = amp, 0.0, 1.0, 0.0, "dwell", dwells.pop(0)
    while now <= t_end:
        out_t.append(now)
        out_s.append(s)
        if mode == "dwell":
            left -= FINE
            if left <= 0:
                mode = "move"
        else:
            target = -side * amp
            dist = abs(target - s)
            brake = v * v / (2 * acc)
            v = max(v - acc * FINE, 0.0) if dist <= brake + v * FINE else min(v + acc * FINE, speed)
            s += -side * v * FINE
            if (s - target) * side <= 0 or (v == 0.0 and dist < 1e-4):
                s, v, side, mode = target, 0.0, -side, "dwell"
                if not dwells:
                    raise ValueError("patrol motion shorter than the requested track")
                left = dwells.pop(0)
        now += FINE
    return np.interp(t + float(m["start_offset_s"]), out_t, out_s)


def _varspeed_offset(m, t):
    amp, acc = float(m["amplitude_m"]), float(m["accel_mps2"])
    speeds, dwells = list(np.asarray(m["leg_speeds_mps"], float)), list(np.asarray(m["dwells_s"], float))
    t_end = float(m["start_offset_s"]) + t[-1] + FINE
    out_t, out_s = [], []
    s, v, side, now, mode, left, speed = amp, 0.0, 1.0, 0.0, "dwell", dwells.pop(0), speeds.pop(0)
    while now <= t_end:
        out_t.append(now)
        out_s.append(s)
        if mode == "dwell":
            left -= FINE
            if left <= 0:
                mode = "move"
        else:
            target = -side * amp
            dist = abs(target - s)
            brake = v * v / (2 * acc)
            v = max(v - acc * FINE, 0.0) if dist <= brake + v * FINE else min(v + acc * FINE, speed)
            s += -side * v * FINE
            if (s - target) * side <= 0 or (v == 0.0 and dist < 1e-4):
                s, v, side, mode = target, 0.0, -side, "dwell"
                if not dwells or not speeds:
                    raise ValueError("varspeed motion shorter than the requested track")
                left, speed = dwells.pop(0), speeds.pop(0)
        now += FINE
    return np.interp(t + float(m["start_offset_s"]), out_t, out_s)


def _random_walk_offset(m, t):
    """(len(t), 3) offsets from the centre."""
    rng = np.random.default_rng(int(m["seed"]))
    w0 = 2 * np.pi / np.asarray(m["natural_periods_s"], float)          # per axis
    zeta, sd = float(m["damping_ratio"]), np.asarray(m["sd_m"], float)
    q = 4 * zeta * w0 ** 3 * sd ** 2                                     # white-noise density for that stationary sd
    x = rng.normal(size=2) * sd
    v = rng.normal(size=2) * sd * w0
    n = int(np.ceil(t[-1] / DT)) + 1
    xs = np.empty((n + 1, 2))
    xs[0] = x
    for i in range(1, n + 1):
        a = -w0 ** 2 * x - 2 * zeta * w0 * v + rng.normal(size=2) * np.sqrt(q / DT)
        v = v + a * DT
        x = x + v * DT
        xs[i] = x
    grid = np.arange(n + 1) * DT
    x1, x2 = np.interp(t, grid, xs[:, 0]), np.interp(t, grid, xs[:, 1])
    axes = np.asarray(m["axes"], float)
    return x1[:, None] * axes[0][None] + x2[:, None] * axes[1][None]


def track(motion, steps, dt=DT):
    """(steps+1, 3) positions at t = 0, dt, ..., steps*dt."""
    t = np.arange(steps + 1) * dt
    if motion["kind"] == "random_walk":
        return np.asarray(motion["center"], float)[None] + _random_walk_offset(motion, t)
    offset = {"quasi_periodic": _quasi_periodic_offset, "patrol": _patrol_offset,
              "varspeed": _varspeed_offset}[motion["kind"]](motion, t)
    return np.asarray(motion["center"], float)[None] + offset[:, None] * np.asarray(motion["direction"], float)[None]


def velocities(positions, dt=DT):
    v = np.gradient(positions, dt, axis=0)
    return v


def hidden(position, occluder):
    """True when the point lies in the occluder slab lo <= (p - centre).normal <= hi."""
    if not occluder:
        return False
    d = float((np.asarray(position, float) - np.asarray(occluder["center"], float)) @ np.asarray(occluder["normal"], float))
    return occluder["lo"] <= d <= occluder["hi"]


class ScriptedDriver:
    """Replays a scripted track with the legacy DynamicObstacleDriver interface."""

    def __init__(self, model, data, config, control_period_s, motion, steps=1300):
        self.config = config
        self.control_period_s = float(control_period_s)
        self.mocap_id = -1
        if model is not None:
            import mujoco
            geom = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, config.geom_name))
            self.mocap_id = int(model.body_mocapid[int(model.geom_bodyid[geom])])
        self.track = track(motion, steps, self.control_period_s)
        self.track_velocity = velocities(self.track, self.control_period_s)
        self._k, self.time_s, self._frozen = 0, 0.0, False
        self.position, self.velocity = self.track[0].copy(), self.track_velocity[0].copy()
        self._acceleration = np.zeros(3)

    def advance(self):
        if not self._frozen:
            self._k = min(self._k + 1, len(self.track) - 1)
            previous = self.velocity
            self.position, self.velocity = self.track[self._k].copy(), self.track_velocity[self._k].copy()
            self._acceleration = (self.velocity - previous) / self.control_period_s
        self.time_s += self.control_period_s
        return self.position.copy(), self.velocity.copy()

    def write_mocap(self, data):
        if self.mocap_id >= 0:
            data.mocap_pos[self.mocap_id] = self.position

    def freeze(self):
        self._frozen = True
        self.velocity = np.zeros(3)
        self._acceleration = np.zeros(3)

    @property
    def acceleration(self):
        return self._acceleration.copy()

    event_index = -1
    event_triggered = False
