"""Obstacle-only training data without the robot (docs/notes/imagination_ac_design_20261022.md, section 21).

The obstacle moves on its own, so its sensor stream and true future can be generated in bulk: fresh
motion parameters from the stage-0 distributions (quasi-periodic sine with jittered half periods, or patrol
with dwells), the geometry (crossing point, direction, frame origin) of a *training* scene, a random
occluder (none / far end / next to the rope) and perception level, run through the stage-0 Sensor at the
control rate. Per decision (K control steps) a sequence holds the measurement (7), corridor features (16)
and the true position; the track continues past the sequence for blind-forecast targets.
sensor="camera" replaces the stage-0 Sensor by the depth cameras' obstacle readings (vision.CameraObstacleSensor:
30 Hz frames available one frame later, velocity fitted over the last frames, reading carried forward by its
age), with position noise 0.2-1.5 mm, a fixed offset up to 3 mm and up to 2% missed frames per sequence.
"""
from __future__ import annotations

import numpy as np

from ..belief import Sensor
from ..motion import track as motion_track, velocities
from ..scene_families import V_PEAK_MAX, _end_occluder, _near_occluder, _perception
from .corridor import CorridorForecast
from .episodes import K, POS, VEL, encode_meas

from .obstacle_contract import FEATS
FUTURE = 32                                    # decisions of true future kept past the last step


def corridor_features(cf, step, origin):
    """16 numbers summarising the corridor estimate (zeros and flag 0 until it is ready)."""
    if not cf.ready():
        return np.zeros(FEATS, np.float32)
    c, axis, s, vs = cf._frame()
    if axis[np.argmax(np.abs(axis))] < 0:     # the principal axis has no sign: fix one
        axis, s, vs = -axis, -s, -vs
    lo, hi, seen_lo, seen_hi = cf._ends(s, vs)
    recent = np.abs(vs[[h[0] >= cf.hist[-1][0] - int(3.0 / 0.008) for h in cf.hist]])
    since = (step - cf.hist[-1][0]) * 0.008
    return np.r_[axis, (c - origin) * POS, lo * POS, hi * POS, float(seen_lo), float(seen_hi), s[-1] * POS, vs[-1] * VEL,
                 np.percentile(recent, 90) * VEL, min(since, 3.0), np.log1p(len(cf.hist)) / 7.0, 1.0].astype(np.float32)


def sample_motion(rng, p, n):
    if rng.uniform() < 0.5:
        amp, base = rng.uniform(0.14, 0.17), rng.uniform(3.0, 5.0)
        halves = np.maximum(base / 2 * (1 + rng.uniform(-0.15, 0.15, size=24)), np.pi * amp / V_PEAK_MAX)
        return dict(kind="quasi_periodic", center=list(map(float, p)), direction=list(map(float, n)), amplitude_m=float(amp),
                    half_periods_s=[float(h) for h in halves], phase0=float(rng.uniform(0, 2 * np.pi))), amp
    amp, speed, dwell = rng.uniform(0.14, 0.17), rng.uniform(0.20, 0.35), rng.uniform(0.3, 1.0)
    cycle = 2 * (2 * amp / speed + speed / 1.5 + dwell)
    return dict(kind="patrol", center=list(map(float, p)), direction=list(map(float, n)), amplitude_m=float(amp),
                speed_mps=float(speed), accel_mps2=1.5, dwells_s=[float(d) for d in dwell * (1 + rng.uniform(-0.15, 0.15, size=24))],
                start_offset_s=float(rng.uniform(0, cycle))), amp


def camera_sensor(occluder, rng, cam=None):
    """cam: CameraObstacleSensor timing (cam_steps, latency_frames, window); default the 30 Hz depth cameras."""
    from .vision import CameraObstacleSensor
    return CameraObstacleSensor(occluder, rng, noise_m=rng.uniform(0.0002, 0.0015), bias_m=rng.uniform(0.0, 0.003),
                                miss=rng.uniform(0.0, 0.02), **(cam or {}))


def sequence(rng, p, n, origin, seconds=12.0, sensor="scene", cam=None):
    """One synthetic obstacle sequence at decision resolution."""
    motion, amp = sample_motion(rng, np.asarray(p, float), np.asarray(n, float))
    u = rng.uniform()
    occluder = None if u < 0.2 else (_end_occluder(rng, p, np.asarray(n, float), amp)[1] if u < 0.6 else _near_occluder(rng, p, np.asarray(n, float))[1])
    sensor = camera_sensor(occluder, rng, cam) if sensor == "camera" else Sensor(_perception(rng), occluder, rng)
    steps = int(seconds / 0.008)
    tr = motion_track(motion, steps + FUTURE * K + K)
    vel = velocities(tr)
    cf = CorridorForecast(seed=0)
    meas, feats, truth = [], [], []
    for n_ in range(steps + 1):
        r = sensor.read(tr[:n_ + 1], vel[:n_ + 1])
        m = None if r is None else (r[1], r[2])
        cf.update(n_, m)
        if n_ % K == 0:
            meas.append(encode_meas(m, origin))
            feats.append(corridor_features(cf, n_, origin))
            truth.append(((tr[n_] - origin) * POS).astype(np.float32))
    future = ((tr[0::K][:len(truth) + FUTURE] - origin) * POS).astype(np.float32)
    return dict(meas=np.stack(meas), feats=np.stack(feats), truth=np.stack(truth), track=future)
