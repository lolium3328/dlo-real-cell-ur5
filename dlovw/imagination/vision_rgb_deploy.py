"""Closed-loop RGB perception in deployment mode (perception preset V4*, perception_rgb_20261110.md section 8).

The simulator only supplies what the real cell would: rendered colour images of the calibrated D435i pair at their
capture times (front at every 4th control step, top `offset_steps` later, as the unsynchronised real streams), and the
TCP positions every control step. Everything else is the deployment front end (rgb_frontend.RGBFrontEnd), fed through
its time-stamped interface, and its results reach the controller only after an assumed latency:
- spheres: per image, available `sphere_latency` control steps after capture (camera exposure + USB + decode +
  2-4 ms processing: 40 ms = 5 steps by default); the reading is the Kalman filter's (position, velocity) at the
  capture time, so carrying it forward by its age (comp=True) gives the present;
- rope: a fit on every new front / top pair (26 ms on one core), available `rope_latency` steps after the pair's
  capture (56 ms: camera pipeline about 30 ms plus the fit), with the known sphere centres (truth here, measured
  by other means on the real cell) kept out of the rope.
track_spheres=False (presets V4r*): no colour tracking of the spheres; the controller gets them from the scene.
The first frame waits for both views and is used at once (the controller does not start before perception has one).
Spheres inside their occluders are left out of the renders (the simulated world's decision, as V2 / V3).
"""
from __future__ import annotations

import numpy as np

from .perception import CAM_STEPS
from .rgb_frontend import RGBFrontEnd
from .vision_rgb import RGBPerception

DT = 0.008


class RGBDeployPerception(RGBPerception):
    def __init__(self, env, sphere_geoms, n_pts=39, seed=0, preset="real", noise_sd=5.0, background=26.0, offset_steps=2,
                 L=0.531, sphere_latency=5, rope_latency=7, rope_period=0.0, q=0.3, track_spheres=True, **_):
        super().__init__(env, sphere_geoms, n_pts=n_pts, seed=seed, preset=preset, noise_sd=noise_sd,
                         background=background, offset_steps=offset_steps, L=L)
        self.fe = RGBFrontEnd(self.camset, self.colours, L=L, rope_period=rope_period, q=q, seed=seed, track_spheres=track_spheres)
        self.sphere_latency, self.rope_latency = int(sphere_latency), int(rope_latency)
        self.snaps = [[] for _ in range(self.J)]       # per sphere: (available step, capture step, (p, v) or None)
        self.rope_queue = []                            # (available step, x, grip)
        self.reading_now = [None] * self.J

    def _capture(self, role, grip, hidden, truth, now=False):
        t = self.steps * DT
        img = self._render(role, hidden)
        ran = self.fe.image(role, t, img, rope=True)
        avail = self.steps + (0 if now else self.sphere_latency)
        row = []
        for j in range(self.J if self.fe.track_spheres else 0):
            s = self.fe.spheres(t)[j]
            self.snaps[j].append((avail, self.steps, s))
            row.append((bool(hidden[j]), None if truth is None else np.asarray(truth[j][0], float).copy(),
                        None if s is None else s[0].copy(), None))
        self.truth_log.append((self.steps, row))
        self.frames_seen += 1
        if ran:
            _, x, g = self.fe.rope()
            self.rope_queue.append((self.steps + (0 if now else self.rope_latency), x, g))
        self.frame_ms.append(self.fe.ms["sphere"][-1] + (self.fe.ms["rope"][-1] if ran else 0.0))

    def _release(self):
        for j, sn in enumerate(self.snaps):
            while sn and sn[0][0] <= self.steps:
                _, k, s = sn.pop(0)
                self.reading_now[j] = None if s is None else (k, s[0].copy(), s[1].copy())
        while self.rope_queue and self.rope_queue[0][0] <= self.steps:
            _, x, g = self.rope_queue.pop(0)
            self.est, self.est_grip = x, g
            self.frame_id = getattr(self, "frame_id", 0) + 1

    def update(self, true_c, grip, hidden, truth, J, L):
        grip = np.asarray(grip, float)
        self._J = J
        self.fe.tcp(self.steps * DT, grip)
        if truth is not None:                               # the obstacle centres are measured on the real cell
            self.fe.obstacles(self.steps * DT, [tr[0] for tr in truth])
        phase = self.steps % CAM_STEPS
        if self.est is None:                                # start: both views now, results used at once
            self._capture("front", grip, hidden, truth, now=True)
            self.fe.last_rope_t = -np.inf
            self._capture("top", grip, hidden, truth, now=True)
        else:
            if phase == 0:
                self._capture("front", grip, hidden, truth)
            if phase == self.offset:
                self._capture("top", grip, hidden, truth)
        self._release()
        if truth is not None:
            now = []
            for j in range(self.J):
                r = self.reading_now[j]
                now.append((np.asarray(truth[j][0], float).copy(), np.asarray(truth[j][1], float).copy(),
                            None if r is None else (r[1] + r[2] * (self.steps - r[0]) * DT, r[2].copy())))
            self.step_log.append((self.steps, now))
        self.steps += 1
        out = self.estimate(grip, J, L)
        self.err.append(float(np.sqrt(((out - np.asarray(true_c)) ** 2).sum(-1).mean())))
        return out

    def obstacle_reading(self, j=0):
        r = self.reading_now[j]
        return None if r is None else (r[0], r[1].copy(), r[2].copy())

    def summary(self, every=10):
        out = super().summary(every)
        out["timing_ms"] = self.fe.timing()
        return out
