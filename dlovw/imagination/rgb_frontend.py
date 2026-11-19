"""Deployment-mode RGB front end: what runs next to the real cell (perception_rgb_20261110.md, section 8).

No simulator objects: the inputs are what the robot computer gets,
- `tcp(t, grip, rot=None)`: the two TCP positions (6,) from the robot's kinematics with their time stamp (s), and
  optionally their orientations (2, 3, 3) world from tool: the rope's learned end clamps then turn with the wrists;
- `image(role, t, rgb)`: one colour image of the "front" or "top" D435i with its capture time stamp (s), in any
  order and at any rate (the two streams are not synchronised);
and the outputs are time-stamped:
- `spheres(t)`: per sphere colour (position, velocity) at time t from its Kalman filter (rgb_sphere_filter), None
  while lost (hidden or out of view);
- `rope()`: (capture time, 39 x 3 centerline, TCP positions at that time) of the latest rope fit, or None.

- `obstacles(t, centres)`: optional, the sphere centres when they are known by other means (then
  track_spheres=False skips the colour tracking); the rope fit keeps the rope out of them.

Spheres (when tracked) are cheap (about 2 ms per image and sphere) and run on every image. The rope fit
(rgb_rope.RopeRGB, about 26 ms per pair on one CPU core) runs on the newest front / top pair whenever a new image
completes one (at most once every `rope_period` s), referred to the pair's mean time: TCP positions interpolated
there, each view fitted at its own capture time with the TCP velocity. Between rope fits the controller carries the
centerline with the grippers (the session's Jacobian).
Compute times are kept per stage (`timing()`), to check them against the latencies assumed in simulation.
"""
from __future__ import annotations

import time
from bisect import bisect_left

import numpy as np

from .rgb_rope import RopeRGB
from .rgb_sphere_filter import SphereFilter

PAIR_MAX_S = 0.05       # a front / top pair further apart than this is not fused for the rope


class RGBFrontEnd:
    def __init__(self, cams, colours, L=0.531, rope_period=0.0, q=0.3, seed=0, track_spheres=True, rope_kw=None):
        self.cams, self.colours = cams, [np.asarray(c, float) for c in colours]
        self.filters = [SphereFilter(q=q) for _ in self.colours]
        self.rope_tracker = RopeRGB(cams, L=L, seed=seed, **(rope_kw or {}))
        self.tcp_rot = None              # latest (2, 3, 3) TCP orientations, if given
        self.rope_period, self.track_spheres = float(rope_period), bool(track_spheres)
        self.known_obstacles = None
        self.latest = {}                 # role -> (t, rgb)
        self.tcp_log = []                # (t, grip), increasing t
        self.rope_est = None
        self.last_rope_t = -np.inf
        self.ms = {"sphere": [], "rope": []}
        self.status = []                 # per image: (role, t, [filter status per sphere])

    # ---- inputs -------------------------------------------------------------------------------------------
    def tcp(self, t, grip, rot=None):
        if self.tcp_log and t <= self.tcp_log[-1][0]:
            return
        if rot is not None:
            self.tcp_rot = np.asarray(rot, float).reshape(2, 3, 3).copy()
        self.tcp_log.append((float(t), np.asarray(grip, float).copy()))
        if len(self.tcp_log) > 2000:
            self.tcp_log = self.tcp_log[-1000:]

    def grip_at(self, t):
        ts = [x[0] for x in self.tcp_log]
        if not ts:
            raise RuntimeError("no TCP positions yet")
        i = bisect_left(ts, t)
        if i == 0:
            return self.tcp_log[0][1].copy()
        if i >= len(ts):
            return self.tcp_log[-1][1].copy()
        (t0, g0), (t1, g1) = self.tcp_log[i - 1], self.tcp_log[i]
        a = (t - t0) / max(t1 - t0, 1e-9)
        return (1 - a) * g0 + a * g1

    def obstacles(self, t, centres):
        """Known obstacle sphere centres (m, 3) at time t (measured by other means on the real cell); the rope fit keeps
        the rope out of them."""
        self.known_obstacles = (float(t), np.asarray(centres, float).reshape(-1, 3).copy())

    def grip_vel_at(self, t, h=0.008):
        return (self.grip_at(t + h) - self.grip_at(t - h)) / (2 * h)

    def image(self, role, t, rgb, rope=True):
        """Process one image. Returns True if a rope fit ran on it."""
        t0 = time.perf_counter()
        st = [f.image(t, rgb, self.cams[role], c) for f, c in zip(self.filters, self.colours)] if self.track_spheres else []
        self.ms["sphere"].append(1000 * (time.perf_counter() - t0))
        self.status.append((role, float(t), st))
        self.latest[role] = (float(t), rgb)
        if not rope or len(self.latest) < 2:
            return False
        (tf, _), (tt, _) = self.latest["front"], self.latest["top"]
        tm = 0.5 * (tf + tt)
        if abs(tf - tt) > PAIR_MAX_S or tm - self.last_rope_t < self.rope_period - 1e-6:
            return False
        t0 = time.perf_counter()
        grip = self.grip_at(tm)
        x = self.rope_tracker.update({r: self.latest[r][1] for r in ("front", "top")}, grip, t=tm,
                                     view_t={r: self.latest[r][0] for r in ("front", "top")}, grip_vel=self.grip_vel_at(tm),
                                     spheres=None if self.known_obstacles is None else self.known_obstacles[1],
                                     grip_rot=self.tcp_rot)
        self.ms["rope"].append(1000 * (time.perf_counter() - t0))
        self.rope_est, self.last_rope_t = (tm, x, grip), tm
        return True

    # ---- outputs ------------------------------------------------------------------------------------------
    def spheres(self, t):
        return [None if f.lost(t) else f.state(t) for f in self.filters]

    def sphere_measured_at(self, j):
        """Time of the last image that updated sphere j (the reading's age is measured from it)."""
        return self.filters[j].t_meas

    def rope(self):
        return self.rope_est

    def timing(self):
        return {k: (float(np.mean(v)), float(np.percentile(v, 95))) if v else None for k, v in self.ms.items()}
