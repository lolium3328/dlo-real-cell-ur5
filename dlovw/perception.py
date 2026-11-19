"""Hand-written perception from front RGB-D (docs/notes/i0_prereg_20261024.md, section 3).

- `ObstacleEstimator`: red mask -> back-projected surface points -> known-radius sphere fit
  (Gauss-Newton from the centroid pushed one radius along the view ray) -> constant-velocity Kalman
  filter at the frame rate. Fewer than MIN_PIXELS valid pixels counts as "not seen".
  Mask pixels whose depth is more than one radius from the mask's median depth are dropped (edge pixels
  are coloured by INTER_AREA but carry background depth from INTER_NEAREST).
- `RopeTracker`: yellow mask -> back-projected points pushed one rope radius along the view ray, kept
  within ROPE_GATE_M of the prior nodes ->
  a simplified CDCPD-style EM tracker for the 39 centreline nodes (soft assignment with an outlier
  weight, second-difference smoothness, a temporal prior on the previous nodes, ends pinned to the
  grippers plus the offsets read at frame 0), resampled to equal arc length after each frame.
All parameters are fixed here before any closed-loop run (pre-registration section 4 allows changes
only to pass the accuracy gate).
"""
from __future__ import annotations

import numpy as np

from .camera import FrontCamera

OBST_RADIUS_M = 0.040
ROPE_RADIUS_M = 0.010
MIN_PIXELS = 6
KF_ACCEL_MPS2 = 2.0          # process noise (white acceleration) of the obstacle filter
KF_MEAS_M = 0.005            # measurement noise of a sphere-fit centre
EM_ITERS, EM_OUTLIER = 10, 0.1
EM_SIGMA0_M = 0.015
SMOOTH, TEMPORAL = 20.0, 1.0
MAX_POINTS = 400
ROPE_GATE_M = 0.08           # rope points farther than this from the prior nodes are dropped


def red_mask(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (r > 0.35) & (r - np.maximum(g, b) > 0.20) & (g < 0.35)


def yellow_mask(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (r > 0.35) & (g > 0.30) & (np.minimum(r, g) - b > 0.15) & (np.abs(r - g) < 0.35)


def _pixels(mask, depth):
    vs, us = np.nonzero(mask & np.isfinite(depth))
    return us.astype(float), vs.astype(float), depth[vs, us]


def fit_sphere(points, rays, radius=OBST_RADIUS_M, iters=5):
    """Centre of a sphere of known radius through visible surface points."""
    c = points.mean(0) + radius * rays.mean(0) / np.linalg.norm(rays.mean(0))
    start = c.copy()
    for _ in range(iters):
        d = points - c
        n = np.linalg.norm(d, axis=1)
        n = np.maximum(n, 1e-9)
        res = n - radius
        J = -d / n[:, None]
        step, *_ = np.linalg.lstsq(J, -res, rcond=None)
        c = c + step
    return c if np.linalg.norm(c - start) < 0.03 else start


class ObstacleEstimator:
    """Measurements (frame step, centre) and a constant-velocity Kalman filter over them."""

    def __init__(self, dt_control=0.008):
        self.dt = float(dt_control)
        self.x, self.P, self.k = None, None, None

    def measure(self, rgb, depth, intr, pose):
        us, vs, z = _pixels(red_mask(rgb), depth)
        if len(z):                              # edge pixels can be red (INTER_AREA) but carry background depth
            keep = np.abs(z - np.median(z)) < OBST_RADIUS_M
            us, vs, z = us[keep], vs[keep], z[keep]
        if len(z) < MIN_PIXELS:
            return None
        pts = FrontCamera.backproject(us, vs, z, intr, pose)
        rays, _ = FrontCamera.rays(us, vs, intr, pose)
        return fit_sphere(pts, rays)

    def update(self, k, centre):
        """Fold in a measurement taken at control step k (None: not seen). Returns (k, p, v) of the
        latest measurement-time estimate, or None before the first sighting."""
        if centre is not None:
            if self.x is None:
                self.x, self.P = np.r_[centre, np.zeros(3)], np.diag([KF_MEAS_M ** 2] * 3 + [0.25] * 3)
            else:
                dt = max(k - self.k, 1) * self.dt
                F = np.eye(6)
                F[:3, 3:] = dt * np.eye(3)
                q = KF_ACCEL_MPS2 ** 2
                Q = np.block([[dt ** 4 / 4 * np.eye(3), dt ** 3 / 2 * np.eye(3)], [dt ** 3 / 2 * np.eye(3), dt ** 2 * np.eye(3)]]) * q
                self.x, self.P = F @ self.x, F @ self.P @ F.T + Q
                H = np.hstack([np.eye(3), np.zeros((3, 3))])
                S = H @ self.P @ H.T + KF_MEAS_M ** 2 * np.eye(3)
                K = self.P @ H.T @ np.linalg.inv(S)
                self.x = self.x + K @ (centre - H @ self.x)
                self.P = (np.eye(6) - K @ H) @ self.P
            self.k = k
        if self.x is None:
            return None
        return self.k, self.x[:3].copy(), self.x[3:].copy()


def resample(points, n):
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, points[:, i]) for i in range(3)], axis=1)


class RopeTracker:
    def __init__(self, centreline0, left0, right0, n=39, rng=None):
        self.Y = np.asarray(centreline0, float).copy()
        self.n = n
        self.off_l = self.Y[0] - np.asarray(left0, float)
        self.off_r = self.Y[-1] - np.asarray(right0, float)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        D = np.zeros((n - 2, n))
        for i in range(n - 2):
            D[i, i:i + 3] = (1.0, -2.0, 1.0)
        self.DtD = D.T @ D

    def points(self, rgb, depth, intr, pose):
        us, vs, z = _pixels(yellow_mask(rgb), depth)
        if len(z) == 0:
            return np.empty((0, 3))
        if len(z) > MAX_POINTS:
            keep = self.rng.choice(len(z), MAX_POINTS, replace=False)
            us, vs, z = us[keep], vs[keep], z[keep]
        pts = FrontCamera.backproject(us, vs, z, intr, pose)
        rays, _ = FrontCamera.rays(us, vs, intr, pose)
        return pts + ROPE_RADIUS_M * rays

    def update(self, X, left, right):
        """One frame of EM tracking on observed centre points X (M, 3); returns the 39 nodes."""
        prev = self.Y.copy()
        ends = (np.asarray(left, float) + self.off_l, np.asarray(right, float) + self.off_r)
        if len(X):                              # gate: edge pixels with background depth are far from the rope
            X = X[np.linalg.norm(X[:, None, :] - prev[None, :, :], axis=-1).min(1) < ROPE_GATE_M]
        Y = prev.copy()
        Y[0], Y[-1] = ends
        if len(X):
            sigma2 = EM_SIGMA0_M ** 2
            M, N = len(X), self.n
            for _ in range(EM_ITERS):
                d2 = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)                  # (M, N)
                g = np.exp(-d2 / (2 * sigma2))
                c = (2 * np.pi * sigma2) ** 1.5 * EM_OUTLIER / (1 - EM_OUTLIER) * N / M
                P = g / (g.sum(1, keepdims=True) + c)
                w = P.sum(0)                                                          # (N,)
                A = np.diag(w) + SMOOTH * self.DtD + TEMPORAL * np.eye(N)
                B = P.T @ X + TEMPORAL * prev
                free = np.arange(1, N - 1)
                fixed = np.array([0, N - 1])
                Yf = np.stack(ends)
                rhs = B[free] - A[np.ix_(free, fixed)] @ Yf
                Y[free] = np.linalg.solve(A[np.ix_(free, free)], rhs)
                Y[0], Y[-1] = ends
                num = (P * ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)).sum()
                sigma2 = max(num / (3 * max(P.sum(), 1e-9)), 1e-6)
        self.Y = resample(Y, self.n)
        return self.Y.copy()
