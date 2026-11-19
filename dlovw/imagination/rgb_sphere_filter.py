"""Per-sphere constant-velocity Kalman filter on single-camera observations with their own time stamps
(perception_rgb_20261110.md, section 7).

Why: the two D435i colour streams are not synchronised (14-20 ms apart). Triangulating the front and top circle centres
as if they were taken at the same instant puts a moving sphere (median 0.18 m/s in m2c) 2-3 mm off, more while it turns.
Here every camera image is one measurement at its own time:
- the ray through the fitted circle centre: two linear constraints e_i . (p - o) = 0 perpendicular to the ray, with a
  pixel error that grows when only a short arc of the outline could be trusted;
- the range along the ray from the apparent size (weak: about 3 % of the range).
Images may come in any order and at any rate; one camera alone still gives the direction, and the filter's position
gives the depth instead of the size range. Measurements far outside the predicted ray (Mahalanobis gate) are rejected;
after REINIT rejections in a row the filter starts again from the next measurement.
"""
from __future__ import annotations

import numpy as np

from . import camera_model as cm
from .rgb_obstacles import MIN_ARC, MIN_SPAN, R_SPHERE, _span, circle_2d, circle_known_radius, fit_circle, trusted_outline

GATE = 16.0          # chi-square, 2 dof (p = 3e-4)
REINIT = 4
REACQUIRE_S = 0.3   # after this long without a measurement the filter starts again
PRIOR_MAX_SD = 0.03  # a prediction less certain than this (m) does not steer the fit
WORKSPACE = (np.array([-0.2, -1.2, -0.3]), np.array([1.6, 1.0, 1.4]))   # world frame (left robot base), generous
R_BIAS_PX = 0.8      # the trusted outline lies about 0.8 px inside the true silhouette (measured on clean renders)


def observe(rgb, cam, colour, prior=None, prior_sd=None):
    """One camera's measurement of the sphere of this colour: dict(o, d, sigma_perp, rng, sigma_rng) or None.

    prior: the filter's predicted centre at the image time (3,), prior_sd its position uncertainty (m). With a prior
    the sphere's image radius is known from the predicted range, and the centre is found by the known-radius RANSAC
    fit within a window around the predicted centre (rgb_obstacles.circle_known_radius); the range row then comes
    from a free fit to the inliers only if that agrees with the prediction. Without one (start, re-acquisition):
    the free trusted-outline fit."""
    f = cam["K"][1]
    if prior is None:
        c2 = circle_2d(rgb, colour)
        if c2 is None:
            return None
        c, r_px, _, span = c2
        r_px = r_px + R_BIAS_PX
        rng = R_SPHERE / np.sin(np.arctan(r_px / f))
        sigma_rng = rng * max(1.5 / r_px, 0.03)
    else:
        px, z = cm.project(np.asarray(prior, float)[None], cam)
        if z[0] <= R_SPHERE:
            return None
        dist = float(np.linalg.norm(np.asarray(prior) - cam_centre(cam)))
        r0 = f * R_SPHERE / np.sqrt(max(dist ** 2 - R_SPHERE ** 2, 1e-6)) - R_BIAS_PX
        shift = min(max(12.0, 4.0 * f * (prior_sd or 0.0) / z[0]), 40.0)
        half = r0 + shift + 12.0
        out = trusted_outline(rgb, colour, roi=(px[0, 0] - half, px[0, 1] - half, px[0, 0] + half + 1, px[0, 1] + half + 1))
        if out is None:
            return None
        fit = circle_known_radius(out[0], r0, px[0], shift)
        if fit is None:
            return None
        c, inl = fit
        span = _span(inl, c)
        if len(inl) < MIN_ARC or span < MIN_SPAN:
            return None
        rng, sigma_rng = dist, np.inf
        if span > 0.5:               # long arc: the free fit on the RANSAC inliers is more precise than the fixed radius
            c_free, r_free, _ = fit_circle(inl)
            if abs(r_free - r0) > 0.2 * r0:              # the prediction's range is wrong: measure without it
                return observe(rgb, cam, colour)
            if np.linalg.norm(c_free - c) < 0.1 * r0:
                c = c_free
            r_px = r_free + R_BIAS_PX                    # the range always from the image, never from the prediction
            rng = R_SPHERE / np.sin(np.arctan(r_px / f))
            sigma_rng = rng * max(1.5 / r_px, 0.03)
    o, d = cm.rays(c[None], cam)
    sigma_px = 0.7 + 3.0 * (1.0 - min(span, 1.0)) ** 2
    return dict(o=np.asarray(o, float), d=d[0], sigma_perp=sigma_px * rng / f, rng=float(rng), sigma_rng=float(sigma_rng))


def cam_centre(cam):
    o, _ = cm.rays(np.array([[cm.W / 2, cm.H / 2]]), cam)
    return np.asarray(o, float).reshape(3)


class SphereFilter:
    def __init__(self, q=0.1, sigma_v0=0.3):
        self.q, self.sigma_v0 = float(q), float(sigma_v0)
        self.x = self.P = None
        self.t = None
        self.t_meas = None          # time of the last accepted measurement
        self.rejects = 0

    def _predict(self, t):
        dt = t - self.t
        if dt <= 0:
            return
        F = np.eye(6)
        F[:3, 3:] = dt * np.eye(3)
        I3, q = np.eye(3), self.q
        Q = np.block([[q * dt ** 3 / 3 * I3, q * dt ** 2 / 2 * I3], [q * dt ** 2 / 2 * I3, q * dt * I3]])
        self.x, self.P, self.t = F @ self.x, F @ self.P @ F.T + Q, t

    def state(self, t):
        """(position, velocity) predicted to time t, or None before the first measurement."""
        if self.x is None:
            return None
        dt = t - self.t
        return self.x[:3] + dt * self.x[3:], self.x[3:].copy()

    def update(self, t, m):
        """Fuse one camera measurement taken at time t (s). Returns True if it was accepted."""
        if m is None:
            return False
        o, d = m["o"], m["d"]
        if self.x is not None and t - self.t_meas > REACQUIRE_S:      # was hidden / out of view: start again
            self.x = None
        if self.x is None:
            p0 = o + m["rng"] * d
            self.x = np.r_[p0, np.zeros(3)]
            s_rng = m["sigma_rng"] if np.isfinite(m["sigma_rng"]) else 0.05 * m["rng"]     # range from an old prediction
            Pp = m["sigma_perp"] ** 2 * (np.eye(3) - np.outer(d, d)) + s_rng ** 2 * np.outer(d, d)
            self.P = np.block([[Pp, np.zeros((3, 3))], [np.zeros((3, 3)), self.sigma_v0 ** 2 * np.eye(3)]])
            self.t = self.t_meas = t
            self.rejects = 0
            return True
        if t < self.t:              # out of order (late image): fuse at the filter time, small error
            t = self.t
        self._predict(t)
        a = np.cross(d, [1.0, 0, 0] if abs(d[0]) < 0.9 else [0, 1.0, 0])
        e1 = a / np.linalg.norm(a)
        e2 = np.cross(d, e1)
        rows = 3 if np.isfinite(m["sigma_rng"]) else 2
        H = np.zeros((3, 6))
        H[0, :3], H[1, :3], H[2, :3] = e1, e2, d
        z = np.array([e1 @ o, e2 @ o, d @ o + m["rng"]])
        R = np.diag([m["sigma_perp"] ** 2, m["sigma_perp"] ** 2, m["sigma_rng"] ** 2 if rows == 3 else 1.0])
        H, z, R = H[:rows], z[:rows], R[:rows, :rows]
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        if y[:2] @ np.linalg.solve(S[:2, :2], y[:2]) > GATE:
            self.rejects += 1
            if self.rejects >= REINIT:
                self.x = None
                return self.update(t, m)
            return False
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)
        self.t_meas, self.rejects = t, 0
        return True

    def prior(self, t):
        """(predicted centre, position sd in m) at time t for observe(), or (None, None) when there is no track."""
        if self.x is None or t - self.t_meas > REACQUIRE_S:
            return None, None
        p, _ = self.state(t)
        sd = float(np.sqrt(np.trace(self.P[:3, :3]) / 3 + (self.q * (t - self.t) ** 3 / 3 if t > self.t else 0.0)))
        if np.any(p < WORKSPACE[0]) or np.any(p > WORKSPACE[1]):     # diverged: start again from the images
            self.x = None
            return None, None
        return (p, sd) if sd < PRIOR_MAX_SD else (None, None)

    def image(self, t, rgb, cam, colour):
        """One camera image taken at time t: measure with the track's prediction as prior, and fuse.
        Returns 'accepted', 'rejected' (gate) or 'none' (no fit)."""
        p, sd = self.prior(t)
        m = observe(rgb, cam, colour, p, sd)
        if m is None and p is not None:          # lost the known-radius fit: try the free fit (re-acquisition)
            m = observe(rgb, cam, colour)
        if m is None:
            return "none"
        return "accepted" if self.update(t, m) else "rejected"

    def lost(self, t, max_gap=0.15):
        """No accepted measurement for max_gap seconds (sphere hidden or out of view)."""
        return self.x is None or t - self.t_meas > max_gap
