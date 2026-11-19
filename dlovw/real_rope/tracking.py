"""Recorded-rope R1 adaptation and explicit partial-observation quality contract.

Image support is evidence at the projected estimate, not oracle visibility or
verified stereo correspondence. Uncertainty growth/stop thresholds are design
values, not calibrated probabilities or a hardware safety controller.
"""
from __future__ import annotations

import cv2
import numpy as np

from .vision import HSV_LO, HSV_HI
from ..imagination import camera_model as cm
from ..imagination.rgb_rope import RopeRGB, chain


MAX_FRAME_AGE_S = .1


class QualityState:
    """Temporal quality from image evidence only; usable by non-R1 frontends too."""
    def __init__(self, n=39, missing_speed_m_s=.15):
        self.last_stereo = np.full(n, np.nan)
        self.last_t = None
        self.missing_speed = missing_speed_m_s

    def update(self, t, support, sd=None):
        support = np.asarray(support, bool)
        if support.shape != (len(self.last_stereo), 2):
            raise ValueError('Expected per-node front/top support (N,2)')
        if self.last_t is not None and t <= self.last_t:
            raise ValueError('Observation times must strictly increase')
        self.last_t = float(t)
        stereo = support.all(1)
        self.last_stereo[stereo] = t
        # Sentinel 2 s avoids infinity in model input; never-observed nodes stay stale.
        age = np.where(np.isfinite(self.last_stereo), t-self.last_stereo, 2.)
        age = np.minimum(age, 2.)
        sigma = np.full(len(age), .005) if sd is None else np.asarray(sd, float).copy()
        sigma = np.sqrt(np.maximum(sigma, .001)**2+(self.missing_speed*age)**2)
        # These two endpoints are known through robot kinematics / the fixed clamp.
        source = np.where(stereo, 'stereo_image_support',
                          np.where(support.any(1), 'single_view_prior', 'temporal_prior'))
        source[[0, -1]] = 'kinematic_endpoint'
        sigma[[0, -1]] = .01  # retained proxy-frame/attachment uncertainty
        interior_age = age[1:-1]
        pause = bool(np.mean(interior_age > .3) > .2 or np.any(interior_age > 1.))
        return dict(image_support=support, stereo_age_s=age, node_sd_m=sigma,
                    source=source, pause_requested=pause,
                    quality_features=np.c_[support.astype(float), age, sigma].astype(np.float32))


class RecordedRopeTracker(RopeRGB):
    """R1 with measured length and bright-yellow masks, without changing legacy R1."""
    def __init__(self, cams, profile, **kwargs):
        defaults = dict(L=profile['length_m'], l_bend=profile['bending_length_m'], shape_sd=.005)
        defaults.update(kwargs)
        super().__init__(cams, **defaults)
        self.quality = QualityState()
        self.image_evidence = {}
        self.last_view_time = {}

    def _observe(self, rgbs, depths=None):
        if depths is not None:
            raise ValueError('Recorded-rope adapter is RGB only')
        out = {}
        self.image_evidence = {}
        for role, rgb in rgbs.items():
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            rope = cv2.inRange(hsv, np.array(HSV_LO, np.uint8), np.array(HSV_HI, np.uint8))
            rope = cv2.morphologyEx(rope, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            _, lab, st, _ = cv2.connectedComponentsWithStats(rope)
            # Retain short fragments around an occluder; offline fitting uses >=100.
            small = st[:, cv2.CC_STAT_AREA] < 30
            small[0] = False  # label 0 is the background
            rope[small[lab]] = 0
            r = rope > 0
            s, v = hsv[..., 1], hsv[..., 2]
            occ = ((v >= 70) | ((s >= 90) & (v >= 45))) & ~cv2.dilate(rope, np.ones((5, 5), np.uint8)).astype(bool)
            occ = cv2.dilate(occ.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
            dt = cv2.distanceTransform((~r).astype(np.uint8), cv2.DIST_L2, 5)
            gy, gx = np.gradient(dt)
            ys, xs = np.nonzero(r[::self.stride, ::self.stride])
            pix = np.stack([xs*self.stride, ys*self.stride], 1).astype(np.float32)
            out[role] = dict(dt=dt, grad=np.stack([gx, gy], -1), occ=occ, pix=pix, n_rope=int(r.sum()))
            self.image_evidence[role] = (dt, occ)
        return out

    def observe(self, rgbs, grip, t, view_t=None, **kwargs):
        if self.quality.last_t is not None and t <= self.quality.last_t:
            raise ValueError('Observation times must strictly increase')
        view_t = {role: float((view_t or {}).get(role, t)) for role in rgbs}
        # Reused/stale frames must not refresh the last-observed timestamp.
        fresh = {r: im for r, im in rgbs.items() if 0 <= t-view_t[r] <= MAX_FRAME_AGE_S
                 and view_t[r] > self.last_view_time.get(r, -np.inf)}
        if fresh:
            estimate = super().update(fresh, grip, t=t, view_t=view_t, **kwargs)
            self.last_view_time.update({r: view_t[r] for r in fresh})
        else:
            if self.x is None:
                raise ValueError('Cannot initialize from missing images')
            grip = np.asarray(grip, float)
            s = np.linspace(0, 1, len(self.x))[:, None]
            prior = self.x+(1-s)*(grip[:3]-self.grip[:3])+s*(grip[3:]-self.grip[3:])
            self.x = chain(prior, grip, self.L/(len(self.x)-1))
            self.grip, self.t = grip.copy(), float(t)
            estimate = self.x.copy()
            self.image_evidence = {}
        support = np.zeros((len(estimate), 2), bool)
        for j, role in enumerate(['front', 'top']):
            if role not in self.image_evidence: continue
            uv, z = cm.project(estimate, self.cams[role])
            u, v = np.rint(uv).astype(int).T
            inside = (z > 0) & (u >= 0) & (u < cm.W) & (v >= 0) & (v < cm.H)
            uc, vc = np.clip(u, 0, cm.W-1), np.clip(v, 0, cm.H-1)
            dt, occ = self.image_evidence[role]
            support[:, j] = inside & (dt[vc, uc] <= 2.) & ~occ[vc, uc]
        quality = self.quality.update(t, support, self.node_sd)
        return dict(centerline=estimate, **quality)
