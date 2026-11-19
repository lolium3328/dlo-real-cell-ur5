"""Closed-loop RGB-only perception (perception preset V3*, docs/notes/perception_rgb_20261110.md).

Same interface and timing as vision_multi.MultiVisionPerception (V2: depth), with what the real cell has:
- the calibrated D435i pair (camera_model "real": poses and intrinsics), colour images only;
- sensor noise (5 grey levels), the dark cloth's brightness (26), and the two views taken 16 ms apart: the front
  image at the camera step k, the top image two control steps later; the frame is processed then and its results
  become available at the next camera step (one frame of latency, as V2); its time stamp is k + 1;
- rope: rgb_rope.RopeRGB (ends at the grippers, known length, occlusion-aware fit);
- spheres: rgb_obstacles.locate (identity by colour, circle fit on the trusted outline, two-ray triangulation);
  velocity from the 5-frame fit of vision.ObstacleFrames.
Spheres inside their occluders are left out of the renders (the simulated world's decision, as V2).
Options: preset (camera model), noise_sd, background, offset_steps (0 = synchronised views), naive (plain sphere fit).
"""
from __future__ import annotations

import time

import mujoco
import numpy as np

from . import camera_model as cm
from .perception import CAM_STEPS
from .rgb_obstacles import locate
from .rgb_rope import RopeRGB
from .vision import ObstacleFrames
from .vision_multi import HIDDEN_GROUP, MultiVisionPerception


class RGBPerception(MultiVisionPerception):
    def __init__(self, env, sphere_geoms, n_pts=39, seed=0, window=5, preset="real", noise_sd=5.0, background=26.0,
                 offset_steps=2, naive=False, L=0.531):
        self.env, self.n = env, n_pts
        m = env.model
        self.camset = cm.PRESETS[preset]
        cm.apply(m, self.camset)
        self.geoms = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in sphere_geoms]
        self.groups = [int(m.geom_group[g]) for g in self.geoms]
        self.colours = [np.array(m.geom_rgba[g], float) for g in self.geoms]
        if background:
            for name in ("table", "front_depth_backdrop", "floor"):
                g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
                if g >= 0:
                    m.geom_rgba[g, :3] = background / 255.0
                    if m.geom_matid[g] >= 0:
                        m.mat_rgba[m.geom_matid[g], :3] = background / 255.0
        self.renderer = mujoco.Renderer(m, cm.H, cm.W)
        self.rng = np.random.default_rng(seed)
        self.noise_sd, self.offset, self.naive = float(noise_sd), int(offset_steps), bool(naive)
        if not 0 <= self.offset < CAM_STEPS:
            raise ValueError("offset_steps must lie within one camera frame")
        self.rope = RopeRGB(self.camset, L=L, seed=seed)
        self.J = len(sphere_geoms)
        self.est = self.est_grip = None
        self.pending = self.front = None
        self.obsts = [ObstacleFrames(window) for _ in range(self.J)]
        self.steps, self.err, self.frame_ms = 0, [], []
        self.frames_seen = 0
        self.truth_log, self.step_log = [], []

    def _render(self, role, hidden):
        m = self.env.model
        for g, grp, h in zip(self.geoms, self.groups, hidden):
            m.geom_group[g] = HIDDEN_GROUP if h else grp
        try:
            mujoco.mj_kinematics(m, self.env.data)         # poses of the current qpos (mj_step leaves them one step old,
            mujoco.mj_camlight(m, self.env.data)           # and unset before the first step); dynamics untouched
            self.renderer.update_scene(self.env.data, camera=cm.MJ_NAMES[role])
            img = cm.warp(self.renderer.render(), self.camset[role])
        finally:
            for g, grp in zip(self.geoms, self.groups):
                m.geom_group[g] = grp
        if self.noise_sd:
            img = np.clip(img.astype(np.float32) + self.rng.normal(0, self.noise_sd, img.shape), 0, 255).astype(np.uint8)
        return img

    def _process(self, grip, hidden, truth):
        t0 = time.perf_counter()
        rgbs = dict(front=self.front[1], top=self._render("top", hidden))
        x = self.rope.update(rgbs, grip)
        dets, row = [], []
        for j, colour in enumerate(self.colours):
            p, ncam = locate(rgbs, self.camset, colour, naive=self.naive)
            dets.append(p)
            row.append((bool(hidden[j]), None if truth is None else np.asarray(truth[j][0], float).copy(), p, None))
        self.truth_log.append((self.steps, row))
        self.frame_ms.append(1000 * (time.perf_counter() - t0) + self.front[2])
        self.frames_seen += 1
        return x, dets

    def update(self, true_c, grip, hidden, truth, J, L):
        grip = np.asarray(grip, float)
        self._J = J
        phase = self.steps % CAM_STEPS
        if phase == 0:
            if self.pending is not None:
                k, x, g, dets = self.pending
                self.est, self.est_grip = x, g
                self.frame_id = getattr(self, "frame_id", 0) + 1
                for ob, d in zip(self.obsts, dets):
                    ob.publish(k, d)
                self.pending = None
            t0 = time.perf_counter()
            self.front = (self.steps, self._render("front", hidden), 1000 * (time.perf_counter() - t0))
            if self.est is None:                     # first frame: both views now, used at once (no truth before it)
                x, dets = self._process(grip, hidden, truth)
                self.est, self.est_grip, self.frame_id = x, grip.copy(), 1
                self.pending, self.front = (self.steps, x, grip.copy(), dets), None
        if self.front is not None and phase == self.offset:
            k = self.front[0]
            x, dets = self._process(grip, hidden, truth)
            self.pending, self.front = (k + self.offset // 2, x, grip.copy(), dets), None
        if truth is not None:
            now = []
            for j, ob in enumerate(self.obsts):
                r = ob.reading()
                now.append((np.asarray(truth[j][0], float).copy(), np.asarray(truth[j][1], float).copy(),
                            None if r is None else (r[1] + r[2] * (self.steps - r[0]) * 0.008, r[2].copy())))
            self.step_log.append((self.steps, now))
        self.steps += 1
        out = self.estimate(grip, J, L)
        self.err.append(float(np.sqrt(((out - np.asarray(true_c)) ** 2).sum(-1).mean())))
        return out

    def close(self):
        self.renderer.close()
