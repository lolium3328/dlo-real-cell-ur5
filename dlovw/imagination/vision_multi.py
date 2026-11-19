"""Depth-camera readings for several obstacles (docs/notes/vision_deployment_design_20261108.md, sections 2-3, step 1).

Differences from vision.VisionPerception (one obstacle):
- Occlusion happens in the rendered images. The scenes' occluders are half-spaces that cannot be rendered as
  objects, so the simulated world (the session, `hidden_now`) decides from the true position whether a sphere
  is inside its occluder, and that sphere is left out of both cameras' renders (geom group 5, not drawn). The
  detector never sees a true position: a sphere it cannot see in the images is simply not detected.
- Every obstacle sphere has its own colour in the model (two spheres: red, blue). The obstacle's identity is
  its colour, so there is no clustering or association step (the design's recommendation for the real robot).
- Per obstacle: pixels of its colour; per camera, pixels whose depth differs from that mask's median depth by
  more than one radius are dropped (edge pixels carrying background depth, as perception.py's I0 pipeline);
  cameras with fewer than MIN_SPHERE_PX pixels left do not count; least-squares centre with the known radius;
  a fit whose RMS residual exceeds MAX_FIT_RMS is not reported. Velocity: 5-frame least-squares slope
  (vision.ObstacleFrames), results available one frame later.
- Statistics against the truth are logged for evaluation only (`truth_log`), after each step, never read back.
"""
from __future__ import annotations

import time

import mujoco
import numpy as np

from .perception import CAM_STEPS
from .vision import CAMS, DT, H, MIN_SPHERE_PX, ROPE_R, SPHERE_R, W, ObstacleFrames, RopeTracker, VisionPerception, fit_sphere, segment

HIDDEN_GROUP = 5                    # not drawn with mujoco's default visual options (groups 0-2 on)
MAX_FIT_RMS = 0.004                 # m: larger residuals mean a partly hidden or merged blob
CHROMA_TOL = 0.12                   # distance in normalised rgb from the sphere's own colour


def colour_mask(rgb, colour):
    """Pixels of an obstacle sphere: chromaticity close to the geom's rgba, not too dark."""
    s = rgb.sum(-1, keepdims=True).clip(1e-6)
    ref = np.asarray(colour[:3], float)
    ref = ref / ref.sum()
    return (np.linalg.norm(rgb / s - ref, axis=-1) < CHROMA_TOL) & (rgb.max(-1) > 0.25)


class MultiCameras:
    """top_view and front_view; per frame the rope points and, per obstacle, its sphere points (world frame)."""

    def __init__(self, model, sphere_geoms, cams=CAMS, depth_noise=0.0, dropout=0.0, seed=0):
        self.model, self.k, self.dropout = model, float(depth_noise), float(dropout)
        self.rng = np.random.default_rng(seed)
        self.ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, c) for c in cams]
        self.geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in sphere_geoms]
        if min(self.geoms) < 0:
            raise ValueError(f"sphere geoms not found: {sphere_geoms}")
        self.groups = [int(model.geom_group[g]) for g in self.geoms]
        self.colours = [np.array(model.geom_rgba[g], float) for g in self.geoms]
        self.rgb, self.depth = mujoco.Renderer(model, H, W), mujoco.Renderer(model, H, W)
        self.depth.enable_depth_rendering()
        v, u = np.mgrid[0:H, 0:W]
        self.u, self.v = u - (W - 1) / 2, -(v - (H - 1) / 2)

    def _render(self, data, cid, hidden):
        for g, grp, h in zip(self.geoms, self.groups, hidden):
            self.model.geom_group[g] = HIDDEN_GROUP if h else grp
        try:
            self.rgb.update_scene(data, camera=cid)
            rgb = self.rgb.render().astype(np.float32) / 255.0
            self.depth.update_scene(data, camera=cid)
            depth = self.depth.render()
        finally:
            for g, grp in zip(self.geoms, self.groups):
                self.model.geom_group[g] = grp
        return rgb, depth

    def capture(self, data, hidden):
        """hidden[j]: sphere j is inside its occluder (decided by the simulated world). Returns, per camera,
        (rope points, [sphere j points], camera position)."""
        out = []
        for cid in self.ids:
            rgb, depth = self._render(data, cid, hidden)
            valid = np.ones(depth.shape, bool)
            if self.k:
                depth = depth + self.rng.normal(0.0, 1.0, depth.shape).astype(np.float32) * (self.k * depth ** 2)
            if self.dropout:
                valid = self.rng.uniform(size=depth.shape) >= self.dropout
            f = 0.5 * H / np.tan(np.radians(self.model.cam_fovy[cid]) / 2)
            R, pos = data.cam_xmat[cid].reshape(3, 3), data.cam_xpos[cid]

            def points(mask):
                d = depth[mask]
                local = np.stack([self.u[mask] * d / f, self.v[mask] * d / f, -d], -1)
                return local @ R.T + pos

            rope_mask, _ = segment(rgb)
            rope = points(rope_mask & valid)
            if len(rope):
                ray = rope - pos
                rope = rope + (np.pi * ROPE_R / 4) * ray / np.linalg.norm(ray, axis=1, keepdims=True)
            spheres = []
            for c in self.colours:
                m = colour_mask(rgb, c) & valid
                if m.sum():
                    m &= np.abs(depth - np.median(depth[m])) <= SPHERE_R
                spheres.append(points(m))
            out.append((rope, spheres, pos.copy()))
        return out

    def close(self):
        for r in (self.rgb, self.depth):
            r.close()


class MultiVisionPerception(VisionPerception):
    """VisionPerception (rope tracker unchanged) with one camera reading per obstacle. `update` takes the
    hidden flags from the simulated world and the true obstacle states for the statistics only."""

    def __init__(self, env, sphere_geoms, n_pts=39, depth_noise=0.0, dropout=0.0, seed=0, window=5):
        self.env, self.tracker, self.n = env, RopeTracker(n_pts), n_pts
        self.cams = MultiCameras(env.model, sphere_geoms, depth_noise=depth_noise, dropout=dropout, seed=seed)
        self.J = len(sphere_geoms)
        self.est = self.est_grip = None
        self.pending = None
        self.obsts = [ObstacleFrames(window) for _ in range(self.J)]
        self.steps, self.err, self.frame_ms = 0, [], []
        self.frames_seen = 0
        self.truth_log = []            # per frame: (step, [(hidden, true centre, detection or None, fit rms)])
        self.step_log = []             # per control step: (step, [(true centre, true velocity, reading now or None)])

    def _process_multi(self, grip, L, hidden, truth):
        t0 = time.perf_counter()
        frames = self.cams.capture(self.env.data, hidden)
        rope = np.concatenate([f[0] for f in frames])
        prior = self.tracker.initial(rope, grip) if self.est is None else self.estimate(grip, self._J, L)
        x = self.tracker.update(rope, grip, prior, L / (self.n - 1))
        dets, row = [], []
        for j in range(self.J):
            pts = [f[1][j] for f in frames if len(f[1][j]) >= MIN_SPHERE_PX]
            det, rms = None, None
            if pts:
                P = np.concatenate(pts)
                c = fit_sphere(P, [f[2] for f in frames])
                rms = float(np.sqrt(np.mean((np.linalg.norm(P - c, axis=1) - SPHERE_R) ** 2)))
                det = c if rms <= MAX_FIT_RMS else None
            dets.append(det)
            row.append((bool(hidden[j]), None if truth is None else np.asarray(truth[j][0], float).copy(), det, rms))
        self.truth_log.append((self.steps, row))
        self.frame_ms.append(1000 * (time.perf_counter() - t0))
        self.frames_seen += 1
        return x, dets

    def update(self, true_c, grip, hidden, truth, J, L):
        """Call after every control step. hidden: per obstacle, inside its occluder now (from the simulated
        world); truth: per obstacle (true centre, true velocity), logged only. Returns the centerline estimate."""
        grip = np.asarray(grip, float)
        self._J = J
        if self.est is None or self.steps % CAM_STEPS == 0:
            if self.pending is not None:
                k, x, g, dets = self.pending
                self.est, self.est_grip = x, g
                self.frame_id = getattr(self, "frame_id", 0) + 1
                for ob, d in zip(self.obsts, dets):
                    ob.publish(k, d)
            x, dets = self._process_multi(grip, L, hidden, truth)
            self.pending = (self.steps, x, grip.copy(), dets)
            if self.est is None:
                self.est, self.est_grip = x, grip.copy()
                self.frame_id = 1
        if truth is not None:
            now = []
            for j, ob in enumerate(self.obsts):
                r = ob.reading()
                now.append((np.asarray(truth[j][0], float).copy(), np.asarray(truth[j][1], float).copy(),
                            None if r is None else (r[1] + r[2] * (self.steps - r[0]) * DT, r[2].copy())))
            self.step_log.append((self.steps, now))
        self.steps += 1
        out = self.estimate(grip, J, L)
        self.err.append(float(np.sqrt(((out - np.asarray(true_c)) ** 2).sum(-1).mean())))
        return out

    def obstacle_reading(self, j=0):
        return self.obsts[j].reading()

    def summary(self, every=10):
        """Reading statistics against the truth (design doc step 1): per frame detection errors and misses,
        per decision (every `every` control steps) the reading carried to the present."""
        fr = [(j, h, t, d, rms) for _, row in self.truth_log for j, (h, t, d, rms) in enumerate(row)]
        vis = [x for x in fr if not x[1]]
        det_err = [float(np.linalg.norm(d - t)) for _, h, t, d, _ in vis if d is not None and t is not None]
        dec = [(t, v, r) for n, now in self.step_log if n % every == 0 for t, v, r in now]
        pos_err = [float(np.linalg.norm(r[0] - t)) for t, v, r in dec if r is not None]
        vel_err = [float(np.linalg.norm(r[1] - v)) for t, v, r in dec if r is not None]
        return dict(frames=self.frames_seen, rope_rmse_mm=1000 * float(np.mean(self.err)) if self.err else None,
                    visible_frames=len(vis), missed_visible=sum(d is None for _, _, _, d, _ in vis),
                    rejected_fit=sum(d is None and rms is not None for _, _, _, d, rms in vis),
                    detected_hidden=sum(d is not None for _, h, _, d, _ in fr if h),
                    det_err_mm=[round(1000 * e, 2) for e in det_err],
                    dec_pos_err_mm=[round(1000 * e, 2) for e in pos_err], dec_vel_err_mps=[round(e, 4) for e in vel_err],
                    frame_ms_mean=float(np.mean(self.frame_ms)) if self.frame_ms else None)


class MultiVisionSensor:
    """Stands in for belief.Sensor for obstacle j: its latest camera detection, with its frame time in the
    belief's steps (as vision.VisionSensor)."""

    def __init__(self, vision, j, occluder):
        self.vision, self.j, self.occluder = vision, int(j), occluder

    def read(self, positions, velocities):
        r = self.vision.obstacle_reading(self.j)
        if r is None:
            return None
        age = self.vision.steps - r[0]
        return max(len(positions) - 1 - age, 0), r[1], r[2]

    def frame_id(self):
        r = self.vision.obstacle_reading(self.j)
        return None if r is None else r[0]
