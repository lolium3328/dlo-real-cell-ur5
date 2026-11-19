"""Depth-camera perception in simulation (docs/notes/perception_robustness_20261023.md, section 9).

Two cameras (top_view, front_view) render RGB and depth (640x480) every CAM_STEPS control steps. The yellow
rope and the red sphere are segmented by colour and back-projected to world points. Every visible point
is moved pi*r/4 further along its viewing ray: the mean depth of a pixel-uniformly sampled visible half
cylinder lies that far in front of its axis.

Rope: a 39-node tracker. Soft EM assignment of the points to the nodes, with the previous estimate
(carried along with the grippers by the session's local Jacobian) as a prior, a light smoothing, then an
inextensible chain with both ends pinned at the grippers (known from the robot's kinematics). Nodes
without nearby points keep the prior.
Sphere: least-squares centre with the known radius. Velocity: least-squares slope over the detections of the
last 0.2 s (at most three frames); a lone detection (e.g. the first one after an occlusion) has velocity zero.
The scenes' occluders are virtual (not rendered), so while the true obstacle is inside its occluder slab
the detector reports nothing, which keeps the scenario's occlusion.
Results of a frame become available one frame later (processing latency). `ObstacleFrames` holds this timing
and the velocity fit; `CameraObstacleSensor` runs the same logic on a true track (position noise instead of
rendering) for the obstacle forecaster's synthetic training data (obstacle_data.py, sensor="camera").
"""
from __future__ import annotations

import mujoco
import numpy as np

from ..motion import hidden
from .perception import CAM_STEPS

DT = 0.008

ROPE_R, SPHERE_R = 0.010, 0.040
CAMS = ("top_view", "front_view")
H, W = 480, 640
SIGMA = 0.006                       # EM assignment width (m)
PRIOR_W = 3.0                       # prior weight, in points
MIN_SPHERE_PX = 40


def segment(rgb):
    """Boolean masks (rope, sphere) from an RGB image in [0, 1]."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rope = (r > 0.45) & (g > 0.35) & (b < 0.30) & (g > 0.6 * r)
    sphere = (r > 0.40) & (g < 0.30) & (b < 0.30) & (g < 0.5 * r)
    return rope, sphere


class Cameras:
    def __init__(self, model, cams=CAMS, depth_noise=0.0, dropout=0.0, seed=0):
        """depth_noise k: per-pixel Gaussian depth noise with sd k * z^2 (z in m), like a structured-light or
        stereo depth camera; dropout: share of pixels with no depth."""
        self.model, self.k, self.dropout = model, float(depth_noise), float(dropout)
        self.rng = np.random.default_rng(seed)
        self.ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, c) for c in cams]
        self.rgb, self.depth = mujoco.Renderer(model, H, W), mujoco.Renderer(model, H, W)
        self.depth.enable_depth_rendering()
        v, u = np.mgrid[0:H, 0:W]
        self.u, self.v = u - (W - 1) / 2, -(v - (H - 1) / 2)

    def capture(self, data):
        """List of (rope points, sphere points) in the world frame, per camera (surface-offset corrected)."""
        out = []
        for cid in self.ids:
            self.rgb.update_scene(data, camera=cid)
            rgb = self.rgb.render().astype(np.float32) / 255.0
            self.depth.update_scene(data, camera=cid)
            depth = self.depth.render()
            valid = np.ones(depth.shape, bool)
            if self.k:
                depth = depth + self.rng.normal(0.0, 1.0, depth.shape).astype(np.float32) * (self.k * depth ** 2)
            if self.dropout:
                valid = self.rng.uniform(size=depth.shape) >= self.dropout
            f = 0.5 * H / np.tan(np.radians(self.model.cam_fovy[cid]) / 2)
            R, pos = data.cam_xmat[cid].reshape(3, 3), data.cam_xpos[cid]
            rope, sphere = segment(rgb)
            rope, sphere = rope & valid, sphere & valid
            res = []
            for mask, offset in ((rope, np.pi * ROPE_R / 4), (sphere, None)):
                d = depth[mask]
                local = np.stack([self.u[mask] * d / f, self.v[mask] * d / f, -d], -1)
                pts = local @ R.T + pos
                if offset is not None and len(pts):
                    ray = pts - pos
                    pts = pts + offset * ray / np.linalg.norm(ray, axis=1, keepdims=True)
                res.append(pts)
            out.append((res[0], res[1], pos.copy()))
        return out


def fit_sphere(pts, cam_positions):
    """Centre of a sphere of known radius from surface points (Gauss-Newton from a shifted centroid)."""
    c = pts.mean(0)
    ray = c - np.mean(cam_positions, 0)
    c = c + (2 * SPHERE_R / 3) * ray / np.linalg.norm(ray)
    for _ in range(10):
        diff = pts - c
        dist = np.linalg.norm(diff, axis=1).clip(1e-9)
        r = dist - SPHERE_R
        Jm = -diff / dist[:, None]
        step = np.linalg.lstsq(Jm, -r, rcond=None)[0]
        c = c + step
        if np.linalg.norm(step) < 1e-6:
            break
    return c


def chain(x, grip, seg, iters=30):
    """Inextensible chain with both ends pinned at the grippers (position-based projection)."""
    x = x.copy()
    x[0], x[-1] = grip[:3], grip[3:]
    n = len(x)
    for _ in range(iters):
        for i in range(n - 1):
            d = x[i + 1] - x[i]
            ln = np.linalg.norm(d)
            if ln < 1e-9:
                continue
            corr = (ln - seg) * d / ln
            if i == 0:
                x[i + 1] -= corr
            elif i + 1 == n - 1:
                x[i] += corr
            else:
                x[i] += 0.5 * corr
                x[i + 1] -= 0.5 * corr
        x[0], x[-1] = grip[:3], grip[3:]
    return x


class RopeTracker:
    def __init__(self, n=39):
        self.n = n

    def initial(self, pts, grip):
        """First estimate: bin the points by their position along the gripper-to-gripper line."""
        a, b = grip[:3], grip[3:]
        t = np.clip((pts - a) @ (b - a) / max(np.dot(b - a, b - a), 1e-9), 0, 1)
        x = np.linspace(a, b, self.n)
        for k, tk in enumerate(np.linspace(0, 1, self.n)):
            sel = np.abs(t - tk) < 0.5 / (self.n - 1)
            if sel.sum() >= 3:
                x[k] = pts[sel].mean(0)
        return x

    def update(self, pts, grip, prior, seg):
        x = prior.copy()
        if len(pts) >= 10:
            for _ in range(5):
                d2 = ((pts[:, None] - x[None]) ** 2).sum(-1)                       # (M, N)
                w = np.exp(-(d2 - d2.min(1, keepdims=True)) / (2 * SIGMA ** 2))
                w = w / w.sum(1, keepdims=True)
                w = w * (d2.min(1, keepdims=True) < (4 * SIGMA) ** 2)              # drop far outliers
                mass = w.sum(0)
                x = (w.T @ pts + PRIOR_W * prior) / (mass + PRIOR_W)[:, None]
        x[1:-1] = 0.25 * x[:-2] + 0.5 * x[1:-1] + 0.25 * x[2:]
        return chain(x, grip, seg)


class ObstacleFrames:
    """Camera-rate obstacle readings: a detection made at frame step k becomes available one frame later; its
    velocity is the least-squares slope over the detections of the last 0.2 s (at most `window` frames, three for
    the 30 Hz camera)."""

    WINDOW_S = 0.2

    def __init__(self, window=3):
        self.window = int(window)
        self.recent, self.det, self.vel = [], None, np.zeros(3)

    def publish(self, k, det):
        """The frame taken at step k becomes available; det is the detected centre or None."""
        if det is None:
            self.det = None
            return
        keep = [(kk, c) for kk, c in self.recent if (k - kk) * DT < self.WINDOW_S]
        self.recent = keep[max(len(keep) - (self.window - 1), 0):] + [(k, np.asarray(det, float))]
        if len(self.recent) >= 2:
            t = np.array([kk for kk, _ in self.recent], float) * DT
            P = np.stack([c for _, c in self.recent])
            tc = t - t.mean()
            self.vel = (tc[:, None] * (P - P.mean(0))).sum(0) / (tc ** 2).sum()
        else:
            self.vel = np.zeros(3)
        self.det = (k, self.recent[-1][1], self.vel.copy())

    def reading(self):
        """(frame step, position, velocity) of the latest available detection, or None."""
        return None if self.det is None else (self.det[0], self.det[1].copy(), self.det[2].copy())


class CameraObstacleSensor:
    """belief.Sensor-like stream from a true track with the camera's timing: a frame every `cam_steps` control
    steps (random phase), available `latency_frames` frames later (0 or 1), Gaussian position noise and a fixed
    offset, an occasional missed frame, nothing while the obstacle is inside its occluder; velocity fitted over
    the last `window` frames. Defaults are the depth cameras of vision.py (30 Hz, one frame, three frames).
    `read` returns the reading carried forward to the present with its velocity (as
    PerceivedActorSession.measurement does with comp=True)."""

    def __init__(self, occluder, rng, noise_m=0.0005, bias_m=0.0, miss=0.0, cam_steps=CAM_STEPS, latency_frames=1, window=3):
        self.occluder, self.rng, self.noise, self.miss = occluder, rng, float(noise_m), float(miss)
        self.cam_steps, self.latency = int(cam_steps), int(latency_frames)
        if self.latency not in (0, 1):
            raise ValueError("latency_frames must be 0 or 1")
        d = rng.normal(size=3)
        self.bias = float(bias_m) * d / np.linalg.norm(d)
        self.phase = int(rng.integers(self.cam_steps))
        self.frames, self.pending = ObstacleFrames(window), None

    def read(self, positions, velocities):
        n = len(positions) - 1
        if (n - self.phase) % self.cam_steps == 0:
            if self.pending is not None:
                self.frames.publish(*self.pending)
                self.pending = None
            p = positions[n]
            seen = not hidden(p, self.occluder) and self.rng.uniform() >= self.miss
            frame = (n, p + self.bias + self.rng.normal(0.0, self.noise, 3) if seen else None)
            if self.latency:
                self.pending = frame
            else:
                self.frames.publish(*frame)
        r = self.frames.reading()
        if r is None:
            return None
        k, pos, vel = r
        return n, pos + vel * (n + 1 - k) * DT, vel

    def frame_id(self):
        """Frame step of the latest available detection, or None (a new value = a new camera reading)."""
        return None if self.frames.det is None else self.frames.det[0]


class VisionPerception:
    """Same interface as perception.RopePerception (update -> centerline estimate), plus the obstacle
    detection for the belief: `obstacle_reading(step)` -> (step of the frame, position, velocity) or None."""

    def __init__(self, env, occluder=None, n_pts=39, depth_noise=0.0, dropout=0.0, seed=0, window=3):
        """window: frames in the obstacle's velocity fit (3 in sections 9-11; 5 from section 13 on)."""
        self.env, self.tracker, self.n = env, RopeTracker(n_pts), n_pts
        self.cams = Cameras(env.model, depth_noise=depth_noise, dropout=dropout, seed=seed)
        self.occluder = occluder
        self.est = self.est_grip = None
        self.pending = None                    # (frame step, centerline, grip, detection) of the last frame
        self.obst = ObstacleFrames(window)
        self.steps, self.err, self.obst_err, self.frame_ms = 0, [], [], []
        self.frames_seen = self.frames_detected = 0

    def _process(self, grip, L, obstacle):
        import time
        t0 = time.perf_counter()
        frames = self.cams.capture(self.env.data)
        rope = np.concatenate([f[0] for f in frames])
        sph = [f[1] for f in frames if len(f[1]) >= MIN_SPHERE_PX]
        seg = L / (self.n - 1)
        if self.est is None:
            prior = self.tracker.initial(rope, grip)
        else:
            prior = self.estimate(grip, self._J, L)
        x = self.tracker.update(rope, grip, prior, seg)
        det = None
        if sph and not (obstacle is not None and hidden(obstacle, self.occluder)):
            c = fit_sphere(np.concatenate(sph), [f[2] for f in frames])
            det = c
            if obstacle is not None:
                self.obst_err.append(float(np.linalg.norm(c - obstacle)))
        self.frame_ms.append(1000 * (time.perf_counter() - t0))
        self.frames_seen += 1
        self.frames_detected += det is not None
        return x, det

    def update(self, true_c, grip, obstacle, J, L):
        """Call after every control step. Returns the current centerline estimate (N, 3)."""
        grip = np.asarray(grip, float)
        self._J = J
        if self.est is None or self.steps % CAM_STEPS == 0:
            if self.pending is not None:       # the previous frame's result becomes available now
                k, x, g, det = self.pending
                self.est, self.est_grip = x, g
                self.frame_id = getattr(self, "frame_id", 0) + 1
                self.obst.publish(k, det)
            x, det = self._process(grip, L, obstacle)
            self.pending = (self.steps, x, grip.copy(), det)
            if self.est is None:               # nothing to show yet: use the first frame at once
                self.est, self.est_grip = x, grip.copy()
                self.frame_id = 1
        self.steps += 1
        out = self.estimate(grip, J, L)
        self.err.append(float(np.sqrt(((out - np.asarray(true_c)) ** 2).sum(-1).mean())))
        return out

    def estimate(self, grip, J, L):
        if self.est is None:
            return None
        x = self.est.copy() if J is None else (self.est.reshape(-1) + J @ (grip - self.est_grip)).reshape(self.n, 3)
        return chain(x, grip, L / (self.n - 1), iters=10)

    def close(self):
        for r in (self.cams.rgb, self.cams.depth):
            r.close()

    def obstacle_reading(self):
        """(frame step, position, velocity) of the latest available detection, or None."""
        return self.obst.reading()


class VisionSensor:
    """Stands in for belief.Sensor: the latest camera detection, with its frame time in the belief's steps."""

    def __init__(self, vision):
        self.vision, self.occluder = vision, vision.occluder

    def read(self, positions, velocities):
        r = self.vision.obstacle_reading()
        if r is None:
            return None
        age = self.vision.steps - r[0]
        return max(len(positions) - 1 - age, 0), r[1], r[2]

    def frame_id(self):
        r = self.vision.obstacle_reading()
        return None if r is None else r[0]
