"""Rope perception model for the robustness test (docs/notes/perception_robustness_20261023.md).

A camera frame every CAM_STEPS control steps shows the true centerline of `latency_frames` frames ago with
- per-point noise sigma_p,
- a smooth shape error: the first three cosine modes along the arc length, total rms sigma_f,
- a fixed per-episode offset of size `bias` (calibration),
- points within OCCLUDE_R of the obstacle centre unobserved (they keep the previous estimate).
Like a rope tracker on a real robot, the output is a plausible curve: the ends are anchored at the grippers
(known from the robot's kinematics; the perturbations taper to zero over the last TAPER points), the
detections are smoothed, and the curve is smoothed further until it is no longer than the known cable
length (the safety filter rejects a centerline longer than 1.02 L). Between frames the estimate is carried
along with the grippers by the session's local Jacobian (x += J dy). sigma_p / sigma_f / bias are therefore
raw detection errors; the effective error is measured and reported (perceived_rmse).
"""
from __future__ import annotations

from collections import deque

import numpy as np

CAM_STEPS = 4                         # 32 ms, about 30 Hz
OCCLUDE_R = 0.06
TAPER = 4                             # points over which perturbations fade to zero at the gripper ends
LENGTH_SLACK = 1.005

PRESETS = {
    "C1": dict(latency_frames=1),
    "C2": dict(latency_frames=1, sigma_p=0.002),
    "C3": dict(latency_frames=1, sigma_p=0.005),
    "C4": dict(latency_frames=1, sigma_p=0.010),
    "C5": dict(latency_frames=1, sigma_f=0.005, bias=0.005),
    "C6": dict(latency_frames=1, sigma_p=0.003, sigma_f=0.003, bias=0.005, occlude=True),
    "C7": dict(latency_frames=2, sigma_p=0.006, sigma_f=0.006, bias=0.010, occlude=True),
    # added after the smoke test showed effective errors of 3-12 mm for C1-C7 (to find where it breaks)
    "C8": dict(latency_frames=3, sigma_p=0.010, sigma_f=0.010, bias=0.015, occlude=True),
    "C9": dict(latency_frames=1, sigma_f=0.015, bias=0.020),
    "V": dict(kind="vision"),        # rendered depth cameras, segmentation and tracking (vision.py)
    "Vp": dict(kind="vision", passive=True),   # vision runs alongside, control still uses the true state
    "Vn": dict(kind="vision", depth_noise=0.004, dropout=0.01),   # depth noise ~2 mm at 0.7 m, 3.6 mm at 0.95 m
    "Vnp": dict(kind="vision", depth_noise=0.004, dropout=0.01, passive=True),
    "Vr": dict(kind="vision", obstacle=False),      # cameras for the rope only; obstacle from the scene's sensor model
    "Vo": dict(kind="vision", rope=False),          # cameras for the obstacle only; rope from the simulator
    "Vc": dict(kind="vision", comp=True),           # + obstacle reading carried forward by its age
    "Vnc": dict(kind="vision", depth_noise=0.004, dropout=0.01, comp=True),
    # control without rendering: the obstacle reading follows the cameras' timing and velocity fit on the true
    # track (vision.CameraObstacleSensor, 0.5 mm noise); rope from the simulator. Vt vs Vo isolates detection.
    "Vt": dict(kind="camtiming", rope=False),
}
# section 13: the real depth pipeline with the 5-frame velocity window chosen in section 12
PRESETS["Vc5"] = dict(kind="vision", comp=True, window=5)
PRESETS["Vnc5"] = dict(kind="vision", depth_noise=0.004, dropout=0.01, comp=True, window=5)
# vision_deployment_design_20261108.md step 1: one camera reading per obstacle (vision_multi.py), spheres left out of the
# renders inside their occluders, identity by colour, 5-frame velocity, readings carried to the present
PRESETS["V2"] = dict(kind="vision2", comp=True, window=5)
PRESETS["V2o"] = dict(kind="vision2", comp=True, window=5, rope=False)      # rope from the simulator
PRESETS["V2n"] = dict(kind="vision2", comp=True, window=5, depth_noise=0.004, dropout=0.01)  # depth noise as Vn
PRESETS["V2p"] = dict(kind="vision2", comp=True, window=5, passive=True)    # cameras alongside, control on the scene sensors
# perception_rgb_20261110.md: colour images only (vision_rgb.py), calibrated real cameras, noise 5, background 26, top view
# 16 ms after the front view; occlusion-aware rope (rgb_rope.RopeRGB) and trusted-outline sphere fits (rgb_obstacles)
PRESETS["V3"] = dict(kind="rgb", comp=True, window=5)
PRESETS["V3o"] = dict(kind="rgb", comp=True, window=5, rope=False)                     # rope from the simulator
PRESETS["V3c"] = dict(kind="rgb", comp=True, window=5, noise_sd=0.0, background=0.0, offset_steps=0)   # clean images
# perception_rgb_20261110.md section 8: deployment mode (vision_rgb_deploy.py): the time-stamped front end, per-image sphere
# Kalman filters, results available after the assumed latency (spheres 40 ms, rope 100 ms after a fit every 0.1 s)
PRESETS["V4"] = dict(kind="rgb", deploy=True, comp=True)
PRESETS["V4o"] = dict(kind="rgb", deploy=True, comp=True, rope=False)                  # rope from the simulator
# rope from the RGB front end, obstacle spheres known (scene sensor; on the real cell measured by other means)
PRESETS["V4r"] = dict(kind="rgb", deploy=True, comp=True, obstacle=False, track_spheres=False)
PRESETS["V4rz"] = dict(kind="rgb", deploy=True, comp=True, obstacle=False, track_spheres=False, rope_latency=0)    # no latency
PRESETS["V4rc"] = dict(kind="rgb", deploy=True, comp=True, obstacle=False, track_spheres=False, noise_sd=0.0, background=0.0,
                       offset_steps=0)                                                 # clean synchronised images
PRESETS["V4z"] = dict(kind="rgb", deploy=True, comp=True, sphere_latency=0, rope_latency=0, rope_period=0.0)  # no latency
# section 12: camera rate x latency sweep (Vt<Hz>, z = no processing latency); velocity window chosen offline per rate
for _steps, _hz, _win in ((4, 30, 5), (2, 60, 9), (1, 120, 17)):
    for _lat in (1, 0):
        PRESETS[f"Vt{_hz}{'' if _lat else 'z'}"] = dict(kind="camtiming", rope=False, cam_steps=_steps, latency_frames=_lat, window=_win)


class RopePerception:
    def __init__(self, latency_frames=1, sigma_p=0.0, sigma_f=0.0, bias=0.0, occlude=False, seed=0, n_pts=39, passive=False,
                 kind="model", **_):
        self.rng = np.random.default_rng(seed)
        self.lat = int(latency_frames) * CAM_STEPS
        self.sigma_p, self.sigma_f, self.occlude, self.n = float(sigma_p), float(sigma_f), bool(occlude), n_pts
        d = self.rng.normal(size=3)
        self.bias = float(bias) * d / np.linalg.norm(d)
        u = np.linspace(0.0, 1.0, n_pts)
        self.modes = np.stack([np.cos(np.pi * k * u) for k in (1, 2, 3)])            # (3, N)
        i = np.arange(n_pts)
        self.taper = np.sin(0.5 * np.pi * np.clip(np.minimum(i, n_pts - 1 - i) / TAPER, 0.0, 1.0)) ** 2
        self.hist = deque(maxlen=self.lat + 1)
        self.est = self.est_grip = None
        self.steps = 0
        self.err = []

    @staticmethod
    def _smooth(x):
        y = x.copy()
        y[1:-1] = 0.25 * x[:-2] + 0.5 * x[1:-1] + 0.25 * x[2:]
        return y

    def _fit(self, x, grip, L):
        """Anchor the ends at the grippers, then smooth until the curve is not longer than the cable."""
        x = x.copy()
        x[0], x[-1] = grip[:3], grip[3:]
        for _ in range(40):
            if np.linalg.norm(np.diff(x, axis=0), axis=1).sum() <= LENGTH_SLACK * L:
                break
            x = self._smooth(x)
        return x

    def _frame(self, true_c, grip, L):
        pert = np.broadcast_to(self.bias, true_c.shape).copy()
        if self.sigma_f:
            amp = self.rng.normal(0.0, self.sigma_f / np.sqrt(3 * 0.5), size=(3, 3))   # mode rms 1/sqrt(2)
            pert += self.modes.T @ amp
        if self.sigma_p:
            noise = self.rng.normal(0.0, self.sigma_p / np.sqrt(3), size=true_c.shape)
            pert += self._smooth(noise)                                                # detections are smoothed
        return self._fit(true_c + pert * self.taper[:, None], grip, L)

    def update(self, true_c, grip, obstacle, J, L):
        """Call after every control step (L: the known cable length). Returns the current estimate (N, 3)."""
        true_c, grip = np.asarray(true_c, float), np.asarray(grip, float)
        self.hist.append((true_c.copy(), grip.copy(), None if obstacle is None else np.asarray(obstacle, float)))
        current = self.estimate(grip, J, L)
        if self.est is None or self.steps % CAM_STEPS == 0:
            c_then, g_then, o_then = self.hist[0]
            new = self._frame(c_then, g_then, L)
            if self.occlude and o_then is not None and current is not None:
                hidden = np.linalg.norm(c_then - o_then, axis=1) < OCCLUDE_R
                carried = (current.reshape(-1) + (0 if J is None else J @ (g_then - grip))).reshape(self.n, 3)
                new[hidden] = carried[hidden]
            self.est, self.est_grip = new, g_then
            self.frame_id = getattr(self, "frame_id", 0) + 1
        self.steps += 1
        out = self.estimate(grip, J, L)
        self.err.append(float(np.sqrt(((out - true_c) ** 2).sum(-1).mean())))
        return out

    def estimate(self, grip, J, L):
        if self.est is None:
            return None
        grip = np.asarray(grip, float)
        x = self.est.copy() if J is None else (self.est.reshape(-1) + J @ (grip - self.est_grip)).reshape(self.n, 3)
        return self._fit(x, grip, L)
