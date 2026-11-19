"""Deployment noise for the cell's policy (docs/notes/cell_policy_20261113.md).

RopeObsModel: what the policy sees of the rope when the centerline comes from the RGB tracker (cell_rgb_rope_20261112.md)
instead of the simulator. The tracker pins both ends (left TCP from kinematics, fixed clamp from the layout), so its
error is a smooth field along the rope that vanishes at the ends and changes slowly from frame to frame:

    e(s, f) = sum_k a_k(f) sin(k pi s) + white,      a_k(f) = rho a_k(f-1) + sqrt(1 - rho^2) sd_k xi

per axis, k = 1..K, sd_k chosen so that the interior rms is `sigma_m` (spectrum ~ 1/k). Frames every `frame_steps`
control steps (30 Hz: 4), each available `latency_steps` after capture; between frames the last estimate is held.
Near the sphere the error grows by `near_gain` within `near_m` of it (occlusion by the sphere and the right wrist),
fitted on the RGB study. cfg None: exact centerline, every step.

sample_noise(rng): one episode's deployment noise (rope_obs, obstacle, exec, dyn), ranges in DEFAULT_RANGES.
"""
from __future__ import annotations

import numpy as np

K_MODES = 4
# rope ranges fitted to the RGB study (cell_rgb_rope_20261112.md, R1 after a sphere-sweep recalibration from 20 mm / 6 deg
# off): interior rms per frame median 2.6 mm (p10 1.3, p90 4.9); p90 node error 6.7 mm within 12 cm of the sphere,
# 4.1 mm further; frame-to-frame correlation of the error 0.96 (mostly the calibration's leftover bias)
DEFAULT_RANGES = dict(
    rope_sigma_m=(0.0013, 0.0050), rope_rho=(0.85, 0.98), rope_white_m=(0.0003, 0.0012), rope_latency_steps=(4, 9),
    rope_near_gain=(1.0, 2.5), rope_near_m=0.12,
    obst_delta_c_m=(0.0005, 0.002), obst_delta_v_mps=(0.005, 0.03), obst_latency_steps=(1, 4), obst_bias_m=0.004,
    exec_latency_steps=(0, 2), exec_gain=(0.92, 1.05),
    stiffness=(0.6, 1.6), damping=(0.6, 1.6), mass=(0.85, 1.2), clamp_offset_m=0.004,
)


class RopeObsModel:
    def __init__(self, cfg, rng):
        self.cfg, self.rng = cfg, rng
        self.hist, self.ready, self.a = {}, [], None
        if cfg:
            k = np.arange(1, K_MODES + 1)
            w = 1.0 / k
            s = np.linspace(0, 1, 39)
            self.basis = np.sin(np.pi * np.outer(k, s))                       # (K, 39)
            rms = np.sqrt(((w[:, None] * self.basis) ** 2).sum(0)[1:-1].mean())
            self.sd = w * float(cfg["sigma_m"]) / rms                          # per mode, per axis
            self.a = rng.normal(size=(K_MODES, 3)) * self.sd[:, None]
            self.phase = int(rng.integers(cfg.get("frame_steps", 4)))

    def push(self, n, c):
        """True centerline at control step n; a camera frame is taken on frame steps."""
        if not self.cfg:
            return
        fs = int(self.cfg.get("frame_steps", 4))
        if (n + self.phase) % fs:
            return
        rho = float(self.cfg["rho"])
        self.a = rho * self.a + np.sqrt(1 - rho ** 2) * self.rng.normal(size=self.a.shape) * self.sd[:, None]
        e = self.basis.T @ self.a
        e[1:-1] += self.rng.normal(size=(37, 3)) * float(self.cfg.get("white_m", 0.0))
        near = self.cfg.get("near")
        if near is not None:                                                    # (centre, gain, radius) of the sphere
            centre, gain, rad = near
            d = np.linalg.norm(c - np.asarray(centre), axis=1)
            e *= (1.0 + (gain - 1.0) * np.clip(1 - d / rad, 0, 1))[:, None]
        self.ready.append((n + int(self.cfg.get("latency_steps", 0)), c + e))
        if len(self.ready) > 8:
            self.ready.pop(0)

    def set_sphere(self, centre):
        if self.cfg and self.cfg.get("near_gain"):
            self.cfg["near"] = (centre, float(self.cfg["near_gain"]), float(self.cfg.get("near_m", 0.12)))

    def current(self, n, truth_now):
        if not self.cfg:
            return truth_now
        avail = [c for t, c in self.ready if t <= n]
        return avail[-1] if avail else (self.ready[0][1] if self.ready else truth_now)


def sample_noise(rng, r=DEFAULT_RANGES):
    u = lambda key: float(rng.uniform(*r[key]))                      # noqa: E731
    ui = lambda key: int(rng.integers(r[key][0], r[key][1] + 1))     # noqa: E731

    def ball(radius):
        d = rng.normal(size=3)
        return (d / np.linalg.norm(d) * rng.uniform(0, radius)).tolist()
    return dict(
        rope_obs=dict(sigma_m=u("rope_sigma_m"), rho=u("rope_rho"), white_m=u("rope_white_m"), frame_steps=4,
                      latency_steps=ui("rope_latency_steps"), near_gain=u("rope_near_gain"), near_m=r["rope_near_m"]),
        obstacle=dict(delta_c_m=u("obst_delta_c_m"), delta_v_mps=u("obst_delta_v_mps"), latency_steps=ui("obst_latency_steps"),
                      bias_m=ball(r["obst_bias_m"])),
        exec=dict(latency_steps=ui("exec_latency_steps"), gain=u("exec_gain")),
        dyn=dict(stiffness=u("stiffness"), damping=u("damping"), mass=u("mass"), clamp_offset_m=ball(r["clamp_offset_m"])),
    )
