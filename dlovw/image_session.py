"""I0 image session (docs/notes/i0_prereg_20261024.md, sections 1-3).

The B2 dodge rule with the go/wait gate (stage-0 `BeliefSession`, arm H0) on the EV information setting
(exact sensing, no virtual occluders), where the controller's rope shape and/or obstacle come from the
front camera through the hand-written pipeline instead of the simulator:
- rope (`rope_source = "image"`): `state()` (CBF, Jacobian update, dodge rule, gate) and the shape
  error in `plan()` use the tracked centreline. A frame arrives every `frame_steps` control steps (one
  frame later at the noisy camera levels); between frames, and over the latency, the estimate is
  carried along with the measured gripper motion (node i moves by (1 - s_i) dLeft + s_i dRight).
- obstacle (`obs_source = "image"`): the stage-0 H0 belief is fed by the camera's Kalman filter
  instead of the simulator sensor, and its extra radius is dc + dv * age (`obstacle_margin`).
Scoring always uses the true state. With both sources "true" the session is the EV arm (truth
passthrough check, pre-registration section 3).
"""
from __future__ import annotations

import copy
import dataclasses
import time

import numpy as np

from .belief import DT, BeliefSession, KalmanBelief
from .camera import FrontCamera
from .perception import ObstacleEstimator, RopeTracker


LENGTH_TOL = 1.02                       # legacy SafetyState rejects centrelines longer than this x L


def cap_length(Y, L):
    """Only when the tracked centreline would fail the legacy length check: Laplacian smoothing of the
    interior nodes (ends fixed) until it is back inside, so episodes that never hit the check are unchanged."""
    arc = lambda P: float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
    if arc(Y) <= L * LENGTH_TOL:
        return Y
    Y = Y.copy()
    for _ in range(500):
        Y[1:-1] = 0.5 * Y[1:-1] + 0.25 * (Y[:-2] + Y[2:])
        if arc(Y) <= L * LENGTH_TOL * 0.9999:
            break
    return Y


class MarginBelief(KalmanBelief):
    """H0 belief whose extra radius is dc + dv * age."""

    def __init__(self, dc, dv):
        super().__init__(dv)
        self.dc = float(dc)

    def now(self, k):
        c, v, grow = super().now(k)
        return c, v, self.dc + grow

    def path(self, k, ts):
        c, grow = super().path(k, ts)
        return c, self.dc + grow


class CameraSensor:
    """Hands the belief a new (step, centre, velocity) when the camera filter folded one in."""

    def __init__(self):
        self.occluder, self.new = None, None

    def read(self, positions, velocities):
        out, self.new = self.new, None
        return out


_ESTIMATORS = {}


def _estimator(ckpt):
    """One loaded model per process (closed-loop workers run many episodes)."""
    if ckpt not in _ESTIMATORS:
        from .i1_models import Estimator
        _ESTIMATORS[ckpt] = Estimator(ckpt)
    return _ESTIMATORS[ckpt]


class ImageSession(BeliefSession):
    arm = "H0"
    rope_source = "true"
    obs_source = "true"
    level = "realistic"
    res = 64
    frame_steps = 4
    obstacle_margin = (0.0, 0.0)

    estimator = "hand"           # I1: "hand" | "rssm" | "cnn" | "gruhand" (learned models from `model_ckpt`)
    model_ckpt = None
    record = False               # I1 data collection: keep per-frame inputs, hand outputs and truth
    start_wait_steps = 0         # I1 data collection: hold still at the start

    def _image_setup(self):
        if hasattr(self, "_cam"):
            return
        rng = np.random.default_rng(int(self.seed) + 104729)
        self._cam = FrontCamera(self.env.model, self.env.data, level=self.level, res=self.res, rng=rng)
        self._latency = 0 if self.level == "clean" else 1
        self._queue = []                       # captured frames waiting out the latency
        self._obs = ObstacleEstimator(DT)
        self._rope, self._rope_Y, self._grip = None, None, None
        n = len(self.frames[-1].centerline)
        self._s = np.linspace(0.0, 1.0, n)[:, None]
        self.image_log = dict(frames=0, obstacle_seen=0, first_seen_step=None)
        hand = self.estimator == "hand"
        self._hand_obs = self.record or self.estimator == "gruhand" or (hand and self.obs_source == "image")
        self._hand_rope = self.record or self.estimator == "gruhand" or (hand and self.rope_source == "image")
        self._learned = None if hand else _estimator(self.model_ckpt)
        self.records = []

    def _gate_blocked(self, x):
        if self.progress.steps < self.start_wait_steps:
            return True
        return super()._gate_blocked(x)

    def _belief_setup(self):
        fresh = not hasattr(self, "_beliefs")
        super()._belief_setup()
        if fresh and self.obs_source == "image":
            if len(self._beliefs) != 1:
                raise ValueError("image obstacle estimation assumes one obstacle")
            self._sensors[0] = CameraSensor()
            self._beliefs[0] = MarginBelief(*self.obstacle_margin)

    def _grippers(self):
        f = self.frames[-1]
        return np.asarray(f.left_pose[:3], float), np.asarray(f.right_pose[:3], float)

    def _carry(self, Y, g_from, g_to):
        return Y + (1 - self._s) * (g_to[0] - g_from[0])[None] + self._s * (g_to[1] - g_from[1])[None]

    def _truth(self, frame):
        rgb, depth, intr, pose = frame
        d = self.dynamic.drivers[0]
        u, v, z = FrontCamera.project(d.position, intr, pose)
        inside = bool(z[0] > 0 and 0 <= u[0] < self.res and 0 <= v[0] < self.res)
        return np.asarray(self.frames[-1].centerline, float).copy(), d.position.copy(), d.velocity.copy(), inside

    def _estimate_obstacles(self):
        if self.dynamic is not None:
            self._image_setup()
            k = getattr(self, "_k", -1) + 1           # the step BeliefSession is about to estimate
            grip = self._grippers()
            if k == 0:                                 # frame 0: start-up calibration from the true shape
                Y0 = np.asarray(self.frames[-1].centerline, float).copy()
                self._offsets = (Y0[0] - grip[0], Y0[-1] - grip[1])
                if self._hand_rope:
                    self._rope = RopeTracker(Y0, *grip, n=len(self._s), rng=self._cam.rng)
                    self._rope_grip = grip
                if self.rope_source == "image":
                    self._rope_Y = Y0.copy()
                if self._learned is not None:
                    self._learned.reset(self._offsets)
                    self._last_cap_grip = None         # the first used frame gets zero motion (as in training)
            elif self.rope_source == "image":
                self._rope_Y = self._carry(self._rope_Y, self._grip, grip)
            self._grip = grip
            if k > 0 and k % self.frame_steps == 0:
                f = self.frames[-1]
                frame = self._cam.process(f.front_rgb, f.front_depth)
                self._queue.append((k, grip, frame, self._truth(frame) if self.record else None))
                if len(self._queue) > self._latency:
                    self._use_frame(*self._queue.pop(0), grip)
        super()._estimate_obstacles()

    def _use_frame(self, k_cap, g_cap, frame, truth, g_now):
        rgb, depth, intr, pose = frame
        self.image_log["frames"] += 1
        centre = meas = rope = None
        if self._hand_obs:
            centre = self._obs.measure(rgb, depth, intr, pose)
            meas = self._obs.update(k_cap, centre)
        if self._hand_rope:
            self._rope.Y = self._carry(self._rope.Y, self._rope_grip, g_cap)    # prior at capture time
            rope = self._rope.update(self._rope.points(rgb, depth, intr, pose), *g_cap)
            self._rope_grip = g_cap
        if self.estimator == "hand":
            obs_new, seen, rope_est = (meas if centre is not None else None), centre is not None, rope
        else:
            last = self._last_cap_grip or g_cap
            action = np.r_[g_cap[0] - last[0], g_cap[1] - last[1]]
            hand = None
            if self.estimator == "gruhand":
                kf = np.r_[meas[1], meas[2]] if meas is not None else np.full(6, np.nan)
                hand = (centre if centre is not None else np.full(3, np.nan), centre is not None, kf, rope)
            rope_est, c, v, vis = self._learned.step(rgb, depth, np.r_[g_cap[0], g_cap[1]], action, hand)
            self._last_cap_grip = g_cap
            seen = vis > 0.5
            if self.estimator == "cnn":                # the hand Kalman filter on the CNN's positions
                obs_new = self._obs.update(k_cap, c) if seen else None
            else:
                obs_new = (k_cap, c, v)
        if self.obs_source == "image" and obs_new is not None:
            if seen:
                self.image_log["obstacle_seen"] += 1
                if self.image_log["first_seen_step"] is None:
                    self.image_log["first_seen_step"] = k_cap
            self._sensors[0].new = obs_new
        if self.rope_source == "image" and rope_est is not None:
            self._rope_Y = self._carry(rope_est, g_cap, g_now)
        if self.record:
            nan3 = np.full(3, np.nan)
            self.records.append(dict(
                k=k_cap, rgb=np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8), depth=depth.astype(np.float16),
                grip=np.r_[g_cap[0], g_cap[1]], hand_meas=centre if centre is not None else nan3,
                hand_seen=centre is not None, hand_kf=np.r_[meas[1], meas[2]] if meas is not None else np.full(6, np.nan),
                hand_rope=rope, true_rope=truth[0], true_centre=truth[1], true_vel=truth[2], in_frame=truth[3]))

    def state(self):
        st = super().state()
        if self.rope_source == "image" and getattr(self, "_rope_Y", None) is not None:
            self._rope_Y = cap_length(self._rope_Y, float(st.L))
            return dataclasses.replace(st, x_k=self._rope_Y.copy())
        return st

    def plan(self):
        """The legacy `ClosedLoopSession.plan` with the shape error taken from `state()`."""
        import control.closed_loop_session as legacy_session
        started = time.perf_counter()
        h, z = self.latent()
        current = (dataclasses.replace(self.frames[-1], centerline=np.asarray(self.state().x_k))
                   if self.rope_source == "image" else self.frames[-1])
        errors = legacy_session.task_errors(current, self.target)
        result = self.planner.plan(h, z, current.left_pose, current.right_pose,
                                   self.target["centerline"], self.target["left_pose"][:3],
                                   self.target["right_pose"][:3], errors["centerline_rmse_m"])
        return result[0], self.planner.previous_solution.detach().cpu().numpy().copy(), {
            **copy.deepcopy(self.planner.last_unified_diagnostics), "plan_wall_s": time.perf_counter() - started}
