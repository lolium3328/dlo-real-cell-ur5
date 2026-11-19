"""Closed-loop episodes in the 2026-11-12 cell for the single-obstacle policy (docs/notes/cell_policy_20261113.md).

Same timing and scoring as the legacy sessions (control step 8 ms, L1 step cap 1.75 mm, decision every 10 steps,
success: centerline RMSE <= 12 mm, p95 <= 25 mm, end error <= 16 mm held 0.4 s; violation: a centerline node closer
than 55 mm to the sphere centre; 7.2 s limit), with the cell's geometry:

- the rope's right end is the fixed clamp (the "right gripper" of the 6-D action never moves);
- the sphere follows its scripted track (the right arm); it does not touch the rope, clearance is scored;
- only rope physics runs (the arms are not simulated; they matter for images only);
- a task is either "reach" (move the left end to a target, success as soon as it holds) or "hold" (keep the shape
  while the sphere passes; success only counts after rec["success_after_s"]).

Deployment noise (rec or `noise` dict; None = exact state, as condition C0 of the legacy studies):
  rope_obs   perceived centerline: RGB tracker error model (noise.py), 30 Hz frames, latency
  obstacle   sphere centre from the right arm: stage-0 Sensor (bounded noise, latency, fixed bias), always visible
  exec       left TCP command latency (control steps) and gain
  dyn        rope bending stiffness / damping / mass scales, clamp offset from its nominal position (the policy is
             given the nominal fixed end)
"""
from __future__ import annotations

import mujoco
import numpy as np

from ..belief import Sensor
from ..motion import track as motion_track, velocities
from . import layout as L
from .cbf import cbf_filter
from .perception import resample
from .scene import scene_xml
from .settle import rope_body_ids

DT = 0.008
CAP = 0.00175
N_SUB = 4
PROTECTED_M = 0.055
CBF_MARGIN_M = 0.005
TOL = dict(rmse=0.012, p95=0.025, end=0.016)
HOLD_STEPS = 50
_MODEL = {}


def _model():
    if "m" not in _MODEL:
        _MODEL["m"] = mujoco.MjModel.from_xml_path(scene_xml())
        _MODEL["nominal"] = dict(stiff=_MODEL["m"].jnt_stiffness.copy(), damp=_MODEL["m"].dof_damping.copy(),
                                 mass=_MODEL["m"].body_mass.copy(), inertia=_MODEL["m"].body_inertia.copy(),
                                 clamp=_MODEL["m"].body("fixed_clamp").pos.copy())
    return _MODEL["m"], _MODEL["nominal"]


def min_jerk(p0, p1, n):
    s = np.linspace(0, 1, n + 1)[1:, None]
    return p0 + (p1 - p0) * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5)


class Frame:
    __slots__ = ("centerline", "left_pose", "right_pose", "obstacle")

    def __init__(self, centerline, left, right, obstacle):
        self.centerline, self.left_pose, self.right_pose, self.obstacle = centerline, left, right, obstacle


class Driver:
    """The sphere: its scripted track at control-step resolution (legacy ScriptedDriver interface)."""

    def __init__(self, motion, n):
        self.track = motion_track(motion, n)
        self.vel = velocities(self.track)
        self.k = 0

    @property
    def position(self):
        return self.track[min(self.k, len(self.track) - 1)]

    @property
    def velocity(self):
        return self.vel[min(self.k, len(self.vel) - 1)]


class Progress:
    def __init__(self):
        self.terminal, self.steps, self.held = "running", 0, 0


class CellSession:
    def __init__(self, rec, max_time_s=7.2, noise=None, seed=0, target=None):
        base, self.nominal = _model()
        self.model = base                                    # shared; per-episode dynamics written in reset
        self.data = mujoco.MjData(self.model)
        self.rec, self.max_steps = rec, int(round(max_time_s / DT))
        self.noise = rec.get("noise") if noise is None else noise
        self.noise = self.noise or {}
        self.rng = np.random.default_rng(seed)
        self.ids = rope_body_ids(self.model)
        self.mocap = self.model.body("left_tcp_mocap").mocapid[0]
        self.fixed = np.asarray(L.FIXED_END_POS, float)      # what the controller knows (nominal)
        self._apply_dynamics()
        start, tgt = np.asarray(rec["start_left"], float), np.asarray(rec["target_left"], float)
        self.target = target or self._settled_target(tgt)
        self._reset_to(start)
        drv = Driver(rec["motion"], self.max_steps + 400)      # + 3.2 s of future for the forecaster targets
        self.dynamic = type("Dyn", (), {"drivers": [drv]})()
        o = self.noise.get("obstacle") or dict(delta_c_m=0.0, delta_v_mps=0.0, latency_steps=0, bias_m=[0.0, 0.0, 0.0])
        self.sensor = Sensor(o, None, np.random.default_rng(seed + 1))
        from .noise import RopeObsModel
        self.rope_obs = RopeObsModel(self.noise.get("rope_obs"), np.random.default_rng(seed + 2))
        ex = self.noise.get("exec") or {}
        self.exec_delay, self.exec_gain = int(ex.get("latency_steps", 0)), float(ex.get("gain", 1.0))
        self.queue = [np.zeros(3)] * self.exec_delay
        self.progress = Progress()
        self.success_after = int(round(float(rec.get("success_after_s", 0.0)) / DT))
        self.frames, self.n = [], 0
        self._meas = None
        self._observe()

    # --- setup -------------------------------------------------------------------------------------------------
    def _apply_dynamics(self):
        m, nom, dyn = self.model, self.nominal, self.noise.get("dyn") or {}
        m.jnt_stiffness[:] = nom["stiff"] * float(dyn.get("stiffness", 1.0))
        m.dof_damping[:] = nom["damp"]
        m.body_mass[:], m.body_inertia[:] = nom["mass"], nom["inertia"]
        rope_dofs = np.r_[[m.jnt_dofadr[m.joint(f"dlo_joint_{i:02d}").id] + k for i in range(1, 25) for k in range(3)]]
        m.dof_damping[rope_dofs] = nom["damp"][rope_dofs] * float(dyn.get("damping", 1.0))
        ms = float(dyn.get("mass", 1.0))
        m.body_mass[self.ids], m.body_inertia[self.ids] = nom["mass"][self.ids] * ms, nom["inertia"][self.ids] * ms
        m.body("fixed_clamp").pos = nom["clamp"] + np.asarray(dyn.get("clamp_offset_m", (0.0, 0.0, 0.0)), float)

    def _run_to(self, d, p_to, move_s, settle_s):
        p0 = d.mocap_pos[self.mocap].copy()
        n = max(1, int(round(move_s / self.model.opt.timestep)))
        for p in min_jerk(p0, np.asarray(p_to, float), n):
            d.mocap_pos[self.mocap] = p
            mujoco.mj_step(self.model, d)
        for _ in range(int(round(settle_s / self.model.opt.timestep))):
            mujoco.mj_step(self.model, d)

    def _reset_to(self, start):
        d = self.data
        mujoco.mj_resetDataKeyframe(self.model, d, self.model.key("home").id)
        mujoco.mj_forward(self.model, d)
        self._run_to(d, start, 1.0, 0.8)
        d.time = 0.0
        mujoco.mj_forward(self.model, d)

    def _settled_target(self, tgt):
        d = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, d, self.model.key("home").id)
        self._run_to(d, tgt, 1.2, 1.2)
        mujoco.mj_forward(self.model, d)
        c = resample(d.xpos[self.ids])
        return dict(centerline=c, left_pose=np.r_[tgt, L.LEFT_TCP_QUAT], right_pose=np.r_[self.fixed, 1.0, 0, 0, 0])

    # --- state -------------------------------------------------------------------------------------------------
    def true_centerline(self):
        return resample(self.data.xpos[self.ids])

    def left(self):
        return self.data.mocap_pos[self.mocap].copy()

    def _observe(self):
        drv = self.dynamic.drivers[0]
        drv.k = self.n
        c = self.true_centerline()
        self.frames.append(Frame(c, np.r_[self.left(), L.LEFT_TCP_QUAT], np.r_[self.fixed, 1.0, 0, 0, 0], drv.position.copy()))
        self.rope_obs.set_sphere(drv.position)
        self.rope_obs.push(self.n, c)
        r = self.sensor.read(drv.track[:self.n + 1], drv.vel[:self.n + 1])
        self._meas = None if r is None else (r[1], r[2])

    def measurement(self, j=0):
        return self._meas

    def perceived_centerline(self):
        return self.rope_obs.current(self.n, self.frames[-1].centerline)

    def errors(self, c=None):
        c = self.frames[-1].centerline if c is None else c
        e = np.linalg.norm(c - self.target["centerline"], axis=1)
        e_end = np.linalg.norm(self.left() - self.target["left_pose"][:3])
        return dict(centerline_rmse_m=float(np.sqrt((e ** 2).mean())), p95_m=float(np.percentile(e, 95)), end_m=float(e_end))

    # --- control -----------------------------------------------------------------------------------------------
    def belief_obstacle(self):
        """(centre, velocity) the safety filter uses: the latest reading carried forward by its latency."""
        if self._meas is None:
            return None
        lat = int((self.noise.get("obstacle") or {}).get("latency_steps", 0))
        return self._meas[0] + self._meas[1] * lat * DT, self._meas[1]

    def step(self, u12, cbf=True):
        """u12: 6-D translation per control step (m); the right half is ignored. Returns (info, None)."""
        u = np.asarray(u12, float)[:3]
        info = dict(cbf_intervention=0.0)
        if cbf:
            ob = self.belief_obstacle()
            lo, hi = L.POLICY_LEFT_WORKSPACE
            u, ci = cbf_filter(u, self.perceived_centerline(), self.left(), self.fixed, [ob] if ob else [],
                               PROTECTED_M + CBF_MARGIN_M, DT, lo, hi, L.MAX_END_DISTANCE_M, CAP)
            info["cbf_intervention"] = ci["intervention"]
        self.queue.append(u * self.exec_gain)
        u_exec = self.queue.pop(0)
        d = self.data
        p0 = d.mocap_pos[self.mocap].copy()
        for k in range(1, N_SUB + 1):
            d.mocap_pos[self.mocap] = p0 + u_exec * k / N_SUB
            mujoco.mj_step(self.model, d)
        self.n += 1
        self._observe()
        err = self.errors()
        info["errors"] = err
        self._score(err)
        return info, None

    def _score(self, err):
        p = self.progress
        p.steps = self.n
        c, o = self.frames[-1].centerline, self.frames[-1].obstacle
        if np.linalg.norm(c - o, axis=1).min() < PROTECTED_M:
            p.terminal = "clearance_violation"
            return
        ok = err["centerline_rmse_m"] <= TOL["rmse"] and err["p95_m"] <= TOL["p95"] and err["end_m"] <= TOL["end"]
        p.held = p.held + 1 if ok and self.n >= self.success_after else 0
        if p.held >= HOLD_STEPS:
            p.terminal = "success"
        elif self.n >= self.max_steps:
            p.terminal = "timeout"

    def close(self):
        pass


def limits():
    """feasible.Limits of the cell: the right 'gripper' is the fixed end (a zero-size box)."""
    from ..imagination.feasible import Limits
    lo, hi = L.POLICY_LEFT_WORKSPACE
    f = tuple(float(x) for x in L.FIXED_END_POS)
    return Limits(tuple(map(float, lo)) + f, tuple(map(float, hi)) + f, float(L.MAX_END_DISTANCE_M), CAP, fixed_right=True)
