"""Run the actor in MuJoCo (docs/notes/imagination_ac_design_20261022.md, sections 7-8).

`ActorSession` is the stage-0 belief session (arm H0: the CBF sees the same occluded, delayed, noisy
sensor as the actor) without SQP planning and without camera rendering after initialisation. The
actor picks a per-control-step gripper translation every K control steps; every control step it goes
through the legacy CBF-QP filter (`Session.step(nominal)`) before execution.

`FastActorSession` runs the same episodes with less work (checked against ActorSession by
scripts/imagination/verify_fast_session.py): no renderers and no planner/RSSM bundle at all, the
target shape can be given instead of re-simulated, and clearance samples use mj_kinematics on a scratch
copy of qpos/mocap instead of a full mj_forward; all bitwise identical to ActorSession. Switching
collision detection off (`contacts_off`) is NOT: some scenes diverge from the first step (2026-10-23
check), so it stays on.
"""
from __future__ import annotations

import copy

import mujoco
import numpy as np
import torch

from .. import legacy  # noqa: F401  (sets sys.path)
import control.closed_loop_session as legacy_session  # noqa: E402
from control.mpc_observation import StateFrame, state_poses  # noqa: E402
from wm.geometry import resample_centerline_by_arclength  # noqa: E402
from ..belief import BeliefSession
from ..sessions import open_session, scene_cfg
from .episodes import CAP, K, build_episode, encode_goal, encode_obs, origin_of
from .feasible import feasible_np, limits_of


class ActorSession(BeliefSession):
    arm = "H0"
    _light = False

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._light = True                   # the RSSM latent is never needed: no more rendering

    def _append_frame(self, action12):
        if not self._light:
            return super()._append_frame(action12)
        left, right, center = state_poses(self.env, int(self.sim_cfg.get("centerline_points", 0)))
        self.frames.append(StateFrame(top_rgb=None, top_depth=None, front_rgb=None, front_depth=None,
                                      left_pose=left, right_pose=right, centerline=center))

    def observation_record(self):
        if not self._light:
            return super().observation_record()
        return dict(time_sim=float(self.env.data.time), previous_actual=self.previous_actual.copy())

    def measurement(self, j=0):
        """The obstacle measurement the sensor delivered this step (position, velocity), or None."""
        b = self._beliefs[j]
        return (b.last[1].copy(), b.last[2].copy()) if b.seen_now else None

    def reading_id(self, j=0):
        """Identity of the latest obstacle reading (the step it was taken), or None while there is none. A camera
        sensor repeats its last frame between frames; the scene sensor gives a new reading every step."""
        sensor, b = self._sensors[j], self._beliefs[j]
        if hasattr(sensor, "frame_id"):
            return sensor.frame_id()
        return b.last[0] if b.seen_now else None


LIGHT_BUNDLE = (None, None, None, "cpu")      # what FastActorSession takes instead of the RSSM bundle


class _NoRenderer:
    def close(self):
        pass


class _NoPlanner:
    """The actor never plans; these are the only planner hooks the session itself calls."""
    context_provider = None

    def set_cable_length(self, length):
        pass

    def set_obstacle_context(self, *args, **kw):
        pass

    def reset(self):
        pass


def _state_frame(env, sim_cfg, *args, **kw):
    left, right, center = state_poses(env, int(sim_cfg.get("centerline_points", 0)))
    return StateFrame(top_rgb=None, top_depth=None, front_rgb=None, front_depth=None,
                      left_pose=left, right_pose=right, centerline=center)


class FastActorSession(ActorSession):
    _light = True
    preset_target = None                     # set per episode by the caller (same for every run of a scene)
    kinematic_samples, contacts_off = True, False

    def __init__(self, *args, **kw):
        saved = legacy_session.make_render_bundle, legacy_session.capture_frame
        legacy_session.make_render_bundle = lambda env, sim_cfg: _NoRenderer()
        legacy_session.capture_frame = lambda env, renderers, sim_cfg, *a, **k: _state_frame(env, sim_cfg)
        try:
            super().__init__(*args, **kw)
        finally:
            legacy_session.make_render_bundle, legacy_session.capture_frame = saved

    def _generate_target(self):
        # first call after the environment is built: switch off collision detection here
        if self.contacts_off:
            self.env.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        self._scratch = mujoco.MjData(self.env.model)
        if self.preset_target is not None:
            self.reference = None
            return copy.deepcopy(self.preset_target)
        return super()._generate_target()

    def _new_planner(self, optimizer=None):
        self.planner = _NoPlanner()

    def _geometry_sample(self):
        if not self.kinematic_samples:
            return super()._geometry_sample()
        m, d, g = self.env.model, self.env.data, self._scratch
        g.qpos[:], g.mocap_pos[:], g.mocap_quat[:] = d.qpos, d.mocap_pos, d.mocap_quat
        mujoco.mj_kinematics(m, g)
        gap, h = self.geometry(g)
        centerline = resample_centerline_by_arclength(np.asarray(g.xpos[self.env.dlo_body_ids]), 39)
        centers = (np.stack([dr.position for dr in self.dynamic.drivers]) if self.dynamic else np.empty((0, 3)))
        return dict(minimums=np.r_[d.time, gap.min(initial=1.), h.min(initial=1.)],
                    centerline=centerline, obstacle_centers=centers.copy())


class PerceivedActorSession(FastActorSession):
    """FastActorSession whose safety filter, local-Jacobian updates and servo (everything that calls
    state()) see the perceived centerline of perception.RopePerception; frames, and therefore success and
    clearance scoring, stay true (docs/notes/perception_robustness_20261023.md)."""
    perception_cfg = None
    _perc = _perceived = None
    _jac_ref = None                 # (frame id, estimate, gripper position) last used for a Jacobian update
    JAC_MIN_DY = 0.002              # update the local Jacobian only over gripper moves of at least 2 mm

    def canonical_initialize(self, *args, **kw):
        super().canonical_initialize(*args, **kw)
        if self.perception_cfg and not self.perception_cfg.get("passive") and self.perception_cfg.get("rope", True):
            # The session updates its local Jacobian from consecutive states every control step. With a
            # perceived state that step also contains each new frame's correction, divided by a gripper
            # move of < 1.75 mm: the Jacobian blows up (|J - J0| reached 13 in a probe) and the tracker,
            # which carries the estimate with J between frames, drifts with it. So per-step updates are
            # switched off and the Jacobian is updated from consecutive measured frames instead.
            jac, orig = self.jacobian, self.jacobian.update
            jac.update = lambda prev, nxt, _orig=orig: _orig(prev, prev)        # zero deltas: skipped
            self._jac_ref = None
    def _append_frame(self, action12):
        super()._append_frame(action12)
        if self.perception_cfg.get("kind") == "camtiming":       # obstacle only, no rendering (_belief_setup)
            return
        if self.perception_cfg.get("kind") in ("vision2", "rgb"):  # one camera reading per obstacle (vision_multi.py, vision_rgb.py)
            return self._append_vision2()
        if self._perc is None:
            if self.perception_cfg.get("kind") == "vision":       # rendered depth cameras (vision.py)
                from .vision import VisionPerception
                occ = ((self.cfg.get("dlovw") or {}).get("occluders") or [None])[0]
                cfg = self.perception_cfg
                self._perc = VisionPerception(self.env, occ, depth_noise=cfg.get("depth_noise", 0.0), dropout=cfg.get("dropout", 0.0),
                                              seed=int(self.seed) + 7907, window=cfg.get("window", 3))
            else:
                from .perception import RopePerception
                self._perc = RopePerception(seed=int(self.seed) + 104729, **self.perception_cfg)
        f = self.frames[-1]
        obstacle = self.dynamic.drivers[0].position if getattr(self, "dynamic", None) else None
        J = np.asarray(self.jacobian.J, float) if getattr(self, "jacobian", None) is not None else None
        L = super().state().L                   # the cable length is a known constant of the rope
        self._perceived = self._perc.update(f.centerline, np.r_[f.left_pose[:3], f.right_pose[:3]], obstacle, J, L)
        if J is not None and not self._rope_from_truth() and self._perc.est is not None:
            fid = getattr(self._perc, "frame_id", 0)
            if self._jac_ref is None:
                self._jac_ref = (fid, self._perc.est.copy(), self._perc.est_grip.copy())
            elif fid != self._jac_ref[0]:
                dy = self._perc.est_grip - self._jac_ref[2]
                if np.linalg.norm(dy) >= self.JAC_MIN_DY:
                    self.jacobian.update_from_deltas((self._perc.est - self._jac_ref[1]).reshape(-1), dy)
                    self._jac_ref = (fid, self._perc.est.copy(), self._perc.est_grip.copy())

    def _occluders(self):
        occ = list((self.cfg.get("dlovw") or {}).get("occluders") or [])
        return occ + [None] * (len(self.dynamic.drivers) - len(occ))

    def _make_vision2(self):
        if self._perc is None:
            from .vision_multi import MultiVisionPerception
            cfg = self.perception_cfg
            if cfg.get("kind") == "rgb":                           # colour images only (vision_rgb.py, vision_rgb_deploy.py)
                from .vision_rgb import RGBPerception
                opts = {k: cfg[k] for k in ("preset", "noise_sd", "background", "offset_steps", "naive") if k in cfg}
                if cfg.get("deploy"):
                    from .vision_rgb_deploy import RGBDeployPerception as RGBPerception  # noqa: N814
                    opts.update({k: cfg[k] for k in ("sphere_latency", "rope_latency", "rope_period", "q", "track_spheres") if k in cfg})
                self._perc = RGBPerception(self.env, [f"obstacle_sphere_{j}" for j in range(len(self.dynamic.drivers))],
                                           seed=int(self.seed) + 7907, window=cfg.get("window", 5), L=FastActorSession.state(self).L, **opts)
                return
            self._perc = MultiVisionPerception(self.env, [f"obstacle_sphere_{j}" for j in range(len(self.dynamic.drivers))],
                                               depth_noise=cfg.get("depth_noise", 0.0), dropout=cfg.get("dropout", 0.0),
                                               seed=int(self.seed) + 7907, window=cfg.get("window", 5))

    def _append_vision2(self):
        """Rendered cameras with every obstacle sphere left out of the images while it is inside its occluder
        (the simulated world's decision); the detector itself never sees a true obstacle position."""
        from ..motion import hidden
        if getattr(self, "dynamic", None) is None:
            return
        drivers = self.dynamic.drivers
        self._make_vision2()
        f = self.frames[-1]
        occ = self._occluders()
        hid = [hidden(d.position, o) for d, o in zip(drivers, occ)]
        truth = [(d.position.copy(), d.velocity.copy()) for d in drivers]
        J = np.asarray(self.jacobian.J, float) if getattr(self, "jacobian", None) is not None else None
        L = FastActorSession.state(self).L
        self._perceived = self._perc.update(f.centerline, np.r_[f.left_pose[:3], f.right_pose[:3]], hid, truth, J, L)
        if J is not None and not self._rope_from_truth() and self._perc.est is not None:
            fid = getattr(self._perc, "frame_id", 0)
            if self._jac_ref is None:
                self._jac_ref = (fid, self._perc.est.copy(), self._perc.est_grip.copy())
            elif fid != self._jac_ref[0]:
                dy = self._perc.est_grip - self._jac_ref[2]
                if np.linalg.norm(dy) >= self.JAC_MIN_DY:
                    self.jacobian.update_from_deltas((self._perc.est - self._jac_ref[1]).reshape(-1), dy)
                    self._jac_ref = (fid, self._perc.est.copy(), self._perc.est_grip.copy())

    def _belief_setup(self):
        fresh = not hasattr(self, "_beliefs")
        super()._belief_setup()
        cfg = self.perception_cfg or {}
        if fresh and cfg.get("kind") in ("vision2", "rgb") and not cfg.get("passive") and cfg.get("obstacle", True):
            from .vision_multi import MultiVisionSensor
            self._make_vision2()
            for j, o in enumerate(self._occluders()):
                self._sensors[j] = MultiVisionSensor(self._perc, j, o)
        if fresh and self._perc is not None and cfg.get("kind") == "vision" and not cfg.get("passive") and cfg.get("obstacle", True):
            from .vision import VisionSensor
            self._sensors[0] = VisionSensor(self._perc)       # the obstacle comes from the cameras too
        if fresh and cfg.get("kind") == "camtiming":
            from .vision import CameraObstacleSensor
            occ = ((self.cfg.get("dlovw") or {}).get("occluders") or [None])[0]
            cam = {k: cfg[k] for k in ("cam_steps", "latency_frames", "window") if k in cfg}
            self._sensors[0] = CameraObstacleSensor(occ, np.random.default_rng(int(self.seed) + 7907), **cam)

    def perceived_centerline(self):
        if self._perceived is None or self._rope_from_truth():
            return np.asarray(self.frames[-1].centerline, float)
        return self._perceived

    def close(self):
        if self._perc is not None and hasattr(self._perc, "close"):
            self._perc.close()
        super().close()

    def measurement(self, j=0):
        """With camera detections of the obstacle the last reading can be several control steps old (a frame
        every 4 steps, available one frame later); with comp=True it is carried forward to the present with
        its velocity before the world model and the obstacle forecaster see it."""
        m = super().measurement(j)
        cfg = self.perception_cfg or {}
        if m is None or not cfg.get("comp") or cfg.get("kind") not in ("vision", "vision2", "rgb"):
            return m
        b = self._beliefs[j]
        age = (self._k - b.last[0]) * 0.008
        return m[0] + m[1] * age, m[1]

    def _rope_from_truth(self):
        cfg = self.perception_cfg or {}
        return cfg.get("passive") or cfg.get("rope") is False

    def state(self):
        st = super().state()
        if self._perceived is None or self._rope_from_truth():
            return st
        return type(st)(np.asarray(self._perceived, float).copy(), st.y_k, st.L)


def open_actor_session(rec, bundle=None, *, max_time_s=7.2, target=None, fast=True, perception=None, **flags):
    """Session for scene `rec`: FastActorSession (no bundle needed) or the reference ActorSession.
    `flags` override FastActorSession switches (kinematic_samples, contacts_off) for checks."""
    cfg = scene_cfg(rec["ends"], rec["profiles"], max_step_m=CAP, scene=rec)
    cfg["protocol"]["max_time_s"] = float(max_time_s)
    if not fast:
        return open_session(cfg, rec["ends"], bundle, rec["seed"], cls=ActorSession)
    base = PerceivedActorSession if perception else FastActorSession
    cls = type("Fast", (base,), dict(flags, preset_target=target, perception_cfg=perception))
    return open_session(cfg, rec["ends"], LIGHT_BUNDLE, rec["seed"], cls=cls)


def project_arms_np(a):
    a = np.asarray(a, float).reshape(2, 3)
    return (a * np.minimum(1.0, 1.0 / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9))).reshape(6)


SERVO_ENTER_M, SERVO_EXIT_M = 0.025, 0.040


def servo_action(s, w_grip=0.5, lam=1e-4):
    """Final approach by the session's local (Broyden) Jacobian: one damped least-squares step towards the
    target centerline and grippers, per gripper capped at the speed limit (units of CAP). A 6x6 solve."""
    st = s.state()
    J = np.asarray(s.jacobian.J, float)
    dx = (np.asarray(s.target["centerline"], float) - np.asarray(st.x_k, float)).reshape(-1)
    dy = np.r_[s.target["left_pose"][:3], s.target["right_pose"][:3]] - np.asarray(st.y_k, float)
    du = np.linalg.solve(J.T @ J + (w_grip + lam) * np.eye(6), J.T @ dx + w_grip * dy)
    return project_arms_np(du / CAP)


def servo_mode(s, on):
    """Hand the final approach to the servo below SERVO_ENTER_M centerline RMSE, back above SERVO_EXIT_M."""
    e = float(np.sqrt(((np.asarray(s.state().x_k) - np.asarray(s.target["centerline"])) ** 2).sum(-1).mean()))
    return e < SERVO_ENTER_M or (on and e < SERVO_EXIT_M)


def oracle_now(actor, drv, n, origin):
    """Privileged forecast for the upper-bound actor: true obstacle positions ORACLE_OFFS decisions ahead."""
    if not actor.extra:
        return None
    from .actor_critic import ORACLE_OFFS
    from .episodes import POS
    tr = np.asarray(drv.track)
    pts = [(tr[min(n + K * k, len(tr) - 1)] - origin) * POS for k in ORACLE_OFFS]
    return torch.as_tensor(np.concatenate(pts), dtype=torch.float32)[None]


def run_episode(rec, wm, actor, bundle=None, *, noise=0.0, seed=0, max_time_s=7.2, target=None, fast=True, servo=False):
    """One episode on scene `rec` with the actor (posterior state from the world model). Returns the
    decision-level episode (episodes.build_episode) with outcome info, plus the target shape used."""
    s = open_actor_session(rec, bundle, max_time_s=max_time_s, target=target, fast=fast)
    rng = np.random.default_rng(seed)
    seq = dict(center=[], left=[], right=[], meas=[], op=[], ov=[])
    try:
        origin = origin_of(s.target)
        goal = torch.as_tensor(encode_goal(s.target, origin))[None]
        drv = s.dynamic.drivers[0]
        lim = limits_of(s)

        def record():
            f = s.frames[-1]
            for k, v in (("center", f.centerline), ("left", f.left_pose[:3]), ("right", f.right_pose[:3]),
                         ("op", drv.position), ("ov", drv.velocity)):
                seq[k].append(np.array(v, float))
            seq["meas"].append(s.measurement(0))

        record()
        state = wm.initial(1, "cpu")
        a_prev, n, interventions, servoing, servo_steps = torch.zeros(1, 6), 0, 0, False, 0
        with torch.no_grad():
            while s.progress.terminal == "running":
                if n % K == 0:
                    if n:
                        grip = np.r_[seq["left"][-1], seq["right"][-1]]
                        a_prev = torch.as_tensor((grip - start) / (K * CAP), dtype=torch.float32)[None]
                    start = np.r_[seq["left"][-1], seq["right"][-1]]
                    o = encode_obs(seq["center"][-1], seq["left"][-1], seq["right"][-1], seq["meas"][-1], origin)
                    state = wm.obs_step(state, a_prev, torch.as_tensor(o)[None])[0]
                    a = actor(wm.feat(state), goal, oracle_now(actor, drv, n, origin))[0].numpy().astype(float)
                    if noise:
                        a = np.clip(a + noise * rng.normal(size=6), -1.0, 1.0)
                    servoing = servo and servo_mode(s, servoing)
                u = servo_action(s) if servoing else feasible_np(a, np.r_[seq["left"][-1], seq["right"][-1]], lim)
                tr, _ = s.step(u * CAP)
                servo_steps += servoing
                interventions += float(tr["cbf_intervention"]) > 1e-6
                n += 1
                record()
        info = dict(id=rec["id"], terminal=s.progress.terminal, steps=int(s.progress.steps),
                    cbf_active_frac=interventions / max(n, 1), noise=float(noise), servo_frac=servo_steps / max(n, 1))
        track = np.asarray(drv.track)
    finally:
        s.close()
    ep = build_episode(seq["center"], seq["left"], seq["right"], seq["meas"], seq["op"], seq["ov"], track, 0, s.target, info,
                       rec, seed)
    ep["target"] = {k: np.array(v) for k, v in s.target.items()}
    return ep
