"""Thin wrappers around the legacy ClosedLoopSession: build configs, run and branch episodes."""
from __future__ import annotations

import copy
import os
import json
import time

import numpy as np

from . import legacy  # noqa: F401  (sets sys.path)
from control.closed_loop_session import ClosedLoopSession  # noqa: E402
from control.mpc_runtime_model import load_model_and_stats  # noqa: E402
from control.mpc_scenarios import Scenario  # noqa: E402
from control.task_value_costs import complete_return  # noqa: E402
from control.task_value_protocol import TaskValueProtocol, sphere_clearances  # noqa: E402
from wm.geometry import resample_centerline_by_arclength  # noqa: E402

ENDPOINT_KEYS = ("initial_left", "initial_right", "target_left", "target_right")
# Task evaluation geometry (55 mm protected radius with the 40 mm spheres), fixed regardless of
# any extra margin the planner/CBF are given through safety.perception_buffer_m.
EVAL_GEOMETRY = dict(rope_radius_m=0.010, clearance_m=0.005, perception_buffer_m=0.0)


class Session(ClosedLoopSession):
    """Legacy session that also works with the local-Jacobian response (gain 0), and that scores
    clearance with EVAL_GEOMETRY rather than the controller's own (possibly inflated) geometry.

    The legacy session always hands `None` as the Jacobian to the unified planner, which only
    works for the full-RSSM response; legacy run_unified.py patched this at runtime instead.
    """

    def __init__(self, cfg, scenario, model_bundle, protocol, seed, **kw):
        """Stage-0 extras in cfg["dlovw"] (set by scene_cfg): scripted obstacle motions replace the
        legacy drivers, and rope stiffness/damping scales are applied right after the environment is
        built, before the target shape is generated, so target and episode share the same rope."""
        extra = cfg.get("dlovw") or {}
        if not extra.get("motions") and not extra.get("rope"):
            super().__init__(cfg, scenario, model_bundle, protocol, seed, **kw)
            return
        import control.closed_loop_session as legacy_session
        from .motion import ScriptedDriver
        make_env, driver_cls = legacy_session.make_env, legacy_session.DynamicObstacleDriver

        def patched_make_env(sim_cfg, env_seed):
            out = make_env(sim_cfg, env_seed)
            if extra.get("rope"):
                scale_rope(out[0], **extra["rope"])
            return out

        def patched_driver(model, data, profile, dt, start):
            motion = (extra.get("motions") or {}).get(profile.geom_name)
            return ScriptedDriver(model, data, profile, dt, motion) if motion else driver_cls(model, data, profile, dt, start)

        legacy_session.make_env, legacy_session.DynamicObstacleDriver = patched_make_env, patched_driver
        try:
            super().__init__(cfg, scenario, model_bundle, protocol, seed, **kw)
        finally:
            legacy_session.make_env, legacy_session.DynamicObstacleDriver = make_env, driver_cls

    # With the local-Jacobian response (gain 0) the unified planner never reads the RSSM latent
    # (`_linearize_response` / `_predict_trials` take the Jacobian branch, the residual observer is
    # off), so inferring it every step only costs time (~14% of a step). Infer once, then reuse.
    cache_latent_gain0 = True

    def _gain0(self):
        return float(self.cfg["mpc"].get("unified", {}).get("rssm_response_gain", 1.0)) == 0.0

    def _new_planner(self, optimizer=None):
        super()._new_planner(optimizer)
        if self._gain0():
            self.planner.context_provider = lambda: (self.state(), self.jacobian.J.copy(), self.safety.filter)

    def latent(self):
        if self.cache_latent_gain0 and self._gain0():
            if getattr(self, "_latent_gain0", None) is None:
                self._latent_gain0 = super().latent()
            return self._latent_gain0
        return super().latent()

    def geometry(self, data=None):
        data = self.env.data if data is None else data
        center = resample_centerline_by_arclength(np.asarray(data.xpos[self.env.dlo_body_ids]), 39)
        centers = (np.stack([d.position for d in self.dynamic.drivers])
                   if self.dynamic else np.empty((0, 3)))
        radii = np.asarray([o.size[0] for o in self.safety.obstacles])
        g = EVAL_GEOMETRY
        return sphere_clearances(center, centers, radii, g["rope_radius_m"], g["clearance_m"], g["perception_buffer_m"])


def scale_rope(env, stiffness_scale=1.0, damping_scale=1.0):
    """Scale the stiffness and damping of every joint on the rope bodies (ball joints in the XML)."""
    model = env.model
    bodies = set(int(b) for b in env.dlo_body_ids)
    ndof = {0: 6, 1: 3, 2: 1, 3: 1}
    for j in range(model.njnt):
        if int(model.jnt_bodyid[j]) in bodies:
            model.jnt_stiffness[j] *= float(stiffness_scale)
            a = int(model.jnt_dofadr[j])
            model.dof_damping[a:a + ndof[int(model.jnt_type[j])]] *= float(damping_scale)


def template_cfg():
    """Base config: the first P2 scene's full cfg (SQP unified planner, CBF, protocol)."""
    cfg = copy.deepcopy(json.load(open(legacy.P2_TEMPLATE_MANIFEST))[0]["cfg"])
    cfg["simulation"].pop("obstacle_positions_m", None)
    return cfg


def scene_cfg(ends, obstacles, *, response_gain=0.0, margin_m=0.0, horizon=None, tube_amax=None, max_step_m=None,
              base=None, scene=None):
    """Full session cfg for given endpoints and obstacle profiles (list of driver dicts).
    `margin_m` inflates only the controller's protected radius (planner + CBF); scoring keeps EVAL_GEOMETRY.
    `horizon` overrides the planner horizon (steps). `max_step_m` lowers the gripper speed cap. `tube_amax` overrides only the acceleration bound of
    the planner's obstacle reachability tube (delta_c + delta_v t + a t^2 / 2); obstacle motion is unchanged."""
    cfg = copy.deepcopy(base or template_cfg())
    sim = cfg["simulation"]
    n = len(obstacles)
    if n not in legacy.XML_BY_OBSTACLES:
        raise ValueError("Scenes need 1-3 obstacles")
    if os.environ.get("DLOVW_SIM_MODEL", "legacy") != "cell":    # the inherited scene (all results so far)
        sim["xml"] = legacy.XML_BY_OBSTACLES[n]
    else:                                                        # corrected to the real cell (imagination/sim_model.py)
        from .imagination.sim_model import scene_xml
        sim["xml"] = scene_xml(n)
    sim["scenario_endpoints"] = {k: [float(v) for v in ends[k]] for k in ENDPOINT_KEYS}
    sim.pop("obstacle_positions_m", None)
    extra = {} if tube_amax is None else {"reachability_a_max_mps2": float(tube_amax)}
    motions = {f"obstacle_sphere_{i}": o["motion"] for i, o in enumerate(obstacles) if o.get("profile") == "scripted"}
    # scripted motions run through ScriptedDriver; the legacy config only needs a valid placeholder
    # with the same bounds (reachability tube, estimator noise), so give it a still constant-velocity profile
    legacy_obstacles = [dict({k: v for k, v in o.items() if k != "motion"}, profile="constant_velocity", speed_mps=0.0)
                        if o.get("profile") == "scripted" else o for o in obstacles]
    sim["dynamic_obstacles"] = {"enabled": True, "obstacles": [dict(o, geom_name=f"obstacle_sphere_{i}", **extra)
                                                               for i, o in enumerate(legacy_obstacles)]}
    knob = (scene or {}).get("knob") or {}
    cfg["dlovw"] = dict(motions=motions, rope=knob.get("rope"), perception=knob.get("perception"),
                        occluders=[d.get("occluder") for d in (scene or {}).get("designs", [])])
    cfg["safety"]["obstacle_geoms"] = [f"obstacle_sphere_{i}" for i in range(n)]
    cfg["mpc"]["unified"]["rssm_response_gain"] = float(response_gain)
    cfg["safety"]["perception_buffer_m"] = float(cfg["safety"].get("perception_buffer_m", 0.0)) + float(margin_m)
    if horizon is not None:
        cfg["mpc"]["horizon"] = int(horizon)
    if max_step_m is not None:            # per-step gripper translation cap (legacy 3.5 mm = 0.44 m/s)
        cfg["simulation"]["max_translation_step_m"] = float(max_step_m)
    return cfg


def open_session(cfg, ends, bundle, seed, cls=None):
    scenario = Scenario(name="dlovw", **{k: np.asarray(ends[k], dtype=float) for k in ENDPOINT_KEYS})
    return (cls or Session)(cfg, scenario, bundle, TaskValueProtocol(**cfg["protocol"]), int(seed))


def run_to_end(session, forced=None, *, reinit_after_forced=True, record=False):
    """Step until terminal. `forced` is an optional (k,6) array of nominal actions for the first
    k steps (passed through the same CBF filter); afterwards the SQP planner runs."""
    rows, mins, traj, plan_s, started = [], [], [], [], time.perf_counter()
    forced = [] if forced is None else list(np.asarray(forced, dtype=float))
    k = 0
    while session.progress.terminal == "running":
        nominal = forced[k] if k < len(forced) else None
        tr, obs = session.step(nominal)
        rows.append(tr["components"])
        mins.append(tr["min_h"])
        if "plan_wall_s" in tr["diagnostics"]:
            plan_s.append(tr["diagnostics"]["plan_wall_s"])
        if record:
            traj.append(dict(centerline=obs["centerline_true"], left=obs["left_pose"][:3],
                             right=obs["right_pose"][:3], obstacles=obs["obstacle_true_centers"]))
        k += 1
        if k == len(forced) and len(forced) and reinit_after_forced and session.progress.terminal == "running":
            session.canonical_initialize()
    ret = complete_return(rows, session.progress.terminal)
    out = dict(terminal=session.progress.terminal, steps=int(session.progress.steps), G=ret["G"],
               min_h=float(np.min(mins)) if mins else None, wall_s=time.perf_counter() - started,
               plan_ms_median=1000 * float(np.median(plan_s)) if plan_s else None,
               plan_ms_p90=1000 * float(np.percentile(plan_s, 90)) if plan_s else None)
    if record:
        out["trajectory"] = traj
    return out


def load_bundle(cfg):
    return load_model_and_stats(cfg)
