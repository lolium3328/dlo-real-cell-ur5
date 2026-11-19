"""Runtime of the cell scene: the left TCP moves one rope end, the right TCP moves the obstacle sphere.

    env = CellEnv()
    obs = env.reset()                                   # keyframe `home`
    obs = env.step(left_tcp_pos, obstacle_pos)          # one control step (default 40 ms)

Commands are world positions (world = left UR base); orientations stay at the home TCP orientations. Each command
is clipped to the layout workspaces, the end-to-end distance limits and a per-step translation limit (10 mm, as the
hardware config's max_translation_step_m), solved by IK and executed by moving the weld mocap / the right arm
linearly over the control step.
"""
from __future__ import annotations

import mujoco
import numpy as np

from . import layout as L
from .arms import Arm
from .scene import mat, scene_xml
from .settle import rope_body_ids


class CellEnv:
    def __init__(self, xml_path=None, control_dt=0.04, max_step_m=0.010):
        self.model = mujoco.MjModel.from_xml_path(xml_path or scene_xml())
        self.data = mujoco.MjData(self.model)
        self.n_sub = max(1, int(round(control_dt / self.model.opt.timestep)))
        self.control_dt = self.n_sub * self.model.opt.timestep
        self.max_step_m = max_step_m
        self.left, self.right = Arm(self.model, "left"), Arm(self.model, "right")
        self.R_left, self.R_right = mat(L.LEFT_TCP_QUAT), mat(L.RIGHT_TCP_QUAT)
        self.mocap = self.model.body("left_tcp_mocap").mocapid[0]
        self.rope_ids = rope_body_ids(self.model)
        self.rope_geoms = np.array([self.model.geom(f"dlo_geom_{i:02d}").id for i in range(len(self.rope_ids))])
        self.obstacle_geom = self.model.geom("obstacle_sphere").id
        self.fixed_end_site = self.model.site("fixed_end").id
        self.key = self.model.key("home").id
        self._renderers = {}
        self.last_ik_error = (0.0, 0.0)

    # --- state ---------------------------------------------------------------------------------------------
    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key)
        mujoco.mj_forward(self.model, self.data)
        self.left.q = self.data.qpos[self.left.qadr].copy()
        self.right.q = self.data.qpos[self.right.qadr].copy()
        return self.observe()

    def left_tcp(self):
        return self.data.mocap_pos[self.mocap].copy()

    def obstacle(self):
        return self.data.geom_xpos[self.obstacle_geom].copy()

    def rope(self):
        """The 25 rope body origins (N, 3), from the left TCP (dlo_00) to the fixed end (dlo_24)."""
        return self.data.xpos[self.rope_ids].copy()

    def clearance(self):
        """Smallest distance between the sphere surface and the rope surface (negative: penetration)."""
        c = self.obstacle()
        best = np.inf
        for g in self.rope_geoms:
            r, half = self.model.geom_size[g, 0], self.model.geom_size[g, 1]
            axis = self.data.geom_xmat[g].reshape(3, 3)[:, 2]
            p = self.data.geom_xpos[g]
            t = np.clip(np.dot(c - p, axis), -half, half)
            best = min(best, np.linalg.norm(c - (p + t * axis)) - r)
        return float(best - L.OBSTACLE_RADIUS_M)

    def observe(self):
        P = self.rope()
        return dict(time=float(self.data.time), rope=P, left_tcp=self.left_tcp(), obstacle=self.obstacle(),
                    clearance=self.clearance(), end_distance=float(np.linalg.norm(P[0] - L.FIXED_END_POS)),
                    weld_error=self.weld_error(), ik_error=self.last_ik_error)

    def weld_error(self):
        """Position mismatch (m) of both welded rope ends against the left TCP mocap and the clamp."""
        P = self.data.xpos[self.rope_ids[[0, -1]]]
        return (float(np.linalg.norm(P[0] - self.left_tcp())),
                float(np.linalg.norm(P[1] - self.data.site_xpos[self.fixed_end_site])))

    # --- commands ------------------------------------------------------------------------------------------
    def clip_left(self, target, current):
        lo, hi = L.LEFT_WORKSPACE
        p = np.clip(np.asarray(target, float), lo, hi)
        v = p - L.FIXED_END_POS
        dist = np.linalg.norm(v)
        p = L.FIXED_END_POS + v / dist * np.clip(dist, L.MIN_END_DISTANCE_M, L.MAX_END_DISTANCE_M)
        return _rate_limit(p, current, self.max_step_m)

    def clip_obstacle(self, target, current):
        lo, hi = L.OBSTACLE_WORKSPACE
        return _rate_limit(np.clip(np.asarray(target, float), lo, hi), current, self.max_step_m)

    def step(self, left_target, obstacle_target):
        p_l0, p_o0 = self.left_tcp(), self.obstacle()
        p_l = self.clip_left(left_target, p_l0)
        p_o = self.clip_obstacle(obstacle_target, p_o0)
        q_l0, q_r0 = self.left.q.copy(), self.right.q.copy()
        q_l, el, rl = self.left.ik(p_l, self.R_left, q_l0, iters=30)
        q_r, er, rr = self.right.ik(p_o - self.R_right @ L.OBSTACLE_CENTRE_IN_TCP, self.R_right, q_r0, iters=30)
        self.last_ik_error = (max(el, er), max(rl, rr))
        for k in range(1, self.n_sub + 1):
            s = k / self.n_sub
            self.data.mocap_pos[self.mocap] = (1 - s) * p_l0 + s * p_l
            self.left.apply(self.data, (1 - s) * q_l0 + s * q_l)
            self.right.apply(self.data, (1 - s) * q_r0 + s * q_r)
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self.observe()

    # --- images --------------------------------------------------------------------------------------------
    def render(self, camera="front_view", width=L.IMAGE_W, height=L.IMAGE_H, depth=False):
        key = (width, height)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, height=height, width=width)
        r = self._renderers[key]
        if depth:
            r.enable_depth_rendering()
        r.update_scene(self.data, camera=camera)
        img = r.render()
        if depth:
            r.disable_depth_rendering()
        return img

    def close(self):
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()


def _rate_limit(p, current, max_step):
    d = p - current
    n = np.linalg.norm(d)
    return p if n <= max_step else current + d * (max_step / n)
