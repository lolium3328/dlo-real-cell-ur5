"""Kinematic UR5e arms of the cell scene: damped least-squares pose IK of the TCP site on a scratch MjData.

The arms do not collide with anything (scene.py), so they are driven kinematically: each control step the IK
solution is written to qpos (velocity zero, the position actuators' ctrl at the same value), as in the legacy
envs/robot_ik.py.
"""
from __future__ import annotations

import itertools

import mujoco
import numpy as np

from .layout import WORK_CENTRE_XY
from .scene import ARM_JOINTS, mat


def rotvec(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).reshape(-1))
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, q, 1.0)
    return v


class Arm:
    def __init__(self, model, side):
        self.model, self.side = model, side
        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_ur5e_{j}") for j in ARM_JOINTS]
        self.qadr = np.array([model.jnt_qposadr[j] for j in jids])
        self.dadr = np.array([model.jnt_dofadr[j] for j in jids])
        self.qmin, self.qmax = model.jnt_range[jids, 0].copy(), model.jnt_range[jids, 1].copy()
        self.act = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_ur5e_{a}")
                             for a in ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")])
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_tcp")
        self.scratch = mujoco.MjData(model)
        self.q = np.zeros(6)

    def fk(self, q):
        d = self.scratch
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)                    # cdof, needed by mj_jacSite
        return d.site_xpos[self.site].copy(), d.site_xmat[self.site].reshape(3, 3).copy()

    def ik(self, pos, R, q0, iters=60, damping=0.02, rot_weight=0.3, tol=(2e-4, 2e-3)):
        """Pose IK from q0. Returns (q, position error m, rotation error rad)."""
        m, d = self.model, self.scratch
        q = np.array(q0, float)
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iters):
            p, Rc = self.fk(q)
            ep, er = pos - p, rotvec(R @ Rc.T)
            if np.linalg.norm(ep) < tol[0] and np.linalg.norm(er) < tol[1]:
                break
            mujoco.mj_jacSite(m, d, jp, jr, self.site)
            J = np.vstack([jp[:, self.dadr], rot_weight * jr[:, self.dadr]])
            e = np.r_[ep, rot_weight * er]
            dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), e)
            step = np.linalg.norm(dq)
            if step > 0.3:
                dq *= 0.3 / step
            q = np.clip(q + dq, self.qmin, self.qmax)
        p, Rc = self.fk(q)
        return q, float(np.linalg.norm(pos - p)), float(np.linalg.norm(rotvec(R @ Rc.T)))

    def ik_global(self, pos, R):
        """IK from a grid of seeds; of the converged solutions, elbow above the shoulder and far from the rope's
        workspace (WORK_CENTRE_XY: less occlusion) preferred."""
        base = self.model.body(f"{self.side}_ur5e_base")
        d = self.scratch
        mujoco.mj_kinematics(self.model, d)
        b = d.xpos[base.id]
        pan0 = np.arctan2(pos[1] - b[1], pos[0] - b[0])
        best = None
        for dpan, lift, elbow, w3 in itertools.product((-0.4, 0.0, 0.4, np.pi), (-1.2, -1.9), (1.2, 2.0),
                                                       (-np.pi / 2, 0.0, np.pi / 2, np.pi)):
            seed = np.array([pan0 + dpan, lift, elbow, -np.pi / 2 - (lift + elbow) + np.pi / 2, -np.pi / 2, w3])
            q, ep, er = self.ik(pos, R, seed, iters=200)
            if ep > 1e-3 or er > 1e-2:
                continue
            elbow = self._body_pos(q, "forearm_link")
            elbow_up = elbow[2] - self._body_pos(q, "upper_arm_link")[2] > 0.05
            away = np.linalg.norm(elbow[:2] - WORK_CENTRE_XY)
            score = ep + 0.1 * er + (0.0 if elbow_up else 1.0) - 0.3 * away + 0.01 * np.abs(q[3:]).sum()
            if best is None or score < best[0]:
                best = (score, q, ep, er)
        if best is None:
            raise RuntimeError(f"{self.side} arm: no IK solution for {pos}")
        return best[1], best[2], best[3]

    def _body_pos(self, q, link):
        self.fk(q)
        return self.scratch.xpos[self.model.body(f"{self.side}_ur5e_{link}").id].copy()

    def apply(self, data, q):
        self.q = np.array(q, float)
        data.qpos[self.qadr] = self.q
        data.qvel[self.dadr] = 0.0
        data.ctrl[self.act] = self.q


def tcp_rotation(quat_wxyz):
    return mat(quat_wxyz)
