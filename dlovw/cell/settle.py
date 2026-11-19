"""The scene's `home` keyframe: both arms at their home TCPs, the rope settled under gravity.

At qpos0 the rope is straight, from the clamp back towards the left arm, and the left TCP mocap sits at its free end
turned so the weld holds; the mocap then moves (1.5 s, smoothstep) to the home TCP pose, pointing down, and the rope
is left to settle until its fastest body moves slower than 1 mm/s (at most 30 s).
"""
from __future__ import annotations

import mujoco
import numpy as np

from . import layout as L
from .arms import Arm
from .scene import N_ROPE, mat, quat


def _slerp(q0, q1, t):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    if np.dot(q0, q1) < 0:
        q1 = -q1
    d = np.clip(np.dot(q0, q1), -1.0, 1.0)
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
    else:
        th = np.arccos(d)
        q = (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / np.sin(th)
    return q / np.linalg.norm(q)


def rope_body_ids(model):
    return np.array([model.body(f"dlo_{i:02d}").id for i in range(N_ROPE)])


def home_arm_qpos(model):
    """IK of both arms at their home TCP poses (the right TCP carries the sphere to OBSTACLE_HOME_POS)."""
    left, right = Arm(model, "left"), Arm(model, "right")
    R_l, R_r = mat(L.LEFT_TCP_QUAT), mat(L.RIGHT_TCP_QUAT)
    ql, *_ = left.ik_global(L.LEFT_HOME_POS, R_l)
    qr, *_ = right.ik_global(L.OBSTACLE_HOME_POS - R_r @ L.OBSTACLE_CENTRE_IN_TCP, R_r)
    return left, right, ql, qr


def settle(model, data, move_s=1.5, max_hold_s=30.0, v_tol=1e-3, log=None):
    left, right, ql, qr = home_arm_qpos(model)
    mujoco.mj_resetData(model, data)
    left.apply(data, ql)
    right.apply(data, qr)
    mid = model.body("left_tcp_mocap").mocapid[0]
    p0, q0 = data.mocap_pos[mid].copy(), data.mocap_quat[mid].copy()
    ids = rope_body_ids(model)
    n_move = int(round(move_s / model.opt.timestep))
    n_max = n_move + int(round(max_hold_s / model.opt.timestep))
    vmax = np.inf
    for k in range(n_max):
        t = min(1.0, k / n_move)
        s = t * t * (3 - 2 * t)
        data.mocap_pos[mid] = (1 - s) * p0 + s * L.LEFT_HOME_POS
        data.mocap_quat[mid] = _slerp(q0, L.LEFT_TCP_QUAT, s)
        left.apply(data, ql)
        right.apply(data, qr)
        mujoco.mj_step(model, data)
        if k > n_move and k % 50 == 0:
            vmax = float(np.abs(data.cvel[ids, 3:]).max())
            if vmax < v_tol:
                break
    if log is not None:
        log(f"settled after {data.time:.2f} s, fastest rope body {vmax * 1e3:.2f} mm/s")
    return dict(qpos=data.qpos.copy(), mpos=data.mocap_pos.copy(), mquat=data.mocap_quat.copy(),
                ctrl=data.ctrl.copy(), time=data.time, vmax=vmax)


def add_home_keyframe(sp, log=print):
    model = sp.compile()
    data = mujoco.MjData(model)
    s = settle(model, data, log=log)
    sp.add_key(name="home", qpos=s["qpos"].tolist(), mpos=s["mpos"].ravel().tolist(),
               mquat=s["mquat"].ravel().tolist(), ctrl=s["ctrl"].tolist())
    return s


__all__ = ["add_home_keyframe", "settle", "home_arm_qpos", "rope_body_ids", "quat"]
