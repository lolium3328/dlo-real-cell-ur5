"""Camera models for the rendered perception studies (docs/notes/perception_rgb_20261110.md).

A camera is its pinhole intrinsics (fx, fy, cx, cy in pixels, 640x480) and its pose T_world_from_optical (OpenCV optical
frame: +x image right, +y image down, +z forward). Two presets:

- "sim": the cameras of the legacy XML (top_view straight down, fovy 60 deg; front_view level, fovy 46 deg).
- "real": the two RealSense D435i of the robot cell, extrinsics from the cell's calibration
  (real_world_collection configs/real_world/sim2real_532mm_hardware.yaml, world = left UR base, as in simulation)
  and intrinsics fitted to the 39 matched pixels / triangulated points of the 34 recorded frames on the USB drive
  (reprojection RMS 0.8 px; vertical field of view 42.5 and 43.0 deg).

`apply(model, cams)` writes a preset's poses into a MuJoCo model and `warp` resamples each render to the exact
intrinsics; `project` / `rays` are the matching analytic projection and back-projection.
"""
from __future__ import annotations

import mujoco
import numpy as np

W, H = 640, 480
ROLES = ("top", "front")
MJ_NAMES = {"top": "top_view", "front": "front_view"}
OPT_FROM_MJ = np.diag([1.0, -1.0, -1.0])          # MuJoCo camera looks along -z with +y up


def _f(fovy_deg):
    return 0.5 * H / np.tan(np.radians(fovy_deg) / 2)


def _sim():
    top = np.eye(4)
    top[:3, :3] = np.diag([1.0, 1.0, 1.0]) @ OPT_FROM_MJ                 # xyaxes 1 0 0 0 1 0
    top[:3, 3] = (0.68, 0.0, 1.04)
    front = np.eye(4)
    front[:3, :3] = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])  # optical x=+x, y=-z, z=+y (xyaxes 1 0 0 0 0 1)
    front[:3, 3] = (0.68, -0.95, 0.38)
    c = ((W - 1) / 2, (H - 1) / 2)
    return {"top": dict(K=(_f(60.0), _f(60.0), *c), T=top), "front": dict(K=(_f(46.0), _f(46.0), *c), T=front)}


REAL = {
    "front": dict(K=(624.0, 617.0, 326.0, 235.3), T=np.array([
        [0.998664162, -0.024744256, 0.045360934, 0.624363607],
        [-0.045787285, -0.016944522, 0.998807493, -0.934892677],
        [-0.023946129, -0.999550201, -0.018054859, 0.322169622],
        [0.0, 0.0, 0.0, 1.0]])),
    "top": dict(K=(601.8, 608.8, 317.5, 252.4), T=np.array([
        [0.995013196, 0.097639550, 0.020377871, 0.618111766],
        [0.096484138, -0.993998460, 0.051554552, -0.086312195],
        [0.025289336, -0.049331319, -0.998462253, 0.998161725],
        [0.0, 0.0, 0.0, 1.0]])),
}
# Recorded depth x depth_scale = true range (real_audit.py on the 34 USB frames): the front D435i reads 0.894 of the
# range (TCPs: 0.898; rope against the RGB estimate: 0.881-0.910 IQR, the same in every episode; its table plane
# 2 cm high, as a 10 % short reading predicts); the top camera's table plane is level at z = 0.03 m: no correction.
REAL["front"]["depth_scale"] = 1.0 / 0.894
REAL["top"]["depth_scale"] = 1.0
PRESETS = {"sim": _sim(), "real": REAL}


def depth_m(raw, cam):
    """Range image corrected by the camera's depth scale."""
    return np.asarray(raw, float) * float(cam.get("depth_scale", 1.0))


def _orthonormal(R):
    u, _, vt = np.linalg.svd(R)
    return u @ vt


def render_focal(cam):
    """Square-pixel focal length rendered before `warp`: 2 % wider than the narrower axis, so the remap stays inside."""
    return 0.98 * min(cam["K"][0], cam["K"][1])


def apply(model, cams):
    """Write the poses of a preset (dict role -> K, T) into the model's top_view / front_view, with a square-pixel
    centred projection of focal `render_focal`. MuJoCo renders one focal length for both axes, so the exact
    intrinsics (fx != fy, off-centre principal point) are produced by `warp` on the rendered image."""
    for role, c in cams.items():
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, MJ_NAMES[role])
        if model.cam_bodyid[i] != 0:
            raise ValueError(f"{MJ_NAMES[role]} is not attached to the world body")
        R = _orthonormal(np.asarray(c["T"], float)[:3, :3]) @ OPT_FROM_MJ
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, R.reshape(-1))
        model.cam_pos[i], model.cam_quat[i] = np.asarray(c["T"], float)[:3, 3], q
        model.cam_sensorsize[i] = (0.0, 0.0)
        model.cam_fovy[i] = np.degrees(2 * np.arctan(0.5 * H / render_focal(c)))


_MAPS = {}


def warp(img, cam, nearest=False):
    """Resample an image rendered by `apply` to the camera's exact intrinsics."""
    import cv2
    key = (tuple(cam["K"]), nearest)
    if key not in _MAPS:
        fx, fy, cx, cy = cam["K"]
        f = render_focal(cam)
        v, u = np.mgrid[0:H, 0:W].astype(np.float32)
        _MAPS[key] = ((u - cx) / fx * f + (W - 1) / 2).astype(np.float32), ((v - cy) / fy * f + (H - 1) / 2).astype(np.float32)
    mx, my = _MAPS[key]
    return cv2.remap(img, mx, my, cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def project(points, cam):
    """Pixels (N, 2) and depth along the optical axis (N,) of world points (N, 3)."""
    T = np.asarray(cam["T"], float)
    R, t = _orthonormal(T[:3, :3]), T[:3, 3]
    q = (np.asarray(points, float) - t) @ R
    fx, fy, cx, cy = cam["K"]
    return np.stack([fx * q[:, 0] / q[:, 2] + cx, fy * q[:, 1] / q[:, 2] + cy], 1), q[:, 2]


def rays(pixels, cam):
    """Origin (3,) and unit directions (N, 3) in the world frame of the viewing rays through pixels (N, 2)."""
    T = np.asarray(cam["T"], float)
    R = _orthonormal(T[:3, :3])
    fx, fy, cx, cy = cam["K"]
    p = np.asarray(pixels, float)
    d = np.stack([(p[:, 0] - cx) / fx, (p[:, 1] - cy) / fy, np.ones(len(p))], 1) @ R.T
    return T[:3, 3].copy(), d / np.linalg.norm(d, axis=1, keepdims=True)
