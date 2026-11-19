"""Front camera for the I0 image setting (docs/notes/i0_prereg_20261024.md, section 2).

Frames are the legacy front RGB-D renders (160x120, or 320x240 for the 128x128 arm), resized as the
release RSSM does (RGB INTER_AREA, depth INTER_NEAREST), then corrupted by one of three camera levels:
- clean: as rendered;
- realistic: per-pixel depth noise sigma(z) = z^2 * 0.08 / (424 px * 0.05 m) (Intel D400 tuning
  white paper, D435 at 848 px / 90 deg / 50 mm / subpixel 0.08), edge pixels (3x3 depth range > 2 cm)
  invalid with probability 0.5, 1% random invalid pixels, RGB noise 0.01; one frame of latency
  (handled by the caller);
- extrinsic: realistic plus a fixed unknown camera pose error (5 mm, 0.3 deg) in the pose the
  estimators use for back-projection.
Invalid depth is NaN. Back-projection follows MuJoCo's camera frame (x right, y up, looking along -z).
"""
from __future__ import annotations

import cv2
import numpy as np

LEVELS = ("clean", "realistic", "extrinsic")
SUBPIXEL, FOCAL_PX, BASELINE_M = 0.08, 424.0, 0.05
EDGE_RANGE_M, EDGE_DROP, RANDOM_DROP, RGB_NOISE = 0.02, 0.5, 0.01, 0.01
POSE_ERR_M, POSE_ERR_DEG = 0.005, 0.3


def _rotation(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


class FrontCamera:
    def __init__(self, model, data, *, level="realistic", res=64, fovy_deg=None, camera="front_view", rng=None):
        import mujoco
        if level not in LEVELS:
            raise ValueError(level)
        self.data, self.level, self.res = data, level, int(res)
        self.cam_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera))
        self.fovy = np.deg2rad(float(model.cam_fovy[self.cam_id]) if fovy_deg is None else fovy_deg)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.pose_error = None
        if level == "extrinsic":
            d = self.rng.normal(size=3)
            shift = POSE_ERR_M * d / np.linalg.norm(d)
            rot = _rotation(self.rng.normal(size=3), np.deg2rad(POSE_ERR_DEG))
            self.pose_error = (shift, rot)

    def intrinsics(self, src_hw):
        """(fx, fy, cx, cy) of the resized image for a source render of size src_hw."""
        h, w = src_hw
        f_src = (h / 2) / np.tan(self.fovy / 2)                   # square pixels in the render
        return f_src * self.res / w, f_src * self.res / h, self.res / 2, self.res / 2

    def pose(self):
        """Camera position and rotation (camera frame -> world) the estimators believe."""
        pos = self.data.cam_xpos[self.cam_id].copy()
        rot = self.data.cam_xmat[self.cam_id].reshape(3, 3).copy()
        if self.pose_error is not None:
            pos, rot = pos + self.pose_error[0], self.pose_error[1] @ rot
        return pos, rot

    def process(self, rgb_src, depth_src):
        """Resized and corrupted (rgb float in [0, 1], depth in m with NaN), plus intrinsics and the
        believed pose at capture time."""
        size = (self.res, self.res)
        rgb = cv2.resize(rgb_src.astype(np.float32) / 255.0, size, interpolation=cv2.INTER_AREA)
        depth = cv2.resize(depth_src.astype(np.float32), size, interpolation=cv2.INTER_NEAREST).astype(np.float64)
        if self.level != "clean":
            sigma = depth ** 2 * SUBPIXEL / (FOCAL_PX * BASELINE_M)
            pad = np.pad(depth, 1, mode="edge")
            win = np.stack([pad[i:i + self.res, j:j + self.res] for i in range(3) for j in range(3)])
            edge = (win.max(0) - win.min(0)) > EDGE_RANGE_M
            depth = depth + sigma * self.rng.normal(size=depth.shape)
            drop = (edge & (self.rng.uniform(size=depth.shape) < EDGE_DROP)) | (self.rng.uniform(size=depth.shape) < RANDOM_DROP)
            depth[drop] = np.nan
            rgb = np.clip(rgb + RGB_NOISE * self.rng.normal(size=rgb.shape), 0.0, 1.0)
        return rgb, depth, self.intrinsics(depth_src.shape[:2]), self.pose()

    @staticmethod
    def rays(us, vs, intr, pose):
        """Unit world directions through pixel centres (us = columns, vs = rows)."""
        fx, fy, cx, cy = intr
        pos, rot = pose
        local = np.stack([(us + 0.5 - cx) / fx, -(vs + 0.5 - cy) / fy, -np.ones(len(us))], axis=1)
        d = local @ rot.T
        return d / np.linalg.norm(d, axis=1, keepdims=True), local

    @staticmethod
    def backproject(us, vs, z, intr, pose):
        """World points for pixels with planar depth z (distance along the optical axis)."""
        fx, fy, cx, cy = intr
        pos, rot = pose
        local = np.stack([(us + 0.5 - cx) / fx * z, -(vs + 0.5 - cy) / fy * z, -z], axis=1)
        return pos[None] + local @ rot.T

    @staticmethod
    def project(points, intr, pose):
        """(u, v, z) pixel coordinates (continuous, pixel centres at +0.5) of world points."""
        fx, fy, cx, cy = intr
        pos, rot = pose
        local = (np.atleast_2d(points) - pos[None]) @ rot
        z = -local[:, 2]
        return local[:, 0] / z * fx + cx, -local[:, 1] / z * fy + cy, z
