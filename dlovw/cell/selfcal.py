"""Camera extrinsics from the right arm's sphere (docs/notes/cell_rgb_rope_20261112.md, section 4).

The ChArUco hand-eye calibration of 2026-11-12 fits well (1.0 / 1.3 px) but fails its independent validation
(about 19 mm / 6 deg). The right arm carries a red sphere whose centre is known from the arm's kinematics and the
base calibration (2.2 mm RMSE): moving it through a few dozen poses and detecting it in each image gives 3D-2D pairs
for a PnP refinement of each camera's pose, with the same world frame as everything else (left UR base).

detect_sphere: red HSV mask, the blob nearest the expected projection; accepted only when the whole sphere is seen
(area at least 85 % of the expected disc, not touching the image edge), so the blob centroid is the projected centre
(within about 1 px at these ranges; the centre of a circle fitted to the blob's outer contour by default, so the
probe on the sphere does not shift it). refine: cv2.solvePnP (iterative, from the current pose) with a RANSAC gate.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import layout as L

RED_LO = [(0, 120, 60), (172, 120, 60)]
RED_HI = [(8, 255, 255), (180, 255, 255)]


def perturb(cam, t_m, r_deg, rng):
    """A copy of cam (K, T) with its pose moved by t_m (random direction) and turned by r_deg (random axis)."""
    T = np.array(cam["T"], float)
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    R, _ = cv2.Rodrigues(ax * np.radians(r_deg))
    d = rng.normal(size=3)
    d /= np.linalg.norm(d)
    T2 = T.copy()
    T2[:3, :3] = R @ T[:3, :3]
    T2[:3, 3] = T[:3, 3] + t_m * d
    return dict(cam, T=T2)


def pose_error(T_est, T_true):
    """(translation mm, rotation deg) between two camera poses."""
    dR = np.asarray(T_est)[:3, :3] @ np.asarray(T_true)[:3, :3].T
    ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
    return 1000 * float(np.linalg.norm(np.asarray(T_est)[:3, 3] - np.asarray(T_true)[:3, 3])), float(ang)


def project(cam, X):
    T = np.asarray(cam["T"], float)
    q = (np.atleast_2d(X) - T[:3, 3]) @ T[:3, :3]
    fx, fy, cx, cy = cam["K"]
    return np.stack([fx * q[:, 0] / q[:, 2] + cx, fy * q[:, 1] / q[:, 2] + cy], 1), q[:, 2]


def detect_sphere(rgb, cam, centre, radius=L.OBSTACLE_RADIUS_M, search_px=120, fit_circle=True):
    """Projected centre (2,) of the sphere if it is wholly visible near its expected place, else None."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in zip(RED_LO, RED_HI):
        m |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    uv, z = project(cam, centre)
    if z[0] <= 0:
        return None
    r_px = cam["K"][0] * radius / z[0]
    d = np.linalg.norm(cen[1:] - uv[0], axis=1)
    k = int(np.argmin(d)) + 1
    if d[k - 1] > search_px:
        return None
    x, y, w, h, area = st[k]
    if x <= 0 or y <= 0 or x + w >= rgb.shape[1] or y + h >= rgb.shape[0]:
        return None
    if area < 0.85 * np.pi * r_px ** 2 or abs(w - h) > 0.15 * max(w, h):
        return None
    if not fit_circle:
        ys, xs = np.nonzero(lab == k)
        return np.array([xs.mean(), ys.mean()])
    # circle through the blob's outer contour: the probe seen on the sphere (a hole or notch) does not move it
    cs, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = max(cs, key=len)[:, 0].astype(np.float64)
    A = np.c_[2 * pts, np.ones(len(pts))]
    sol = np.linalg.lstsq(A, (pts ** 2).sum(1), rcond=None)[0]
    return sol[:2] + 0.5                                 # contour pixels sit half a pixel inside the boundary


def refine(cam, X, uv, reproj_px=4.0):
    """Camera pose from 3D sphere centres X (N, 3) and their pixels uv (N, 2), starting from cam's pose. Returns
    (cam with the new T, info)."""
    T = np.asarray(cam["T"], float)
    R_cw = T[:3, :3].T
    t_cw = -R_cw @ T[:3, 3]
    rvec, _ = cv2.Rodrigues(R_cw)
    tvec = t_cw.reshape(3, 1).copy()
    fx, fy, cx, cy = cam["K"]
    Km = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    X, uv = np.asarray(X, np.float64), np.asarray(uv, np.float64)
    ok, rvec, tvec, inl = cv2.solvePnPRansac(X, uv, Km, None, rvec=rvec, tvec=tvec, useExtrinsicGuess=True,
                                             iterationsCount=200, reprojectionError=reproj_px, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or inl is None or len(inl) < 6:
        return cam, dict(ok=False, n=len(X))
    idx = inl.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(X[idx], uv[idx], Km, None, rvec, tvec)
    R_cw, _ = cv2.Rodrigues(rvec)
    T2 = np.eye(4)
    T2[:3, :3] = R_cw.T
    T2[:3, 3] = (-R_cw.T @ tvec).ravel()
    res = np.linalg.norm(project(dict(cam, T=T2), X[idx])[0] - uv[idx], axis=1)
    return dict(cam, T=T2), dict(ok=True, n=len(X), inliers=len(idx), rms_px=float(np.sqrt((res ** 2).mean())))


SWEEP_BOX = (np.array([-0.22, -0.86, 0.18]), np.array([0.26, -0.40, 0.62]))   # larger than the task's: depth spread


def sweep_poses(rng, n, box=SWEEP_BOX):
    """n sphere centres spread over the sweep box (a jittered 3 x 3 x 3 grid, then uniform)."""
    lo, hi = box
    g = np.stack(np.meshgrid(*[np.linspace(0.15, 0.85, 3)] * 3, indexing="ij"), -1).reshape(-1, 3)
    rng.shuffle(g)
    u = np.r_[g, rng.uniform(0, 1, size=(max(0, n - len(g)), 3))][:n]
    u = np.clip(u + rng.uniform(-0.08, 0.08, size=u.shape), 0, 1)
    return lo + u * (hi - lo)


def sweep(cams_belief, n=30, seed=0, noise_sd=5.0, background=26.0, kin_bias_m=0.002, kin_noise_m=0.001, box=SWEEP_BOX,
          fit_circle=True, tilt_deg=50.0):
    """Simulated calibration sweep: the right arm puts the sphere at n poses (left arm at home; probe tilted by tilt_deg
    towards -x so the wrist is not between the top camera and the sphere), both cameras take a
    frame at each, the sphere is detected with the believed cameras and each camera's pose is refined from the pairs.
    The sphere centre the robot reports carries the base calibration's error (a fixed offset of kin_bias_m plus
    kin_noise_m per pose). Returns (refined cams, info per camera)."""
    from .env import CellEnv
    rng = np.random.default_rng(seed)
    env = CellEnv(control_dt=0.04, max_step_m=0.02)
    env.clip_obstacle = lambda target, current: np.asarray(target, float)    # the sweep box, not the task's
    # probe tilted towards -x (the right wrist then sits beside the sphere, not above it: the top camera sees it)
    th = np.radians(tilt_deg)
    Ry = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    env.R_right = Ry @ env.R_right
    m = env.model
    for name in ("table", "backdrop"):
        m.geom_rgba[m.geom(name).id, :3] = background / 255.0
    env.reset()
    bias = rng.normal(size=3)
    bias *= kin_bias_m / np.linalg.norm(bias)
    pairs = {r: ([], []) for r in cams_belief}
    for p in sweep_poses(rng, n, box):
        for _ in range(60):                                   # move there (at most 2.4 s), then let it settle
            obs = env.step(L.LEFT_HOME_POS, p)
            if np.linalg.norm(obs["obstacle"] - p) < 1e-3:
                break
        for _ in range(5):
            obs = env.step(L.LEFT_HOME_POS, p)
        reported = obs["obstacle"] + bias + rng.normal(size=3) * kin_noise_m
        for role, cam in cams_belief.items():
            img = env.render(f"{role}_view")
            img = np.clip(img.astype(np.float32) + rng.normal(0, noise_sd, img.shape), 0, 255).astype(np.uint8)
            uv = detect_sphere(img, cam, reported, fit_circle=fit_circle)
            if uv is not None:
                pairs[role][0].append(reported)
                pairs[role][1].append(uv)
    env.close()
    out, info = {}, {}
    for role, cam in cams_belief.items():
        X, uv = pairs[role]
        if len(X) < 8:
            out[role], info[role] = cam, dict(ok=False, n=len(X))
            continue
        out[role], info[role] = refine(cam, np.array(X), np.array(uv))
    return out, info
