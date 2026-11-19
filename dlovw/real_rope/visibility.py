"""Calibrated field-of-view bounds, separate from geometric occlusion tests."""
from __future__ import annotations

import mujoco
import numpy as np

from ..imagination.camera_model import _orthonormal, project


def frustum(cam, margin_px=20., width=640, height=480, near_m=.05):
    """Unit world planes A @ x >= b for an inset pinhole image and near plane."""
    fx, fy, cx, cy = cam['K']
    T = np.asarray(cam['T'])
    local = np.array([[fx, 0, cx-margin_px], [-fx, 0, width-1-margin_px-cx],
                      [0, fy, cy-margin_px], [0, -fy, height-1-margin_px-cy], [0, 0, 1.]])
    A = local @ _orthonormal(T[:3, :3]).T
    A /= np.linalg.norm(A, axis=1)[:, None]
    b = A @ T[:3, 3]
    b[-1] += near_m
    return A, b


def curve_margin(points, cams, radius_m, margin_px=20., uncertainty_m=.01):
    """Smallest signed plane clearance of the entire polyline's inflated tube."""
    out = {}
    for role, cam in cams.items():
        A, b = frustum(cam, margin_px)
        out[role] = float((np.asarray(points) @ A.T - b).min() - radius_m - uncertainty_m)
    return out


def envelope_margin(tcp, fixed, length, cams, radius_m, margin_px=20., uncertainty_m=.01):
    """Sufficient FOV bound for EVERY inextensible curve between two endpoints.

    Any curve point has sum of distances to endpoints <= L. Its ellipsoid has
    centre c, shape matrix Q = ((L²-|d|²) I + d dᵀ)/4, support sqrt(aᵀ Q a).
    An empty certified set does not imply actual gravity-loaded curves are invisible.
    """
    tcp = np.atleast_2d(np.asarray(tcp, float))
    d = tcp - fixed
    c = (tcp + fixed)/2
    b2 = (length**2 - np.sum(d*d, axis=1))/4
    out = {}
    for role, cam in cams.items():
        A, b = frustum(cam, margin_px)
        support = np.sqrt(np.maximum(0, b2[:, None] + (d @ A.T)**2/4))
        v = (c @ A.T - b - support).min(axis=1) - radius_m - uncertainty_m
        out[role] = np.where(b2 >= 0, v, -np.inf)
    return out


def ray_visibility(model, data, curve, cams, rope_geom_ids, radius_m):
    """Simulation truth only: nearest rendered surface on each centreline ray.

    Match default rendering groups 0,1,2 (not duplicate collision geometry 3).
    A different strand closer than radius+4mm is classified as self-occlusion.
    This is a point-ray test, not a guarantee about segmentation or hardware.
    """
    out = {}
    group = np.array([1, 1, 1, 0, 0, 0], np.uint8)
    hit = np.zeros(1, np.int32)
    rope_ids = set(map(int, rope_geom_ids))
    for role, cam in cams.items():
        uv, depth = project(curve, cam)
        eye = np.asarray(cam['T'])[:3, 3].copy()
        visible, labels = [], []
        for p, pixel, z in zip(curve, uv, depth):
            if z <= 0 or not (0 <= pixel[0] <= 639 and 0 <= pixel[1] <= 479):
                visible.append(False); labels.append('outside'); continue
            v = p-eye
            dist = float(np.linalg.norm(v))
            t = mujoco.mj_ray(model, data, eye, v/dist, group, True, -1, hit)
            gid = int(hit[0])
            local_rope = gid in rope_ids and t >= 0 and abs(dist-t) <= radius_m+.004
            visible.append(local_rope)
            labels.append('rope' if local_rope else 'self' if gid in rope_ids else
                          (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or 'arm')
                          if gid >= 0 else 'no_hit')
        out[role] = dict(visible=visible, labels=labels)
    return out
