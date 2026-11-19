"""RGB-only obstacle spheres (known radius 40 mm, identity by colour) from the two cameras (perception_rgb_20261110.md).

Per camera and sphere colour: chromaticity mask (as vision_multi.colour_mask). A circle is fitted to the silhouette's
outline, but only to outline pixels whose outside neighbour is dark background: where the outline borders the rope,
the robot, the other sphere or the image edge it is the occluder's edge, not the sphere's, and is left out
(this is what keeps a partly hidden sphere from being pulled towards its visible half). The fit is algebraic (Kasa),
refined by Gauss-Newton on geometric distances with one round of outlier rejection; it is rejected if fewer than
MIN_ARC px of trusted outline remain or they span less than MIN_SPAN of the circle.

3D: the ray through the circle centre of each camera that accepted a fit, and the range from the apparent size
(range = R / sin(half-angle), half-angle from the radius in px). With two cameras the centre is the least-squares
point of the two rays, with the size ranges as weak priors; with one camera, ray and size range alone.
`naive=True` gives the plain variant for comparison: mask centroid and radius from the mask area.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import camera_model as cm

R_SPHERE = 0.040
CHROMA_TOL = 0.12
MIN_PX, MIN_ARC, MIN_SPAN = 60, 25, 0.25      # mask pixels, trusted outline pixels, share of the circumference
DARK_V = 70                                    # background (dark cloth / table) is darker than this


def colour_mask(rgb, colour):
    """Largest blob of pixels whose chromaticity (r, g, b) / (r + g + b) lies within CHROMA_TOL of the colour's."""
    ch = [rgb[..., i].astype(np.float32) for i in range(3)]
    tot = ch[0] + ch[1] + ch[2]
    ref = np.asarray(colour[:3], np.float32)
    ref = ref / ref.sum()
    d2 = (ch[0] - ref[0] * tot) ** 2 + (ch[1] - ref[1] * tot) ** 2 + (ch[2] - ref[2] * tot) ** 2
    m = (d2 < (CHROMA_TOL * tot) ** 2) & (rgb.max(-1) > 0.25 * 255)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.zeros(m.shape, bool)
    return lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))


def fit_circle(pts):
    x, y = pts[:, 0], pts[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    c = np.linalg.lstsq(A, x * x + y * y, rcond=None)[0]
    cx, cy = c[0], c[1]
    r = np.sqrt(max(c[2] + cx * cx + cy * cy, 1e-6))
    for _ in range(2):
        for _ in range(8):
            d = pts - (cx, cy)
            dist = np.linalg.norm(d, axis=1).clip(1e-9)
            J = np.c_[-d / dist[:, None], -np.ones(len(d))]
            step = np.linalg.lstsq(J, -(dist - r), rcond=None)[0]
            cx, cy, r = cx + step[0], cy + step[1], r + step[2]
        res = np.abs(np.linalg.norm(pts - (cx, cy), axis=1) - r)
        keep = res < max(1.5, 3 * np.median(res))
        if keep.all() or keep.sum() < MIN_ARC:
            break
        pts = pts[keep]
    return np.array([cx, cy]), float(r), pts


def trusted_outline(rgb, colour, roi=None):
    """Outline pixels (n, 2) of the colour's silhouette whose outside neighbour is dark background, and the mask, or
    None. roi = (x0, y0, x1, y1): look only there (with a track the sphere's image position is known)."""
    x0, y0 = 0, 0
    if roi is not None:
        x0, y0 = max(int(roi[0]), 0), max(int(roi[1]), 0)
        x1, y1 = min(int(roi[2]), cm.W), min(int(roi[3]), cm.H)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        rgb = rgb[y0:y1, x0:x1]
    m = colour_mask(rgb, colour)
    if m.sum() < MIN_PX:
        return None
    hsv_v = rgb.max(-1)
    mu8 = m.astype(np.uint8)
    edge = mu8 - cv2.erode(mu8, np.ones((3, 3), np.uint8))
    outside = cv2.dilate(mu8, np.ones((5, 5), np.uint8)) - cv2.dilate(mu8, np.ones((3, 3), np.uint8))
    dark = (hsv_v < DARK_V) & (outside > 0)
    # an outline pixel is trusted when the ring just outside it, within 2 px, is dark background and in the image
    dark_near = cv2.dilate(dark.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    bright_near = cv2.dilate(((outside > 0) & ~dark).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    ys, xs = np.nonzero((edge > 0) & dark_near & ~bright_near)
    xs, ys = xs + x0, ys + y0
    border = (xs < 3) | (xs > cm.W - 4) | (ys < 3) | (ys > cm.H - 4)
    return np.stack([xs[~border], ys[~border]], 1).astype(float), m


def _span(pts, c):
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    return len(np.unique(np.floor((ang + np.pi) / (2 * np.pi) * 36))) / 36


def circle_2d(rgb, colour, naive=False):
    """(centre px (2,), radius px, n trusted outline px, share of the circumference covered) or None."""
    out = trusted_outline(rgb, colour)
    if out is None:
        return None
    pts, m = out
    if naive:
        ys, xs = np.nonzero(m)
        return np.array([xs.mean(), ys.mean()]), float(np.sqrt(m.sum() / np.pi)), int(m.sum()), 1.0
    if len(pts) < MIN_ARC:
        return None
    c, r, used = fit_circle(pts)
    span = _span(used, c)
    if span < MIN_SPAN or not (2 < r < 200):
        return None
    return c, r, len(used), span


def circle_known_radius(pts, r0, c0, max_shift, tol=1.5, trials=300, rng=None):
    """Centre of a circle of known radius r0 through the outline pixels, near the predicted centre c0: RANSAC over
    point pairs (each pair gives two candidate centres), then least squares on the inliers with the radius fixed.
    Outline pixels of an occluder's edge (wrist, the robot's light-blue caps) do not lie on that circle and drop out.
    Returns (centre, inlier points) or None."""
    if len(pts) < MIN_ARC:
        return None
    rng = rng or np.random.default_rng(0)
    i, j = rng.integers(len(pts), size=(2, trials))
    a, b = pts[i], pts[j]
    mid, dv = 0.5 * (a + b), b - a
    h2 = r0 ** 2 - 0.25 * (dv ** 2).sum(1)
    ok = (h2 > 0) & ((dv ** 2).sum(1) > 16)
    if not ok.any():
        return None
    nrm = np.c_[-dv[ok, 1], dv[ok, 0]] / np.linalg.norm(dv[ok], axis=1, keepdims=True)
    h = np.sqrt(h2[ok])[:, None]
    cand = np.concatenate([mid[ok] + h * nrm, mid[ok] - h * nrm])
    cand = cand[np.linalg.norm(cand - c0, axis=1) < max_shift]
    if not len(cand):
        return None
    res = np.abs(np.linalg.norm(pts[None] - cand[:, None], axis=2) - r0)          # (candidates, points)
    best = cand[int(np.argmax((res < tol).sum(1)))]
    c = best
    for _ in range(5):
        inl = pts[np.abs(np.linalg.norm(pts - c, axis=1) - r0) < tol]
        if len(inl) < 3:
            return None
        d = inl - c
        dist = np.linalg.norm(d, axis=1).clip(1e-9)
        Jm = -d / dist[:, None]
        c = c + np.linalg.lstsq(Jm, -(dist - r0), rcond=None)[0]
    inl = pts[np.abs(np.linalg.norm(pts - c, axis=1) - r0) < 2.5 * tol]      # wider: keep the noisy outline for the final fit
    return c, inl


def locate(rgbs, cams, colour, naive=False):
    """3D centre (3,) and the number of cameras used, or (None, 0)."""
    obs = []
    for role in cm.ROLES:
        c2 = circle_2d(rgbs[role], colour, naive)
        if c2 is None:
            continue
        o, d = cm.rays(c2[0][None], cams[role])
        fy = cams[role]["K"][1]
        rng = R_SPHERE / np.sin(np.arctan(c2[1] / fy))
        obs.append((o, d[0], rng))
    if not obs:
        return None, 0
    if len(obs) == 1:
        o, d, rng = obs[0]
        return o + rng * d, 1
    # least squares over p: distance to both rays (weight 1) + range along each ray to its size range (weight 0.05)
    A, b = [], []
    for o, d, rng in obs:
        P = np.eye(3) - np.outer(d, d)
        A.append(P)
        b.append(P @ o)
        A.append(0.05 * d[None])
        b.append(0.05 * np.array([d @ o + rng]))
    return np.linalg.lstsq(np.vstack(A), np.concatenate(b), rcond=None)[0], len(obs)
