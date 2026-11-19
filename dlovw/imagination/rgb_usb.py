"""R0: the real cell's RGB-only rope centerline (USB drive, real_world_collection/scripts/auto_yellow_rope_centerline.py),
ported unchanged in method and defaults to the simulator's cameras (camera_model.py).

Per frame, no memory: yellow HSV mask (H 15-45, S >= 60, V >= 40; open 3, close 9; components >= 200 px, keep the
largest; erode 1, retried without erosion if the path breaks) -> Zhang-Suen skeleton -> longest skeleton path ->
39 front samples at equal image arc length -> monotonic epipolar match into the top path (300 candidates, prior
weight 400 on the normalised position, both directions, the one with the smaller mean ray gap) -> closest-point
triangulation of the two rays -> ordered by increasing world x. No depth is used.
"""
from __future__ import annotations

import heapq

import cv2
import numpy as np

from . import camera_model as cm

DEFAULTS = dict(hue_min=15, hue_max=45, sat_min=60, val_min=40, open_kernel=3, close_kernel=9, min_component_area_px=200,
                keep_largest_components=1, mask_erode_iterations=1, thinning_iterations=80, points=39,
                match_candidates=300, match_prior_weight=400.0, epipolar_near_m=0.20, epipolar_far_m=1.50)


def segment_yellow(rgb, p=DEFAULTS, erode=None):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([p["hue_min"], p["sat_min"], p["val_min"]], np.uint8), np.array([p["hue_max"], 255, 255], np.uint8))
    for op, k in ((cv2.MORPH_OPEN, p["open_kernel"]), (cv2.MORPH_CLOSE, p["close_kernel"])):
        if k > 0:
            mask = cv2.morphologyEx(mask, op, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.zeros(mask.shape, np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.flatnonzero(areas >= p["min_component_area_px"]) + 1
    if len(keep) == 0:
        keep = np.array([int(np.argmax(areas)) + 1])
    if p["keep_largest_components"] > 0 and len(keep) > p["keep_largest_components"]:
        keep = keep[np.argsort(stats[keep, cv2.CC_STAT_AREA])[-p["keep_largest_components"]:]]
    mask = np.isin(labels, keep).astype(np.uint8) * 255
    it = p["mask_erode_iterations"] if erode is None else erode
    if it > 0:
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=it)
    return mask


def thin(mask, max_iterations=80):
    """Zhang-Suen thinning (as the original)."""
    img = (mask > 0).astype(np.uint8)
    img[[0, -1], :] = 0
    img[:, [0, -1]] = 0
    for _ in range(max_iterations):
        prev = img.copy()
        for step in (0, 1):
            p2, p4, p6, p8 = np.roll(img, -1, 0), np.roll(img, 1, 1), np.roll(img, 1, 0), np.roll(img, -1, 1)
            p3, p5 = np.roll(p2, 1, 1), np.roll(p6, 1, 1)
            p7, p9 = np.roll(p6, -1, 1), np.roll(p2, -1, 1)
            nb = [p2, p3, p4, p5, p6, p7, p8, p9]
            count = sum(nb)
            trans = sum(((a == 0) & (b == 1)).astype(np.uint8) for a, b in zip(nb, nb[1:] + nb[:1]))
            cond = ((p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)) if step == 0 else ((p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0))
            rm = (img == 1) & (count >= 2) & (count <= 6) & (trans == 1) & cond
            rm[[0, -1], :] = False
            rm[:, [0, -1]] = False
            img[rm] = 0
        if np.array_equal(img, prev):
            break
    return img * 255


def longest_path(skel):
    ys, xs = np.nonzero(skel > 0)
    if len(xs) < 2:
        raise RuntimeError("skeleton has fewer than two pixels")
    pts = np.stack([xs, ys], 1).astype(float)
    index = np.full(skel.shape, -1, np.int32)
    index[ys, xs] = np.arange(len(pts))
    adj = [[] for _ in range(len(pts))]
    for i, (x, y) in enumerate(pts.astype(int)):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if (dx or dy) and 0 <= y + dy < skel.shape[0] and 0 <= x + dx < skel.shape[1] and index[y + dy, x + dx] >= 0:
                    adj[i].append((int(index[y + dy, x + dx]), float(np.hypot(dx, dy))))
    deg = np.array([len(a) for a in adj])
    ends = np.flatnonzero(deg == 1)
    starts = ends if len(ends) >= 2 else np.arange(len(pts))
    best = (-np.inf, None, None)
    for s in starts:
        dist = np.full(len(pts), np.inf)
        prev = np.full(len(pts), -1)
        dist[s] = 0.0
        heap = [(0.0, int(s))]
        while heap:
            dc, c = heapq.heappop(heap)
            if dc != dist[c]:
                continue
            for nb, w in adj[c]:
                if dc + w < dist[nb]:
                    dist[nb], prev[nb] = dc + w, c
                    heapq.heappush(heap, (dc + w, nb))
        cand = ends if len(ends) >= 2 else np.flatnonzero(np.isfinite(dist))
        t = int(cand[np.argmax(dist[cand])])
        if dist[t] > best[0]:
            best = (dist[t], t, prev)
    if best[2] is None or not np.isfinite(best[0]):       # e.g. erosion split the mask: unreachable end points
        raise RuntimeError("no connected skeleton path")
    path, c = [], best[1]
    while c >= 0:
        path.append(c)
        c = int(best[2][c])
    return pts[path[::-1]]


def resample(xy, n):
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cum = np.r_[0.0, np.cumsum(seg)]
    if cum[-1] <= 1e-9:
        return np.repeat(xy[:1], n, 0)
    t = np.linspace(0, cum[-1], n)
    return np.stack([np.interp(t, cum, xy[:, 0]), np.interp(t, cum, xy[:, 1])], 1)


def triangulate(cams, front_xy, top_xy):
    """Midpoints of the closest points of the two viewing rays, and the ray gaps."""
    o1, d1 = cm.rays(front_xy, cams["front"])
    o2, d2 = cm.rays(top_xy, cams["top"])
    w = o1 - o2
    b = (d1 * d2).sum(1)
    dd, e = d1 @ w, d2 @ w
    den = np.maximum(1 - b * b, 1e-12)
    s, t = (b * e - dd) / den, (e - b * dd) / den
    p1, p2 = o1 + s[:, None] * d1, o2 + t[:, None] * d2
    return 0.5 * (p1 + p2), np.linalg.norm(p1 - p2, axis=1)


def _epi_lines(cams, front_xy, p):
    o, d = cm.rays(front_xy, cams["front"])
    a, _ = cm.project(o + p["epipolar_near_m"] * d, cams["top"])
    b, _ = cm.project(o + p["epipolar_far_m"] * d, cams["top"])
    l = np.cross(np.c_[a, np.ones(len(a))], np.c_[b, np.ones(len(b))])
    return l / np.linalg.norm(l[:, :2], axis=1, keepdims=True)


def epipolar_match(cams, front_xy, top_xy, p):
    n, m = len(front_xy), len(top_xy)
    lines = _epi_lines(cams, front_xy, p)
    dist = np.abs(top_xy @ lines[:, :2].T + lines[:, 2]).T                       # (n, m)
    prior = np.abs(np.linspace(0, 1, m)[None] - (np.arange(n) / max(n - 1, 1))[:, None])
    local = dist ** 2 + p["match_prior_weight"] * prior ** 2
    dp = np.full((n, m), np.inf)
    prev = np.full((n, m), -1)
    dp[0] = local[0]
    for i in range(1, n):
        run = np.minimum.accumulate(dp[i - 1])                                    # best over j' <= j-1 (strictly earlier)
        arg = np.zeros(m, int)
        best = 0
        for j in range(m):
            if dp[i - 1, j] < dp[i - 1, best]:
                best = j
            arg[j] = best
        dp[i, 1:] = local[i, 1:] + run[:-1]
        prev[i, 1:] = arg[:-1]
    end = int(np.argmin(dp[-1]))
    idx = [end]
    for i in range(n - 1, 0, -1):
        idx.append(int(prev[i, idx[-1]]))
    return top_xy[np.array(idx[::-1])], float(dp[-1, end] / n)


def estimate(rgbs, cams, p=DEFAULTS):
    """rgbs: dict role -> uint8 RGB image. Returns (centerline (39, 3), info) or raises RuntimeError."""
    paths, fallbacks = {}, []
    for role in cm.ROLES:
        mask = segment_yellow(rgbs[role], p)
        try:
            paths[role] = longest_path(thin(mask, p["thinning_iterations"]))
        except RuntimeError:
            paths[role] = longest_path(thin(segment_yellow(rgbs[role], p, erode=0), p["thinning_iterations"]))
            fallbacks.append(role)
    front = resample(paths["front"], p["points"])
    top_dense = resample(paths["top"], p["match_candidates"])
    best = None
    for name, cand in (("epipolar", top_dense), ("epipolar-reversed", top_dense[::-1])):
        top, cost = epipolar_match(cams, front, cand, p)
        pts, gap = triangulate(cams, front, top)
        if best is None or (gap.mean(), cost) < (best[0], best[1]):
            best = (gap.mean(), cost, name, pts, gap)
    pts, gap = best[3], best[4]
    if pts[0, 0] > pts[-1, 0]:
        pts, gap = pts[::-1], gap[::-1]
    return pts, dict(method=best[2], ray_gap_mean=float(gap.mean()), ray_gap_p95=float(np.percentile(gap, 95)),
                     length=float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()), fallbacks=fallbacks)
