"""Safety filter for the cell's single moving rope end (left TCP), every control step.

The legacy CBF-QP (control/cbf_qp.py) acts on two grippers with an online local Jacobian. Here one end is fixed, so
a rope point moves, to first order, by w_i u with w_i = 1 - s_i (s_i its arc fraction from the left TCP): the same
blend the RGB tracker uses for node motion. For every rope point near the obstacle (its believed centre o and
velocity v, inflated radius R):

    h_i = |p_i - o| - R,    n_i . (w_i u - v dt) >= -gamma h_i        (discrete-time CBF, n_i = (p_i - o) / |p_i - o|)

The filtered action is the closest u to the nominal one that satisfies these rows, the workspace box and the stretch
limit (linear rows), solved by Hildreth's dual coordinate ascent (3 unknowns, a few dozen rows); rows that cannot all
hold are relaxed in least-squares sense by the iteration limit. The per-step L1 cap is applied last by scaling, which
keeps every row that held at u = 0 with margin.
"""
from __future__ import annotations

import numpy as np

GAMMA = 0.10              # per control step (8 ms)
ACTIVE_M = 0.10           # rows only for points this close to the inflated surface


def rope_weights(n):
    return 1.0 - np.linspace(0.0, 1.0, n)


def hildreth(u_nom, A, b, iters=200, tol=1e-9):
    """min |u - u_nom|^2 s.t. A u >= b (rows of A, entries of b)."""
    if len(A) == 0:
        return u_nom.copy(), 0
    # The control problem has three coordinates. Keep them as scalars to
    # avoid making NumPy row views and temporaries in each sweep.
    rows = list(zip(A.tolist(), b.tolist(), (A * A).sum(1).tolist()))
    lam = [0.0] * len(rows)
    x, y, z = u_nom.tolist()
    sweeps = iters
    for it in range(iters):
        change = 0.0
        for k, ((a0, a1, a2), rhs, norm2) in enumerate(rows):
            if norm2 < 1e-12:
                continue
            new = max(0.0, lam[k] + (rhs - (a0*x + a1*y + a2*z)) / norm2)
            d = new - lam[k]
            if d:
                x += d*a0
                y += d*a1
                z += d*a2
                lam[k] = new
                change = max(change, abs(d))
        if change < tol:
            sweeps = it + 1
            break
    return np.array([x, y, z]), sweeps


def cbf_filter(u_nom, rope, left, fixed, obstacles, radius, dt, ws_low, ws_high, stretch_max, cap, gamma=GAMMA, norm='l1'):
    """u_nom (3,) m per control step; rope (N, 3) the believed centerline from the left TCP; obstacles list of
    (centre, velocity); radius the inflated protected radius. Returns (u, info)."""
    w = rope_weights(len(rope))
    rows, rhs, h_min = [], [], np.inf
    for c, v in obstacles:
        d = rope - c
        r = np.linalg.norm(d, axis=1)
        h = r - radius
        h_min = min(h_min, float(h.min()))
        for i in np.flatnonzero(h < ACTIVE_M):
            if w[i] < 1e-3:
                continue
            n = d[i] / max(r[i], 1e-9)
            rows.append(w[i] * n)
            rhs.append(-gamma * h[i] + float(n @ v) * dt)
    for k in range(3):                                   # workspace box of the left TCP
        e = np.zeros(3)
        e[k] = 1.0
        rows += [e, -e]
        rhs += [ws_low[k] - left[k], left[k] - ws_high[k]]
    s = left - fixed                                     # stretch: linearised |left + u - fixed| <= stretch_max
    sn = np.linalg.norm(s)
    rows.append(-s / max(sn, 1e-9))
    rhs.append(sn - stretch_max)
    A, b = np.array(rows, float), np.array(rhs, float)
    u, it = hildreth(np.asarray(u_nom, float), A, b)
    magnitude = np.abs(u).sum() if norm == 'l1' else np.linalg.norm(u) if norm == 'l2' else None
    if magnitude is None:
        raise ValueError(f'Unknown action norm {norm}')
    if magnitude > cap:
        u = u * (cap / magnitude)
    changed = float(np.linalg.norm(u - u_nom))
    return u, dict(h_min=h_min, rows=len(A) - 7, iters=it, intervention=changed)
