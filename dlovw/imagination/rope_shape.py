"""The rope's shape model from its shapes alone (perception_rgb_20261110.md, section 11).

A hanging inextensible rod at rest has the shape that minimises bending + gravity energy under its length
constraints, and per unit weight that energy depends on one length only, the bending length
l_b = (EI / (rho g))^(1/3):  E / (rho g) = l_b^3 / 2 * integral(kappa^2 ds) + integral(z ds).
Stiffness and weight never need to be known separately, and l_b can be read off shapes that were seen: cut a stretch
out of a seen curve, refill it from its two neighbours, the arc length and this energy (`inpaint`), and take the l_b
that refills best (`calibrate`). Motion, end clamps and contacts make any l_b only an effective value; the error of
the refill at that value is the size of what the model cannot explain.
The end clamps the same way: cut the stretch next to a seen end, refill it with the clamp spring of stiffness k
(relative to one bending joint) towards the tangent the tracker would hold there (learned on other frames), and take
the k that refills best (`calibrate_clamp`). k says how much a clamp gives: how far the true end tangent strays from
the learned one with the configuration.
"""
from __future__ import annotations

import numpy as np

from .rgb_rope import chain


def inpaint(x, lo, hi, l_bend, iters=150, step=0.002, clamp=None, clamp_k=0.0):
    """Nodes lo..hi of curve x (n, 3) refilled: the rest held, segment lengths kept, bending + gravity minimised.
    Starts from the straight chord between the held neighbours (no knowledge of the cut stretch). clamp: (tangent at
    node 0, tangent at node n-1), world frame, either None: a spring clamp_k * l_b^3 / seg^4 on node 1 (n-2) towards
    end + seg * tangent, as in RopeRGB."""
    x = np.array(x, float)
    n = len(x)
    seg = float(np.median(np.linalg.norm(np.diff(x, axis=0), axis=1)))
    a, b = x[lo - 1].copy(), x[hi + 1].copy()
    m = hi - lo + 1
    t = np.linspace(0, 1, m + 2)[1:-1, None]
    x[lo:hi + 1] = a + t * (b - a) - np.c_[0 * t, 0 * t, 0.02 * np.sin(np.pi * t)]     # small sag to break symmetry
    k3 = l_bend ** 3 / seg ** 4
    trip = [k for k in range(n - 2) if k + 2 >= lo and k <= hi]                       # bending terms touching the cut
    free = np.arange(3 * lo, 3 * (hi + 1))
    for _ in range(iters):
        g = np.zeros((n, 3))
        Hb = np.zeros((3 * n, 3 * n))
        for k in trip:
            dd = x[k] - 2 * x[k + 1] + x[k + 2]
            for j, c in ((k, 1.0), (k + 1, -2.0), (k + 2, 1.0)):
                g[j] += k3 * c * dd
            for j1, c1 in ((k, 1.0), (k + 1, -2.0), (k + 2, 1.0)):
                for j2, c2 in ((k, 1.0), (k + 1, -2.0), (k + 2, 1.0)):
                    Hb[3 * j1:3 * j1 + 3, 3 * j2:3 * j2 + 3] += k3 * c1 * c2 * np.eye(3)
        g[lo:hi + 1, 2] += 1.0
        for end, nxt, tang in ((0, 1, None if clamp is None else clamp[0]), (n - 1, n - 2, None if clamp is None else clamp[1])):
            if tang is not None and clamp_k > 0 and lo <= nxt <= hi:
                g[nxt] += clamp_k * k3 * (x[nxt] - x[end] - seg * np.asarray(tang, float))
                Hb[3 * nxt:3 * nxt + 3, 3 * nxt:3 * nxt + 3] += clamp_k * k3 * np.eye(3)
        segs = range(lo - 1, hi + 1)
        C = np.zeros((len(segs), 3 * n))
        cv = np.zeros(len(segs))
        for r, k in enumerate(segs):
            d = x[k + 1] - x[k]
            C[r, 3 * k:3 * k + 3], C[r, 3 * k + 3:3 * k + 6] = -2 * d, 2 * d
            cv[r] = d @ d - seg ** 2
        Hf = Hb[np.ix_(free, free)] + 1e-3 * np.eye(len(free))
        Cf = C[:, free]
        K = np.block([[Hf, Cf.T], [Cf, np.zeros((len(segs), len(segs)))]])
        sol = np.linalg.lstsq(K, np.r_[-g.reshape(-1)[free], -cv], rcond=None)[0]
        dx = sol[:len(free)].reshape(-1, 3)
        mx = np.linalg.norm(dx, axis=1).max()
        x[lo:hi + 1] += dx * min(1.0, step / max(mx, 1e-12))
        sub = chain(x[lo - 1:hi + 2], np.r_[a, b], seg, iters=30)
        x[lo:hi + 1] = sub[1:-1]
    return x


def calibrate(curves, grid=(0.01, 0.02, 0.03, 0.05, 0.08, 0.12), cut=8, cuts_per_curve=3, rng=None, reference=None):
    """Bending length that best refills cut stretches of the given curves (list of (n, 3)). Returns (best l_b,
    {l_b: mean refill error (m)}). reference: optional true curves (same order) to score the refill against instead
    of the curves themselves (simulation check)."""
    rng = rng or np.random.default_rng(0)
    jobs = []
    for ci, x in enumerate(curves):
        n = len(x)
        for _ in range(cuts_per_curve):
            lo = int(rng.integers(3, n - cut - 3))
            jobs.append((ci, lo, lo + cut - 1))
    err = {}
    for lb in grid:
        e = []
        for ci, lo, hi in jobs:
            y = inpaint(curves[ci], lo, hi, lb)
            ref = curves[ci] if reference is None else reference[ci]
            e.append(np.linalg.norm(y[lo:hi + 1] - ref[lo:hi + 1], axis=1).mean())
        err[lb] = float(np.mean(e))
    return min(err, key=err.get), err


def calibrate_clamp(curves, tangents, l_bend, grid=(0.0, 0.3, 1.0, 3.0, 10.0, 30.0), cut=8, reference=None, skip_tip=1):
    """Clamp stiffness that best refills the stretch next to a seen end. curves: list of (n, 3); tangents: per curve
    (tangent at node 0, at node n-1), world frame, the one the tracker would hold (learned on other frames), either
    None to skip that end. Returns (best k, {k: mean refill error (m)}, {k: per-end errors}). reference: optional true
    curves to score against (simulation check). The cut stretch is nodes 1..cut from the end; the skip_tip nodes next
    to the end are refilled but not scored (on the real cell the probe hides node 1 in the top view)."""
    jobs = [(ci, e) for ci, tt in enumerate(tangents) for e in (0, 1) if tt[e] is not None]
    err, per = {}, {}
    for k in grid:
        e_ = []
        for ci, e in jobs:
            x = curves[ci]
            n = len(x)
            lo, hi = (1, cut) if e == 0 else (n - 1 - cut, n - 2)
            cl = (tangents[ci][0], None) if e == 0 else (None, tangents[ci][1])
            y = inpaint(x, lo, hi, l_bend, clamp=cl, clamp_k=k)
            ref = x if reference is None else reference[ci]
            sc = slice(lo + skip_tip, hi + 1) if e == 0 else slice(lo, hi + 1 - skip_tip)
            e_.append(np.linalg.norm(y[sc] - ref[sc], axis=1).mean())
        err[k], per[k] = float(np.mean(e_)) if e_ else float("nan"), np.array(e_)
    return min(err, key=err.get), err, per


def bending_length(EI, mu, g=9.81):
    """l_b (m) from the bending stiffness EI (N m^2) and the mass per length mu (kg/m)."""
    return float((EI / (mu * g)) ** (1.0 / 3.0))


def sim_bending_length(model, prefix="dlo_"):
    """l_b of the simulated rope from its own parameters, no fitting: a chain of rigid links with ball-joint springs,
    EI = joint stiffness x link length (k theta = EI kappa ds), mu = link mass / link length. Returns (l_b, parts)."""
    import mujoco
    ids = [b for b in range(model.nbody) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(prefix)]
    seg = float(np.median(np.linalg.norm(model.body_pos[ids[1:]], axis=1)))
    k = float(np.median([model.jnt_stiffness[j] for j in range(model.njnt) if model.jnt_bodyid[j] in ids[1:]]))
    mu = float(np.median(model.body_mass[ids])) / seg
    g = float(-model.opt.gravity[2])
    EI = k * seg
    return bending_length(EI, mu, g), dict(link_m=seg, joint_k=k, EI=EI, mu=mu)


def refill(x, lo, hi, l_bend, iters=80, step=0.005, start=None):
    """inpaint, fast enough to run online: the same energy and constraints solved on the window around the cut
    (nodes lo-2 .. hi+2) only. start: initial nodes lo..hi (warm start, e.g. the refill at a neighbouring l_b).
    Returns the refilled nodes lo..hi (m, 3)."""
    x = np.asarray(x, float)
    n = len(x)
    seg = float(np.median(np.linalg.norm(np.diff(x, axis=0), axis=1)))
    w0, w1 = max(lo - 2, 0), min(hi + 2, n - 1)
    xw = x[w0:w1 + 1].copy()
    a, b = lo - w0, hi - w0                                  # free nodes in window coordinates
    m = b - a + 1
    if start is None:
        t = np.linspace(0, 1, m + 2)[1:-1, None]
        xw[a:b + 1] = xw[a - 1] + t * (xw[b + 1] - xw[a - 1]) - np.c_[0 * t, 0 * t, 0.02 * np.sin(np.pi * t)]
    else:
        xw[a:b + 1] = start
    nw = len(xw)
    D = np.zeros((nw - 2, nw))
    for i in range(nw - 2):
        D[i, i:i + 3] = (1.0, -2.0, 1.0)
    k3 = l_bend ** 3 / seg ** 4
    Hb = k3 * np.kron(D.T @ D, np.eye(3))
    fr = np.arange(3 * a, 3 * (b + 1))
    Hf = Hb[np.ix_(fr, fr)] + 1e-3 * np.eye(3 * m)
    gz = np.zeros(3 * nw)
    gz[3 * a + 2:3 * (b + 1):3] = 1.0
    for _ in range(iters):
        g = Hb @ xw.reshape(-1) + gz
        d = np.diff(xw[a - 1:b + 2], axis=0)                 # segments a-1 .. b
        C = np.zeros((m + 1, 3 * nw))
        for r in range(m + 1):
            k = a - 1 + r
            C[r, 3 * k:3 * k + 3], C[r, 3 * k + 3:3 * k + 6] = -2 * d[r], 2 * d[r]
        Cf = C[:, fr]
        K = np.block([[Hf, Cf.T], [Cf, np.zeros((m + 1, m + 1))]])
        sol = np.linalg.solve(K, np.r_[-g[fr], -((d ** 2).sum(1) - seg ** 2)])
        dx = sol[:3 * m].reshape(m, 3)
        mx = np.linalg.norm(dx, axis=1).max()
        xw[a:b + 1] += dx * min(1.0, step / max(mx, 1e-12))
        if mx < 1e-5:
            break
    return xw[a:b + 1].copy()


class OnlineBendingLength:
    """l_b read online from the rope being tracked, no offline fit to a particular rope: on frames where a stretch and
    its two neighbours on each side are seen in both views, the stretch is cut, refilled (`refill`) at every l_b of a
    log grid (warm-started along the grid) and each refill's error is kept as a running mean; the estimate is the
    minimum, interpolated in log l_b. Before min_n stretches it stays at the starting value."""

    def __init__(self, l_start=0.08, grid=None, cut=8, alpha=0.05, min_n=5, seed=0):
        self.grid = np.geomspace(0.02, 0.2, 13) if grid is None else np.asarray(grid, float)
        self.cut, self.alpha, self.min_n = int(cut), float(alpha), int(min_n)
        self.err, self.n, self.l = np.zeros(len(self.grid)), 0, float(l_start)
        self.rng = np.random.default_rng(seed)

    def observe(self, x, seen):
        """x (n, 3) the fitted curve, seen (n,) nodes seen in both views. Returns True if a stretch was used."""
        n, c = len(x), self.cut
        ok = np.convolve(seen.astype(int), np.ones(c + 4, int), "valid") == c + 4     # stretch + 2 neighbours each side
        starts = np.flatnonzero(ok) + 2                                                # lo of each usable stretch
        starts = starts[(starts >= 3) & (starts + c - 1 <= n - 4)]                     # away from the clamped ends
        if not len(starts):
            return False
        lo = int(self.rng.choice(starts))
        hi = lo + c - 1
        e, prev = np.zeros(len(self.grid)), None
        for j, lb in enumerate(self.grid):
            prev = refill(x, lo, hi, lb, start=prev)
            e[j] = np.linalg.norm(prev - x[lo:hi + 1], axis=1).mean()
        self.n += 1
        a = max(1.0 / self.n, self.alpha)
        self.err = e if self.n == 1 else (1 - a) * self.err + a * e
        if self.n >= self.min_n:
            j = int(np.clip(np.argmin(self.err), 1, len(self.grid) - 2))
            y0, y1, y2 = self.err[j - 1:j + 2]
            u = np.log(self.grid)
            den = y0 - 2 * y1 + y2
            off = 0.5 * (y0 - y2) / den if den > 1e-12 else 0.0
            self.l = float(np.exp(u[j] + np.clip(off, -1, 1) * (u[j + 1] - u[j])))
        return True
