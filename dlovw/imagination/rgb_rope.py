"""R1: RGB-only rope tracker that keeps working when parts of the rope are hidden (perception_rgb_20261110.md).

Why R0 (rgb_usb.py, the real cell's method) fails: it treats the longest skeleton path of the largest yellow blob in
each image as the whole rope, spreads 39 points over it at equal image arc length and pairs the two views by
epipolar order. When the gripper fingers hide the ends, a sphere or a wrist splits the rope, or the rope leaves the
image, the visible piece is stretched over the whole rope and the pairs no longer belong to the same rope point.

R1 instead fits the 39 3D nodes of a rope of known length, with both ends at the grippers (robot kinematics), to the
two images directly:
- per camera, the yellow mask (same HSV thresholds as R0, every blob kept), its distance transform, and an
  occluder mask: pixels that are neither rope nor dark background (robot, obstacle spheres), dilated by 3 px;
- EM in the image plane: visible rope pixels are softly assigned to the projected nodes, each node is pulled to the
  mean of its pixels (this spreads the nodes along the visible rope, whatever is missing);
- a node whose projection lies on background is pulled to the nearest rope pixel (distance-transform term), while a
  node that projects onto an occluder or outside the image gets no pull from that camera: it is unobserved there;
- every node is tied to its prior (the previous estimate carried with the grippers) and to its neighbours (second
  differences), and each iteration ends with the inextensible chain pinned at both grippers (vision.chain).
Unobserved stretches therefore follow the prior and the shape of the observed rope instead of being invented.

Occluded stretches (section 11 of the notes): a node behind an occluder gets no image term from that camera (before,
rope pixels beside the occluder pulled it); where a view is missing, the generic smoothing and most of the pull to the
last frame give way to the rope's own statics, bending + gravity with one parameter, the bending length l_b
(rope_shape.py: read online off this rope's seen stretches, OnlineBendingLength; no material constants and no offline
fit to a particular rope; l_bend is only the starting value; off by default: it converges to 0.083 from 0.03-0.15
but its frame-to-frame scatter, 0.07-0.11, costs more than it gains: sphere-hidden p90 4.1 mm vs 3.3 mm fixed), with the segment lengths as constraints (KKT). Each
node's sd comes with the estimate (`node_sd`): the fit's own information plus a floor and, where a view is missing,
the shape model's error (shape_sd). l_bend=None restores the previous behaviour (with occ_blind=False).
The ends are clamped, not hinged: each end's tangent is learned online from frames where the nodes next to it are
seen (quadratic fit through the first six nodes, running average; or given as `clamp`, calibrated beforehand like
l_bend), and held by a rotational spring clamp_k times one bending joint (the sim's welds give a little: the right
end's tangent moves 7-12 deg with the configuration). Refilling end stretches of seen shapes
(rope_shape.calibrate_clamp) rules out hinged ends (k = 0: 8 mm sim, 6 mm real) and leaves k = 3-30 flat on both; 3
is the best of that range against simulation truth (the spring also steadies seen ends, node 1 next to the tool). The tangents are kept in each tool's frame when the tool orientations are given
(`update(..., grip_rot=...)`), so a clamp turns with its wrist; without them, in the world frame.

Accuracy changes (section 9 of the notes): every 3rd rope pixel on a regular grid instead of 700 random ones (the
random subset made the fit jitter by 1-2 mm); the two views are fitted at their own capture times (`update(...,
t, view_t, grip_vel)`: each view sees the nodes moved by the grippers' velocity blended along the rope; a 16 ms
offset otherwise costs 1.2 mm); known obstacle spheres are kept out of the rope (`spheres`); smoothing 1e6, 4
iterations per frame. 26 ms per frame pair on one CPU core.
The first frame starts from R0 when R0's result is plausible (length within 15 % of the rope, mean ray gap < 1 cm),
otherwise from a sagging curve between the grippers. No depth is used.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import camera_model as cm

N = 39
L_ROPE = 0.531
HSV_LO, HSV_HI = (15, 60, 40), (45, 255, 255)


def masks(rgb):
    """Rope mask (bool), its distance transform (px) with gradient, and the occluder mask (bool)."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    rope = cv2.inRange(hsv, np.array(HSV_LO, np.uint8), np.array(HSV_HI, np.uint8))
    rope = cv2.morphologyEx(rope, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(rope, connectivity=8)
    small = np.flatnonzero(st[1:, cv2.CC_STAT_AREA] < 30) + 1
    if len(small):
        rope[np.isin(lab, small)] = 0
    r = rope > 0
    s, v = hsv[..., 1].astype(np.int32), hsv[..., 2].astype(np.int32)
    occ = ((v >= 70) | ((s >= 90) & (v >= 45))) & ~cv2.dilate(rope, np.ones((5, 5), np.uint8)).astype(bool)
    occ = cv2.dilate(occ.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
    dt = cv2.distanceTransform((~r).astype(np.uint8), cv2.DIST_L2, 5)
    gy, gx = np.gradient(dt)
    return r, dt, np.stack([gx, gy], -1), occ


def project_jac(x, cam):
    """Pixels (N, 2), depth (N,) and Jacobians d pixel / d world point (N, 2, 3)."""
    T = np.asarray(cam["T"], float)
    R, t = cm._orthonormal(T[:3, :3]), T[:3, 3]
    fx, fy, cx, cy = cam["K"]
    q = (x - t) @ R
    z = q[:, 2]
    u = np.stack([fx * q[:, 0] / z + cx, fy * q[:, 1] / z + cy], 1)
    Jq = np.zeros((len(x), 2, 3))
    Jq[:, 0, 0], Jq[:, 0, 2] = fx / z, -fx * q[:, 0] / z ** 2
    Jq[:, 1, 1], Jq[:, 1, 2] = fy / z, -fy * q[:, 1] / z ** 2
    return u, z, Jq @ R.T


def sag_init(grip, L=L_ROPE, n=N):
    """A parabola hanging between the grippers with the rope's arc length (straight if they are a rope apart)."""
    a, b = grip[:3], grip[3:]
    s = np.linspace(0, 1, n)
    base = a[None] + s[:, None] * (b - a)[None]
    chord = np.linalg.norm(b - a)
    lo, hi = 0.0, L
    for _ in range(40):                                       # depth h such that the parabola's length is L
        h = 0.5 * (lo + hi)
        p = base.copy()
        p[:, 2] -= 4 * h * s * (1 - s)
        if np.linalg.norm(np.diff(p, axis=0), axis=1).sum() < L:
            lo = h
        else:
            hi = h
    return p if chord < L else base


def chain(x, grip, seg, iters=60):
    """Inextensible chain pinned at the grippers, as vision.chain but vectorised (even / odd segments alternately)."""
    x = np.array(x, float)
    x[0], x[-1] = grip[:3], grip[3:]
    n = len(x)
    idx = [np.arange(p, n - 1, 2) for p in (0, 1)]
    wa = [np.where(i == 0, 0.0, np.where(i + 1 == n - 1, 1.0, 0.5))[:, None] for i in idx]
    wb = [np.where(i + 1 == n - 1, 0.0, np.where(i == 0, 1.0, 0.5))[:, None] for i in idx]
    for _ in range(iters):
        for i, a, b in zip(idx, wa, wb):
            d = x[i + 1] - x[i]
            ln = np.linalg.norm(d, axis=1, keepdims=True)
            corr = (ln - seg) * d / np.maximum(ln, 1e-9)
            x[i] += a * corr
            x[i + 1] -= b * corr
    return x


CLAMP_EMA = 0.05       # running average weight of a newly seen end tangent
SD_FLOOR_M = 0.001     # node sd floor
PHYS_STEP_M = 0.004    # largest node move per iteration with the physical energy
DEPTH_OCC_PX = 12      # depth this close to an occluder (tool, wrist) is smeared by it on the D435i: not used
DEPTH_MAX_M = 1.52     # the recorder clips depth here
DEPTH_GATE_M = 0.03    # a range residual further than this from the median residual is not the rope
DEPTH_MAX_OFFSET_M = 0.25


def constrained_step(H, b, x, seg):
    """Gauss-Newton step dx (3N,) minimising 1/2 dx'H dx - b'dx with the end nodes fixed and every segment's length
    held to first order (|x_k+1 - x_k|^2 = seg^2, linearised): the KKT system. The constraint forces are the rope's
    tension, which carries gravity: on the true hanging shape the gravity term pushes no node off."""
    n = len(x)
    free = np.arange(3, 3 * (n - 1))                       # nodes 1 .. n-2
    d = np.diff(x, axis=0)                                 # (n-1, 3)
    C = np.zeros((n - 1, 3 * n))
    for k in range(n - 1):
        C[k, 3 * k:3 * k + 3] = -2 * d[k]
        C[k, 3 * k + 3:3 * k + 6] = 2 * d[k]
    c = (d ** 2).sum(1) - seg ** 2
    Cf = C[:, free]
    m = len(free)
    K = np.zeros((m + n - 1, m + n - 1))
    K[:m, :m] = H[np.ix_(free, free)]
    K[:m, m:] = Cf.T
    K[m:, :m] = Cf
    rhs = np.r_[b[free], -c]
    sol = np.linalg.lstsq(K, rhs, rcond=None)[0] if not np.all(np.isfinite(K)) else np.linalg.solve(K, rhs)
    dx = np.zeros(3 * n)
    dx[free] = sol[:m]
    return dx


def node_sd(H, x, seg):
    """Per-node position sd (m, sqrt of the 3 x 3 block trace) from the last Gauss-Newton system with the length
    constraints and the pinned ends: the top-left block of the inverse KKT matrix. Image terms count 1 px as one
    standard deviation; the shape model is taken as exact, so this is a floor where nodes are not seen."""
    n = len(x)
    free = np.arange(3, 3 * (n - 1))
    d = np.diff(x, axis=0)
    C = np.zeros((n - 1, 3 * n))
    for k in range(n - 1):
        C[k, 3 * k:3 * k + 3], C[k, 3 * k + 3:3 * k + 6] = -2 * d[k], 2 * d[k]
    m = len(free)
    K = np.zeros((m + n - 1, m + n - 1))
    K[:m, :m] = H[np.ix_(free, free)]
    K[:m, m:] = C[:, free].T
    K[m:, :m] = C[:, free]
    try:
        cov = np.linalg.inv(K)[:m, :m]
    except np.linalg.LinAlgError:
        return None
    v = np.clip(np.diag(cov), 0, None).reshape(-1, 3).sum(1)
    return np.r_[0.0, np.sqrt(v), 0.0]


def end_tangent_fit(x, end, k=6):
    """Direction in which the curve leaves its end node: derivative at the end of a quadratic through the first k
    nodes (the end node held exactly), so one node's error near the gripper does not set the direction."""
    idx = np.arange(k) if end == 0 else np.arange(len(x) - 1, len(x) - 1 - k, -1)
    s_ = np.arange(k, dtype=float)
    p = x[idx] - x[idx[0]]
    A = np.c_[s_[1:], s_[1:] ** 2]
    coef = np.linalg.lstsq(A, p[1:], rcond=None)[0]
    t = coef[0]
    return t / max(np.linalg.norm(t), 1e-9)


CLEAR_M = 0.040 + 0.010 - 0.002     # sphere radius + rope radius, less 2 mm of contact softness


def push_out(x, spheres):
    """Move nodes that lie inside a known sphere (centre distance < CLEAR_M) radially onto its surface."""
    x = x.copy()
    for c in np.asarray(spheres, float).reshape(-1, 3):
        d = x - c
        r = np.linalg.norm(d, axis=1)
        inside = r < CLEAR_M
        if inside.any():
            x[inside] = c + d[inside] * (CLEAR_M / np.maximum(r[inside], 1e-9))[:, None]
    return x


class RopeRGB:
    """update(rgbs, grip) -> (39, 3) centerline estimate from the two RGB images and the gripper positions."""

    def __init__(self, cams, L=L_ROPE, iters=4, first_iters=25, sigma=(6.0, 2.5), w_dt=0.5, tau_px=12.0,
                 lam_prior=4e4, lam_smooth=1e6, max_samples=None, seed=0, motion_mode="grip", stride=3, lam_depth=2.8e4,
                 first_prior=1.0, l_bend=0.08, w_phys=1e2, phys_all=False, miss_prior=0.05, occ_blind=True,
                 phys_outside=True, shape_sd=0.002, learn_clamp=True, clamp=None, clamp_k=3.0,
                 online_lb=False, lb_every=4):
        self.cams, self.L = cams, float(L)
        self.iters, self.first_iters, self.sigma, self.w_dt, self.tau = iters, first_iters, sigma, w_dt, tau_px
        self.lam_prior, self.lam_smooth, self.max_samples = lam_prior, lam_smooth, max_samples
        self.rng = np.random.default_rng(seed)
        self.x = self.grip = self.t = self.v_hist = None
        self.node_sd = None
        self.motion_mode, self.stride, self.lam_depth = motion_mode, int(stride), float(lam_depth)
        self.first_prior = float(first_prior)
        self.l_bend, self.w_phys = (None if l_bend is None else float(l_bend)), float(w_phys)
        self.phys_all, self.miss_prior, self.occ_blind = bool(phys_all), float(miss_prior), bool(occ_blind)
        self.phys_outside = bool(phys_outside)      # False: the physics fills occluded nodes only (not out of view)
        self.shape_sd = float(shape_sd)             # error of the shape model where a view is missing (rope_shape)
        self.learn_clamp = bool(learn_clamp)        # end tangents (clamps) learned from seen ends, tool frame
        self.clamp_k = float(clamp_k)               # clamp stiffness relative to one bending joint of the rope
        self.end_t, self.end_n = [None, None], [0, 0]
        self.lb_est, self.lb_every, self.n_frames = None, int(lb_every), 0
        if online_lb and self.l_bend is not None:   # l_b read off this rope's seen stretches (l_bend: the start)
            from .rope_shape import OnlineBendingLength
            self.lb_est = OnlineBendingLength(l_start=self.l_bend, seed=seed)
        self.rot = [np.eye(3), np.eye(3)]           # world from tool, per end (identity: tangents in the world frame)
        if clamp is not None:                       # calibrated beforehand (a fixture property, like l_bend)
            self.end_t = [np.asarray(c, float) / np.linalg.norm(c) for c in clamp]
            self.end_n = [int(1 / CLAMP_EMA)] * 2
        self.axes = {r: cm._orthonormal(np.asarray(c["T"], float)[:3, :3])[:, 2] for r, c in cams.items()}
        D = np.zeros((N - 2, N))
        for i in range(N - 2):
            D[i, i:i + 3] = (1.0, -2.0, 1.0)
        self.S = np.kron(D.T @ D, np.eye(3))
        self.D = D
        self.info = {}

    def _observe(self, rgbs, depths=None):
        obs = {}
        for role in [r for r in cm.ROLES if r in rgbs]:
            r, dt, g, occ = masks(rgbs[role])
            ys, xs = np.nonzero(r[::self.stride, ::self.stride])     # regular grid (random subsets add jitter)
            ys, xs = ys * self.stride, xs * self.stride
            if self.max_samples and len(xs) > self.max_samples:
                k = self.rng.choice(len(xs), self.max_samples, replace=False)
                ys, xs = ys[k], xs[k]
            obs[role] = dict(dt=dt, grad=g, occ=occ, pix=np.stack([xs, ys], 1).astype(np.float32), n_rope=int(r.sum()))
            if depths is not None and depths.get(role) is not None:
                D = np.asarray(depths[role], float)
                core = cv2.erode(r.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
                near_occ = cv2.dilate(occ.astype(np.uint8), np.ones((2 * DEPTH_OCC_PX + 1,) * 2, np.uint8)) > 0
                ok = core & ~near_occ & (D > 0.1) & (D < DEPTH_MAX_M)
                obs[role]["dmap"] = np.where(ok, D, np.nan)
        return obs

    def _depth_terms(self, o, cam_axis, ui, vi, z, use):
        """Range residuals (N,) from the depth image around each node's pixel (median of rope-core depths in a 7 x 7
        window), NaN where there is no usable depth or the residual is off the median residual by DEPTH_GATE_M."""
        dy, dx = np.mgrid[-3:4, -3:4]
        vv = np.clip(vi[:, None] + dy.ravel()[None], 0, cm.H - 1)
        uu = np.clip(ui[:, None] + dx.ravel()[None], 0, cm.W - 1)
        win = o["dmap"][vv, uu]
        cnt = np.isfinite(win).sum(1)
        med = np.full(len(z), np.nan)
        good = use & (cnt >= 5)
        if good.any():
            med[good] = np.nanmedian(win[good], axis=1)
        r = med - z
        if np.isfinite(r).sum() >= 3:      # gate around the common offset: a whole curve off in range is corrected,
            c = np.nanmedian(r)            # a stray pixel (another strand, the tool) is not
            r[~((np.abs(r - c) < DEPTH_GATE_M) & (abs(c) < DEPTH_MAX_OFFSET_M))] = np.nan
        else:
            r[:] = np.nan
        return r

    def _iterate(self, x, prior, obs, grip, iters, shift=None, spheres=None, lam_prior=None):
        seg = self.L / (N - 1)
        lp = self.lam_prior if lam_prior is None else lam_prior
        for it in range(iters):
            sig = self.sigma[0] + (self.sigma[1] - self.sigma[0]) * it / max(iters - 1, 1)
            H = lp * np.eye(3 * N) + self.lam_smooth * self.S
            b = lp * (prior - x).reshape(-1) - self.lam_smooth * (self.S @ x.reshape(-1))
            seen, observed, out_of = {}, {}, {}
            for role, o in obs.items():
                u, z, J = project_jac(x + (shift[role] if shift else 0.0), self.cams[role])   # where it was at that view's time
                ui, vi = np.round(u[:, 0]).astype(int), np.round(u[:, 1]).astype(int)
                inside = (z > 0) & (ui >= 0) & (ui < cm.W) & (vi >= 0) & (vi < cm.H)
                uc, vc = np.clip(ui, 0, cm.W - 1), np.clip(vi, 0, cm.H - 1)
                free = inside & ~o["occ"][vc, uc]
                res, wt = np.zeros((N, 2)), np.zeros(N)
                if len(o["pix"]):                                   # E-step: rope pixels -> projected nodes
                    pu = (o["pix"][:, None] - u[None].astype(np.float32))
                    d2 = (pu * pu).sum(-1)
                    d2[:, ~inside] = np.inf
                    near = d2.min(1) < (3 * sig) ** 2
                    w = np.exp(-(d2[near] - d2[near].min(1, keepdims=True)) / (2 * sig ** 2))
                    w /= w.sum(1, keepdims=True)
                    mass = w.sum(0)
                    has = mass > 1e-6
                    res[has] = (w.T @ o["pix"][near])[has] / mass[has, None] - u[has]
                    wt = np.minimum(mass, 8.0) / 8.0
                    if self.occ_blind:              # a node behind an occluder or outside the image is not seen here:
                        wt[~free] = 0.0             # rope pixels beside the occluder belong to its seen neighbours
                dtv = o["dt"][vc, uc]
                pull = free & (dtv > 1.5) & (wt < 0.25)            # node on background: towards the nearest rope pixel
                g = o["grad"][vc, uc]
                gn = np.linalg.norm(g, axis=1).clip(1e-6)
                res[pull] = -(np.minimum(dtv[pull], self.tau) / gn[pull])[:, None] * g[pull]
                wt[pull] = self.w_dt
                seen[role] = int(((wt > 0) & ~pull).sum())
                observed[role] = ((wt > 0) & ~pull) | (~inside if not self.phys_outside else False)
                out_of[role] = ~inside
                if "dmap" in o and self.lam_depth > 0:          # range along the camera axis from the depth image
                    rd = self._depth_terms(o, self.axes[role], ui, vi, z, (wt > 0) & ~pull & inside)
                    a = self.axes[role]
                    for n_ in np.flatnonzero(np.isfinite(rd)):
                        H[3 * n_:3 * n_ + 3, 3 * n_:3 * n_ + 3] += self.lam_depth * wt[n_] * np.outer(a, a)
                        b[3 * n_:3 * n_ + 3] += self.lam_depth * wt[n_] * a * rd[n_]
                ar = np.arange(N)
                H.reshape(N, 3, N, 3)[ar, :, ar, :] += wt[:, None, None] * np.einsum("nij,nik->njk", J, J)
                b += (wt[:, None] * np.einsum("nij,ni->nj", J, res)).reshape(-1)
            if self.l_bend is None:
                x = x + np.linalg.solve(H, b).reshape(N, 3)
            else:                                            # gravity + bending, segment lengths as constraints
                # only where a view is missing: the physics fills the unseen direction between seen neighbours
                # (a local elastica); seen stretches follow the images and are not pulled by a model that is only
                # approximately right (end clamps, motion)
                miss = ~np.logical_and.reduce([observed.get(r, np.zeros(N, bool)) for r in cm.ROLES])
                miss[[0, -1]] = False
                if self.phys_all:
                    miss[1:-1] = True
                if miss.any():
                    trip = np.flatnonzero(miss[:-2] | miss[1:-1] | miss[2:])
                    Dm = self.D[trip]
                    Sm = np.kron(Dm.T @ Dm, np.eye(3))
                    # there the physics replaces the generic smoothing and most of the pull to the last frame
                    mi = np.repeat(miss, 3)
                    H = H - self.lam_smooth * Sm - (1 - self.miss_prior) * lp * np.diag(mi.astype(float))
                    b = b + self.lam_smooth * (Sm @ x.reshape(-1)) - (1 - self.miss_prior) * lp * mi * (prior - x).reshape(-1)
                    a = self.l_bend ** 3 / seg ** 4
                    gz = np.zeros(3 * N)
                    gz[2::3] = miss.astype(float)
                    H = H + self.w_phys * a * Sm
                    b = b - self.w_phys * (a * (Sm @ x.reshape(-1)) + gz)
                if self.learn_clamp:                      # the clamps: end tangents learned from seen shapes
                    a = self.l_bend ** 3 / seg ** 4
                    for end, nxt, tang in ((0, 1, self.end_t[0]), (N - 1, N - 2, self.end_t[1])):
                        if tang is not None:
                            sl = slice(3 * nxt, 3 * nxt + 3)
                            H[sl, sl] += self.clamp_k * self.w_phys * a * np.eye(3)
                            tw = self.rot[0 if end == 0 else 1] @ tang
                            b[sl] -= self.clamp_k * self.w_phys * a * (x[nxt] - (x[end] + seg * tw))
                dx = constrained_step(H, b, x, seg).reshape(N, 3)
                mx = np.linalg.norm(dx, axis=1).max()        # trust region: gravity is a constant force and the
                x = x + dx * min(1.0, PHYS_STEP_M / max(mx, 1e-12))   # linearised constraints allow long steps
            x = chain(x, grip, seg)
            if spheres is not None and len(spheres):
                for _ in range(2):
                    x = chain(push_out(x, spheres), grip, seg)
        sd = node_sd(H, x, seg)
        if sd is not None:                      # + what the image terms do not know: a floor (the rope's end and
            unseen = ~np.logical_and.reduce([observed.get(r, np.zeros(N, bool)) & ~out_of.get(r, np.zeros(N, bool))
                                             for r in cm.ROLES])      # centre definitions) and, where a view is
            unseen[[0, -1]] = False                                   # missing, the shape model's own error
            sd = np.sqrt(sd ** 2 + SD_FLOOR_M ** 2 + unseen * self.shape_sd ** 2)
        self.node_sd = sd
        self.last_observed = np.logical_and.reduce([observed.get(r, np.zeros(N, bool)) for r in cm.ROLES])
        return x, seen

    def node_velocity(self, grip_vel):
        """Node velocities (N, 3): the grippers' velocities blended along the rope (motion_mode 'grip'), or that plus
        the smoothed change of the estimate between fits ('hist')."""
        s = np.linspace(0, 1, N)[:, None]
        v = (1 - s) * grip_vel[:3] + s * grip_vel[3:] if grip_vel is not None else np.zeros((N, 3))
        if self.motion_mode == "hist" and self.v_hist is not None:
            v = self.v_hist
        return v

    def update(self, rgbs, grip, t=None, view_t=None, grip_vel=None, spheres=None, depths=None, grip_rot=None):
        """t: the time the estimate refers to; view_t: role -> capture time of that image (the two cameras are not
        synchronised: each view is fitted to the rope moved by its node velocity over view_t - t); grip_vel (6,);
        spheres: known obstacle centres (m, 3), which the rope cannot enter; depths: role -> depth image (m, aligned to
        the colour image, scale-corrected: camera_model depth_scale), optional per camera; rgbs may hold one view;
        grip_rot: (2, 3, 3) world-from-tool rotations of the two TCPs (the clamps' frames), optional."""
        grip = np.asarray(grip, float)
        if grip_rot is not None:
            self.rot = [np.asarray(R, float).reshape(3, 3) for R in grip_rot]
        obs = self._observe(rgbs, depths)
        shift = None
        if t is not None and view_t is not None and self.x is not None:
            v = self.node_velocity(None if grip_vel is None else np.asarray(grip_vel, float))
            shift = {r: v * (view_t[r] - t) for r in obs}
        if self.x is None:
            x0, how = sag_init(grip, self.L), "sag"
            try:
                from .rgb_usb import estimate
                p0, inf0 = estimate(rgbs, self.cams)
                if abs(inf0["length"] - self.L) < 0.15 * self.L and inf0["ray_gap_mean"] < 0.01:
                    x0, how = chain(p0, grip, self.L / (N - 1)), "r0"
            except (RuntimeError, KeyError, ValueError, IndexError):
                pass
            # the first frame's prior is only a starting curve (R0 or a sag): weak, the images and depths decide
            x, seen = self._iterate(x0, x0, obs, grip, self.first_iters, spheres=spheres, lam_prior=self.first_prior * self.lam_prior)
            self.info = dict(init=how)
        else:
            s = np.linspace(0, 1, N)[:, None]                       # carry the estimate with the grippers
            prior = self.x + (1 - s) * (grip[:3] - self.grip[:3]) + s * (grip[3:] - self.grip[3:])
            prior = chain(prior, grip, self.L / (N - 1))
            x, seen = self._iterate(prior, prior, obs, grip, self.iters, shift, spheres)
            self.info = {}
        self.n_frames += 1
        if self.lb_est is not None and self.n_frames % self.lb_every == 0 and getattr(self, "last_observed", None) is not None:
            if self.lb_est.observe(x, self.last_observed):
                self.l_bend = self.lb_est.l
        if self.learn_clamp and getattr(self, "last_observed", None) is not None:   # clamp tangents from seen ends
            for e, near in enumerate((np.arange(2, 6), np.arange(N - 6, N - 2))):
                if self.last_observed[near].all():               # the image decides the curve there
                    tn = self.rot[e].T @ end_tangent_fit(x, 0 if e == 0 else N - 1)
                    self.end_n[e] += 1
                    a_ = max(1.0 / self.end_n[e], CLAMP_EMA)
                    v = tn if self.end_t[e] is None else (1 - a_) * self.end_t[e] + a_ * tn
                    self.end_t[e] = v / np.linalg.norm(v)
        if t is not None and self.t is not None and t > self.t:
            vh = (x - self.x) / (t - self.t)
            self.v_hist = vh if self.v_hist is None else 0.5 * (self.v_hist + vh)
        self.x, self.grip, self.t = x, grip.copy(), t
        self.info.update(observed_front=seen.get("front", 0), observed_top=seen.get("top", 0),
                         **{f"rope_px_{r}": o["n_rope"] for r, o in obs.items()})
        return x.copy()
