"""Single-obstacle task scenes for the cell (docs/notes/cell_policy_20261113.md).

A scene: left TCP start and target (reach: 6-18 cm apart; hold: target = start), and the sphere's scripted motion.
The right arm sweeps the sphere across the rope's way and parks it (patrol: one pass, or there and back in half of
the reach scenes, then it stays; 0.15-0.33 m/s, hold scenes 0.12-0.25 m/s), along a direction square to the rope, through a point of the
obstacle-blind reference rope at arc fraction 0.15-0.55 from the left end (nearer the fixed end the rope can hardly
move). Accepted when:
- the obstacle-blind reference (straight to the target at the step cap, then holding) violates the clearance,
  so every scene needs avoidance;
- the sphere starts >= 30 mm outside the protected radius of the start rope;
- reach: after the reference arrives (+0.5 s), the sphere keeps >= 15 mm outside the protected radius of the
  target shape (it does not sit on the target, legacy gate m2c); hold: the pass ends by 5.0 s and its end point is
  >= 30 mm clear of the target shape; success counts from 0.2 s after the pass ends;
- the track stays in the right arm's box (ARM_BOX) and >= 80 mm from the left TCP and the clamp, and >= 60 mm from
  the left probe (TCP to 30 cm above it) along the reference.
"""
from __future__ import annotations

import numpy as np

from . import layout as L
from .session import CAP, DT, PROTECTED_M, CellSession, min_jerk  # noqa: F401

ARM_BOX = (np.array([-0.24, -0.94, 0.14]), np.array([0.26, -0.30, 0.56]))   # right-arm IK reaches all of it (probe down)
T_MAX_S = 7.2
STATIC = dict(kind="quasi_periodic", center=[0.6, -0.62, 0.4], direction=[1.0, 0.0, 0.0], amplitude_m=0.0,
              half_periods_s=[2.0] * 40, phase0=0.0)


def reference(rec, seconds=T_MAX_S):
    """Obstacle-blind rollout: straight to the target at the step cap (L1), then hold. Returns centerlines (n+1, 39, 3),
    left TCP (n+1, 3), the step it arrived and the session's target shape."""
    s = CellSession(dict(rec, motion=STATIC), max_time_s=seconds + 1)
    cs, lefts, arrive = [s.frames[-1].centerline], [s.left()], None
    tgt = np.asarray(rec["target_left"], float)
    for n in range(int(round(seconds / DT))):
        d = tgt - s.left()
        l1 = np.abs(d).sum()
        u = d if l1 <= CAP else d * CAP / l1
        if arrive is None and l1 < 1e-4:
            arrive = n
        s.step(np.r_[u, 0, 0, 0], cbf=False)
        cs.append(s.frames[-1].centerline)
        lefts.append(s.left())
    return np.array(cs), np.array(lefts), (arrive if arrive is not None else len(cs) - 1), s.target


def _perp_basis(t):
    t = t / np.linalg.norm(t)
    a = np.array([1.0, 0, 0]) if abs(t[0]) < 0.9 else np.array([0, 0, 1.0])
    e1 = np.cross(t, a)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(t, e1)


def _motion(rng, kind, c, n):
    """The right arm sweeps the sphere across the rope's way and parks it: patrol from c + A n to c - A n (one pass),
    or there and back (two passes, reach scenes only, half of them), then it stays. Returns (motion, end of the last
    pass in s)."""
    amp, d0 = rng.uniform(0.12, 0.17), rng.uniform(0.3, 1.8)
    speed = rng.uniform(0.12, 0.25) if kind == "hold" else rng.uniform(0.15, 0.33)
    passes = 1 if kind == "hold" or rng.uniform() < 0.5 else 2
    dwells = [float(d0)] + ([float(rng.uniform(0.3, 1.0))] if passes == 2 else []) + [30.0] * 4
    leg = 2 * amp / speed + speed / 1.5
    end = d0 + passes * leg + (dwells[1] if passes == 2 else 0.0)
    return dict(kind="patrol", center=c.tolist(), direction=n.tolist(), amplitude_m=float(amp), speed_mps=float(speed),
                accel_mps2=1.5, dwells_s=dwells, start_offset_s=0.0, passes=passes), end


def _seg_dist(p, a, b):
    ab = b - a                                            # row-wise: point k against segment k
    t = np.clip(((p - a) * ab).sum(1) / (ab * ab).sum(1), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def propose(rng, kind, index):
    """One candidate scene (not yet checked)."""
    lo, hi = L.POLICY_LEFT_WORKSPACE
    m = 0.01
    while True:
        start = rng.uniform(lo + m, hi - m)
        if kind == "hold":
            target = start
            break
        target = rng.uniform(lo + m, hi - m)
        if 0.06 <= np.linalg.norm(target - start) <= 0.18:
            break
    return dict(id=f"cell_{kind}_{index:04d}", version=f"cell_{kind}", index=int(index), task=kind,
                start_left=start.tolist(), target_left=target.tolist())


REJECT = {}


def _no(reason):
    REJECT[reason] = REJECT.get(reason, 0) + 1


def make_scene(rng, kind, index, tries=40):
    """An accepted scene, or None after `tries` obstacle draws on one start/target pair (reasons counted in REJECT)."""
    from ..motion import track as motion_track
    rec = propose(rng, kind, index)
    ref, lefts, arrive, target = reference(rec)
    n_steps = int(round(T_MAX_S / DT))
    for _ in range(tries):
        k = int(rng.integers(int(0.6 / DT), int(3.5 / DT)))
        rope = ref[min(k, len(ref) - 1)]
        i = int(rng.integers(int(0.15 * 38), int(0.55 * 38) + 1))
        tang = rope[min(i + 1, 38)] - rope[max(i - 1, 0)]
        e1, e2 = _perp_basis(tang)
        th = rng.uniform(0, 2 * np.pi)
        n = np.cos(th) * e1 + np.sin(th) * e2
        motion, pass_end = _motion(rng, kind, rope[i], n)
        if pass_end > 5.0:
            _no("pass_late")
            continue
        tr = motion_track(motion, n_steps)
        T = min(len(tr), len(ref))
        tr, ref_t, lefts_t = tr[:T], ref[:T], lefts[:T]
        if np.any(tr < ARM_BOX[0]) or np.any(tr > ARM_BOX[1]):
            _no("arm_box")
            continue
        d_ref = np.linalg.norm(ref_t - tr[:, None], axis=2).min(1)
        if d_ref[0] < PROTECTED_M + 0.03 or d_ref.min() >= PROTECTED_M:
            _no("start_close" if d_ref[0] < PROTECTED_M + 0.03 else "no_crossing")
            continue
        if np.linalg.norm(tr[0] - ref_t[0], axis=1).min() < PROTECTED_M + 0.03:
            _no("start_close")
            continue
        if min(np.linalg.norm(tr - lefts_t, axis=1).min(), np.linalg.norm(tr - L.FIXED_END_POS, axis=1).min()) < 0.08:
            _no("near_tcp_or_clamp")
            continue
        if _seg_dist(tr, lefts_t, lefts_t + [0, 0, 0.30]).min() < 0.06:
            _no("near_probe")
            continue
        dt_tgt = np.linalg.norm(target["centerline"][None] - tr[:, None], axis=2).min(1)
        if kind == "hold":
            k_end = int(round(pass_end / DT))
            if dt_tgt[k_end:].min() < PROTECTED_M + 0.03:
                _no("pass_end_on_target")
                continue
            rec["success_after_s"] = float(pass_end + 0.2)
        else:
            if dt_tgt[min(arrive + int(0.5 / DT), len(dt_tgt) - 1):].min() < PROTECTED_M + 0.015:
                _no("on_target")
                continue
            rec["success_after_s"] = 0.0
        rec.update(motion=motion, crossing=dict(arc_index=i, ref_step=k, ref_min_clearance_m=float(d_ref.min() - PROTECTED_M)),
                   ref_arrive_s=float(arrive * DT), accept=True)
        return rec
    return None


def oracle_plans(rec, waits=(-0.4, 0.0, 0.4)):
    """Plans that know the sphere's future: straight to the target; or to one of 27 dodge points of the workspace
    (3 x 3 x 3 grid, those far from where the sphere first crosses first) at the step cap, wait there until the
    sphere has left the reference rope's surroundings (last time within 15 cm, + `waits`), then to the target.
    Each plan is (dodge point or None, time to head for the target in s)."""
    from ..motion import track as motion_track
    ref, _, _, _ = reference(rec)
    tr = motion_track(rec["motion"], len(ref))
    T = min(len(tr), len(ref))
    near = np.flatnonzero(np.linalg.norm(ref[:T] - tr[:T, None], axis=2).min(1) < 0.15)
    t_free = (near[-1] + 1) * DT if len(near) else 0.0
    lo, hi = L.POLICY_LEFT_WORKSPACE
    grid = [np.array(p) for p in np.stack(np.meshgrid(*[np.linspace(lo[k] + 0.005, hi[k] - 0.005, 3) for k in range(3)],
                                                       indexing="ij"), -1).reshape(-1, 3)]
    c0 = np.mean(tr[near], 0) if len(near) else tr[0]
    grid.sort(key=lambda p: -np.linalg.norm(p - c0))
    return [(None, 0.0)] + [(p, t_free + w) for p in grid for w in waits]


def plan_action(plan, s, target):
    """Per-step command (3,) of an oracle plan in session s."""
    dodge, t_back = plan
    goal = target if dodge is None or s.n * DT >= t_back else dodge
    d = goal - s.left()
    l1 = np.abs(d).sum()
    return d if l1 <= CAP else d * CAP / l1


def oracle(rec, plans=None):
    """(solved, index of the first plan that succeeds, plans tried) with exact state and no safety filter."""
    plans = oracle_plans(rec) if plans is None else plans
    tgt = np.asarray(rec["target_left"], float)
    for k, plan in enumerate(plans):
        s = CellSession(rec)
        while s.progress.terminal == "running":
            s.step(np.r_[plan_action(plan, s, tgt), 0, 0, 0], cbf=False)
        if s.progress.terminal == "success":
            return True, k, k + 1
    return False, None, len(plans)
