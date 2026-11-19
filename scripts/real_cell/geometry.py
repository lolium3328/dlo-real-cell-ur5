"""Conservative capsule clearance checks (arm-arm, ball-arm, arm-post, arm-table, arm self).

UR does not provide collision checking between robots or against the environment; these are approximations from
link origins with inflated radii (not UR CAD collision geometry). Cameras, stands, cables and the backdrop are not
modelled.
"""
import numpy as np

# capsule i joins point i and i+1 of Kin.points: base->shoulder, shoulder->upper_arm(offset), upper_arm, forearm,
# wrist_1->wrist_2, wrist_2->wrist_3, wrist_3->TCP (flange, plate, probe)
RADII = np.array([.075, .065, .065, .055, .05, .05, .04])
NAMES = ['base', 'shoulder', 'upper_arm', 'forearm', 'wrist1', 'wrist2', 'tool']
BALL_R = .04                             # default only; callers pass cfg.sphere_radius (--sphere-radius)


def seg_seg(p1, q1, p2, q2):
    """Distance between segments p1q1 and p2q2 (broadcasting over leading dims; degenerate segments allowed)."""
    d1, d2, r = q1-p1, q2-p2, p1-p2
    a, e = (d1*d1).sum(-1), (d2*d2).sum(-1)
    f, c, b = (d2*r).sum(-1), (d1*r).sum(-1), (d1*d2).sum(-1)
    den, eps = a*e-b*b, 1e-12
    s = np.where(den > eps, np.clip((b*f-c*e)/np.maximum(den, eps), 0, 1), 0.)
    t = (b*s+f)/np.maximum(e, eps)
    s = np.where(t < 0, np.clip(-c/np.maximum(a, eps), 0, 1), np.where(t > 1, np.clip((b-c)/np.maximum(a, eps), 0, 1), s))
    t = np.clip(t, 0, 1)
    return np.linalg.norm((p1+d1*s[..., None])-(p2+d2*t[..., None]), axis=-1)


def _ends(P):
    return P[..., :-1, :], P[..., 1:, :]


def arm_arm_gap(PL, PR):
    """PL (nL,8,3), PR (nR,8,3) -> gap (nR,nL) [m, surface to surface], worst pair (left name, right name)."""
    AL, BL = _ends(PL)
    AR, BR = _ends(PR)
    best, worst = np.full((len(PR), len(PL)), np.inf), (np.inf, None)
    for i in range(7):
        for j in range(7):
            d = seg_seg(AR[:, None, j], BR[:, None, j], AL[None, :, i], BL[None, :, i])-RADII[i]-RADII[j]
            best = np.minimum(best, d)
            if d.min() < worst[0]:
                worst = (float(d.min()), (NAMES[i], NAMES[j]))
    return best, worst[1]


def pair_gap(PL, PR):
    """One left and one right configuration, (8,3) each -> min capsule surface gap [m] (all 49 pairs at once)."""
    AL, BL = _ends(PL)
    AR, BR = _ends(PR)
    return float((seg_seg(AL[:, None], BL[:, None], AR[None], BR[None])-RADII[:, None]-RADII[None]).min())


def ball_arm_gap(centres, P, ball_r=BALL_R):
    """centres (n,3) ball path, P (m,8,3) arm configurations -> gap (n,m)."""
    A, B = _ends(P)
    c = centres[:, None, None, :]
    d = seg_seg(c, c, A[None], B[None])-RADII-ball_r
    return d.min(-1)


def post_gap(P, post_xy, post_top, post_r):
    """P (n,8,3) -> (n,) gap to the vertical fixture post (base capsule excluded)."""
    A, B = _ends(P)
    bottom = np.array([post_xy[0], post_xy[1], 0.])
    top = np.array([post_xy[0], post_xy[1], post_top])
    d = seg_seg(A[:, 1:], B[:, 1:], bottom, top)-RADII[1:]-post_r
    return d.min(-1)


def table_gap(P):
    """(n,) height of the lowest capsule surface above the table (z = 0), shoulder and base excluded."""
    A, B = _ends(P)
    z = np.minimum(A[..., 2], B[..., 2])-RADII
    return z[:, 2:].min(-1)


def self_gap(P):
    """(n,) min gap between capsules of one arm that are >= 3 links apart."""
    A, B = _ends(P)
    best = np.full(len(P), np.inf)
    for i in range(7):
        for j in range(i+3, 7):
            best = np.minimum(best, seg_seg(A[:, i], B[:, i], A[:, j], B[:, j])-RADII[i]-RADII[j])
    return best
