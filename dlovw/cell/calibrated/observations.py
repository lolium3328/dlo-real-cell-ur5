"""Causal kinematics-to-ball readings and explicit quality-aware tensor layout."""
import numpy as np
import torch

from .config import Config, sphere_from_tcp
from ...imagination.episodes import ROPE, POS, VEL, encode_obs, encode_goal

OBS = 130+39*4
MEAS_SLICE = slice(ROPE, ROPE+7)
QUALITY_SLICE = slice(ROPE+7, OBS)


class EvidenceMemory:
    """Carry latest image support to a faster control clock without refreshing capture times."""
    def __init__(self):
        self.times = np.full(2, -np.inf)
        self.flags = np.zeros((39, 2), bool)

    def update(self, t, quality, view_times):
        for j, role in enumerate(('front', 'top')):
            if role in view_times and view_times[role] > self.times[j]:
                self.times[j] = view_times[role]
                self.flags[:, j] = quality['image_support'][:, j]
        support = self.flags & ((t-self.times) <= .1)[None]
        quality['image_support'] = support
        quality['quality_features'][:, :2] = support.astype(np.float32)
        quality['image_support_capture_s'] = self.times.copy()
        quality['source'] = np.where(support.all(1), 'recent_stereo_image_support',
                                     np.where(support.any(1), 'recent_single_view_support', 'temporal_prior'))
        quality['source'][[0, -1]] = 'kinematic_endpoint'
        return quality


# Per-node tracking error bound (p99, m) by evidence state, measured on paired RGB tracking (2026-11-15,
# docs/notes/quality_gate_analysis_20261115.md): error saturates within 0.1-0.3 s of losing support, so the bound
# depends on the view state and the anchored-gap radius, not on how long the node has been unseen.
BOUND_STEREO_M, BOUND_STEREO_NEAR_M, BOUND_NEAR_BALL_M, BOUND_SINGLE_M = .0095, .014, .025, .027
BOUND_GAP_M = ((.01, .022), (.02, .030), (.03, .035))
BOUND_GAP_MAX_M = .043
BOUND_END_M = .002


def node_bound(quality, ball_surface_m=None):
    """(39,) error bound for the reported centreline, from the latest support and QualityGate's gap radius."""
    support = np.asarray(quality['image_support'], bool)
    radius = np.asarray(quality.get('anchored_gap_radius_m', np.full(39, np.inf)))
    gap = np.full(39, BOUND_GAP_MAX_M)
    for limit, value in reversed(BOUND_GAP_M):
        gap = np.where(radius < limit, value, gap)
    stereo = np.full(39, BOUND_STEREO_M)
    if ball_surface_m is not None:
        stereo = np.where(np.asarray(ball_surface_m) < BOUND_NEAR_BALL_M, BOUND_STEREO_NEAR_M, stereo)
    u = np.where(support.all(1), stereo, np.where(support.any(1), BOUND_SINGLE_M, gap))
    u[[0, -1]] = BOUND_END_M
    return u


class QualityGate:
    """Shared image-evidence age gate, with explicit kinematically constrained clamp zones.
    mode 'age' (v1-v39): the three age rules below. mode 'bound': pause only on global loss; local uncertainty is
    handled by the CBF through node_bound()."""
    def __init__(self, length=Config().material['length_m'], mode='age'):
        self.last_support = np.full(39, np.nan)
        self.length = length
        self.mode = mode

    def update(self, t, quality):
        any_support = np.asarray(quality['image_support'], bool).any(1)
        if 'image_support_capture_s' in quality:
            times = np.asarray(quality['image_support_capture_s'])
            supported_t = np.where(quality['image_support'], times[None], -np.inf).max(1)
            self.last_support[any_support] = np.fmax(self.last_support[any_support], supported_t[any_support])
        else:
            self.last_support[any_support] = t
        age = np.where(np.isfinite(self.last_support[2:-2]), t-self.last_support[2:-2], 2.)
        self.age = age
        curve = np.asarray(quality['centerline'])
        # A short occluded span can stay bounded by visible neighbours and the
        # kinematic endpoints even with old stereo support. Its inextensible
        # ellipse radius is a geometric prior, never a new image measurement.
        recent = np.isfinite(self.last_support) & (t-self.last_support <= .1)
        anchors = np.flatnonzero(recent | (np.arange(39) == 0) | (np.arange(39) == 38))
        bounded = np.zeros(39, bool)
        radius = np.full(39, np.inf)
        for a, b in zip(anchors[:-1], anchors[1:]):
            if b <= a+1:
                continue
            length = (b-a)*self.length/38
            chord = np.linalg.norm(curve[b]-curve[a])
            bound = np.sqrt(max(0., length*length-chord*chord))*.5+.005
            radius[a+1:b] = bound
            if bound <= .02:
                bounded[a+1:b] = True
        unsupported = ~bounded[2:-2]
        quality['anchored_gap_radius_m'] = radius
        global_loss = bool((age > .3).mean() > .8)
        if self.mode == 'bound':
            return global_loss
        return bool(global_loss or ((age > .3) & unsupported).mean() > .2 or ((age > 1.) & unsupported).any())


class BallReadings:
    def __init__(self, seed=0, deploy=False, config=Config()):
        self.config, self.rng, self.deploy = config, np.random.default_rng(seed), deploy
        self.bias = self.rng.normal(size=3)
        self.bias *= self.rng.uniform(0, .004)/max(np.linalg.norm(self.bias), 1e-9) if deploy else 0.
        self.latency = int(self.rng.integers(1, 5)) if deploy else 0
        self.log, self.last, self.filtered = [], None, np.zeros(3)

    def push(self, t, tcp, rotation, *, frame='left_base', valid=True):
        if self.log and t <= self.log[-1][0]:
            raise ValueError('Non-monotonic robot timestamp')
        p = sphere_from_tcp(tcp, rotation, self.config, frame=frame)
        if self.deploy:
            p = p+self.bias+self.rng.normal(0, .0005, 3)
        self.log.append((float(t), p, bool(valid)))
        idx = len(self.log)-1-self.latency
        if idx < 0:
            return None
        time, centre, good = self.log[idx]
        if not good:
            return None
        if self.last is not None:
            dt = time-self.last[0]
            if dt <= 0:
                raise ValueError('Reused robot sample')
            v = (centre-self.last[1])/dt
            self.filtered = .25*v+.75*self.filtered
        self.last = (time, centre.copy())
        return centre.copy(), self.filtered.copy(), max(0., t-time)


def encode(curve, left, fixed, ball, quality, origin):
    base = encode_obs(curve, left, fixed, None if ball is None else ball[:2], origin)
    q = np.asarray(quality, np.float32).copy()
    if q.shape != (39, 4):
        raise ValueError(f'quality must be (39,4), got {q.shape}')
    q[:, 2] = np.minimum(q[:, 2], 2.)/2.
    q[:, 3] = np.minimum(q[:, 3], .3)*100.
    out = np.r_[base, q.ravel()].astype(np.float32)
    if not np.isfinite(out).all():
        raise ValueError('Non-finite observation')
    return out


def goal_vector(rec, config=Config()):
    fixed, target = np.asarray(config.material['fixed_end_m']), np.asarray(rec['target_left'])
    origin = (fixed+target)*.5
    goal = encode_goal(dict(centerline=rec['goal_curve'], left_pose=target, right_pose=fixed), origin)
    return goal, origin


def filter_states(fc, measurements):
    B, T, J = measurements.shape[:3]
    o, states = fc.initial(B*J, measurements.device), []
    for i in range(T):
        o = fc.step(o, measurements[:, i].reshape(B*J, 7), None)
        states.append(o.reshape(B, J, -1))
    return torch.stack(states, 1)
