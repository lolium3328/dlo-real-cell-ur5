"""Occlusion-aware headless tracking (vision validation, 2026-11-14): a cheaper stand-in for rendering + RGB tracking.

Validated against RecordedRopeTracker on paired v37 episodes (docs/notes/vision_validation_v37_20261114.md). The
v37 HeadlessTracking draws stationary Gaussian noise; real errors are dominated by nodes the cameras cannot see
(right arm and ball in front of the rope, mostly in the top view), where the tracker carries a stale estimate.
This model reproduces that mechanism instead of a noise distribution:

  - visibility: MuJoCo rays from each nominal camera to every node (plus a small image-plane margin, as the
    tracker's bright-occluder mask is dilated), with both arms posed by IK (left TCP at the rope's left end,
    right TCP from the ball centre) and the rope's own geoms excluded;
  - frames at image_hz with the image latency under deployment, as RGBObservations;
  - estimate update per frame: stereo-visible nodes snap to truth + smooth AR(1) noise with a lognormal scale
    (and, under deployment, the triangulation bias of the perturbed cameras); single-view nodes are corrected only
    across that view's ray; unseen nodes keep the tracker's prior (moved with the grippers, inextensible chain);
  - tracker drift on nodes without stereo support: a smooth, heavy-tailed OU offset (tau 2 s) revealed in full
    on unseen nodes and along the viewing ray on single-view nodes, amplified within ~10 mm of the ball (where
    the ball's own pixels bias the fit) and there partly present on stereo nodes too;
  - quality features / pause from the tracker's own QualityState + EvidenceMemory + QualityGate.
Not reproduced: tracker lock loss with full visibility (rare, see the note) and segmentation noise.

Integration (main session's decision): Session(..., headless=True) would construct OcclusionTracking instead of
HeadlessTracking and call update(curve, sphere, t); to enforce pauses, Session.step's `paused` must also read the
headless quality (currently only RGB pauses are applied).
"""
import copy

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from ...imagination.rgb_rope import chain
from ...real_rope.tracking import QualityState
from ...real_rope.vision import cameras
from ..arms import Arm
from ..scene import mat
from .config import ROOT, right_rotation, tcp_from_sphere
from .observations import EvidenceMemory, QualityGate

ROLES = ('front', 'top')
STEREO_SD_M = .00055          # stereo-visible node noise (per axis), times a lognormal knot scale
STEREO_SDLOG = .7              # fitted to real clean stereo errors: p50/p90/p99 0.8/2.4/5.0 mm
DRIFT_SD_M = .0045             # tracker drift on nodes without stereo support (per axis), lognormal knot scale
DRIFT_SDLOG = .5               # fitted to real unsupported-node errors: p50/p90/p99 7.8/20.4/30.8 mm
DRIFT_TAU_S = 2.              # slow field, revealed at once: real errors are full-size within 0.1-0.3 s of
                               # losing support and then persist (lag-1 decision correlation 0.96)
BALL_DECAY_M = .01             # ball proximity (node-ball surface distance) scale of the extra error near the ball:
BALL_GAIN_HIDDEN = 1.         #   drift x (1 + exp(-d/10 mm)) on unsupported nodes (real p50 15 mm within 25 mm)
BALL_GAIN_STEREO = 3.         #   min(1, 3 exp(-d/10 mm)) of the drift also on stereo nodes (real p99 18 mm there)
SINGLE_GAIN = .5               # fraction of the cross-ray error a single view removes per frame
RHO_FRAME = .97                # frame-to-frame correlation of the stereo noise (0.93 per 80 ms decision)
KNOTS = 10


class RayVisibility:
    """Per-node, per-camera line of sight in the calibrated scene (nominal cameras)."""
    def __init__(self, config, margin_px=3.):
        self.config, self.margin_px = config, margin_px
        self.model = mujoco.MjModel.from_xml_path(str(config.path/'scene.xml'))
        for g in range(self.model.ngeom):
            if self.model.body(self.model.geom_bodyid[g]).name.startswith('dlo_'):
                self.model.geom_group[g] = 5                      # rope never occludes itself here
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key('home').id)
        self.left, self.right = Arm(self.model, 'left'), Arm(self.model, 'right')
        self.ql, self.qr = self.data.qpos[self.left.qadr].copy(), self.data.qpos[self.right.qadr].copy()
        self.Rl, self.Rr = mat(config.material['left_tcp_quaternion_wxyz']), right_rotation(config)
        self.cams = cameras(ROOT/config.material['cameras'])
        self.group = np.array([1, 0, 1, 0, 0, 0], np.uint8)       # what the renderer draws
        self.rope_r = config.material['radius_m']

    def pose(self, left, sphere):
        self.ql = self.left.ik(left, self.Rl, self.ql, iters=30)[0]
        self.qr = self.right.ik(tcp_from_sphere(sphere, self.Rr, self.config), self.Rr, self.qr, iters=40)[0]
        self.left.apply(self.data, self.ql)
        self.right.apply(self.data, self.qr)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)

    def __call__(self, curve, sphere):
        curve = np.asarray(curve, float)
        self.pose(curve[0], sphere)
        out = np.zeros((len(curve), 2), bool)
        for j, role in enumerate(ROLES):
            T = np.asarray(self.cams[role]['T'], float)
            origin, R = T[:3, 3], T[:3, :3]
            d = curve-origin
            depth = d@R[:, 2]
            off = self.margin_px*depth[:, None]/self.cams[role]['K'][0]
            targets = [curve]+[curve+s*off*R[:, a] for a in (0, 1) for s in (-1, 1)] if self.margin_px else [curve]
            vec = np.concatenate([t-origin for t in targets])
            length = np.linalg.norm(vec, axis=1)
            vec = vec/length[:, None]
            n = len(vec)
            geom, dist = np.zeros(n, np.int32), np.zeros(n)
            mujoco.mj_multiRay(self.model, self.data, origin, vec.ravel(), self.group, 1, -1, geom, dist, None, n, 10.)
            clear = (geom < 0) | (dist > length-self.rope_r-.002)
            out[:, j] = clear.reshape(len(targets), len(curve)).all(0)
        return out


def perturbed_cameras(rng, cams):
    """Same draw as RGBObservations under deployment (1.5 mm, 0.15 deg)."""
    cams = copy.deepcopy(cams)
    for cam in cams.values():
        cam['T'][:3, 3] += rng.normal(0., .0015, 3)
        cam['T'][:3, :3] = Rotation.from_rotvec(rng.normal(0., np.deg2rad(.15), 3)).as_matrix()@cam['T'][:3, :3]
    return cams


def triangulation_bias(points, true_cams, used_cams):
    """Where two-view triangulation with the perturbed cameras puts each true point (midpoint of the two rays)."""
    origins, dirs = [], []
    for role in ROLES:
        Tt, Tu = np.asarray(true_cams[role]['T'], float), np.asarray(used_cams[role]['T'], float)
        local = (points-Tt[:3, 3])@Tt[:3, :3]                     # camera-frame ray of the true pixel
        d = local@Tu[:3, :3].T
        origins.append(np.broadcast_to(Tu[:3, 3], points.shape)); dirs.append(d/np.linalg.norm(d, axis=1, keepdims=True))
    (o1, o2), (d1, d2) = origins, dirs
    w = o1-o2
    a, b, c = (d1*d1).sum(1), (d1*d2).sum(1), (d2*d2).sum(1)
    dd, e = (d1*w).sum(1), (d2*w).sum(1)
    den = np.maximum(a*c-b*b, 1e-12)
    s, t = (b*e-c*dd)/den, (a*e-b*dd)/den
    return .5*(o1+s[:, None]*d1+o2+t[:, None]*d2)-points


class OcclusionTracking:
    """Drop-in for HeadlessTracking with update(curve, sphere, t); returns the RGBObservations quality dict."""
    def __init__(self, config, seed, deploy, visibility=None, stereo_sd=STEREO_SD_M, single_gain=SINGLE_GAIN,
                 drift_sd=DRIFT_SD_M):
        self.config, self.deploy = config, deploy
        self.rng = np.random.default_rng(seed)
        self.vis = visibility or RayVisibility(config)
        self.sd, self.drift_sd, self.single_gain = stereo_sd, drift_sd, single_gain
        self.rays = {r: np.asarray(self.vis.cams[r]['T'], float)[:3, 3] for r in ROLES}
        self.bias_cams = perturbed_cameras(self.rng, self.vis.cams) if deploy else None
        s = np.linspace(0, KNOTS-1, 39)
        i = np.minimum(s.astype(int), KNOTS-2)
        self.basis = np.zeros((39, KNOTS))
        self.basis[np.arange(39), i], self.basis[np.arange(39), i+1] = 1-s+i, s-i
        self.basis /= np.linalg.norm(self.basis, axis=1, keepdims=True)
        self.noise, self.drift = self.rng.standard_normal((2, KNOTS, 3))
        self.log_scale = self.rng.standard_normal((2, KNOTS))        # stereo noise, drift
        self.rho_drift = np.exp(-1/(config.image_hz*DRIFT_TAU_S))
        self.offset = np.zeros((39, 3))
        self.latency = config.image_latency_s if deploy else 0.
        self.seg = config.material['length_m']/38
        self.quality_state, self.evidence = QualityState(), EvidenceMemory()
        self.gate = QualityGate(config.material['length_m'], getattr(config, 'gate_mode', 'age'))
        self.queue, self.next_frame, self.x, self.grip, self.first = [], 0., None, None, True
        self.last_t = None

    def _frame(self, curve, sphere):
        visible = self.vis(curve, sphere)
        target = curve.copy()
        if self.deploy:
            target += triangulation_bias(curve, self.vis.cams, self.bias_cams)
        ar = lambda x, rho: rho*x+np.sqrt(1-rho**2)*self.rng.standard_normal(x.shape)
        self.noise, self.drift = ar(self.noise, RHO_FRAME), ar(self.drift, self.rho_drift)
        self.log_scale = ar(self.log_scale, np.array([[RHO_FRAME], [self.rho_drift]]))
        scale = np.exp(self.log_scale*np.array([[STEREO_SDLOG], [DRIFT_SDLOG]]))
        noise = self.sd*(self.basis@(scale[0, :, None]*self.noise))
        drift = self.drift_sd*(self.basis@(scale[1, :, None]*self.drift))
        ball_d = np.linalg.norm(curve-sphere, axis=1)-self.config.sphere_radius
        return visible, target+noise, drift, ball_d

    def _apply(self, visible, target, drift, ball_d, grip):
        if self.x is None:
            self.x = target.copy()                                     # stationary pre-roll, as Session's RGB setup
        stereo = visible.all(1)
        self.x[stereo] = target[stereo]
        proximity = np.exp(-np.maximum(ball_d, 0.)/BALL_DECAY_M)
        near = 1+BALL_GAIN_HIDDEN*proximity
        self.offset = np.where(visible.any(1)[:, None], 0., drift*near[:, None])    # unseen: full drift
        self.offset[stereo] = np.minimum(1., BALL_GAIN_STEREO*proximity[stereo, None])*drift[stereo]
        if stereo.all():
            return
        for j, role in enumerate(ROLES):
            one = visible[:, j] & ~stereo
            if one.any():
                ray = target[one]-self.rays[role]
                ray /= np.linalg.norm(ray, axis=1, keepdims=True)
                err = target[one]-self.x[one]
                cross = err-(err*ray).sum(1, keepdims=True)*ray
                self.x[one] += self.single_gain*cross
                self.offset[one] = near[one, None]*(drift[one]*ray).sum(1, keepdims=True)*ray   # single view: depth
        self.x = chain(self.x, grip, self.seg)

    def update(self, curve, sphere, t):
        curve = np.asarray(curve, float)
        grip = np.r_[curve[0], curve[-1]]
        if self.x is not None:                                         # carry the estimate with the grippers
            s = np.linspace(0, 1, 39)[:, None]
            self.x = self.x+(1-s)*(grip[:3]-self.grip[:3])+s*(grip[3:]-self.grip[3:])
        self.grip = grip
        if t+1e-9 >= self.next_frame:
            delay = 0. if self.first else self.latency
            self.queue.append((t+delay, t)+self._frame(curve, sphere))
            self.next_frame, self.first = t+1/self.config.image_hz, False
        ready = [r for r in self.queue if r[0] <= t+1e-9]
        support = np.zeros((39, 2), bool)
        captured = {}
        if ready:
            _, cap, visible, target, drift, ball_d = ready[-1]
            self.queue = [r for r in self.queue if r[0] > t+1e-9]
            self._apply(visible, target, drift, ball_d, grip)
            support, captured = visible, {r: cap for r in ROLES}
        if self.last_t is not None and t <= self.last_t:
            t = self.last_t+1e-6
        self.last_t = t
        quality = self.quality_state.update(t, support, None)
        # Error level (2026-11-15): the RGB error this model was fitted to is too large (user: the real data it traces
        # back to was recorded after the calibration board came loose; representative mean ~2.6 mm). Scale the
        # whole error field about the current true centreline by Config.occlusion_error_scale.
        k = getattr(self.config, 'occlusion_error_scale', 1.)
        quality['centerline'] = curve+k*(self.x+self.offset-curve)
        quality = self.evidence.update(t, quality, captured)
        quality['pause_requested'] = self.gate.update(t, quality)
        if not getattr(self.config, 'occlusion_quality', True):     # ablation: training's constant quality features
            from .headless import QUALITY
            quality['quality_features'] = QUALITY.copy()
        quality['headless'] = 'occlusion_v1'
        return quality
