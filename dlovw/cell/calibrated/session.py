"""Closed-loop refined rope, causal ball readings and delayed RGB observations."""
import numpy as np

from ...motion import track
from ...real_rope.visibility import frustum
from ...real_rope.vision import cameras
from ..cbf import cbf_filter
from .config import Config, ROOT, right_rotation, tcp_from_sphere
from .constraints import execute_action, in_domain, CalibratedLimits, clip_norm
from .physics import RopeState, surface_distance
from .observations import BallReadings, encode, goal_vector, node_bound
from .render import SceneView, RGBObservations
from .headless import HeadlessTracking


class Session:
    def __init__(self, rec, config=Config(), seed=0, deploy=False, visual=True, headless=False):
        self.rec, self.config, self.seed = rec, config, seed
        self.rope = RopeState(rec['start_left'], config, seed, deploy, qpos=rec.get('start_qpos'))
        self.goal, self.origin = goal_vector(rec, config)
        n = round(config.max_time_s/config.dt)+config.horizon*config.decision_steps+8
        self.track = track(rec['motion'], n, config.dt)
        self.n, self.held, self.terminal = 0, 0, 'running'
        self.ball_sensor = BallReadings(seed+1, deploy, config)
        self.previous = np.zeros(3)
        self.rng = np.random.default_rng(seed+2)
        self.gain = self.rng.uniform(.92, 1.05) if deploy else 1.
        self.delay = int(self.rng.integers(0, 3)) if deploy else 0
        self.commands = [np.zeros(3)]*self.delay
        self.limits = CalibratedLimits.from_config(config)
        self.view = SceneView(config) if visual else None
        self.rgb = RGBObservations(config, self.view, seed+3, deploy) if visual else None
        self.headless = None
        if headless:
            if config.headless_model == 'occlusion':
                from .headless_occlusion import OcclusionTracking
                self.headless = OcclusionTracking(config, seed+3, deploy)
            else:
                self.headless = HeadlessTracking(config, seed+3, deploy)
        self.occlusion = headless and config.headless_model == 'occlusion'
        self.quality, self.ball = None, None
        self.min_surface = np.inf
        self.out_of_domain, self.out_of_fov, self.pause_steps = 0, 0, 0
        self.out_of_image = 0
        self.max_speed, self.max_acceleration, self.max_ik_error = 0., 0., 0.
        cams = cameras(ROOT/config.material['cameras'])
        self.planes = [frustum(v, config.margin_px) for v in cams.values()]
        self.raw_planes = [frustum(v, 0.) for v in cams.values()]
        if self.rgb:
            # Observe a stationary setup before starting the task clock; never
            # initialize the tracker with the simulation's centreline.
            self.rgb.next_frame = -.64
            for t in np.arange(-.64, 0., .04):
                self.rgb.update(float(t), self.rope, self.track[0], np.zeros(3))
        self.sample()

    @property
    def t(self):
        return self.n*self.config.dt

    def sphere(self):
        return self.track[self.n]

    def sample(self, stress=None):
        c = self.config
        self.ball = self.ball_sensor.push(self.t, tcp_from_sphere(self.sphere(), right_rotation(c), c), right_rotation(c))
        if self.rgb:
            self.quality = self.rgb.update(self.t, self.rope, self.sphere(), self.previous/c.dt, stress=stress)
            self.max_ik_error = max(self.max_ik_error, self.view.last_ik[0])
        curve = self.rope.curve()
        if self.headless:
            self.quality = (self.headless.update(curve, self.sphere(), self.t) if self.occlusion
                            else self.headless.update(curve))
        self.min_surface = min(self.min_surface, surface_distance(curve, self.sphere(), c.sphere_radius,
                                                                 c.material['radius_m']))
        inside = in_domain(self.rope.left(), c.slices)
        fov = all((curve@A.T-b).min() >= c.material['radius_m'] for A, b in self.planes)
        raw_fov = all((curve@A.T-b).min() >= c.material['radius_m'] for A, b in self.raw_planes)
        self.out_of_domain += int(not inside)
        self.out_of_fov += int(not fov)
        self.out_of_image += int(not raw_fov)

    def observation(self):
        if self.quality is None:
            raise RuntimeError('Policy observations require RGB, not simulation truth')
        return encode(self.quality['centerline'], self.rope.left(), self.rope.fixed,
                      self.ball, self.quality['quality_features'], self.origin)

    def error(self):
        e = np.linalg.norm(self.rope.curve()-np.asarray(self.rec['goal_curve']), axis=1)
        return dict(rmse=float(np.sqrt((e*e).mean())), p95=float(np.percentile(e, 95)),
                    end=float(np.linalg.norm(self.rope.left()-self.rec['target_left'])))

    def step(self, u, cbf=True, stress=None):
        c = self.config
        u = np.asarray(u, float)[:3]
        if not np.isfinite(u).all():
            self.terminal = 'invalid_action'
            return
        paused = bool((self.rgb or (self.occlusion and c.headless_pause)) and self.quality['pause_requested'])
        if paused:
            u = np.zeros(3)
            self.pause_steps += 1
        if cbf and not paused and self.quality is not None and self.ball is not None:
            p, v, age = self.ball
            margin = c.cbf_margin
            if c.gate_mode == 'bound':
                centre = np.asarray(self.quality['centerline'])
                surface = np.linalg.norm(centre-(p+v*age), axis=1)-c.sphere_radius-c.material['radius_m']
                margin = node_bound(self.quality, surface)
            u, _ = cbf_filter(u, self.quality['centerline'], self.rope.left(), self.rope.fixed,
                              [(p+v*age, v)], c.protected+margin, c.dt,
                              self.limits.ws_low[:3], self.limits.ws_high[:3], self.limits.stretch_max, c.cap, norm='l2')
        self.commands.append(u)
        delayed = self.commands.pop(0)
        try:
            executed = execute_action(delayed, self.rope.left(), self.rope.fixed, self.previous, c, self.gain)
        except RuntimeError:
            self.terminal = 'actuation_infeasible'
            return
        self.max_speed = max(self.max_speed, np.linalg.norm(executed)/c.dt)
        self.max_acceleration = max(self.max_acceleration, np.linalg.norm(executed-self.previous)/c.dt**2)
        self.previous = executed
        self.rope.advance(executed)
        self.n += 1
        self.sample(stress)
        error = self.error()
        near = surface_distance(self.rope.curve(), self.sphere(), c.sphere_radius, c.material['radius_m'])
        ok = error['rmse'] <= .012 and error['p95'] <= .025 and error['end'] <= .016
        self.held = self.held+1 if ok and self.t >= self.rec['event_end_s']+.2 else 0
        if near < c.clearance:
            self.terminal = 'clearance_violation'
        elif self.out_of_domain:
            self.terminal = 'workspace_violation'
        elif not np.isfinite(self.rope.data.qpos).all():
            self.terminal = 'invalid_state'
        elif self.held*c.dt >= c.hold_s:
            self.terminal = 'success'
        elif self.t >= c.max_time_s:
            self.terminal = 'timeout'

    def result(self):
        return dict(id=self.rec['id'], task=self.rec['task'], terminal=self.terminal, steps=self.n,
                    min_surface_m=self.min_surface, out_of_domain=self.out_of_domain, out_of_fov=self.out_of_fov,
                    margin_violation_samples=self.out_of_fov, out_of_image_samples=self.out_of_image,
                    frame_bounds_semantics='out_of_fov/margin_violation_samples use 20px inset; out_of_image_samples use raw image bounds',
                    pause_steps=self.pause_steps, max_speed=self.max_speed, max_acceleration=self.max_acceleration,
                    max_ik_error_m=self.max_ik_error, **self.error())

    def close(self):
        if self.view:
            self.view.close()
