"""Real-input-only controller. Returns bounded commands; never sends robot commands."""
from dataclasses import fields
import time
import numpy as np
import torch

from ...real_rope.tracking import RecordedRopeTracker
from ...real_rope.vision import cameras
from ...real_rope.visibility import curve_margin
from ..cost import CostCfg
from ..planner import CellPlanner
from .checkpoint import load
from .config import Config, ROOT
from .constraints import CalibratedLimits, execute_action, in_domain
from .observations import BallReadings, QualityGate, EvidenceMemory, encode, MEAS_SLICE


class Controller:
    def __init__(self, checkpoint, goal_curve, target_left, threads=1, commit=None):
        """commit: optional (abs, rel, risk) planner commitment (v39 demo: 3, 0.1, 0.2); None = r10 switch rule."""
        torch.set_num_threads(threads)
        self.nets, self.metadata = load(checkpoint)
        torch.manual_seed(int(self.metadata.get('seed', 0)))
        known = {f.name for f in fields(Config)}
        raw = {k:v for k,v in self.metadata['config']['config'].items() if k in known}
        self.config = Config(**raw)
        if self.config.provenance()['inputs'] != self.metadata['config']['inputs']:
            raise ValueError('Deployment physical inputs differ from checkpoint')
        self.wm, self.fc, self.actor, self.critic = self.nets
        self.lim = CalibratedLimits.from_config(self.config)
        self.planner = CellPlanner(self.wm, self.fc, self.actor, self.critic, self.lim,
                                   cost=CostCfg(protected_m=self.config.protected, segment_distance=True),
                                   horizon=self.config.plan_horizon,
                                   commit=dict(zip(('abs', 'rel', 'risk'), commit)) if commit else None)
        self.tracker = RecordedRopeTracker(cameras(ROOT/self.config.material['cameras']), self.config.material)
        self.ball = BallReadings(config=self.config)
        self.gate = QualityGate(self.config.material['length_m'])
        self.evidence = EvidenceMemory()
        self.fixed = np.array(self.config.material['fixed_end_m'])
        self.target, self.target_curve = np.asarray(target_left, float), np.asarray(goal_curve, float)
        if (not in_domain(self.target, self.config.slices) or self.target_curve.shape != (39, 3)
                or not np.isfinite(self.target_curve).all()):
            raise ValueError('Invalid target')
        if (np.linalg.norm(self.target_curve[0]-self.target) > .016
                or np.linalg.norm(self.target_curve[-1]-self.fixed) > .016):
            raise ValueError('Goal curve endpoints differ from declared grippers')
        margins = curve_margin(self.target_curve, self.tracker.cams, self.config.material['radius_m'],
                               self.config.margin_px, 0.)
        if min(margins.values()) < 0:
            raise ValueError('Goal curve outside the declared camera domain')
        from ...imagination.episodes import encode_goal
        self.origin = (self.fixed+self.target)*.5
        self.goal = torch.tensor(encode_goal(dict(centerline=self.target_curve, left_pose=self.target, right_pose=self.fixed), self.origin))[None]
        self.state, self.belief = self.wm.initial(1, 'cpu'), self.fc.initial(1)
        self.last_t, self.previous_left, self.decision_left = None, None, None
        self.previous_u, self.action = np.zeros(3), np.zeros(6)
        self.last_decision_t, self.first_t = -np.inf, None
        self.latencies = []
        self.decision_latencies = []
        self.last_right_timestamp, self.cached_ball = None, None

    def step(self, t, left, right_tcp, right_rotation, rgbs, view_times=None, *, right_frame='left_base',
             right_timestamp=None, left_timestamp=None):
        """One timestamped control sample; xyz poses in metres, rotation a base-from-tool matrix."""
        started = time.monotonic()
        try:
            t, left = float(t), np.asarray(left, float)
        except (ValueError, TypeError, OverflowError):
            return dict(command_xyz=np.zeros(3), status='invalid_input')
        if not np.isfinite(t) or left.shape != (3,) or not np.isfinite(left).all():
            return dict(command_xyz=np.zeros(3), status='invalid_input')
        if self.last_t is not None and t <= self.last_t:
            return dict(command_xyz=np.zeros(3), status='invalid_clock')
        if not in_domain(left, self.config.slices):
            return dict(command_xyz=np.zeros(3), status='outside_workspace')
        try:
            rt = t if right_timestamp is None else float(right_timestamp)
            lt = t if left_timestamp is None else float(left_timestamp)
        except (ValueError, TypeError, OverflowError):
            return dict(command_xyz=np.zeros(3), status='invalid_robot_clock')
        if not np.isfinite(rt) or not np.isfinite(lt) or rt > t+1e-6 or lt > t+1e-6:
            return dict(command_xyz=np.zeros(3), status='invalid_robot_clock')
        if t-rt > .1 or t-lt > .1:
            return dict(command_xyz=np.zeros(3), status='robot_stale')
        dt = self.config.dt if self.last_t is None else t-self.last_t
        if dt > .1:
            self.previous_u[:] = 0.
            self.last_t = t
            return dict(command_xyz=np.zeros(3), status='control_gap')
        velocity = np.zeros(3) if self.previous_left is None else (left-self.previous_left)/dt
        self.last_t, self.previous_left = t, left.copy()
        try:
            if not isinstance(rgbs, dict):
                raise ValueError('Images must be keyed by front/top')
            for role, rgb in rgbs.items():
                if role not in ('front', 'top') or np.asarray(rgb).shape != (480, 640, 3) or np.asarray(rgb).dtype != np.uint8:
                    raise ValueError('Expected uint8 RGB 640x480')
                capture = float((view_times or {}).get(role, t))
                if not np.isfinite(capture) or capture > t+1e-6:
                    raise ValueError('Invalid camera timestamp')
            if self.last_right_timestamp is not None and rt < self.last_right_timestamp:
                raise ValueError('Robot clock went backwards')
            if self.last_right_timestamp is None or rt > self.last_right_timestamp:
                self.cached_ball = self.ball.push(rt, right_tcp, right_rotation, frame=right_frame)
                self.last_right_timestamp = rt
            ball = None if self.cached_ball is None else (self.cached_ball[0], self.cached_ball[1], t-rt)
            quality = self.tracker.observe(rgbs, np.r_[left, self.fixed], t, view_t=view_times,
                                           grip_vel=np.r_[velocity, 0, 0, 0])
            quality = self.evidence.update(t, quality, {r:(view_times or {}).get(r, t) for r in rgbs})
        except (ValueError, RuntimeError, TypeError):
            self.action[:] = 0.
            return dict(command_xyz=np.zeros(3), status='observation_fault')
        self.first_t = t if self.first_t is None else self.first_t
        pause = self.gate.update(t, quality)
        status = 'tracking'
        if t-self.first_t < .64:
            status = 'warming_up'
        elif pause or ball is None:
            status = 'observation_stale'
        decision = False
        with torch.no_grad():
            if status == 'tracking' and t-self.last_decision_t >= self.config.decision_dt-1e-6:
                decision = True
                prev = np.zeros(6) if self.decision_left is None else np.r_[(left-self.decision_left)/(self.config.cap*self.config.decision_steps), 0, 0, 0]
                obs = encode(quality['centerline'], left, self.fixed, ball, quality['quality_features'], self.origin)
                ap = torch.tensor(prev, dtype=torch.float32)[None]
                self.state = self.wm.obs_step(self.state, ap, torch.tensor(obs)[None])[0]
                self.belief = self.fc.step(self.belief, torch.tensor(obs[MEAS_SLICE])[None], None)
                _, self.action, _ = self.planner.choose(self.state, [self.belief], self.goal,
                                                       np.r_[left, self.fixed], np.r_[self.target, self.fixed], self.origin, ap)
                self.decision_left, self.last_decision_t = left.copy(), t
        raw = self.lim.project(self.action, np.r_[left,self.fixed])[:3]*self.config.cap if status == 'tracking' else np.zeros(3)
        if status == 'tracking' and ball is not None:
            from ..cbf import cbf_filter
            p, v, age = ball
            raw, _ = cbf_filter(raw, quality['centerline'], left, self.fixed, [(p+v*age, v)],
                                self.config.protected+self.config.cbf_margin, self.config.dt,
                                self.lim.ws_low[:3], self.lim.ws_high[:3], self.lim.stretch_max, self.config.cap, norm='l2')
        try:
            command = execute_action(raw, left, self.fixed, self.previous_u, self.config)
        except RuntimeError:
            return dict(command_xyz=np.zeros(3), status='constraint_infeasible', stop_required=True)
        self.previous_u = command
        elapsed = (time.monotonic()-started)*1000
        self.latencies.append(elapsed)
        if decision:
            self.decision_latencies.append(elapsed)
        return dict(command_xyz=command, status=status, elapsed_ms=elapsed,
                    decision=decision, sphere_position=None if ball is None else ball[0], quality=quality)
