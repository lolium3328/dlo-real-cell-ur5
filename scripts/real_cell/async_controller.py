"""Asynchronous version of dlovw.cell.calibrated.deployment.Controller for the real cell (same networks, tracker,
gate, CBF and limits).

Three threads instead of one sequential step, so the 125 Hz low level never waits for planning or image tracking:

* tracker thread - owns the RecordedRopeTracker / EvidenceMemory / QualityGate. Full update on every new camera
  frame (~25 ms), cheap gripper-following update otherwise; publishes the newest quality + pause flag.
* planner thread - every decision_dt: world-model / forecaster update and CellPlanner.choose on the newest snapshot
  (~55-80 ms on CPU); publishes the action.
* step() (caller's 125 Hz loop) - input and clock checks, ball readings, status, projection of the newest action,
  CBF on the newest centreline, execute_action. While a decision is being computed the previous action keeps running.

Simulated with scripts/real_cell/latency_sim.py (async:L:Q): the demo scenes keep 5/5 at the measured p50 latency.

time_scale k (runner --time-scale): the whole cell runs k times slower than the trained policy. Every time the
controller sees (step t, robot and camera timestamps, decision_dt, warm-up, staleness and clock checks) is VIRTUAL
time = wall time / k, so ball, arm and decision period are all k times slower while velocities, ages and the
per-decision displacement look exactly as in training. Real-cell 2026-11-19 (Windows PC): tracker ~120 ms, planner
~370 ms, frame -> action ~0.46 s, i.e. ~8 of the 10 control steps of a decision at k = 6 (latency_sim async:8: 5/5).
Timing-trace fields and the returned quality_age stay in WALL seconds.
"""
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dlovw.cell.calibrated.deployment import Controller
from dlovw.cell.calibrated.constraints import execute_action, in_domain
from dlovw.cell.calibrated.observations import encode, MEAS_SLICE
from dlovw.cell.cbf import cbf_filter
from dlovw.real_rope.tracking import MAX_FRAME_AGE_S

QUALITY_MAX_AGE = .1                     # virtual s; older tracker output (tracker thread stalled) counts as stale
WARMUP_S = .64


class AsyncController(Controller):
    def __init__(self, *a, time_scale=1., **kw):
        super().__init__(*a, **kw)
        if not time_scale >= 1.:
            raise ValueError(f'time_scale must be >= 1, got {time_scale}')
        self.k = float(time_scale)           # wall seconds per virtual second
        self.lock = threading.Lock()
        self.stopped = False
        self.fault = None                    # exception text from a worker thread -> observation_fault
        self.published = None                # (t, quality, pause, tracker_id, published_s)
        self.grip = None                     # (t, left, velocity) from step(), for the tracker
        self.status = 'warming_up'
        self.decisions = 0
        self.last_adopted = 0

    # ---------------------------------------------------------------- threads
    def start(self, cams, T0):
        self.cams, self.T0 = cams, T0
        self.threads = [threading.Thread(target=self._guard, args=(f,), daemon=True)
                        for f in (self._tracker_loop, self._planner_loop)]
        for th in self.threads:
            th.start()

    def stop(self):
        self.stopped = True
        for th in getattr(self, 'threads', []):
            th.join(1.)

    def _guard(self, f):
        try:
            f()
        except Exception as e:               # noqa: BLE001 - any worker failure stops the controller
            traceback.print_exc()
            self.fault = repr(e)
            if self.timing_trace is not None:
                self.timing_trace.record('worker_error', worker=f.__name__, error=self.fault)

    def _now(self):
        """Virtual time (controller logic)."""
        return (time.perf_counter()-self.T0)/self.k

    def _wall(self):
        """Wall time since T0 (timing trace)."""
        return time.perf_counter()-self.T0

    def _tracker_loop(self):
        seen, last_t = {}, -np.inf
        trace = self.timing_trace
        last_wait = None
        while not self.stopped:
            with self.lock:
                grip = self.grip
            if grip is None:
                time.sleep(.002)
                continue
            frames = self.cams.latest()
            frames_read_s = self._wall() if trace is not None else None
            imgs = {r: f for r, (f, ft) in frames.items() if ft > seen.get(r, -np.inf)}
            for r, im in imgs.items():
                if r not in ('front', 'top') or im.shape != (480, 640, 3) or im.dtype != np.uint8:
                    raise ValueError(f'camera {r}: expected uint8 RGB 640x480, got {im.shape} {im.dtype}')
            t = max(self._now(), last_t+1e-4)
            input_frames = {r: (frames[r][1]-self.T0)/self.k for r in imgs}       # virtual
            vt = {r: min(v, t) for r, v in input_frames.items()}
            k = self.k
            if self.tracker.x is None and not all(
                    r in vt and t-vt[r] <= MAX_FRAME_AGE_S for r in ('front', 'top')):
                if trace is not None:
                    waiting = tuple(r for r in ('front', 'top') if r not in vt or t-vt[r] > MAX_FRAME_AGE_S)
                    if waiting != last_wait:
                        trace.record('tracker_wait', waiting_for=waiting,
                                     input_frame_s={r: v*k for r, v in input_frames.items()})
                        last_wait = waiting
                time.sleep(.002)
                continue
            grip_s, left, vel = grip
            tracker_id, gate_detail = None, None
            if trace is not None:
                tracker_id = trace.record('tracker_start', tracker_sample_s=t*k, grip_sample_s=grip_s*k,
                                          frames_read_s=frames_read_s,
                                          input_frame_s={r: v*k for r, v in input_frames.items()},
                                          view_s={r: v*k for r, v in vt.items()})
                gate_detail = {}
                cpu_start = time.thread_time()
            quality = self.tracker.observe(imgs, np.r_[left, self.fixed], t, view_t=vt, grip_vel=np.r_[vel, 0, 0, 0])
            observed_s = self._wall() if trace is not None else None
            quality = self.evidence.update(t, quality, vt)
            pause = self.gate.update(t, quality, diagnostics=gate_detail)
            computed_s = self._wall() if trace is not None else None
            seen.update({r: frames[r][1] for r in imgs})
            last_t = t
            with self.lock:
                published_s = self._wall() if trace is not None else None
                self.published = (t, quality, pause, tracker_id, published_s)
            if trace is not None:
                if gate_detail.get('support_frame_s'):                          # virtual -> wall
                    gate_detail['support_frame_s'] = [None if s is None else s*k for s in gate_detail['support_frame_s']]
                trace.record('tracker_publish', published_s, tracker_id=tracker_id, tracker_sample_s=t*k,
                             observed_s=observed_s, computed_s=computed_s,
                             thread_cpu_s=time.thread_time()-cpu_start, gate_pause=pause,
                             **gate_detail)
            if not imgs:
                time.sleep(.008)             # no new frame: follow the gripper at ~125 Hz

    def _planner_loop(self):
        last = -np.inf
        trace = self.timing_trace
        while not self.stopped:
            t = self._now()
            with self.lock:
                pub, grip, status, ball = self.published, self.grip, self.status, self.cached_ball
            if status != 'tracking' or pub is None or grip is None or ball is None or t-last < self.config.decision_dt-1e-6:
                time.sleep(.002)
                continue
            qt, quality, _, tracker_id, tracker_published_s = pub
            left = grip[1]
            right_s = self.last_right_timestamp
            ball_obs = (ball[0], ball[1], t-right_s)
            plan_id, k = None, self.k
            if trace is not None:
                plan_id = trace.record('planner_start', t*k, tracker_id=tracker_id, tracker_sample_s=qt*k,
                                       tracker_published_s=tracker_published_s, grip_sample_s=grip[0]*k,
                                       right_sample_s=right_s*k, control_status=status)
                cpu_start = time.thread_time()
            with torch.no_grad():
                prev = np.zeros(6) if self.decision_left is None else \
                    np.r_[(left-self.decision_left)/(self.config.cap*self.config.decision_steps), 0, 0, 0]
                obs = encode(quality['centerline'], left, self.fixed, ball_obs, quality['quality_features'], self.origin)
                ap = torch.tensor(prev, dtype=torch.float32)[None]
                self.state = self.wm.obs_step(self.state, ap, torch.tensor(obs)[None])[0]
                self.belief = self.fc.step(self.belief, torch.tensor(obs[MEAS_SLICE])[None], None)
                model_ready_s = self._wall() if trace is not None else None
                _, action, _ = self.planner.choose(self.state, [self.belief], self.goal, np.r_[left, self.fixed],
                                                   np.r_[self.target, self.fixed], self.origin, ap)
            self.decision_left, last = left.copy(), t
            computed_s = self._wall()
            self.decision_latencies.append((computed_s-t*k)*1000)      # wall ms
            with self.lock:
                self.action = np.asarray(action, float).copy()
                self.decisions += 1
                if trace is not None:
                    published_s = self._wall()
                    self.action_trace = dict(plan_id=plan_id, plan_published_s=published_s)
            if trace is not None:
                trace.record('planner_publish', published_s, plan_id=plan_id, tracker_id=tracker_id,
                             decision_number=self.decisions, model_ready_s=model_ready_s, computed_s=computed_s,
                             thread_cpu_s=time.thread_time()-cpu_start)

    # ---------------------------------------------------------------- 125 Hz low level
    def step(self, t, left, right_tcp, right_rotation, rgbs=None, view_times=None, *, right_frame='left_base',
             right_timestamp=None, left_timestamp=None):
        started = time.perf_counter()
        zero = np.zeros(3)
        try:
            t, left = float(t), np.asarray(left, float)
        except (ValueError, TypeError, OverflowError):
            return dict(command_xyz=zero, status='invalid_input')
        if not np.isfinite(t) or left.shape != (3,) or not np.isfinite(left).all():
            return dict(command_xyz=zero, status='invalid_input')
        if self.last_t is not None and t <= self.last_t:
            return dict(command_xyz=zero, status='invalid_clock')
        if not in_domain(left, self.config.slices):
            return dict(command_xyz=zero, status='outside_workspace')
        try:
            rt = t if right_timestamp is None else float(right_timestamp)
            lt = t if left_timestamp is None else float(left_timestamp)
        except (ValueError, TypeError, OverflowError):
            return dict(command_xyz=zero, status='invalid_robot_clock')
        if not np.isfinite(rt) or not np.isfinite(lt) or rt > t+1e-6 or lt > t+1e-6:
            return dict(command_xyz=zero, status='invalid_robot_clock')
        if t-rt > .1 or t-lt > .1:
            return dict(command_xyz=zero, status='robot_stale')
        dt = self.config.dt if self.last_t is None else t-self.last_t
        if dt > .1:
            self.previous_u[:] = 0.
            self.last_t = t
            return dict(command_xyz=zero, status='control_gap')
        velocity = np.zeros(3) if self.previous_left is None else (left-self.previous_left)/dt
        self.last_t, self.previous_left = t, left.copy()
        if self.fault is not None:
            return dict(command_xyz=zero, status='observation_fault', fault=self.fault)
        try:
            if self.last_right_timestamp is not None and rt < self.last_right_timestamp:
                raise ValueError('Robot clock went backwards')
            if self.last_right_timestamp is None or rt > self.last_right_timestamp:
                cached = self.ball.push(rt, right_tcp, right_rotation, frame=right_frame)
                with self.lock:
                    self.cached_ball, self.last_right_timestamp = cached, rt
        except (ValueError, RuntimeError, TypeError) as exc:
            return dict(command_xyz=zero, status='observation_fault', fault=repr(exc))
        ball = None if self.cached_ball is None else (self.cached_ball[0], self.cached_ball[1], t-rt)
        with self.lock:
            self.grip = (t, left.copy(), velocity)
            pub, action, decisions = self.published, self.action.copy(), self.decisions
            action_trace = self.action_trace
            inputs_read_s = self._wall() if self.timing_trace is not None else None
        if pub is None:                      # tracker not initialised yet
            return dict(command_xyz=zero, status='warming_up', elapsed_ms=(time.perf_counter()-started)*1000,
                        timing=dict(inputs_read_s=inputs_read_s, tracker_id=None, ball_available=ball is not None)
                        if self.timing_trace is not None else {})
        qt, quality, pause, tracker_id, tracker_published_s = pub
        self.first_t = t if self.first_t is None else self.first_t
        status = 'tracking'
        if t-self.first_t < WARMUP_S:
            status = 'warming_up'
        elif pause or ball is None or t-qt > QUALITY_MAX_AGE:
            status = 'observation_stale'
        timing = {}
        if self.timing_trace is not None:
            timing = dict(tracker_id=tracker_id, tracker_sample_s=qt*self.k, tracker_published_s=tracker_published_s,
                          inputs_read_s=inputs_read_s, quality_max_age_s=QUALITY_MAX_AGE*self.k,
                          gate_pause=pause, ball_available=ball is not None, decision_number=decisions,
                          **action_trace)
            if status == 'observation_stale':
                timing['stale_reasons'] = [reason for reason, triggered in
                                           (('result_age', t-qt > QUALITY_MAX_AGE),
                                            ('quality_gate', pause), ('ball_missing', ball is None)) if triggered]
        with self.lock:
            self.status = status
        raw = self.lim.project(action, np.r_[left, self.fixed])[:3]*self.config.cap if status == 'tracking' else zero
        if status == 'tracking' and ball is not None:
            p, v, age = ball
            raw, _ = cbf_filter(raw, quality['centerline'], left, self.fixed, [(p+v*age, v)],
                                self.config.protected+self.config.cbf_margin, self.config.dt,
                                self.lim.ws_low[:3], self.lim.ws_high[:3], self.lim.stretch_max, self.config.cap, norm='l2')
        try:
            command = execute_action(raw, left, self.fixed, self.previous_u, self.config)
        except RuntimeError:
            return dict(command_xyz=zero, status='constraint_infeasible', stop_required=True, timing=timing)
        self.previous_u = command
        elapsed = (time.perf_counter()-started)*1000
        self.latencies.append(elapsed)
        adopted = decisions != self.last_adopted
        self.last_adopted = decisions
        return dict(command_xyz=command, status=status, elapsed_ms=elapsed, decision=adopted,
                    decision_ms=self.decision_latencies[-1] if adopted and self.decision_latencies else None,
                    sphere_position=None if ball is None else ball[0], quality=quality, quality_age=(t-qt)*self.k,
                    timing=timing)
