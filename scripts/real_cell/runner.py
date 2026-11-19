"""Scene execution for the real cell: two loops so a slow planning step never stalls the robots.

* Servo loop (caller's thread, 125 Hz): all robot I/O. Reads both arms, publishes a timestamped snapshot, runs the
  guards (faults, follow error, online two-arm / ball-left-arm capsule clearance, timeout) and streams servoL (left:
  the current commanded target) and servoJ (right: the planned joints at the ball clock).
* Control thread: newest snapshot (+ camera frames for the sync Controller) -> Controller.step with the COMMANDED left
  position (as in simulation) -> left target += command, projected onto the workspace domain if it is up to
  --snap-m outside. Status aborts, success and the ball-rope clearance live here. With AsyncController the tracker
  and the planner run in their own threads (async_controller.py).

The ball clock starts when the Controller first reports 'tracking' (as in simulation, where the tracker sees 0.64 s
of the stationary setup and then policy and ball start together at t = 0).

--time-scale k (args.time_scale, 1 = real time): the scene runs k times slower than simulated. The controller, the
ball clock s, the scene limits (max_time_s, hold_s, event end, --max-start-s) use VIRTUAL time = wall time / k; the
control thread runs every k servo periods (one simulated 8 ms step) and the servo loop interpolates the left target
over those k periods. Hardware guards (follow error, --max-stale-s, servo period) and every logged *_s time stay in
wall seconds; s is virtual.
"""
import json
import sys
import threading
import time
import traceback
from collections import deque

import numpy as np

import errors as E
import geometry as G
from planning import right_q_at, sphere_at
from timing_trace import TimingTrace
from video import VideoRecorder
from robots import NoCameras, PERIOD, T_RIGHT, Rp_to_pose, pose_to_Rp, tool_rotation_error
from dlovw.cell.calibrated.constraints import bounds_at
from dlovw.real_rope.tracking import MAX_FRAME_AGE_S

ABORT_STATUS = ('invalid_input', 'invalid_clock', 'outside_workspace', 'observation_fault', 'robot_stale',
                'invalid_robot_clock')


class Shared:
    def __init__(self, left_target):
        self.lock = threading.Lock()
        self.terminal = 'running'
        self.snap = None                         # newest robot sample, poses, servo step ID and publish time
        self.left_target = left_target
        self.target_step_id = None
        self.t0_run = None                       # ball clock start
        self.refused = 0                         # control steps refused by the workspace-domain check
        self.snapped = 0                         # commanded targets projected back onto the domain
        self.detail, self.exc = {}, None         # measured values / exception of the first ending (error report)

    def finish(self, why, exc=None, *, source=None, step_id=None, **detail):
        with self.lock:
            if self.terminal == 'running':
                self.terminal, self.detail, self.exc = why, detail, exc
                self.terminal_at = time.perf_counter()
                self.terminal_source, self.terminal_step_id = source, step_id

    @property
    def running(self):
        return self.terminal == 'running'


def snap_to_domain(p, slices):
    """Nearest-ish point of the workspace domain (z clipped, then x/y clipped at that z) and the distance moved.
    Many scene starts / targets lie exactly on a slab plane, where a 0.1 mm robot error would otherwise read as
    'outside_workspace'."""
    p = np.asarray(p, float)
    z = float(np.clip(p[2], slices[0]['z_m'], slices[-1]['z_m']))
    lo, hi = bounds_at(z, slices)
    q = np.r_[np.clip(p[:2], lo, hi), z]
    return q, float(np.linalg.norm(q-p))


def surface_distance(curve, sphere, sphere_r, rope_r):
    return float(np.linalg.norm(np.asarray(curve)-sphere, axis=1).min()-sphere_r-rope_r)


def ball_centre(pr, cfg):
    """World ball centre from the right arm's actual TCP pose (right-base frame)."""
    R, p = pose_to_Rp(pr)
    Rw, pw = T_RIGHT[:3, :3] @ R, T_RIGHT[:3, :3] @ p+T_RIGHT[:3, 3]
    return pw+Rw @ np.asarray(cfg.sphere_offset, float)


def time_scale(args):
    k = float(getattr(args, 'time_scale', 1.))
    if not k >= 1.:
        raise ValueError(f'--time-scale must be >= 1, got {k}')
    return k


def servo_loop(sh, plan, left, right, kin, cfg, args, T0, log):
    Rl, hist = plan['Rl'], deque()
    nxt = time.perf_counter()
    prev_t = None
    k = time_scale(args)
    seg_id, seg_a, seg_b, seg_n, sp = object(), None, None, 0, None   # left setpoint: seg_a -> seg_b over k periods
    while sh.running:
        nxt += PERIOD
        wait = nxt-time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        else:
            nxt = time.perf_counter()               # late: do not burst to catch up
        t = time.perf_counter()-T0
        servo_step_id = len(log)+1
        row = dict(t=round(t, 4), servo_step_id=servo_step_id, sample_s=t,
                   period_s=None if prev_t is None else t-prev_t)
        prev_t = t
        log.append(row)
        bad = [n for n, arm in (('left', left), ('right', right)) if arm.faulted()]
        if bad:
            sh.finish('robot_fault', source='servo', step_id=servo_step_id, faulted_arms=bad)
            row['faulted_arms'] = bad
            break
        row['left_read_started_s'] = time.perf_counter()-T0
        ql, pl = left.q(), left.tcp()
        row['left_read_s'] = time.perf_counter()-T0
        qr, pr = right.q(), right.tcp()
        row['right_read_s'] = time.perf_counter()-T0
        with sh.lock:
            published_s = time.perf_counter()-T0
            sh.snap = (t, ql, pl, pr, qr, servo_step_id, published_s)
            target, t0, target_step_id = sh.left_target.copy(), sh.t0_run, sh.target_step_id
        s = 0. if t0 is None else (t-t0)/k
        q_cmd = right_q_at(plan, s)
        # left setpoint: each new commanded target is approached linearly over k servo periods (k = 1: the target)
        if target_step_id != seg_id or seg_b is None or not np.array_equal(target, seg_b):
            seg_id, seg_a, seg_b, seg_n = target_step_id, (target if sp is None else sp).copy(), target, 0
        seg_n += 1
        sp = seg_a+(seg_b-seg_a)*min(1., seg_n/k)
        # follow error against the commands of the last --follow-window s (unknown servo tracking delay)
        hist.append((t, sp, q_cmd))
        while hist[0][0] < t-args.follow_window:
            hist.popleft()
        lag_l = min(float(np.linalg.norm(pl[:3]-h[1])) for h in hist)
        lag_r = min(float(np.abs(qr-h[2]).max()) for h in hist)
        row.update(s=round(s, 4), left=pl[:3].tolist(), left_cmd=target.tolist(), qr=qr.tolist(),
                   qr_cmd=q_cmd.tolist(), lag_l=round(lag_l, 5), lag_r=round(lag_r, 5),
                   snapshot_published_s=published_s, target_step_id=target_step_id, ball_start_s=t0)
        if k != 1.:
            row['left_sp'] = sp.tolist()
        if lag_r > args.max_follow_rad:
            sh.finish('right_lag_abort', source='servo', step_id=servo_step_id, 右臂关节偏差_rad=round(lag_r, 4), 阈值_rad=args.max_follow_rad,
                      joint=int(np.argmax(np.abs(qr-q_cmd))), qr=np.round(qr, 4).tolist(), qr_cmd=np.round(q_cmd, 4).tolist())
        if lag_l > args.max_follow_m:
            sh.finish('left_lag_abort', source='servo', step_id=servo_step_id, 左臂位置偏差_mm=round(lag_l*1e3, 1), 阈值_mm=args.max_follow_m*1e3,
                      left=np.round(pl[:3], 4).tolist(), left_cmd=np.round(sp, 4).tolist())
        if not args.no_capsule:                  # actual joints of both arms, every servo step
            PL = kin.points('left', ql)
            gap_arm = G.pair_gap(PL, kin.points('right', qr))
            gap_ball = float(G.ball_arm_gap(ball_centre(pr, cfg)[None], PL[None]).min())
            row.update(gap_arm=round(gap_arm, 4), gap_ball=round(gap_ball, 4))
            if min(gap_arm, gap_ball) < args.abort_gap_m:
                sh.finish('abort_arm_clearance', source='servo', step_id=servo_step_id, 两臂间隙_mm=round(gap_arm*1e3, 1), 球与左臂间隙_mm=round(gap_ball*1e3, 1),
                          阈值_mm=args.abort_gap_m*1e3)
        if s >= cfg.max_time_s:
            sh.finish('timeout', source='servo', step_id=servo_step_id, 时限_s=cfg.max_time_s)
        if t0 is None and t/k > args.max_start_s:
            sh.finish('abort_never_tracking', source='servo', step_id=servo_step_id, 等待_s=round(t, 2))
        if not sh.running:
            break
        row['left_servo_started_s'] = time.perf_counter()-T0
        left.servo_l(Rp_to_pose(Rl, sp), args.lookahead, args.gain)
        row['left_servo_returned_s'] = row['right_servo_started_s'] = time.perf_counter()-T0
        right.servo_j(q_cmd, args.lookahead, args.gain)
        row['right_servo_returned_s'] = time.perf_counter()-T0


def control_loop(sh, ctl, cams, rec, cfg, plan, args, T0, log):
    goal, target_left = np.asarray(rec['goal_curve'], float), np.asarray(rec['target_left'], float)
    seen, stale_since, ok_since = {}, None, None
    step_id, prev_t = None, None
    threaded = hasattr(ctl, 'start')             # AsyncController: own tracker / planner threads read the cameras
    k = time_scale(args)
    nxt = time.perf_counter()
    try:
        if threaded:
            ctl.start(cams, T0)
        while sh.running:
            nxt += PERIOD*k                      # one simulated control step
            wait = nxt-time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            else:
                nxt = time.perf_counter()
            with sh.lock:
                snap, t0, cmd = sh.snap, sh.t0_run, sh.left_target.copy()
            if snap is None:
                continue
            rt, _, pl, pr, _, servo_step_id, published_s = snap
            t = time.perf_counter()-T0
            step_id = len(log)+1
            row = dict(t=round(t, 4), rt=round(rt, 4), s=round(0. if t0 is None else (t-t0)/k, 4),
                       sample_s=t, step_id=step_id, servo_step_id=servo_step_id, robot_sample_s=rt,
                       robot_snapshot_published_s=published_s, robot_sample_age_s=t-rt, ball_start_s=t0,
                       period_s=None if prev_t is None else t-prev_t, status='step_incomplete')
            prev_t = t
            log.append(row)
            frames = {} if threaded else cams.latest()
            imgs = {r: f for r, (f, ft) in frames.items() if ft > seen.get(r, -np.inf)}
            vt = {r: frames[r][1]-T0 for r in imgs}
            seen.update({r: frames[r][1] for r in imgs})
            Rrb, prb = pose_to_Rp(pr)
            # The controller gets the COMMANDED left position, as in simulation (where the arm is exactly where it was
            # commanded): its acceleration / braking limits, CBF and velocity estimate integrate its own commands.
            # The actual arm trails it by the servo delay; the follow-error guard (servo loop) bounds the difference.
            row['step_started_s'] = time.perf_counter()-T0
            res = ctl.step(t/k, cmd, prb, Rrb, imgs, {r: v/k for r, v in vt.items()}, right_frame='right_base',
                           right_timestamp=rt/k, left_timestamp=rt/k)            # virtual clock
            row['step_returned_s'] = time.perf_counter()-T0
            d_in = float(np.linalg.norm(pl[:3]-cmd))
            status, u = res['status'], np.asarray(res['command_xyz'], float)
            row.update(res.get('timing', {}))
            row.update(status=status, u=u.tolist(),
                       actual_vs_cmd_mm=round(d_in*1e3, 3),
                       decision=bool(res.get('decision')), ms=res.get('elapsed_ms'),
                       decision_ms=res.get('decision_ms', res.get('elapsed_ms') if res.get('decision') else None),
                       quality_age=res.get('quality_age'),
                       stop_required=bool(res.get('stop_required')), fault=res.get('fault'),
                       plan_active=status == 'tracking' and row.get('plan_id') is not None)
            if not np.isfinite(u).all() or np.linalg.norm(u) > cfg.cap*1.001:
                row['u'] = [x if np.isfinite(x) else None for x in row['u']]
                sh.finish('abort_bad_command', source='control', step_id=step_id, command=row['u'], 上限_m=cfg.cap)
                break
            if res.get('stop_required') or status in ABORT_STATUS:
                sh.finish(f'abort_{status}', source='control', step_id=step_id, status=status,
                          fault=res.get('fault'), left_cmd=np.round(cmd, 4).tolist(),
                          机器人数据年龄_s=round(t-rt, 3))
                break
            t_now = time.perf_counter()-T0
            if status == 'observation_stale':
                stale_since = t_now if stale_since is None else stale_since
                row.update(stale_since_s=stale_since, stale_duration_s=t_now-stale_since)
                if t_now-stale_since > args.max_stale_s:
                    sh.finish('abort_stale_too_long', source='control', step_id=step_id,
                              失效时长_s=round(t_now-stale_since, 2), 限制_s=args.max_stale_s,
                              跟踪数据年龄_s=res.get('quality_age'))
                    break
            else:
                stale_since = None
            with sh.lock:
                if t0 is None and status == 'tracking':
                    sh.t0_run = t0 = t           # ball clock starts with the first policy step
                new, d_cmd = snap_to_domain(sh.left_target+u, cfg.slices)
                refused = d_cmd > args.snap_m
                if refused:
                    sh.refused += 1              # far outside: hold the commanded target
                else:
                    sh.left_target = new         # slightly outside (target ahead of the lagging arm): slide along
                    sh.target_step_id = step_id
                    row['target_published_s'] = time.perf_counter()-T0
                    sh.snapped += d_cmd > 1e-9
            s = 0. if t0 is None else (t-t0)/k
            row.update(s=round(s, 4), ball_start_s=t0, refused=refused, snap_cmd_mm=round(d_cmd*1e3, 3))
            q = res.get('quality')
            if q is not None and t0 is not None and status in ('tracking', 'observation_stale'):
                curve = np.asarray(q['centerline'], float)
                e = np.linalg.norm(curve-goal, axis=1)
                rmse, p95 = float(np.sqrt((e*e).mean())), float(np.percentile(e, 95))
                end = float(np.linalg.norm(pl[:3]-target_left))
                sphere = res.get('sphere_position')
                sphere = sphere_at(plan, s) if sphere is None else np.asarray(sphere, float)
                near = surface_distance(curve, sphere, cfg.sphere_radius, cfg.material['radius_m'])
                row.update(rmse=rmse, p95=p95, end=end, surface=near)
                ok = rmse <= .012 and p95 <= .025 and end <= .016 and s >= rec['event_end_s']+.2
                ok_since = (s if ok_since is None else ok_since) if ok else None
                if near < args.abort_surface_m:
                    sh.finish('abort_surface_clearance', source='control', step_id=step_id,
                              球绳估计间隙_mm=round(near*1e3, 1), 阈值_mm=args.abort_surface_m*1e3,
                              ball=np.round(sphere, 4).tolist())
                elif ok_since is not None and s-ok_since >= cfg.hold_s:
                    sh.finish('success', source='control', step_id=step_id,
                              rmse_mm=round(rmse*1e3, 1), p95_mm=round(p95*1e3, 1), end_mm=round(end*1e3, 1))
            row['control_completed_s'] = time.perf_counter()-T0
    except Exception as e:                       # noqa: BLE001 - any controller error stops the scene
        traceback.print_exc()
        sh.finish(f'controller_error: {e!r}', exc=e, source='control', step_id=step_id)
    finally:
        if threaded:
            ctl.stop()


def run_scene(rec, cfg, plan, left, right, cams, kin, args, factory, out_dir):
    """Run one scene; every ending other than success writes an error report (errors.py). Returns the terminal."""
    Rl, sid = plan['Rl'], rec['id']
    preparation_started = time.perf_counter()
    camera_wait_started, camera_ready = None, None
    frames = {}
    timing_path = out_dir/f'{sid}_timing.jsonl'

    def early(terminal, **detail):
        trace = TimingTrace(preparation_started)
        trace.record('scene_terminal', terminal=terminal, source='preparation', detail=detail,
                     camera_wait_started_s=None if camera_wait_started is None else camera_wait_started-trace.origin,
                     cached_frame_s={r: stamp-trace.origin for r, (_, stamp) in frames.items()})
        save_trace(trace, timing_path)
        E.report(out_dir, sid, terminal, detail=detail, left=left, right=right)
        return terminal

    rot = tool_rotation_error(left.tcp(), Rl)
    if rot > .03:
        return early('orientation_changed', 姿态变化_rad=round(rot, 3))
    ctl = factory(rec)
    if not isinstance(cams, NoCameras):          # the tracker initialises from the first frames of both views
        t_wait = camera_wait_started = time.perf_counter()
        while True:
            frames = cams.latest()
            now = time.perf_counter()
            fresh = sorted(r for r, (_, stamp) in frames.items() if 0 <= now-stamp <= MAX_FRAME_AGE_S)
            if 'front' in fresh and 'top' in fresh:
                camera_ready = now
                break
            if now-t_wait > 10:
                return early('camera_timeout', 已收到的相机=sorted(frames), 有新鲜画面的相机=fresh)
            time.sleep(.05)
    start, d0 = snap_to_domain(left.tcp()[:3], cfg.slices)
    if d0 > args.snap_m:
        return early('outside_workspace_at_start', 超出工作区_mm=round(d0*1e3, 2), left=np.round(left.tcp()[:3], 4).tolist())
    sh = Shared(start)
    servo_log, control_log = [], []
    old_switch = sys.getswitchinterval()
    sys.setswitchinterval(5e-4)                  # keep the servo loop on time while the planner holds the GIL
    T0 = time.perf_counter()  # same clock as the camera timestamp and asynchronous workers
    trace = TimingTrace(T0)
    trace.record('scene_start', 0., preparation_started_s=preparation_started-T0,
                 camera_wait_started_s=None if camera_wait_started is None else camera_wait_started-T0,
                 camera_ready_s=None if camera_ready is None else camera_ready-T0,
                 cached_frame_s={r: stamp-T0 for r, (_, stamp) in frames.items()},
                 controller=type(ctl).__name__, cameras=type(cams).__name__, period_s=PERIOD,
                 time_scale=time_scale(args), control_period_s=PERIOD*time_scale(args),
                 camera_latency_s=getattr(cams, 'latency', None),
                 max_start_s=args.max_start_s, max_stale_s=args.max_stale_s, left_step_cap_m=cfg.cap,
                 abort_surface_m=args.abort_surface_m, abort_gap_m=args.abort_gap_m)
    ctl.timing_trace = trace
    has_camera_trace = hasattr(cams, 'timing_trace')
    if has_camera_trace:
        cams.timing_trace = trace
    video = None
    if getattr(args, 'video', True) and not isinstance(cams, NoCameras):
        video = VideoRecorder(cams, out_dir, sid, T0).start()
    worker = threading.Thread(target=control_loop, args=(sh, ctl, cams, rec, cfg, plan, args, T0, control_log),
                              daemon=True)
    worker.start()
    try:
        servo_loop(sh, plan, left, right, kin, cfg, args, T0, servo_log)
    except KeyboardInterrupt:
        sh.finish('user_interrupt', source='servo')
    except Exception as e:                       # noqa: BLE001
        traceback.print_exc()
        sh.finish(f'servo_error: {e!r}', exc=e, source='servo', step_id=len(servo_log) or None)
    finally:
        sh.finish('stopped', source='runner')
        stop_started_s = trace.now()
        left.stop()
        left_stop_returned_s = trace.now()
        right.stop()
        right_stop_returned_s = trace.now()
        worker.join(2.)
        if video is not None:
            vs = video.stop()
            print(f"  video: {vs['files']} {vs['frames']} frames" + (f", error {vs['error']}" if vs['error'] else ''))
        sys.setswitchinterval(old_switch)
        worker_alive = worker.is_alive()
        if worker_alive:
            control_log = [row.copy() for row in control_log.copy()]
        if has_camera_trace:
            cams.timing_trace = None
        trace.record('scene_terminal', sh.terminal_at-T0, terminal=sh.terminal, source=sh.terminal_source,
                     step_id=sh.terminal_step_id, ball_start_s=sh.t0_run, stop_started_s=stop_started_s,
                     left_stop_returned_s=left_stop_returned_s, right_stop_returned_s=right_stop_returned_s,
                     join_returned_s=trace.now(), worker_alive=worker_alive)
        save_trace(trace, timing_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir/f'{sid}.jsonl').write_text('\n'.join(json.dumps(r) for r in control_log))
        (out_dir/f'{sid}_servo.jsonl').write_text('\n'.join(json.dumps(r) for r in servo_log))
    summary = report_scene(sh, servo_log, control_log)
    if sh.terminal != 'success':
        E.report(out_dir, sid, sh.terminal, detail=sh.detail, left=left, right=right, servo_log=servo_log,
                 control_log=control_log, exc=sh.exc,
                 extra=dict(run_summary=summary, terminal_s=sh.terminal_at-T0,
                            terminal_source=sh.terminal_source, terminal_step_id=sh.terminal_step_id))
    return sh.terminal


def save_trace(trace, path):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        trace.save(path)
    except (OSError, ValueError) as exc:
        print(f'  timing log error ({path}): {exc}', flush=True)


def report_scene(sh, servo_log, control_log):
    dec = [r['decision_ms'] for r in control_log if r.get('decision_ms')]
    st = np.diff([r['t'] for r in servo_log]) if len(servo_log) > 1 else np.zeros(1)
    msg = (f"  finished: {sh.terminal}, {len(servo_log)} servo / {len(control_log)} control steps, "
           f"decision p95 {np.percentile(dec, 95) if dec else 0:.1f} ms, servo period max {st.max()*1e3:.1f} ms, "
           f"refused {sh.refused}, snapped {sh.snapped}")
    gaps = [r['gap_arm'] for r in servo_log if 'gap_arm' in r]
    if gaps:
        msg += f", min arm gap {min(gaps)*1e3:.0f} mm, min ball-left gap {min(r['gap_ball'] for r in servo_log)*1e3:.0f} mm"
    lags = [r['lag_r'] for r in servo_log if 'lag_r' in r]
    msg += f", max right follow {max(lags):.3f} rad" if lags else ''
    print(msg)
    return dict(servo_steps=len(servo_log), control_steps=len(control_log), servo_period_max_ms=round(float(st.max())*1e3, 1),
                decision_p95_ms=round(float(np.percentile(dec, 95)), 1) if dec else None, refused=sh.refused,
                snapped=sh.snapped, min_gap_arm_mm=round(min(gaps)*1e3, 1) if gaps else None,
                max_right_follow_rad=round(max(lags), 4) if lags else None)
