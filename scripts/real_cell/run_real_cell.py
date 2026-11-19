"""Real dual-UR5 (CB3) cell runner for the v39 r10 policy on the v41 UR5 demo scene (test_reach_0087).

Per scene: read-only feasibility check -> confirm 1 -> home both arms -> confirm 2 -> run
(left arm: closed-loop r10 policy via servoL; right arm: the scene's scripted ball track, planned offline with the
controller's own IK and sent with servoJ) -> stop, next scene. The run itself (servo loop + control thread, guards)
is in runner.py.

World frame = left UR base (as in simulation). Both TCPs must be configured as the probe tips ([0, 0, 0.165]). Tool
orientations are the ones measured before planning and are held fixed (homing never rotates the tool).

Safety: preflight.py runs first (package integrity, software, BLAS, planning latency self-test, robot modes / speed
slider / TCP / ur_rtde script, cameras) and blocks motion on any FAIL. Every failure writes an error report with a
code (errors.py). See 部署说明.md in the package root for the full procedure.

    python scripts/real_cell/run_real_cell.py --preflight-only --left-ip A --right-ip B  # safety checks only
    python scripts/real_cell/run_real_cell.py --touch-check --left-ip A --right-ip B     # read-only, Local Control
    python scripts/real_cell/run_real_cell.py --check-only --left-ip A --right-ip B      # no motion, all scenes
    python scripts/real_cell/run_real_cell.py --left-ip A --right-ip B                   # the default scene(s), DEFAULT_SCENES
    python scripts/real_cell/run_real_cell.py --dry-run [--check-only]                  # mock robots, stub policy
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    # before numpy: multithreaded BLAS made each tracker update ~12x slower and starved the 125 Hz loops
    os.environ.setdefault(_v, '1')
import argparse
import json
import sys
import time
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import errors as E
import preflight
from robots import Kin, MockUR, NoCameras, RealSense, T_RIGHT, UR, URReceive, tool_rotation_error, wrap
from planning import make_plan, print_report
from runner import run_scene
from dlovw.cell.calibrated.config import Config

# ---- fill in ---------------------------------------------------------------------------------------------------
LEFT_IP = ''
RIGHT_IP = ''
# ------------------------------------------------------------------------------------------------------------------
SCENE_DIR = 'outputs/cell_calibrated_20261113/v42_ur5'
CKPT = 'outputs/cell_calibrated_20261113/v39_occlusion/policy/seed_0/ckpt_r60.pt'
DEFAULT_SCENES = ['test_reach_0087']                      # v41: the scene rechecked on UR5 arms
COMMIT = (3., .1, .2)
TIME_SCALE = 6.                                           # see --time-scale


class StubController:
    """--dry-run policy: heads the left end to the target at the speed cap through the real execute_action
    (acceleration, workspace, slab limits); goal curve as rope estimate."""
    def __init__(self, rec, cfg):
        self.rec, self.cfg, self.goal = rec, cfg, np.asarray(rec['goal_curve'], float)
        self.fixed, self.prev, self.t0 = np.array(cfg.material['fixed_end_m']), np.zeros(3), None

    def step(self, t, left, right_tcp, right_rotation, images, view_times=None, **kw):
        from dlovw.cell.calibrated.constraints import execute_action
        self.t0 = t if self.t0 is None else self.t0
        tracking = t-self.t0 >= .64
        d = np.asarray(self.rec['target_left'], float)-np.asarray(left)
        n = np.linalg.norm(d)
        raw = (d if n < self.cfg.cap else d/n*self.cfg.cap) if tracking else np.zeros(3)
        try:
            self.prev = execute_action(raw, np.asarray(left, float), self.fixed, self.prev, self.cfg)
        except RuntimeError:                         # pushing into a boundary target (the real policy slows down)
            self.prev = np.zeros(3)
        return dict(command_xyz=self.prev, status='tracking' if tracking else 'warming_up',
                    quality=dict(centerline=self.goal))


def ask(prompt):
    return input(prompt).strip().lower()


def home(left, right, plan, rec, args):
    """Left: straight moveL with the tool orientation unchanged. Right: moveJ to the planned joint start."""
    dist = np.linalg.norm(left.tcp()[:3]-np.asarray(rec['start_left']))
    print(f"  左臂归位: {dist*1000:.0f} mm -> {np.round(rec['start_left'], 4)} (moveL)")
    if dist > args.max_home_m:
        raise RuntimeError(f'左臂归位距离 {dist:.2f} m 超过 --max-home-m {args.max_home_m} m；先人工移近')
    left.move_l(plan['left_home_pose'], args.home_speed, args.home_acc)
    print(f"  右臂归位: 最大关节移动 {np.abs(plan['Q'][0]-right.q()).max():.2f} rad (moveJ)")
    right.move_j(plan['Q'][0], args.home_joint_speed, args.home_joint_acc)
    time.sleep(.3)


def verify_home(left, right, plan, rec, args):
    """Problems after homing: left position, left joint branch vs the simulation, right joints vs the plan."""
    bad = []
    e = np.linalg.norm(left.tcp()[:3]-np.asarray(rec['start_left']))
    if e > .004:
        bad.append(f'left TCP {e*1000:.1f} mm from the start point')
    dev = float(np.abs(wrap(left.q()-plan['ref_l'])).max())
    if dev > args.branch_tol:
        bad.append(f'left joints differ from the simulated IK solution by {dev:.2f} rad (different branch)')
    e = float(np.abs(right.q()-plan['Q'][0]).max())
    if e > .01:
        bad.append(f'right joints {e:.3f} rad from the planned start')
    for name, arm, R in (('left', left, plan['Rl']), ('right', right, plan['Rr_w'])):
        if tool_rotation_error(arm.tcp(), R if name == 'left' else T_RIGHT[:3, :3].T @ R) > .03:
            bad.append(f'{name} tool orientation changed during homing')
    return bad


def positive_threads(value):
    threads = int(value)
    if threads < 1:
        raise argparse.ArgumentTypeError('Torch thread count must be at least 1')
    return threads


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scenes', default=','.join(DEFAULT_SCENES))
    ap.add_argument('--left-ip', default=LEFT_IP)
    ap.add_argument('--right-ip', default=RIGHT_IP)
    ap.add_argument('--checkpoint', default=CKPT)
    ap.add_argument('--scene-dir', default=SCENE_DIR)
    ap.add_argument('--out', default='outputs/real_runs')
    ap.add_argument('--preflight-only', action='store_true', help='run the safety checks (robots, cameras, software) and exit')
    ap.add_argument('--touch-check', action='store_true', help='read both TCP poses in Local Control to check the base transform')
    ap.add_argument('--skip-preflight', action='store_true', help='do not block motion on preflight failures (not recommended)')
    ap.add_argument('--check-only', action='store_true', help='read-only: IK, limits, clearances; no motion')
    ap.add_argument('--no-capsule', action='store_true', help='skip the two-arm capsule clearance checks (plan and run)')
    ap.add_argument('--ignore-check', action='store_true', help='run even when the feasibility check fails')
    ap.add_argument('--dry-run', action='store_true', help='mock robots, no cameras, stub policy')
    ap.add_argument('--torch-threads', type=positive_threads, default=1,
                    help='Torch CPU threads for both the recorded-frame self-test and the live planner')
    ap.add_argument('--sync-controller', action='store_true',
                    help='original single-thread Controller (planning stalls the low level; not recommended)')
    ap.add_argument('--ik-dt', type=float, default=.04, help='s; sampling of the planned right-arm joint trajectory')
    ap.add_argument('--min-gap-m', type=float, default=0., help='required capsule surface gap (0 = just no collision)')
    ap.add_argument('--max-joint-speed', type=float, default=1.5, help='rad/s limit on the planned right trajectory')
    ap.add_argument('--branch-tol', type=float, default=.35, help='rad; joint deviation from the simulated IK branch')
    ap.add_argument('--home-speed', type=float, default=.05, help='m/s (left moveL)')
    ap.add_argument('--home-acc', type=float, default=.1)
    ap.add_argument('--home-joint-speed', type=float, default=.3, help='rad/s (right moveJ)')
    ap.add_argument('--home-joint-acc', type=float, default=.5)
    ap.add_argument('--max-home-m', type=float, default=.6)
    ap.add_argument('--lookahead', type=float, default=.1)
    ap.add_argument('--gain', type=float, default=300.)
    ap.add_argument('--max-follow-m', type=float, default=.03, help='abort if the left TCP is this far from the command')
    ap.add_argument('--max-follow-rad', type=float, default=.05, help='abort if a right joint is this far from the command')
    ap.add_argument('--follow-window', type=float, default=.2,
                    help='s; follow errors are measured against the closest command of this recent window (servo delay)')
    ap.add_argument('--abort-gap-m', type=float, default=0., help='abort if the online capsule gap (arm-arm, ball-left arm) < this')
    ap.add_argument('--max-start-s', type=float, default=5., help='abort if the controller is not tracking by then')
    ap.add_argument('--snap-m', type=float, default=.001,
                    help='measured / commanded left positions this close outside the workspace domain are projected onto it')
    ap.add_argument('--max-stale-s', type=float, default=3.)
    ap.add_argument('--abort-surface-m', type=float, default=.002, help='abort if estimated ball-rope surface gap < this')
    ap.add_argument('--camera-latency', type=float, default=0.)
    ap.add_argument('--sphere-radius', type=float, default=None,
                    help='m; radius of the ball actually mounted (trained: 0.04). Sets the protected distance of the '
                         'planner and CBF, the ball-rope abort and the ball-arm checks; the policy is not retrained')
    ap.add_argument('--sphere-centre-from-tip', type=float, default=None,
                    help='m; ball centre along the probe axis from the tip, + = beyond the tip (away from the flange). '
                         'Default: Config.sphere_offset, measured 2026-11-19 = 0.025 (the policy was trained at -0.025; '
                         'it sees only the ball centre)')
    ap.add_argument('--no-video', dest='video', action='store_false',
                    help='do not record the front / top camera videos of each run (<scene>_front.mp4, _top.mp4)')
    ap.add_argument('--time-scale', type=float, default=TIME_SCALE,
                    help='run the scene this many times slower than simulated (ball, left arm, decision period, scene '
                         'time limits); 1 = real time. Default 6: the 2026-11-19 cell PC needs ~0.46 s from camera '
                         'frame to action, ~8 of the 10 simulated steps of a decision at 6x (runner.py)')
    a = ap.parse_args()
    a.commit = COMMIT
    if not a.time_scale >= 1.:
        ap.error('--time-scale must be >= 1')
    if a.sync_controller and a.time_scale != 1.:
        ap.error('--sync-controller supports only --time-scale 1')
    if a.sphere_radius is not None and not .01 <= a.sphere_radius <= .1:
        ap.error('--sphere-radius must be 0.01-0.1 m')
    a.ball = {}                                  # Config overrides for scene checks, runner and Controller alike
    if a.sphere_radius is not None:
        a.ball['sphere_radius'] = a.sphere_radius
    if a.sphere_centre_from_tip is not None:
        a.ball['sphere_offset'] = (0., 0., a.sphere_centre_from_tip)
    return a


def connect(a, kin, out_dir):
    """Robots and cameras, or a report and exit."""
    if a.dry_run:
        return MockUR(kin, 'left'), MockUR(kin, 'right'), NoCameras()
    try:
        left, right = UR(a.left_ip, 'left'), UR(a.right_ip, 'right')
    except ConnectionError as e:
        E.report(out_dir, 'startup', 'connection_failed', code='E501', detail={'错误': str(e)})
        sys.exit(1)
    need_cams = not a.check_only
    try:
        cams = RealSense(a.camera_latency) if need_cams else NoCameras()
    except Exception as e:                           # noqa: BLE001
        E.report(out_dir, 'startup', 'camera_failed', code='E203', detail={'错误': repr(e)}, left=left, right=right)
        left.close(), right.close()
        sys.exit(1)
    return left, right, cams


def run_touch_check(a, cfg, out_dir):
    def receiver(ip, side):
        return MockUR(kin, side) if a.dry_run else URReceive(ip, side)

    kin = Kin(cfg.path/'scene.xml') if a.dry_run else None
    try:
        with closing(receiver(a.left_ip, 'left')) as left, closing(receiver(a.right_ip, 'right')) as right:
            ok, errs = preflight.touch_check(left, right)
    except ConnectionError as e:
        E.report(out_dir, 'touch_check', 'connection_failed', code='E501', detail={'错误': str(e)})
        sys.exit(1)
    except ValueError as e:
        E.report(out_dir, 'touch_check', 'invalid_input', code='E206', detail={'错误': str(e)})
        sys.exit(1)
    except KeyboardInterrupt as e:
        E.report(out_dir, 'touch_check', 'user_interrupt', code='E901', exc=e)
        sys.exit(130)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/'touch_check.json').write_text(json.dumps(dict(ok=ok, errors_m=errs), indent=1), encoding='utf-8')
    if not ok:
        sys.exit(1)


def make_factory(a, cfg):
    if a.dry_run:
        return lambda rec: StubController(rec, cfg)
    if a.check_only:
        return None
    # the cell's ball (cfg: Config defaults + --sphere-* options), never the checkpoint's training-time values
    ball = dict(sphere_radius=cfg.sphere_radius, sphere_offset=tuple(cfg.sphere_offset))
    if a.sync_controller:
        from dlovw.cell.calibrated.deployment import Controller
        return lambda rec: Controller(a.checkpoint, rec['goal_curve'], rec['target_left'],
                                      threads=a.torch_threads, commit=COMMIT, ball=ball)
    from async_controller import AsyncController
    return lambda rec: AsyncController(a.checkpoint, rec['goal_curve'], rec['target_left'],
                                       threads=a.torch_threads, commit=COMMIT, time_scale=a.time_scale, ball=ball)


def run_one(rec, cfg, kin, left, right, cams, a, factory, out_dir, summary):
    """Plan, confirm, home, confirm, run one scene. Returns 'quit' to stop the session."""
    sid, m = rec['id'], rec['motion']
    print(f"\n=== {sid} ({rec['task']}), seed {rec['seed']} ===")
    print(f"  左臂起点 {np.round(rec['start_left'], 4)} -> 目标 {np.round(rec['target_left'], 4)}"
          f"（{np.linalg.norm(np.subtract(rec['target_left'], rec['start_left']))*1000:.0f} mm），"
          f"球事件结束 {rec['event_end_s']:.1f} s，球中心 {np.round(m['center'], 3)}")
    plan = make_plan(rec, cfg, kin, left, right, a)
    print_report(rec, plan)
    summary[sid] = dict(ok=plan['ok'], fail=plan['fail'], warn=plan['warn'], info=plan['info'])
    if a.check_only:
        return None
    if not plan['ok']:
        E.report(out_dir, sid, 'infeasible', code='E601', detail={'fail': plan['fail']}, left=left, right=right)
        if not a.ignore_check:
            return None
    c = ask('  [1/2] 摆好绳子、清空工作区后输入 y 开始归位（s 跳过，q 退出）: ')
    if c != 'y':
        return 'quit' if c == 'q' else None
    try:
        home(left, right, plan, rec, a)
    except Exception as e:                           # noqa: BLE001
        left.stop(), right.stop()
        E.report(out_dir, sid, 'homing_failed', code='E602', detail={'错误': str(e)}, left=left, right=right, exc=e)
        return None if ask('  继续下一个场景？y/n: ') == 'y' else 'quit'
    bad = verify_home(left, right, plan, rec, a)
    if bad:
        E.report(out_dir, sid, 'homing_check_failed', code='E602', detail={'问题': bad}, left=left, right=right)
        if not a.ignore_check:
            return None
    c = ask('  [2/2] 绳子静止、归位确认后输入 y 开始规划并运动（s 跳过，q 退出）: ')
    if c != 'y':
        return 'quit' if c == 'q' else None
    term = run_scene(rec, cfg, plan, left, right, cams, kin, a, factory, out_dir)
    summary[sid]['terminal'] = term
    if term not in ('success', 'timeout') and ask('  场景中止；继续下一个场景？y/n: ') != 'y':
        return 'quit'
    return None


def main():
    a = parse_args()
    os.chdir(Path(__file__).resolve().parents[2])    # all package paths are relative to its root
    if not a.dry_run and not (a.left_ip and a.right_ip):
        sys.exit('在脚本开头填写 LEFT_IP / RIGHT_IP（或用 --left-ip / --right-ip）。')
    cfg = replace(Config(out=a.scene_dir), plan_commit=COMMIT, **a.ball)
    recs = {r['id']: r for r in json.loads((cfg.path/'tasks/test/scenes.json').read_text())['scenes']}
    print(f'球：半径 {cfg.sphere_radius*1e3:.1f} mm，球心在探针尖前方（远离法兰）{cfg.sphere_offset[2]*1e3:.1f} mm，'
          f'保护距离 {cfg.protected*1e3:.1f} mm'
          + ('。不是默认值：先用 --check-only 和仿真复验' if a.ball else ''))
    ids = a.scenes.split(',')
    missing = [i for i in ids if i not in recs]
    if missing:
        sys.exit(f'未知场景: {missing}')
    out_dir = Path(a.out)/time.strftime('%Y%m%d_%H%M%S')
    print(f'日志目录: {out_dir}')
    k = a.time_scale
    print(f'时间缩放 {k:g}x：球和左臂按仿真速度的 1/{k:g} 运动（左臂最大 {cfg.speed/k*1e3:.1f} mm/s），'
          f'决策周期 {cfg.decision_dt*k:.2f} s，场景时限 {cfg.max_time_s*k:.0f} s'
          + ('' if k == 1 else '；日志里的 s 是仿真时间，其余 *_s 是真实时间'))
    if a.touch_check:
        run_touch_check(a, cfg, out_dir)
        return
    kin = Kin(cfg.path/'scene.xml')
    left, right, cams = connect(a, kin, out_dir)
    summary = {}
    try:
        if not a.skip_preflight:
            real = not a.dry_run
            ok, res = preflight.run(cfg, recs, a, out_dir, left if real else None, right if real else None,
                                    cams if real and not a.check_only else None, selftest=not a.check_only)
            if a.preflight_only:
                if not ok:
                    sys.exit(1)
                return
            if not ok and not a.check_only:
                E.report(out_dir, 'preflight', 'preflight_failed', code='E700',
                         detail={r['name']: r['msg'] for r in res if r['level'] == 'FAIL'}, left=left, right=right)
                return
        factory = make_factory(a, cfg)
        for rec in (recs[i] for i in ids):
            if run_one(rec, cfg, kin, left, right, cams, a, factory, out_dir, summary) == 'quit':
                break
    except KeyboardInterrupt as e:
        left.stop(), right.stop()
        E.report(out_dir, 'session', 'user_interrupt', code='E901', left=left, right=right, exc=e)
    except Exception as e:                           # noqa: BLE001 - always stop the arms and leave a report
        left.stop(), right.stop()
        E.report(out_dir, 'session', 'session_error', code='E402', detail={'错误': repr(e)}, left=left, right=right, exc=e)
        raise
    finally:
        left.stop()
        right.stop()
        cams.close()
        left.close()
        right.close()
        if summary:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir/'feasibility.json').write_text(
                json.dumps(summary, indent=1, default=float, ensure_ascii=False), encoding='utf-8')
            print('\n场景汇总:', {k: v.get('terminal', 'OK' if v['ok'] else 'FAIL') for k, v in summary.items()})


if __name__ == '__main__':
    main()
