"""Software-in-the-loop check of the real-cell runner: the real runner + controller (AsyncController by default) on
mock robots, against a simulated cell in REAL TIME.

* physics thread (125 Hz): the simulated rope follows the mock left arm's actual TCP; true ball-rope clearance logged.
* render thread (~30 Hz): renders both camera views of a rope snapshot (ball from the mock right arm), delivered
  after the simulated image latency, with the capture time as the frame time (like RealSense).
* at the end: the TRUE rope error (rmse, p95, end) and clearance vs what the runner reported.

Tests timing, threading and the closed loop; not the robots, the real cameras or the real rope.

--time-scale k slows the run as on the real cell (the simulated rope still runs in real time, so this also tests
that the slowed cell behaves like the trained one). --tracker-delay-s / --planner-delay-s add wall-clock sleeps to
every image tracker update / planner decision to emulate a slower PC (cell PC 2026-11-19: tracker ~0.12 s, planner
~0.37 s; this machine ~0.03 / ~0.06 s).

    nice -n 10 python scripts/real_cell/sil_check.py --scenes test_hold_0116 [--sync-controller]
    nice -n 10 python scripts/real_cell/sil_check.py --scenes test_reach_0087 --tracker-delay-s .09 --planner-delay-s .31
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')                # as in run_real_cell.py
import argparse
import json
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
for sub in ('', 'scripts/cell', 'scripts/real_cell'):
    sys.path.insert(0, str(ROOT/sub))
import numpy as np
import run_real_cell as M
from async_controller import AsyncController
from planning import make_plan
from robots import Kin, MockUR
from runner import run_scene
from dlovw.cell.calibrated.config import Config
from dlovw.cell.calibrated.deployment import Controller
from dlovw.cell.calibrated.physics import RopeState, surface_distance
from dlovw.cell.calibrated.render import SceneView


class SlowController(AsyncController):
    """AsyncController with extra wall-clock time in each image tracker update and each planner decision."""
    def __init__(self, *a, tracker_delay=0., planner_delay=0., **kw):
        super().__init__(*a, **kw)
        observe, choose = self.tracker.observe, self.planner.choose

        def slow_observe(imgs, *x, **y):
            out = observe(imgs, *x, **y)
            if imgs:
                time.sleep(tracker_delay)
            return out

        def slow_choose(*x, **y):
            out = choose(*x, **y)
            time.sleep(planner_delay)
            return out
        self.tracker.observe, self.planner.choose = slow_observe, slow_choose


class SimCell:
    def __init__(self, rec, cfg, kin_xml, left, right, seed):
        self.cfg, self.left, self.right = cfg, left, right
        self.kin = Kin(kin_xml)                   # own MuJoCo data: never shared with the servo thread
        self.rope = RopeState(rec['start_left'], cfg, seed, True, qpos=rec.get('start_qpos'))
        self.view = None                          # created in the render thread (GL contexts are thread-bound)
        self.offset = np.asarray(cfg.sphere_offset, float)
        self.goal, self.target = np.asarray(rec['goal_curve'], float), np.asarray(rec['target_left'], float)
        self.lock, self.stopped, self.frames, self.pending = threading.Lock(), False, {}, []
        self.min_surface, self.lag = np.inf, []
        self.snap = None
        self.T = None

    def _left(self):
        return self.kin.tcp('left', self.left._q.copy())[0]

    def _ball(self):
        p, R = self.kin.tcp('right', self.right._q.copy())
        return p+R @ self.offset

    def start(self):
        self.T = time.perf_counter()
        self.threads = [threading.Thread(target=f, daemon=True) for f in (self._physics, self._render)]
        for th in self.threads:
            th.start()

    def _physics(self):
        c, nxt = self.cfg, time.perf_counter()
        while not self.stopped:
            nxt += c.dt
            w = nxt-time.perf_counter()
            if w > 0:
                time.sleep(w)
            else:
                self.lag.append(-w)
                nxt = time.perf_counter()
            target, ball = self._left(), self._ball()
            self.rope.advance(target-self.rope.left())
            curve = self.rope.curve()
            self.min_surface = min(self.min_surface, surface_distance(curve, ball, c.sphere_radius, c.material['radius_m']))
            with self.lock:
                self.snap = (time.perf_counter(), self.rope.data.qpos.copy(), self.rope.data.mocap_pos.copy(), ball, curve)

    def _render(self):
        period, latency = 1/30., self.cfg.image_latency_s
        self.view = SceneView(self.cfg)
        try:
            self._render_loop(period, latency)
        finally:
            self.view.close()

    def _render_loop(self, period, latency):
        nxt = time.perf_counter()
        while not self.stopped:
            nxt += period
            w = nxt-time.perf_counter()
            time.sleep(max(w, 0.))
            with self.lock:
                snap = self.snap
            if snap is None:
                continue
            tc, qpos, mocap, ball, _ = snap
            rope = SimpleNamespace(data=SimpleNamespace(qpos=qpos, mocap_pos=mocap), left=lambda m=mocap: m[0].copy())
            self.view.sync(rope, ball)
            imgs = self.view.images()
            now = time.perf_counter()
            with self.lock:
                self.pending += [(tc+latency, r, im, tc) for r, im in imgs.items()]
                due = [p for p in self.pending if p[0] <= now]
                self.pending = [p for p in self.pending if p[0] > now]
                for _, r, im, t in due:
                    self.frames[r] = (im, t)

    def latest(self):
        now = time.perf_counter()
        with self.lock:
            due = [p for p in self.pending if p[0] <= now]
            self.pending = [p for p in self.pending if p[0] > now]
            for _, r, im, t in due:
                self.frames[r] = (im, t)
            return dict(self.frames)

    def truth(self):
        with self.lock:
            curve = self.snap[4]
        e = np.linalg.norm(curve-self.goal, axis=1)
        return dict(rmse_mm=round(float(np.sqrt((e*e).mean()))*1e3, 1), p95_mm=round(float(np.percentile(e, 95))*1e3, 1),
                    end_mm=round(float(np.linalg.norm(self.rope.left()-self.target))*1e3, 1),
                    min_surface_mm=round(float(self.min_surface)*1e3, 1),
                    physics_late_steps=len(self.lag), physics_max_late_ms=round(max(self.lag, default=0.)*1e3, 1))

    def close(self):
        self.stopped = True
        for th in self.threads:
            th.join(2.)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scenes', default='test_hold_0116')
    ap.add_argument('--sync-controller', action='store_true')
    ap.add_argument('--torch-threads', type=M.positive_threads, default=1)
    ap.add_argument('--time-scale', type=float, default=M.TIME_SCALE)
    ap.add_argument('--tracker-delay-s', type=float, default=0.)
    ap.add_argument('--planner-delay-s', type=float, default=0.)
    ap.add_argument('--no-video', action='store_true')
    ap.add_argument('--out', default='outputs/real_runs/sil')
    a = ap.parse_args()
    if a.sync_controller and (a.time_scale != 1. or a.tracker_delay_s or a.planner_delay_s):
        ap.error('--sync-controller supports only --time-scale 1 and no added delays')
    cfg = replace(Config(out=str(ROOT/M.SCENE_DIR)), plan_commit=M.COMMIT)
    recs = {r['id']: r for r in json.loads((cfg.path/'tasks/test/scenes.json').read_text())['scenes']}
    kin = Kin(cfg.path/'scene.xml')
    args = SimpleNamespace(ik_dt=.04, no_capsule=False, min_gap_m=0., max_joint_speed=1.5, branch_tol=.35, lookahead=.1,
                           gain=300., max_follow_m=.03, max_follow_rad=.05, follow_window=.2, abort_gap_m=0.,
                           max_start_s=5., max_stale_s=3., abort_surface_m=.002, snap_m=.001, time_scale=a.time_scale,
                           video=not a.no_video)
    extra = {} if a.sync_controller else dict(time_scale=a.time_scale, tracker_delay=a.tracker_delay_s,
                                              planner_delay=a.planner_delay_s)
    extra['ball'] = dict(sphere_radius=cfg.sphere_radius, sphere_offset=tuple(cfg.sphere_offset))   # the cell's ball
    cls = Controller if a.sync_controller else SlowController
    out = Path(a.out)/time.strftime('%Y%m%d_%H%M%S')
    summary = {}
    for sid in a.scenes.split(','):
        rec = recs[sid]
        left, right = MockUR(kin, 'left'), MockUR(kin, 'right')
        plan = make_plan(rec, cfg, kin, left, right, args)
        left.move_l(plan['left_home_pose'], 0, 0)
        right.move_j(plan['Q'][0], 0, 0)
        cell = SimCell(rec, cfg, cfg.path/'scene.xml', left, right, int(rec['seed']) % (1 << 30))
        cell.start()
        print(f'--- {sid} ({cls.__name__})', flush=True)
        factory = lambda r: cls(str(ROOT/M.CKPT), r['goal_curve'], r['target_left'],
                                threads=a.torch_threads, commit=M.COMMIT, **extra)  # noqa: E731
        try:
            term = run_scene(rec, cfg, plan, left, right, cell, kin, args, factory, out)
        finally:
            truth = cell.truth()
            cell.close()
        summary[sid] = dict(terminal=term, truth=truth)
        print(f'    runner: {term}; truth: {truth}', flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
