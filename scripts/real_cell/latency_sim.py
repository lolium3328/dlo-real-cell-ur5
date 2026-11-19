"""How much does real planning latency change the v39 policy's outcome? Simulated deploy episodes (same seeds) with a
latency model per 80 ms decision cycle (10 control steps):

* hold:L  - the controller is busy planning for the first L steps of every cycle: nothing is executed (commanded
            left target stays put) while ball and rope keep moving. This is the current single-thread Controller.
* async:L - the 125 Hz low level never waits: for the first L steps it keeps executing the PREVIOUS decision's action
            (projected, CBF-filtered and acceleration-limited every step as usual); the new action takes over at step L.
* :Q      - optional tracker compute delay: the low level and the planner see the tracker output Q steps late (the
            tracker in its own thread), on top of the simulated image latency.

The policy's own state sees all of this exactly as the real Controller would (it measures the displacement).
Measured on this machine (CPU, 1 thread, commit 3/.1/.2): decision p50 ~55 ms (L~7), p95 ~81 ms (L~10); image
tracking steps p95 ~27 ms (Q~3).

    nice -n 19 python scripts/real_cell/latency_sim.py --conds hold:0:0,async:7:0,async:7:3 --workers 2
"""
import argparse
import json
import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from dlovw.cell.calibrated.config import Config
from dlovw.cell.calibrated import data as D, session as S
from dlovw.cell.calibrated.checkpoint import load
from dlovw.cell.calibrated.constraints import CalibratedLimits
from dlovw.cell.cost import CostCfg
from dlovw.cell.planner import CellPlanner
from dlovw.cell.calibrated.session import surface_distance

DEMO = 'test_reach_0087'
_W = {}


class DelayedPlanner:
    """Remembers the previous decision's action for the async low level."""
    def __init__(self, inner):
        self.inner, self.prev, self.cur = inner, np.zeros(6), np.zeros(6)

    def choose(self, *a, **k):
        name, act, scores = self.inner.choose(*a, **k)
        self.prev, self.cur = self.cur, np.asarray(act, float).copy()
        return name, act, scores

    def __getattr__(self, k):
        return getattr(self.inner, k)


class LatencySession(S.Session):
    mode, L, Q, planner = 'hold', 0, 0, None

    def sample(self, stress=None):
        super().sample(stress)
        if self.Q and self.quality is not None:
            buf = self.__dict__.setdefault('_qbuf', deque())
            buf.append(self.quality)
            while len(buf) > self.Q+1:
                buf.popleft()
            self.quality = buf[0]

    def step(self, u, cbf=True, stress=None):
        c = self.config
        busy = self.n % c.decision_steps < self.L
        if not busy:
            return super().step(u, cbf, stress)
        if self.mode == 'async':                 # low level keeps running the previous action
            u = self.limits.project(self.planner.prev, np.r_[self.rope.left(), self.rope.fixed])[:3]*c.cap
            return super().step(u, cbf, stress)
        # hold: controller busy, nothing executed, previous command / delay queue untouched
        self.rope.advance(np.zeros(3))
        self.n += 1
        self.sample(stress)
        error = self.error()
        near = surface_distance(self.rope.curve(), self.sphere(), c.sphere_radius, c.material['radius_m'])
        ok = error['rmse'] <= .012 and error['p95'] <= .025 and error['end'] <= .016
        self.held = self.held+1 if ok and self.t >= self.rec['event_end_s']+.2 else 0
        if near < c.clearance:
            self.terminal = 'clearance_violation'
        elif not np.isfinite(self.rope.data.qpos).all():
            self.terminal = 'invalid_state'
        elif self.held*c.dt >= c.hold_s:
            self.terminal = 'success'
        elif self.t >= c.max_time_s:
            self.terminal = 'timeout'


def job(j):
    rec, config, checkpoint, cond = j
    mode, L, Q = cond
    torch.set_num_threads(1)
    if _W.get('checkpoint') != checkpoint:
        _W['nets'], _ = load(checkpoint)
        _W['checkpoint'] = checkpoint
        D.Session = LatencySession
    wm, fc, actor, critic = _W['nets']
    planner = DelayedPlanner(CellPlanner(wm, fc, actor, critic, CalibratedLimits.from_config(config),
                                         cost=CostCfg(protected_m=config.protected, segment_distance=True),
                                         horizon=config.plan_horizon, commit=dict(abs=3., rel=.1, risk=.2)))
    LatencySession.mode, LatencySession.L, LatencySession.Q, LatencySession.planner = mode, L, Q, planner
    e = D.run_episode(rec, config, seed=int(rec['seed']) % (1 << 30), deploy=True, nets=_W['nets'], planner=planner,
                      kind='heads')
    i = e['info']
    return dict(id=rec['id'], task=rec['task'], cond=f'{mode}:{L}:{Q}', terminal=i['terminal'],
                duration_s=round(i['steps']*config.dt, 2), min_surface_mm=round(i['min_surface_m']*1e3, 1),
                rmse_mm=round(i['rmse']*1e3, 1), p95_mm=round(i['p95']*1e3, 1), end_mm=round(i['end']*1e3, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scenes', default=DEMO, help="comma-separated ids, or 'all' (every accepted test scene)")
    ap.add_argument('--conds', default='hold:0:0,hold:6:0,hold:8:0', help='mode:L:Q,...')
    ap.add_argument('--scene-dir', default='outputs/cell_calibrated_20261113/v42_ur5')
    ap.add_argument('--policy', default='outputs/cell_calibrated_20261113/v39_occlusion/policy/seed_0/ckpt_r60.pt')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--out', default='outputs/real_runs/latency_sim')
    ap.add_argument('--tag', default='')
    ap.add_argument('--rgb', action='store_true', help='render + RecordedRopeTracker instead of the headless tracking model')
    ap.add_argument('--sphere-radius', type=float, default=None,
                    help='m; a ball other than the trained 0.04 (clearance, planner cost and CBF all use it)')
    a = ap.parse_args()
    ball = {} if a.sphere_radius is None else dict(sphere_radius=a.sphere_radius)
    cfg = replace(Config(out=a.scene_dir), plan_commit=(3., .1, .2), headless=not a.rgb, **ball)
    scenes = json.loads((cfg.path/'tasks/test/scenes.json').read_text())['scenes']
    recs = [r for r in scenes if r.get('accept', True)] if a.scenes == 'all' else \
        [{r['id']: r for r in scenes}[i] for i in a.scenes.split(',')]
    conds = [(m, int(L), int(Q)) for m, L, Q in (c.split(':') for c in a.conds.split(','))]
    jobs = [(r, cfg, str(ROOT/a.policy), c) for r in recs for c in conds]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(a.workers, mp_context=get_context('spawn')) as pool:
        for row in pool.map(job, jobs):
            rows.append(row)
            print(json.dumps(row), flush=True)
    tag = a.tag or ('demo' if a.scenes == DEMO else 'all' if a.scenes == 'all' else 'custom')
    (out/f'{tag}.json').write_text(json.dumps(rows, indent=1))
    for m, L, Q in conds:
        r = [x for x in rows if x['cond'] == f'{m}:{L}:{Q}']
        print(f'{m}:{L}:{Q}: {sum(x["terminal"] == "success" for x in r)}/{len(r)} success', flush=True)


if __name__ == '__main__':
    main()
