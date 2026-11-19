"""Replay the deployed v39 policy in simulation on the demo scenes and check two-arm collisions along the ACTUAL path.

Per 80 ms decision frame: left TCP (policy path) and ball (scripted) -> IK for both arms -> capsule gaps
(arm-arm, ball-left-arm) time-aligned. Also reports the earlier mesh-based arm-arm minimum (capped at 50 mm).

    nice -n 19 python scripts/real_cell/replay_check.py --workers 2
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for sub in ('', 'scripts/cell', 'scripts/real_cell'):
    sys.path.insert(0, str(ROOT/sub))
import numpy as np
import check_demo_cells as C
import geometry as G
from robots import Kin
from dlovw.cell.calibrated.config import Config, right_rotation, tcp_from_sphere
from dlovw.cell.scene import mat

DEFAULT = 'test_reach_0087'


def job(j):
    res = C._job(j)
    log = C._W['session'].log
    return res, np.array([x[0] for x in log]), np.array([x[1] for x in log])


def frames_gap(kin, cfg, left, ball, tool_radius=None):
    Rl, Rr = mat(cfg.material['left_tcp_quaternion_wxyz']), right_rotation(cfg)
    ql, qr = kin.home['left'].copy(), kin.home['right'].copy()
    PL, PR = [], []
    for l, b in zip(left, ball):
        ql, _, _ = kin.arms['left'].ik(l, Rl, ql, iters=60)
        qr, _, _ = kin.arms['right'].ik(tcp_from_sphere(b, Rr, cfg), Rr, qr, iters=60)
        PL.append(kin.points('left', ql))
        PR.append(kin.points('right', qr))
    PL, PR = np.array(PL), np.array(PR)
    radii = G.RADII.copy()
    if tool_radius is not None:
        radii[-1] = tool_radius
    AL, BL, AR, BR = PL[:, :-1], PL[:, 1:], PR[:, :-1], PR[:, 1:]
    arm = np.full(len(PL), np.inf)
    worst = (np.inf, None)
    for i in range(7):
        for j in range(7):
            d = G.seg_seg(AR[:, j], BR[:, j], AL[:, i], BL[:, i])-radii[i]-radii[j]
            arm = np.minimum(arm, d)
            if d.min() < worst[0]:
                worst = (float(d.min()), (G.NAMES[i], G.NAMES[j]))
    c = ball[:, None, :]
    ball_gap = (G.seg_seg(c, c, AL, BL)-radii-G.BALL_R).min(-1)
    return arm, ball_gap, worst[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scenes', default=DEFAULT)
    ap.add_argument('--scene-dir', default='outputs/cell_calibrated_20261113/v42_ur5')
    ap.add_argument('--policy', default='outputs/cell_calibrated_20261113/v39_occlusion/policy/seed_0/ckpt_r60.pt')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--out', default='outputs/real_runs/replay_check')
    a = ap.parse_args()
    cfg = replace(Config(out=a.scene_dir), plan_commit=(3., .1, .2))
    recs = {r['id']: r for r in json.loads((cfg.path/'tasks/test/scenes.json').read_text())['scenes']}
    ids = a.scenes.split(',')
    jobs = [(recs[i], cfg, str(ROOT/a.policy), True) for i in ids]
    with ProcessPoolExecutor(a.workers, mp_context=get_context('spawn')) as pool:
        results = list(pool.map(job, jobs))
    kin = Kin(cfg.path/'scene.xml')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for sid, (res, left, ball) in zip(ids, results):
        np.savez(out/f'{sid}.npz', left=left, ball=ball)
        row = dict(terminal=res['terminal'], duration_s=res['duration_s'], mesh_arm_arm_mm=res['mins_mm']['arm_arm'])
        for tag, tr in (('r40', None), ('r20', .02)):
            arm, bg, pair = frames_gap(kin, cfg, left, ball, tr)
            row[f'capsule_{tag}'] = dict(arm_arm_mm=round(float(arm.min())*1e3, 1), t_s=round(int(arm.argmin())*cfg.decision_dt, 2),
                                         pair=pair, ball_left_arm_mm=round(float(bg.min())*1e3, 1))
        summary[sid] = row
        print(sid, json.dumps(row), flush=True)
    (out/'summary.json').write_text(json.dumps(summary, indent=1, default=float))


if __name__ == '__main__':
    main()
