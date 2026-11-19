"""Left / right TCP trajectories of one scene from a simulated deployment run, in the UR controllers' own frames.

The v39 policy runs the scene in the headless simulation (as latency_sim.py) with the deployment's latency model:
the new action takes over L of the 10 control steps after each observation (default 8 = the cell PC at
--time-scale 6). Every 8 ms simulated step is written with the left TCP (the rope's left end, policy-driven) and the
ball (scripted track) converted to robot poses:

* left_*  : left TCP in the LEFT controller base [x y z rx ry rz] (m, rotation vector) + IK joints (rad), sim tool
            orientation held as at the home keyframe (probe down), as the runner holds it
* right_* : right TCP in the RIGHT controller base + joints, exactly as the runner's plan commands them (make_plan:
            the scene's ball track through tcp_from_sphere and the controller IK, sampled every --ik-dt)
* ball_*  : ball centre in the world (= left base) frame; surface_mm: ball-rope surface gap in the simulation

t_sim_s is simulated time (ball clock, policy and ball start together at 0); t_real_s = t_sim_s * --time-scale is
when the same state is reached on the real cell. This is ONE closed-loop simulated run: replaying the left column
open loop on the real cell is not the policy (the real rope and ball timing differ), use it for reference / checks.

    python scripts/real_cell/export_sim_trajectory.py --scene test_reach_0087
"""
import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
for sub in ('', 'scripts/real_cell'):
    sys.path.insert(0, str(ROOT/sub))
import numpy as np
import torch

import latency_sim as LS
import run_real_cell as M
from planning import make_plan, right_q_at
from robots import Kin, MockUR, Rp_to_pose
from dlovw.cell.calibrated import data as D
from dlovw.cell.calibrated.checkpoint import load
from dlovw.cell.calibrated.config import Config
from dlovw.cell.calibrated.constraints import CalibratedLimits
from dlovw.cell.calibrated.session import surface_distance
from dlovw.cell.cost import CostCfg
from dlovw.cell.planner import CellPlanner


class RecordingSession(LS.LatencySession):
    """LatencySession that keeps (t, left TCP, ball centre, ball-rope surface gap) of every control step."""
    log = None

    def step(self, u, cbf=True, stress=None):
        if not RecordingSession.log:
            RecordingSession.log.append(self._row())
        out = super().step(u, cbf, stress)
        RecordingSession.log.append(self._row())
        return out

    def _row(self):
        c = self.config
        gap = surface_distance(self.rope.curve(), self.sphere(), c.sphere_radius, c.material['radius_m'])
        return self.t, self.rope.left().copy(), self.sphere().copy(), gap


def simulate(rec, cfg, checkpoint, latency_steps):
    torch.set_num_threads(1)
    nets, _ = load(checkpoint)
    wm, fc, actor, critic = nets
    planner = LS.DelayedPlanner(CellPlanner(wm, fc, actor, critic, CalibratedLimits.from_config(cfg),
                                            cost=CostCfg(protected_m=cfg.protected, segment_distance=True),
                                            horizon=cfg.plan_horizon, commit=dict(abs=3., rel=.1, risk=.2)))
    RecordingSession.mode, RecordingSession.L, RecordingSession.Q = 'async', latency_steps, 0
    RecordingSession.planner, RecordingSession.log = planner, []
    D.Session = RecordingSession
    e = D.run_episode(rec, cfg, seed=int(rec['seed']) % (1 << 30), deploy=True, nets=nets, planner=planner, kind='heads')
    return e['info'], RecordingSession.log


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scene', default='test_reach_0087')
    ap.add_argument('--scene-dir', default=M.SCENE_DIR)
    ap.add_argument('--checkpoint', default=M.CKPT)
    ap.add_argument('--latency-steps', type=int, default=8, help='of the 10 steps per decision (async model)')
    ap.add_argument('--time-scale', type=float, default=M.TIME_SCALE)
    ap.add_argument('--sphere-radius', type=float, default=None)
    ap.add_argument('--out', default=None, help='default: <scene-dir>/trajectories')
    a = ap.parse_args()
    ball = {} if a.sphere_radius is None else dict(sphere_radius=a.sphere_radius)
    cfg = replace(Config(out=str(ROOT/a.scene_dir)), plan_commit=M.COMMIT, headless=True, **ball)
    rec = {r['id']: r for r in json.loads((cfg.path/'tasks/test/scenes.json').read_text())['scenes']}[a.scene]
    info, log = simulate(rec, cfg, str(ROOT/a.checkpoint), a.latency_steps)
    print(f"simulated {a.scene}: {info['terminal']}, {len(log)} rows, min surface {info['min_surface_m']*1e3:.1f} mm")

    kin = Kin(cfg.path/'scene.xml')
    left, right = MockUR(kin, 'left'), MockUR(kin, 'right')           # both at the scene's home keyframe
    args = SimpleNamespace(ik_dt=.04, no_capsule=False, min_gap_m=0., max_joint_speed=1.5, branch_tol=.35)
    plan = make_plan(rec, cfg, kin, left, right, args)               # as run_real_cell --dry-run --check-only
    if not plan['ok']:
        raise RuntimeError(f"the runner feasibility check fails for this scene: {plan['fail']}")
    Rl = plan['Rl']                                                  # left tool held as at home (probe down)
    q, rows, worst_ik = plan['qs'], [], 0.
    for t, p, ball_c, gap in log:
        q_new = left.ik(Rp_to_pose(Rl, p), q)
        if q_new is None:
            raise RuntimeError(f'left IK failed at t = {t:.3f} s, {np.round(p, 4)}')
        q = q_new
        lp = left.fk(q)
        worst_ik = max(worst_ik, float(np.linalg.norm(lp[:3]-p)))
        qr = right_q_at(plan, t)
        rows.append([round(t, 3), round(t*a.time_scale, 3), *np.round(lp, 6), *np.round(q, 6),
                     *np.round(right.fk(qr), 6), *np.round(qr, 6), *np.round(ball_c, 6), round(gap*1e3, 2)])
    out = Path(a.out) if a.out else cfg.path/'trajectories'
    out.mkdir(parents=True, exist_ok=True)
    stem = f'{a.scene}_sim_tcp'
    head = (['t_sim_s', 't_real_s'] + [f'left_{k}' for k in ('x', 'y', 'z', 'rx', 'ry', 'rz')]
            + [f'left_q{i}' for i in range(1, 7)] + [f'right_{k}' for k in ('x', 'y', 'z', 'rx', 'ry', 'rz')]
            + [f'right_q{i}' for i in range(1, 7)] + ['ball_x', 'ball_y', 'ball_z', 'surface_mm'])
    with open(out/f'{stem}.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerows(rows)
    meta = dict(scene=a.scene, checkpoint=a.checkpoint, scene_dir=a.scene_dir, terminal=info['terminal'],
                min_surface_mm=round(info['min_surface_m']*1e3, 1), final_rmse_mm=round(info['rmse']*1e3, 1),
                final_end_mm=round(info['end']*1e3, 1), rows=len(rows), dt_sim_s=cfg.dt,
                latency_model=f'async:{a.latency_steps}:0 (new action after {a.latency_steps} of {cfg.decision_steps} steps)',
                time_scale=a.time_scale, sphere_radius_m=cfg.sphere_radius, sphere_offset_m=list(cfg.sphere_offset),
                tcp_offset=[0, 0, .165, 0, 0, 0], left_ik_max_error_mm=round(worst_ik*1e3, 3),
                plan_ok=plan['ok'], plan_fail=plan['fail'], plan_warn=plan['warn'],
                frames=dict(left='left UR controller base (= simulation world)',
                            right='right UR controller base (T_WORLD_FROM_RIGHT_BASE in dlovw/cell/layout.py)',
                            ball='world (= left base)', pose='[x y z rx ry rz]: metres and UR rotation vector'),
                columns=head, note=__doc__.split('\n\n')[3].strip())
    (out/f'{stem}.json').write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(f'wrote {out/stem}.csv / .json (left IK max error {worst_ik*1e3:.3f} mm, plan ok {plan["ok"]})')


if __name__ == '__main__':
    main()
