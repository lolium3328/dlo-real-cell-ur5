"""Disjoint measured-domain reach/hold scenes and recorded oracle screening."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time
import mujoco

import numpy as np

from ...motion import track
from ...real_rope.vision import cameras
from ...real_rope.visibility import curve_margin
from .config import Config, ROOT, digest
from .constraints import bounds_at, clip_norm, segment_ok
from .physics import RopeState, surface_distance
from .session import Session

_VIEW = None


def _perp_basis(t):
    t = t/max(np.linalg.norm(t), 1e-12)
    a = np.array([1., 0, 0]) if abs(t[0]) < .9 else np.array([0., 0, 1])
    e = np.cross(t, a)
    e /= np.linalg.norm(e)
    return e, np.cross(t, e)


def _motion(rng, kind, centre, direction, scale=1.):
    """scale < 1 slows the ball by time-scaling the whole motion by 1/scale (speed * scale, acceleration * scale^2,
    dwells / scale), keeping its path; the random draws are unchanged. scale = 1 reproduces the original tasks."""
    amplitude, dwell = rng.uniform(.12, .17), rng.uniform(.3, 1.8)
    speed = rng.uniform(.12, .25) if kind == 'hold' else rng.uniform(.15, .33)
    passes = 1 if kind == 'hold' or rng.random() < .5 else 2
    middle = float(rng.uniform(.3, 1.))
    speed, acc, dwell, middle = speed*scale, 1.5*scale**2, dwell/scale, middle/scale
    end = dwell+passes*(2*amplitude/speed+speed/acc)+(middle if passes == 2 else 0.)
    return dict(kind='patrol', center=centre.tolist(), direction=direction.tolist(), amplitude_m=float(amplitude),
                speed_mps=float(speed), accel_mps2=float(acc), dwells_s=[float(dwell)]+([middle] if passes == 2 else [])+[30./scale]*4,
                start_offset_s=0., passes=passes), float(end)


def sample_point(rng, config):
    z = float(rng.choice([r['z_m'] for r in config.slices]))
    lo, hi = bounds_at(z, config.slices)
    return np.r_[rng.uniform(lo+.001, hi-.001), z]


ONSET_M = .01             # v38 causal oracle: the ball has visibly started to move


def onset_time(motion, config):
    """First time the ball is ONSET_M from its initial position, plus one decision of reaction (causal trigger)."""
    ball = track(motion, round(config.max_time_s/config.dt), config.dt)
    moved = np.flatnonzero(np.linalg.norm(ball-ball[0], axis=1) >= ONSET_M)
    return (float(moved[0])*config.dt if len(moved) else np.inf)+config.decision_dt


def action_for_plan(session, plan):
    """Clairvoyant oracle (v1-v37): head for the dodge point from t = 0 until `wait`. With rec['onset_s'] (v38,
    Config.causal_oracle) it follows the reference until the ball has started to move, then dodges."""
    dodge, wait = plan
    target = np.asarray(session.rec['target_left'])
    onset = session.rec.get('onset_s', 0.)
    goal = target if dodge is None or session.t >= wait or session.t < onset else np.asarray(dodge)
    return clip_norm(goal-session.rope.left(), session.config.cap)


def oracle_plans(rec, config):
    rows, grid = config.slices, []
    for row in rows:
        for x in np.linspace(row['x_low_m']+.002, row['x_high_m']-.002, 3):
            for y in np.linspace(row['y_low_m']+.002, row['y_high_m']-.002, 3):
                p = np.array([x, y, row['z_m']])
                if segment_ok(np.asarray(rec['start_left']), p, rows):
                    grid.append(p)
    cross = np.asarray(rec['motion']['center'])
    grid.sort(key=lambda p: -np.linalg.norm(p-cross))
    return [(None, 0.)]+[(p.tolist(), rec['event_end_s']+w) for w in (-.4, 0., .4) for p in grid]


def run_oracle(rec, plan, config):
    session = Session(rec, config, visual=False)
    try:
        while session.terminal == 'running':
            session.step(action_for_plan(session, plan), cbf=False)
        result = session.result()
        result['accepted'] = result['terminal'] == 'success' and not result['out_of_fov']
        return result
    finally:
        session.close()


def _arm_geoms(model):
    """Left arm (links + tool) and right arm (links + tool + ball) geoms for the arm-arm distance (v40)."""
    def side(prefix, extra=()):
        return [g for g in range(model.ngeom) if model.geom(g).name in extra
                or (model.body(model.geom_bodyid[g]).name.startswith(prefix+'_ur5e') and model.geom_group[g] == 2)
                or model.body(model.geom_bodyid[g]).name == prefix+'_tool']
    return side('left'), side('right', ('obstacle_sphere',))


def geom_distance(model, data, a, b, distmax=.05):
    """mj_geomDistance capped at distmax. It returns a spurious 0 for some mesh pairs with a large distmax (a pair
    measured 0.0 at distmax 0.3 m and >= 0.05 m at distmax 0.05 m), so a 0 is re-measured with a 1 cm cap."""
    x = mujoco.mj_geomDistance(model, data, a, b, distmax, None)
    if x == 0.:
        x = mujoco.mj_geomDistance(model, data, a, b, .01, None)
    return x


def arm_distance(model, data, geoms, distmax=.05):
    """Min distance (m, negative = penetration, capped at distmax) between the two arms. The simulated arms are
    kinematic and never collide, so without this check a task can need the arms to pass through each other."""
    return min(geom_distance(model, data, a, b, distmax) for a in geoms[0] for b in geoms[1])


def _right_capsules(model, data):
    """Right arm collision capsules (links) + probe as (a, b, radius) segments."""
    geoms = [g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE and (
        (model.body(model.geom_bodyid[g]).name.startswith('right_ur5e') and model.geom_group[g] == 3)
        or model.geom(g).name == 'right_probe')]
    out = []
    for g in geoms:
        c, ax = data.geom_xpos[g], data.geom_xmat[g].reshape(3, 3)[:, 2]
        r, h = model.geom_size[g, :2]
        out.append((c-h*ax, c+h*ax, r))
    return out


def right_rope_distance(model, data, curve, rope_radius):
    """Min surface distance (m) between the rope nodes and the right arm's links and probe (the ball excluded:
    surface_distance covers it). The simulated rope only collides with the ball."""
    best = np.inf
    for a, b, r in _right_capsules(model, data):
        ab = b-a
        t = np.clip(((curve-a)@ab)/(ab@ab), 0, 1)
        best = min(best, float(np.min(np.linalg.norm(curve-(a+t[:, None]*ab), axis=1)))-r-rope_radius)
    return best


def verify_geometry(rec, plan, config):
    """Recheck the new mount in full-cell IK, independently of the rope-only oracle."""
    global _VIEW
    from .render import SceneView
    if _VIEW is None or _VIEW.config.out != config.out:
        _VIEW = SceneView(config, render=False)
        _VIEW.arm_geoms, _VIEW.home = _arm_geoms(_VIEW.model), (_VIEW.ql.copy(), _VIEW.qr.copy())
    _VIEW.ql, _VIEW.qr = (q.copy() for q in _VIEW.home)            # same IK warm start for every task
    s = Session(rec, config, visual=False)
    max_error, tool_clearance, arm_clear, rope_clear = 0., np.inf, np.inf, np.inf
    try:
        while s.terminal == 'running':
            if s.n % 10 == 0:
                _VIEW.sync(s.rope, s.sphere())
                max_error = max(max_error, _VIEW.last_ik[0])
                for name in ('left_probe', 'left_plate', 'fixture_post'):
                    distance = mujoco.mj_geomDistance(_VIEW.model, _VIEW.data,
                                _VIEW.model.geom('obstacle_sphere').id, _VIEW.model.geom(name).id, .2, None)
                    tool_clearance = min(tool_clearance, distance)
                if config.arm_clearance > 0:
                    arm_clear = min(arm_clear, arm_distance(_VIEW.model, _VIEW.data, _VIEW.arm_geoms))
                rope_clear = min(rope_clear, right_rope_distance(_VIEW.model, _VIEW.data, s.rope.curve(), config.material['radius_m']))
            s.step(action_for_plan(s, plan), cbf=False)
        return dict(accepted=bool(s.terminal == 'success' and max_error <= .001 and tool_clearance >= .005
                                  and arm_clear >= config.arm_clearance and rope_clear >= config.right_rope_clearance),
                    max_ik_position_m=float(max_error), min_ball_tool_fixture_m=float(tool_clearance),
                    min_arm_arm_m=float(arm_clear), min_right_arm_rope_m=float(rope_clear))
    finally:
        s.close()


def make_candidate(job):
    config, split, kind, index, seed = job
    rng, rejected = np.random.default_rng(seed), Counter()
    rec = dict(id=f'{split}_{kind}_{index:04d}', split=split, task=kind, seed=seed, accept=False)
    start = sample_point(rng, config)
    target = start.copy()
    if kind == 'reach':
        for _ in range(100):
            target = sample_point(rng, config)
            if .04 <= np.linalg.norm(start-target) <= .16 and segment_ok(start, target, config.slices):
                break
        else:
            return dict(**rec, rejection='start_goal_path')
    initial, goal = RopeState(start, config), RopeState(target, config)
    goal.move(target, settle=4.)
    rec.update(start_left=start.tolist(), target_left=target.tolist(), start_qpos=initial.data.qpos.tolist(),
               goal_curve=goal.curve().tolist(), event_end_s=0.,
               motion=dict(kind='quasi_periodic', center=[2., -1., 1.], direction=[1., 0, 0],
                           amplitude_m=0., half_periods_s=[2.]*40, phase0=0.))
    ref_session = Session(rec, config, visual=False)
    ref_curves = [ref_session.rope.curve()]
    # Reference must remain running after reaching its shape to cover the ball event.
    for _ in range(round(6/config.ball_speed_scale/config.dt)):
        ref_session.step(clip_norm(target-ref_session.rope.left(), config.cap), cbf=False)
        ref_curves.append(ref_session.rope.curve())
    ref_session.close()
    ref = np.array(ref_curves)
    cams = cameras(ROOT/config.material['cameras'])
    if min(curve_margin(goal.curve(), cams, config.material['radius_m'], config.margin_px, 0.).values()) < 0:
        return dict(**rec, rejection='goal_fov')
    for _ in range(24):
        k = int(rng.integers(round(.6/config.dt), round(3.5/config.dt)))
        curve = ref[min(k, len(ref)-1)]
        j = int(rng.integers(6, config.crossing_node_max+1))   # crossing node (0 = moving end, 38 = fixed end)
        e1, e2 = _perp_basis(curve[j+1]-curve[j-1])
        theta = rng.uniform(0, 2*np.pi)
        motion, event_end = _motion(rng, kind, curve[j], np.cos(theta)*e1+np.sin(theta)*e2, config.ball_speed_scale)
        if event_end > 5./config.ball_speed_scale:
            rejected['event_late'] += 1
            continue
        ball = track(motion, len(ref)-1, config.dt)
        lo, hi = np.array([-.24, -.94, .14]), np.array([.26, -.30, .56])
        if (ball < lo).any() or (ball > hi).any():
            rejected['ball_box'] += 1
            continue
        start_clear = surface_distance(ref[0], ball[0], config.sphere_radius, config.material['radius_m'])
        end_clear = surface_distance(goal.curve(), ball[-1], config.sphere_radius, config.material['radius_m'])
        if start_clear < config.clearance+.03 or end_clear < config.clearance+.03:
            rejected['endpoint_clearance'] += 1
            continue
        reference_d = np.linalg.norm(ref-ball[:, None], axis=2).min(-1)-config.protected
        if reference_d.min() >= 0:
            rejected['no_crossing'] += 1
            continue
        rec.update(motion=motion, event_end_s=float(event_end), crossing_node=j, reference_clearance_m=float(reference_d.min()),
                   rejections=dict(rejected))
        if config.causal_oracle:
            rec['onset_s'] = onset_time(motion, config)
        break
    else:
        return dict(**rec, rejection='motion_draws_exhausted', rejections=dict(rejected))
    trials = []
    for plan in oracle_plans(rec, config)[:config.max_oracle_plans]:
        result = run_oracle(rec, plan, config)
        trials.append(result)
        if plan[0] is None and result['terminal'] != 'clearance_violation':     # v37: the recheck criteria, folded in
            return dict(**rec, rejection='reference_does_not_require_avoidance', plans_tried=len(trials))
        if result['accepted'] and result['max_acceleration'] <= config.acceleration+1e-6:
            geometry = verify_geometry(rec, plan, config)
            if geometry['accepted']:
                rec.update(accept=True, oracle_plan=plan, oracle_result=result, plans_tried=len(trials), geometry=geometry,
                           screening='braking_constraints_v1')
                return rec
    return dict(**rec, rejection='oracle_not_found', plans_tried=len(trials),
                oracle_failures=dict(Counter(r['terminal'] for r in trials)))


def _job(job):
    try:
        return make_candidate(job)
    except Exception as exc:
        return dict(id=f'{job[1]}_{job[2]}_{job[3]:04d}', split=job[1], task=job[2],
                    accept=False, rejection='exception', error=repr(exc), seed=job[4])


def generate(config, split, reach, hold, workers):
    config.freeze()
    out = config.path/'tasks'/split
    out.mkdir(parents=True, exist_ok=True)
    manifest = out/'scenes.json'
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old['counts'] != [reach, hold] or old['inputs'] != config.provenance()['inputs']:
            raise ValueError('Task cache settings differ; use a new run directory')
        return old['summary']
    seed = config.seed+['train', 'validation', 'test'].index(split)
    rng = np.random.default_rng(seed)
    jobs = [(config, split, kind, i, int(rng.integers(1<<31))) for kind, n in [('reach', reach), ('hold', hold)] for i in range(n)]
    rows, begin = [], time.monotonic()
    with ProcessPoolExecutor(workers) as pool, (out/'candidates.jsonl').open('a') as stream:
        futures = [pool.submit(_job, j) for j in jobs]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            stream.write(json.dumps(row)+'\n'); stream.flush()
            print(f'{split}: {len(rows)}/{len(jobs)}, accepted={sum(r["accept"] for r in rows)}', flush=True)
    summary = dict(proposed=len(rows), accepted=sum(r['accept'] for r in rows),
                   reasons=dict(Counter(r.get('rejection', 'accepted') for r in rows)), wall_s=time.monotonic()-begin)
    manifest.write_text(json.dumps(dict(protocol='calibrated_dynamic_oracle_v1', split=split, seed=seed, counts=[reach, hold],
                                        inputs=config.provenance()['inputs'], scenes=sorted(rows, key=lambda r:r['id']),
                                        summary=summary), indent=2))
    return summary


def _rescreen(job):
    rec, config = job
    if rec.get('event_end_s', 0.) <= 0:
        return dict(rec, accept=False, rechecked=True)
    try:
        reference = run_oracle(rec, (None, 0.), config)
        if reference['terminal'] != 'clearance_violation':
            return dict(rec, accept=False, rejection='new_reference_does_not_require_avoidance', rechecked=True)
        trials = []
        for plan in oracle_plans(rec, config):
            result = run_oracle(rec, plan, config)
            trials.append(result)
            if result['accepted'] and result['max_acceleration'] <= config.acceleration+1e-6:
                geometry = verify_geometry(rec, plan, config)
                if geometry['accepted']:
                    return dict(rec, accept=True, oracle_plan=plan, oracle_result=result, plans_tried=len(trials),
                                rechecked=True, screening='braking_constraints_v1', geometry=geometry)
        return dict(rec, accept=False, rejection='rechecked_oracle_not_found', rechecked=True,
                    oracle_failures=dict(Counter(r['terminal'] for r in trials)))
    except Exception as exc:
        return dict(rec, accept=False, rejection='rescreen_exception', error=repr(exc), rechecked=True)


def rescreen(config, source, workers):
    config.freeze()
    source = Path(source)
    raw = json.loads(source.read_text())
    split = raw['split']
    out = config.path/'tasks'/split; out.mkdir(parents=True, exist_ok=True)
    complete = out/'scenes.json'
    if complete.exists():
        return json.loads(complete.read_text())['summary']
    rows, prior = [], {}
    log = out/'rechecked.jsonl'
    if log.exists():
        for line in log.read_text().splitlines():
            r = json.loads(line); prior[r['id']] = r
    rows.extend(prior.values())
    todo = [(r, config) for r in raw['scenes'] if r['id'] not in prior]
    with ProcessPoolExecutor(workers) as pool, log.open('a') as stream:
        for future in as_completed([pool.submit(_rescreen, j) for j in todo]):
            row = future.result(); rows.append(row)
            stream.write(json.dumps(row)+'\n'); stream.flush()
            print(f'recheck {split}: {len(rows)}/{len(raw["scenes"])} accepted={sum(r["accept"] for r in rows)}', flush=True)
    summary = dict(proposed=len(rows), accepted=sum(r['accept'] for r in rows),
                   reasons=dict(Counter(r.get('rejection', 'accepted') if not r['accept'] else 'accepted' for r in rows)))
    complete.write_text(json.dumps(dict(protocol='calibrated_dynamic_braking_v1', split=split, seed=raw['seed'],
                                        counts=raw['counts'], inputs=config.provenance()['inputs'],
                                        config=config.provenance(), source=str(source), source_sha256=digest(source),
                                        scenes=sorted(rows, key=lambda r:r['id']), summary=summary), indent=2))
    return summary
