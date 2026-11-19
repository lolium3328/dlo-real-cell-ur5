"""Frozen-scenario paired evaluations and explicit deployment acceptance metrics."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from multiprocessing import get_context
import numpy as np
import torch

from ..planner import CellPlanner
from ..cost import CostCfg
from .checkpoint import load
from .constraints import CalibratedLimits
from .data import run_episode
from .config import Config, digest, verify_source

_NETS = {}


def evaluate_job(job):
    rec, config, checkpoint, condition, arm = job
    torch.set_num_threads(1)
    key = str(checkpoint)
    if arm == 'heads' and _NETS.get('key') != key:
        verify_source(config.provenance()['source_sha256'])
        nets, _ = load(checkpoint)
        _NETS.clear(); _NETS.update(key=key, nets=nets)
    seed = int(rec['seed']) % (1<<30)
    try:
        nets = _NETS['nets'] if arm == 'heads' else None
        planner = None
        if nets:
            wm, fc, actor, critic = nets
            planner = CellPlanner(wm, fc, actor, critic, CalibratedLimits.from_config(config),
                                   cost=CostCfg(protected_m=config.protected, segment_distance=True),
                                   horizon=config.plan_horizon,
                                   commit=dict(zip(('abs', 'rel', 'risk'), config.plan_commit)) if config.plan_commit else None)
        episode = run_episode(rec, config, seed=seed, deploy=condition != 'clean', nets=nets, planner=planner, kind=arm)
        return dict(episode['info'], arm=arm, condition=condition,
                    start_height=rec['start_left'][2], target_height=rec['target_left'][2])
    except Exception as exc:
        return dict(id=rec['id'], task=rec['task'], seed=seed, arm=arm, condition=condition,
                    terminal='error', error=repr(exc), start_height=rec['start_left'][2], target_height=rec['target_left'][2])


def wilson(successes, count):
    if not count:
        return [0., 1.]
    z, p = 1.95996398454, successes/count
    d = 1+z*z/count
    centre = (p+z*z/(2*count))/d
    half = z*np.sqrt(p*(1-p)/count+z*z/(4*count*count))/d
    return [float(centre-half), float(centre+half)]


def summarize(rows):
    counts = Counter(r['terminal'] for r in rows)
    n = len(rows)
    times = [x for r in rows for x in r.get('decision_ms', [])]
    return dict(n=n, outcomes=dict(counts), success_rate=counts['success']/max(n, 1),
                success_wilson95=wilson(counts['success'], n),
                workspace_violations=sum(r.get('out_of_domain', 0) for r in rows),
                fov_violation_samples=sum(r.get('out_of_fov', 0) for r in rows),
                margin_violation_samples=sum(r.get('margin_violation_samples', r.get('out_of_fov', 0)) for r in rows),
                actual_out_of_image_samples=sum(r.get('out_of_image_samples', 0) for r in rows),
                decision_p95_ms=float(np.percentile(times, 95)) if times else None,
                decision_samples=len(times),
                max_speed=max((r.get('max_speed', 0.) for r in rows), default=0.),
                max_acceleration=max((r.get('max_acceleration', 0.) for r in rows), default=0.))


def evaluate(config, checkpoint, split='test', workers=10, arms=('heads', 'cbf', 'oracle'), limit=0, tag=''):
    config.provenance()  # Snapshot evaluator source before starting worker processes.
    if limit < 0 or (split == 'test' and limit):
        raise ValueError('Development subsets must use validation with a positive limit')
    manifest_path = config.path/'tasks'/split/'scenes.json'
    manifest = json.loads(manifest_path.read_text())
    accepted = [r for r in manifest['scenes'] if r['accept']]
    recs = accepted
    if limit:
        rng = np.random.default_rng(713)
        recs = []
        for kind, n in [('reach', (limit+1)//2), ('hold', limit//2)]:
            group = [r for r in accepted if r['task'] == kind]
            recs.extend(group[int(i)] for i in rng.permutation(len(group))[:n])
    if not recs:
        raise RuntimeError('No accepted evaluation scenes')
    out = checkpoint.parent/('eval_'+split+tag)
    if limit:
        out = checkpoint.parent/('eval_'+split+'_development')/checkpoint.stem
    out.mkdir(parents=True, exist_ok=True)
    destination = out/'rows.jsonl'
    jobs = [(r, config, checkpoint, condition, arm) for condition in ('clean', 'deploy') for arm in arms for r in recs]
    rows = []
    with ProcessPoolExecutor(workers, mp_context=get_context('spawn')) as pool, destination.open('w') as stream:
        futures = [pool.submit(evaluate_job, job) for job in jobs]
        for future in as_completed(futures):
            row = future.result(); rows.append(row)
            stream.write(json.dumps(row)+'\n'); stream.flush()
            print(f'{len(rows)}/{len(jobs)} {row["arm"]} {row["condition"]} {row["terminal"]}', flush=True)
    verify_source(config.provenance()['source_sha256'])
    report = dict(protocol='calibrated_cell_deploy_v1', checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                  manifest_sha256=digest(manifest_path), selected_ids=[r['id'] for r in recs], subset_seed=713 if limit else None, split=split,
                  development=bool(limit) or split != 'test', config=config.provenance(),
                  proposed=manifest['summary']['proposed'], accepted=len(accepted), evaluated=len(recs),
                  groups={condition:{arm:summarize([r for r in rows if r['arm']==arm and r['condition']==condition])
                                     for arm in arms} for condition in ('clean', 'deploy')})
    report['by_task'] = {condition:{arm:{kind:summarize([r for r in rows if r['arm']==arm and r['condition']==condition
                                                       and r['task']==kind]) for kind in ('reach', 'hold')}
                                   for arm in arms} for condition in ('clean', 'deploy')}
    report['by_target_height'] = {condition:{arm:{str(z):summarize([r for r in rows if r['arm']==arm and r['condition']==condition
                                                                   and r.get('target_height')==z])
                                                for z in (.43, .48, .53)} for arm in arms} for condition in ('clean', 'deploy')}
    main = report['groups']['deploy'].get('heads')
    report['performance_acceptance'] = bool(main and not report['development'] and main['success_rate'] >= .95
        and main['outcomes'].get('clearance_violation', 0) == 0 and main['outcomes'].get('error', 0) == 0
        and main['workspace_violations'] == 0 and main['decision_p95_ms'] is not None and main['decision_p95_ms'] <= 80
        and main['max_speed'] <= config.speed+1e-8 and main['max_acceleration'] <= config.acceleration+1e-5)
    (out/'summary.json').write_text(json.dumps(report, indent=2))
    return report
