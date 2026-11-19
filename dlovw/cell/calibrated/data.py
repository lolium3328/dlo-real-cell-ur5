"""RGB episodes and padded training batches; shared by runtime and training."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import pickle
import time
import fcntl
import os
import numpy as np
import torch

from ...imagination.episodes import ROPE, POS, VEL, encode_meas, encode_obs, future_index
from ...imagination.r10 import obstacle_inputs
from .config import Config
from .observations import OBS, MEAS_SLICE
from .session import Session
from .tasks import action_for_plan
from .constraints import clip_norm


def run_episode(rec, config=Config(), *, seed=0, deploy=False, nets=None, planner=None, noise=0., kind='oracle'):
    torch.manual_seed(seed)
    s = Session(rec, config, seed, deploy, visual=not config.headless, headless=config.headless)
    obs, truths, robot, acts, obst, balls = [], [], [], [], [], []
    state, belief = None, None
    a, begin = np.zeros(6), time.monotonic()
    decision_ms = []
    choices = []
    if nets:
        wm, fc, actor, _ = nets
        state, belief = wm.initial(1, 'cpu'), fc.initial(1)
    rng = np.random.default_rng(seed+100)
    head = int(rng.integers(8))
    previous_grip = None

    def record():
        grip = np.r_[s.rope.left(), s.rope.fixed]
        obs.append(s.observation())
        truths.append(encode_obs(s.rope.curve(), s.rope.left(), s.rope.fixed, None, s.origin)[:ROPE])
        robot.append(grip)
        # Only the supervision sees true velocity, computed from current/past track.
        velocity = np.zeros(3) if s.n == 0 else (s.track[s.n]-s.track[s.n-1])/config.dt
        obst.append(np.r_[(s.sphere()-s.origin)*POS, velocity*VEL])
        balls.append(s.sphere().copy())
        return grip

    try:
        with torch.no_grad():
            while s.terminal == 'running':
                if s.n % config.decision_steps == 0:
                    grip = record()
                    actual = np.zeros(6) if previous_grip is None else (grip-previous_grip)/(config.cap*config.decision_steps)
                    if previous_grip is not None:
                        acts.append(actual)
                    previous_grip = grip
                    if nets:
                        started = time.monotonic()
                        ap = torch.tensor(actual, dtype=torch.float32)[None]
                        state = wm.obs_step(state, ap, torch.tensor(obs[-1])[None])[0]
                        belief = fc.step(belief, torch.tensor(obs[-1][MEAS_SLICE])[None], None)
                        if planner:
                            name, a, scores = planner.choose(state, [belief], torch.tensor(s.goal)[None], grip,
                                                             np.r_[rec['target_left'], s.rope.fixed], s.origin, ap)
                            choices.append(dict(t=s.t, head=name, scores=np.asarray(scores).tolist(),
                                                survival=np.asarray(planner.last['survival']).tolist()))
                        else:
                            ox = obstacle_inputs(fc, belief[:, None], wm.rope_out(state))
                            a = actor(torch.cat(state[:2], -1), torch.tensor(s.goal)[None], ox, torch.ones(1, 1))[0, head].numpy()
                            a = np.clip(a+noise*rng.normal(size=6), -1, 1)
                        decision_ms.append((time.monotonic()-started)*1000)
                if kind == 'oracle':
                    command = action_for_plan(s, rec['oracle_plan'])
                elif kind == 'cbf':
                    command = clip_norm(np.asarray(rec['target_left'])-s.rope.left(), config.cap)
                else:
                    command = s.limits.project(a, np.r_[s.rope.left(), s.rope.fixed])[:3]*config.cap
                s.step(command, cbf=(kind != 'oracle'))
            # Record the final partial decision with its actual interval duration.
            grip = record()
            last_steps = s.n % config.decision_steps or config.decision_steps
            acts.append((grip-previous_grip)/(config.cap*last_steps))
        # Replay only causal robot measurements for supervision of the predictor.
        from .observations import BallReadings
        from .config import right_rotation, tcp_from_sphere
        sensor = BallReadings(seed+1, deploy, config)
        future_meas = []
        for k, centre in enumerate(s.track):
            reading = sensor.push(k*config.dt, tcp_from_sphere(centre, right_rotation(config), config), right_rotation(config))
            if k % config.decision_steps == 0:
                future_meas.append(encode_meas(None if reading is None else reading[:2], s.origin))
        result = dict(**s.result(), source=kind, deploy=deploy, wall_s=time.monotonic()-begin,
                      decision_ms=decision_ms, choices=choices, seed=seed,
                      dynamics=s.rope.randomization,
                      head=head if kind == 'actor' else None)
        return dict(obs=np.stack(obs), rope_truth=np.stack(truths), act=np.asarray(acts, np.float32),
                    truth=np.asarray(obst, np.float32), goal=s.goal, origin=s.origin.astype(np.float32),
                    final_partial=bool(s.n % config.decision_steps),
                    track=((s.track[::config.decision_steps]-s.origin)*POS).astype(np.float32),
                    track_meas=np.asarray(future_meas, np.float32), info=result)
    finally:
        s.close()


def batch(episodes, horizon, device):
    B, T = len(episodes), max(len(e['obs']) for e in episodes)
    out = dict(obs=np.zeros((B, T, OBS), np.float32), rope_truth=np.zeros((B, T, ROPE), np.float32),
               act_prev=np.zeros((B, T, 6), np.float32), truth=np.zeros((B, T, 6), np.float32),
               mask=np.zeros((B, T), np.float32), goal=np.stack([e['goal'] for e in episodes]),
               future=np.zeros((B, T, horizon, 3), np.float32), future_meas=np.zeros((B, T, horizon, 7), np.float32),
               origin=np.stack([e['origin'] for e in episodes]))
    for b, e in enumerate(episodes):
        n = len(e['obs'])
        for key in ('obs', 'rope_truth', 'truth'):
            out[key][b, :n], out[key][b, n:] = e[key], e[key][-1]
        out['act_prev'][b, 1:n] = e['act'][:n-1]
        out['mask'][b, :n] = 1.
        if e.get('final_partial', False):
            out['mask'][b, n-1] = 0.
        idx = future_index(T, horizon, len(e['track']))
        out['future'][b], out['future_meas'][b] = e['track'][idx], e['track_meas'][idx]
    return {k: torch.as_tensor(v, device=device) for k, v in out.items()}


def _demo(job):
    rec, config, seed, deploy, kind = job
    try:
        return run_episode(rec, config, seed=seed, deploy=deploy, kind=kind)
    except Exception as exc:
        return dict(error=repr(exc), id=rec['id'], source=kind, seed=seed)


def _collect(config=Config(), workers=12):
    manifest = json.loads((config.path/'tasks/train/scenes.json').read_text())
    recs = [r for r in manifest['scenes'] if r['accept']]
    if not recs:
        raise RuntimeError('No accepted training scenes')
    rng = np.random.default_rng(config.seed)
    jobs = [(r, config, int(rng.integers(1<<30)), bool(rng.random() < config.deploy_fraction), kind)
            for r in recs for kind in ('oracle', 'oracle', 'cbf')]
    out = config.path/'demos'; out.mkdir(exist_ok=True)
    pending, damaged = [], []
    for i, job in enumerate(jobs):
        path = out/f'{i:06d}.pkl'
        if not path.exists():
            pending.append((i, job)); continue
        try:
            with path.open('rb') as handle:
                cached = pickle.load(handle)
            if cached['info']['seed'] != job[2] or cached['obs'].shape[-1] != OBS:
                raise ValueError('Cached episode identity differs')
        except (EOFError, ValueError, KeyError, pickle.UnpicklingError) as exc:
            archive = config.path/'development'/'damaged_demo_cache'
            archive.mkdir(parents=True, exist_ok=True)
            path.rename(archive/f'{i:06d}_{time.time_ns()}.pkl')
            damaged.append(dict(index=i, error=repr(exc), status='damaged_cache_retried'))
            pending.append((i, job))
    failures = []
    with ProcessPoolExecutor(workers) as pool, (out/'index.jsonl').open('a') as stream:
        for row in damaged:
            stream.write(json.dumps(row)+'\n'); stream.flush()
        futures = {pool.submit(_demo, j):i for i,j in pending}
        for f in as_completed(futures):
            index, ep = futures[f], f.result()
            path = out/f'{index:06d}.pkl'
            if 'error' in ep:
                row = dict(index=index, **ep)
                failures.append(row)
            else:
                temporary = path.with_suffix('.tmp')
                with temporary.open('wb') as handle:
                    pickle.dump(ep, handle)
                os.replace(temporary, path)
                row = dict(index=index, path=str(path), **ep['info'])
            stream.write(json.dumps(row)+'\n'); stream.flush()
            print(f'demo {index}: {row.get("terminal", row.get("error"))}', flush=True)
    if failures:
        raise RuntimeError(f'{len(failures)} demo jobs failed; retry resumes existing files and preserves index errors')
    record = dict(jobs=len(jobs), completed=len(list(out.glob('*.pkl'))), config=config.provenance())
    (out/'completed.json').write_text(json.dumps(record, indent=2))
    return record


def collect(config=Config(), workers=12):
    """Cooperate with the pipeline while allowing independent RGB collection to overlap."""
    config.freeze()
    out = config.path/'demos'; out.mkdir(exist_ok=True)
    with (out/'collection.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if (out/'completed.json').exists():
            record = json.loads((out/'completed.json').read_text())
            if record['config']['inputs'] != config.provenance()['inputs']:
                raise ValueError('Completed demo inputs differ')
            print(f'Demos already completed: {record["completed"]}', flush=True)
            return record
        return _collect(config, workers)
