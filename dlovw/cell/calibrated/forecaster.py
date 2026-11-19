"""GPU-only standalone forecaster on causal kinematic readings of new-cell motions."""
from concurrent.futures import ProcessPoolExecutor
import json
import pickle
import time
import fcntl
from multiprocessing import get_context
import numpy as np
import torch

from ...motion import track
from ...imagination.episodes import encode_meas, POS, VEL
from ...imagination.obstacle_model import ObstacleForecaster, forecaster_loss
from .observations import BallReadings
from .config import Config, require_cuda, right_rotation, tcp_from_sphere
from .tasks import _motion


def _sequence(job):
    rec, seed, config = job
    rng = np.random.default_rng(seed)
    motion, _ = _motion(rng, rec['task'], np.array(rec['motion']['center']), np.array(rec['motion']['direction']),
                        config.ball_speed_scale)
    count = round(min(12/config.ball_speed_scale, config.max_time_s)/config.dt)
    tr = track(motion, count+config.horizon*config.decision_steps+config.decision_steps, config.dt)
    origin = (np.asarray(rec['target_left'])+config.material['fixed_end_m'])*.5
    sensor = BallReadings(seed, True, config)
    meas, truths = [], []
    for k, centre in enumerate(tr[:count+1]):
        r = sensor.push(k*config.dt, tcp_from_sphere(centre, right_rotation(config), config), right_rotation(config))
        if k % config.decision_steps == 0:
            meas.append(encode_meas(None if r is None else r[:2], origin))
            truths.append((centre-origin)*POS)
    return dict(meas=np.array(meas, np.float32), truth=np.array(truths, np.float32),
                track=((tr[::config.decision_steps]-origin)*POS).astype(np.float32), seed=seed)


def _batch(rows, device):
    return {k:torch.as_tensor(np.stack([r[k] for r in rows]), device=device) for k in ('meas', 'truth', 'track')} | {
        'feats':torch.zeros(len(rows), len(rows[0]['meas']), 16, device=device),
        'mask':torch.ones(len(rows), len(rows[0]['meas']), device=device)}


def report_predictions(model, rows, device, config):
    """Do not let long parked suffixes hide moving/future-moving forecast errors."""
    results = {k:{g:[] for g in ('all', 'moving_now', 'will_move', 'stationary_now', 'current_measured')} for k in (5, 10, 15)}
    baseline = {k:[] for k in results}
    with torch.no_grad():
        for start in range(0, len(rows), 32):
            b = _batch(rows[start:start+32], device)
            B, T = b['meas'].shape[:2]
            o, states = model.initial(B, device), []
            for t in range(T):
                o = model.step(o, b['meas'][:, t], None)
                states.append(o)
            o = torch.stack(states, 1).reshape(B*T, -1)
            movement = torch.zeros(B, T, device=device, dtype=torch.bool)
            movement[:, 1:] = (b['truth'][:, 1:]-b['truth'][:, :-1]).norm(dim=-1) > .001
            for k in range(1, 16):
                o = model.step(o, torch.zeros(B*T, 7, device=device), None)
                if k not in results:
                    continue
                w, mu, sd = model.dist(o)
                prediction = (w[..., None]*mu).sum(-2).reshape(B, T, 3)
                target = b['track'][:, torch.arange(T, device=device)+k]
                error = (prediction-target).norm(dim=-1)/POS*1000
                future_moves = (target-b['truth']).norm(dim=-1) > .01*POS
                measured = b['meas'][..., 6] > 0
                groups = dict(all=torch.ones_like(movement), moving_now=movement, will_move=future_moves,
                              stationary_now=~movement, current_measured=measured)
                for name, mask in groups.items():
                    results[k][name].extend(error[mask].cpu().tolist())
                cv = b['meas'][..., :3]+b['meas'][..., 3:6]/VEL*POS*(k*config.decision_dt)
                baseline[k].extend(((cv-target).norm(dim=-1)/POS*1000)[measured].cpu().tolist())
    def stats(values):
        return dict(n=len(values), mean_mm=float(np.mean(values)), median_mm=float(np.median(values)),
                    p95_mm=float(np.percentile(values, 95))) if values else dict(n=0)
    return dict(protocol='causal_forecaster_phase_validation_v1', sequences=len(rows), independent_frames=False,
                horizon={str(k):dict(seconds=k*config.decision_dt, gru={g:stats(v) for g,v in groups.items()},
                                    constant_velocity_current_measured=stats(baseline[k])) for k,groups in results.items()},
                config=config.provenance())


def _train(config=Config(), n=6000, steps=4000, workers=12):
    device = require_cuda()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    config.freeze()
    out = config.path/'forecaster'; out.mkdir(exist_ok=True)
    recs = [r for r in json.loads((config.path/'tasks/train/scenes.json').read_text())['scenes'] if r['accept']]
    if not recs:
        raise RuntimeError('No training scenes for forecaster')
    path = out/f'data_{n}.pkl'
    if path.exists():
        with path.open('rb') as handle:
            rows = pickle.load(handle)
    else:
        rng = np.random.default_rng(config.seed)
        jobs = [(recs[int(rng.integers(len(recs)))], int(rng.integers(1<<31)), config) for _ in range(n)]
        with ProcessPoolExecutor(workers, mp_context=get_context('spawn')) as pool:
            rows = list(pool.map(_sequence, jobs, chunksize=16))
        with path.open('wb') as handle:
            pickle.dump(rows, handle)
    valid, data = rows[:max(1, n//10)], rows[max(1, n//10):]
    torch.manual_seed(0)
    rng, model = np.random.default_rng(0), ObstacleForecaster(feats=False).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    checkpoint = out/'forecaster.pt'
    first = 1
    if checkpoint.exists():
        previous = torch.load(checkpoint, map_location=device, weights_only=False)
        if previous['config']['inputs'] != config.provenance()['inputs']:
            raise ValueError('Forecaster checkpoint input mismatch')
        if previous['steps'] >= steps:
            return checkpoint
        model.load_state_dict(previous['state'])
        if 'optimizer' in previous:
            opt.load_state_dict(previous['optimizer'])
            rng.bit_generator.state = previous['numpy_rng']
            torch.set_rng_state(previous['torch_rng'].cpu())
            torch.cuda.set_rng_state(previous['cuda_rng'].cpu())
        first = previous['steps']+1
    begin = time.monotonic()
    with (out/'log.jsonl').open('a') as stream:
        for step in range(first, steps+1):
            b = _batch([data[i] for i in rng.integers(len(data), size=64)], device)
            loss, metrics = forecaster_loss(model, b, H=config.horizon)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite forecaster loss')
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.)
            opt.step()
            if step % 100 == 0 or step == steps:
                with torch.no_grad():
                    _, val = forecaster_loss(model, _batch(valid[:64], device), H=config.horizon)
                row = dict(step=step, train=metrics, validation=val, wall_s=time.monotonic()-begin,
                           device=str(device), gpu=torch.cuda.get_device_name(0))
                stream.write(json.dumps(row)+'\n'); stream.flush()
                print(json.dumps(row), flush=True)
                torch.save(dict(state=model.state_dict(), config=config.provenance(), steps=step, valid_seeds=[r['seed'] for r in valid],
                                optimizer=opt.state_dict(), numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                                cuda_rng=torch.cuda.get_rng_state()),
                           out/'forecaster.pt')
    report = report_predictions(model, valid, device, config)
    (out/'validation_report.json').write_text(json.dumps(report, indent=2))
    return out/'forecaster.pt'


def train(config=Config(), n=6000, steps=4000, workers=12):
    require_cuda()
    out = config.path/'forecaster'; out.mkdir(exist_ok=True)
    with (out/'training.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _train(config, n, steps, workers)
