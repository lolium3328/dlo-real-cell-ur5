"""Full GPU training with calibrated RGB data, causal predictor, and CPU collectors."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from multiprocessing import get_context
import pickle
import os
import time
import numpy as np
import torch

from .config import Config, VERSION, require_cuda, verify_source
from .observations import MEAS_SLICE, filter_states
from .data import batch
from .model import QualityWM
from .constraints import CalibratedLimits
from .checkpoint import collect_job
from ...imagination import actor_critic
from ...imagination.obstacle_model import ObstacleForecaster
from ...imagination.r10 import SetActorCritic, obstacle_inputs
from ...imagination.world_model import wm_loss
from ..cost import CostCfg, reward_v2


def train(config=Config(), seed=0, workers=10, smoke=False):
    device = require_cuda()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    actor_critic.PROGRESS = 'exp'
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config.freeze()
    name = f'seed_{seed}'+('_smoke' if smoke else '')
    out = config.path/'policy'/name
    out.mkdir(parents=True, exist_ok=True)
    source = config.provenance()['source_sha256']
    stamp = out.parent/'training_source.json'
    if stamp.exists():
        if json.loads(stamp.read_text())['source_sha256'] != source:
            raise ValueError('Policy seeds must share frozen training sources; use a new run directory')
    else:
        stamp.write_text(json.dumps(config.provenance(), indent=2))
    if (out/'completed.json').exists():
        return json.loads((out/'completed.json').read_text())
    files = sorted((config.path/'demos').glob('*.pkl'))
    replay = []
    for path in files:
        with path.open('rb') as handle:
            replay.append(pickle.load(handle))
    for path in sorted((out/'episodes').glob('*.pkl')):
        with path.open('rb') as handle:
            replay.append(pickle.load(handle))
    demos = [e for e in replay if e['info']['source'] == 'oracle' and e['info']['terminal'] == 'success']
    if not demos:
        raise RuntimeError('No successful calibrated RGB demonstrations')
    recs = [r for r in json.loads((config.path/'tasks/train/scenes.json').read_text())['scenes'] if r['accept']]
    wm_kw = dict(h=256, z=32, hid=256, ho=128)
    wm = QualityWM(**wm_kw).to(device)
    opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
    fc = ObstacleForecaster(feats=False).to(device)
    fc.load_state_dict(torch.load(config.path/'forecaster/forecaster.pt', map_location=device, weights_only=False)['state'])
    fc.requires_grad_(False)
    ac = SetActorCritic(wm.h_dim+wm.z_dim, device, heads=8)
    ac.reward_fn = reward_v2(CostCfg(protected_m=config.protected, segment_distance=True))
    ac.cont_scale, ac.truncate, ac.state_clip = config.cont_scale, config.grad_truncate, config.state_grad_clip
    lim = CalibratedLimits.from_config(config)
    wm_steps, bc_steps = (3, 3) if smoke else (config.wm_pretrain, config.bc_pretrain)
    rounds, updates, count = (2, 3, 2) if smoke else (config.rounds, config.train_steps, config.collect)
    size, starts = (2, 8) if smoke else (config.batch, config.starts)
    log = (out/'log.jsonl').open('a')
    sampled_groups = {}
    begin = time.monotonic()
    progress = dict(wm=0, bc=0, round=0)
    recovery = out/'recovery.pt'
    if recovery.exists():
        saved = torch.load(recovery, map_location=device, weights_only=False)
        if saved['config']['inputs'] != config.provenance()['inputs']:
            raise ValueError('Recovery inputs changed')
        if saved['config']['source_sha256'] != source:
            raise ValueError('Recovery training sources changed; use a new run directory')
        for key, module in [('wm', wm), ('actor', ac.actor), ('critic', ac.critic), ('target', ac.target)]:
            module.load_state_dict(saved[key])
        opt.load_state_dict(saved['opt_wm'])
        ac.opt_a.load_state_dict(saved['opt_a']); ac.opt_c.load_state_dict(saved['opt_c'])
        ac.scale, progress = saved['ac_scale'], saved['progress']
        rng.bit_generator.state = saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng'].cpu())
        torch.cuda.set_rng_state(saved['cuda_rng'].cpu())

    def emit(**row):
        row.update(seed=seed, elapsed_s=time.monotonic()-begin)
        log.write(json.dumps(row)+'\n'); log.flush()
        print(json.dumps(row), flush=True)

    def sample(pool, n=size):
        key = id(pool)
        if key not in sampled_groups or sampled_groups[key][0] != len(pool):
            groups = {kind:np.array([i for i,e in enumerate(pool) if e['info']['task'] == kind]) for kind in ('reach', 'hold')}
            if any(not len(group) for group in groups.values()):
                raise RuntimeError('Both reach and hold data are required')
            sampled_groups[key] = (len(pool), groups)
        groups = sampled_groups[key][1]
        indices = np.r_[rng.choice(groups['reach'], n//2), rng.choice(groups['hold'], n-n//2)]
        rng.shuffle(indices)
        return batch([pool[i] for i in indices], config.horizon, device)

    def world_update(b):
        loss, metrics, states = wm_loss(wm, b, rec_scale=100.)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite world model loss')
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), 100.)
        opt.step()
        return metrics, states

    def bc_batch(n=size):
        b = sample(demos, n)
        with torch.no_grad():
            states = wm.observe(b['obs'], b['act_prev'])[0]
            o = filter_states(fc, b['obs'][:, :, MEAS_SLICE][:, :, None])
            ox = obstacle_inputs(fc, o, wm.rope_out(states))
        rf = torch.cat(states[:2], -1)
        T = rf.shape[1]
        act_next = torch.cat([b['act_prev'][:, 1:], torch.zeros_like(b['act_prev'][:, :1])], 1)
        valid = torch.cat([b['mask'][:, 1:], torch.zeros_like(b['mask'][:, :1])], 1)
        return rf, b['goal'][:, None].expand(-1, T, -1), ox, torch.ones(*ox.shape[:3], device=device), act_next, valid

    def start_states(b, states):
        bi, ti = torch.nonzero(b['mask'] > 0, as_tuple=True)
        indices = torch.as_tensor(rng.choice(len(bi), min(starts, len(bi)), replace=False), device=device)
        bi, ti = bi[indices], ti[indices]
        with torch.no_grad():
            o_seq = filter_states(fc, b['obs'][:, :, MEAS_SLICE][:, :, None])
        return (tuple(x[bi, ti] for x in states), o_seq[bi, ti], torch.ones(len(bi), 1, device=device), b['goal'][bi],
                b['future'][bi, ti][:, :, None], b['future_meas'][bi, ti][:, :, None], b['act_prev'][bi, ti], b['origin'][bi])

    def save(round_):
        path = out/f'ckpt_r{round_}.pt'
        torch.save(dict(interface=VERSION, config=config.provenance(), wm=wm.state_dict(), wm_kw=wm_kw,
                        forecaster=fc.state_dict(), actor=ac.actor.state_dict(), critic=ac.critic.state_dict(),
                        heads=8, seed=seed, round=round_, cost='v2_segments', progress='exp'), path)
        return path

    def recoverable():
        verify_source(source)
        temporary = out/'recovery.tmp'
        torch.save(dict(wm=wm.state_dict(), actor=ac.actor.state_dict(), critic=ac.critic.state_dict(),
                        target=ac.target.state_dict(), opt_wm=opt.state_dict(), opt_a=ac.opt_a.state_dict(),
                        opt_c=ac.opt_c.state_dict(), ac_scale=ac.scale, progress=dict(progress), config=config.provenance(),
                        numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()), temporary)
        os.replace(temporary, recovery)

    emit(phase='start', gpu=torch.cuda.get_device_name(0), demos=len(replay), successful_demos=len(demos),
         scenes=len(recs), smoke=smoke, config=config.provenance())
    try:
        for step in range(progress['wm']+1, wm_steps+1):
            metrics, _ = world_update(sample(replay))
            progress['wm'] = step
            if step % 100 == 0 or step == wm_steps:
                emit(phase='wm_pretrain', step=step, **metrics)
                recoverable()
        for group in opt.param_groups:
            group['lr'] = 1e-4
        for step in range(progress['bc']+1, bc_steps+1):
            loss = ac.bc_step(*bc_batch())
            progress['bc'] = step
            if step % 100 == 0 or step == bc_steps:
                emit(phase='bc_pretrain', step=step, loss=loss)
                recoverable()
        ckpt = save(progress['round'])
        with ProcessPoolExecutor(workers, mp_context=get_context('spawn')) as pool:
            for round_ in range(progress['round']+1, rounds+1):
                scene_groups = {kind:[r for r in recs if r['task'] == kind] for kind in ('reach', 'hold')}
                jobs = []
                for i in range(count):
                    group = scene_groups['reach' if i % 2 == 0 else 'hold']
                    jobs.append((group[int(rng.integers(len(group)))], config, ckpt, int(rng.integers(1<<30))))
                futures = [pool.submit(collect_job, j) for j in jobs]
                metrics = []
                for step in range(updates):
                    b = sample(replay)
                    m, states = world_update(b)
                    m.update(ac.step(wm, fc, *start_states(b, states), config.horizon, lim,
                                     bc=bc_batch(), w_bc=.85**round_))
                    metrics.append(m)
                fresh, errors = [], []
                for future in as_completed(futures):
                    e = future.result()
                    if 'error' in e:
                        errors.append(e)
                    else:
                        fresh.append(e)
                attempts = count
                for retry in range(2):
                    if len(fresh) >= count:
                        break
                    replacement = []
                    for i in range(count-len(fresh)):
                        group = scene_groups['reach' if i % 2 == 0 else 'hold']
                        replacement.append((group[int(rng.integers(len(group)))], config, ckpt, int(rng.integers(1<<30))))
                    attempts += len(replacement)
                    for future in as_completed([pool.submit(collect_job, j) for j in replacement]):
                        e = future.result()
                        if 'error' in e:
                            errors.append(e)
                        else:
                            fresh.append(e)
                if len(fresh) < count:
                    raise RuntimeError(f'Only {len(fresh)}/{count} valid collection episodes after {attempts} attempts: {errors[:3]}')
                fresh.sort(key=lambda e:e['info']['seed'])
                replay.extend(fresh)
                episodes = out/'episodes'; episodes.mkdir(exist_ok=True)
                for i, episode in enumerate(fresh):
                    with (episodes/f'r{round_:02d}_{i:03d}.pkl').open('wb') as handle:
                        pickle.dump(episode, handle)
                progress['round'] = round_
                ckpt = save(round_)
                recoverable()
                emit(phase='round', round=round_, replay=len(replay), collected=len(fresh), collection_attempts=attempts, errors=errors,
                     outcomes=dict(Counter(e['info']['terminal'] for e in fresh)),
                     actor_grad_max=float(np.max([m['actor_grad'] for m in metrics])),
                     actor_grad_p50=float(np.median([m['actor_grad'] for m in metrics])),
                     **{key:float(np.mean([m[key] for m in metrics if m[key] is not None]))
                        for key in metrics[0] if metrics[0][key] is not None})
        (out/'completed.json').write_text(json.dumps(dict(rounds=rounds, checkpoint=str(ckpt), smoke=smoke,
                                                         elapsed_s=time.monotonic()-begin, config=config.provenance()), indent=2))
        emit(phase='completed', checkpoint=str(ckpt))
    finally:
        log.close()
