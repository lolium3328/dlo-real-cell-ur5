"""Version checked CPU inference bundles for collectors and deployment."""
import torch
from .model import QualityWM
from .config import VERSION, verify_source
from ...imagination.r10 import SetActor, SetCritic
from ...imagination.obstacle_model import ObstacleForecaster
from ...imagination import actor_critic


def load(path):
    actor_critic.PROGRESS = 'exp'
    st = torch.load(path, map_location='cpu', weights_only=False)
    if st['interface'] != VERSION:
        raise ValueError('Unsupported calibrated policy interface')
    wm = QualityWM(**st['wm_kw'])
    fc = ObstacleForecaster(feats=False)
    actor, critic = SetActor(wm.h_dim+wm.z_dim, st['heads']), SetCritic(wm.h_dim+wm.z_dim)
    for name, module in [('wm', wm), ('forecaster', fc), ('actor', actor), ('critic', critic)]:
        module.load_state_dict(st[name])
        module.eval()
    return (wm, fc, actor, critic), st


_CACHE = {}


def collect_job(job):
    import numpy as np
    from .data import run_episode
    rec, config, path, seed = job
    torch.set_num_threads(1)
    if _CACHE.get('path') != str(path):
        verify_source(config.provenance()['source_sha256'])
        nets, _ = load(path)
        _CACHE.clear()
        _CACHE.update(path=str(path), nets=nets)
    deploy = bool(np.random.default_rng(seed+991).random() < config.deploy_fraction)
    try:
        return run_episode(rec, config, seed=seed, deploy=deploy, nets=_CACHE['nets'], noise=.3, kind='actor')
    except Exception as exc:
        return dict(error=repr(exc), id=rec['id'], seed=seed)
