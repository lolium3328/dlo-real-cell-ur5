"""Quality-conditioned RSSM with unchanged 123-D physical reconstruction target."""
import torch

from ...imagination.world_model import FactoredWM, mlp
from ...imagination.episodes import ROPE
from .observations import OBS, MEAS_SLICE, QUALITY_SLICE


class QualityWM(FactoredWM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        hid = kwargs.get('hid', 256)
        self.enc = mlp(ROPE+156, hid, hid)

    def obs_step(self, state, a, obs):
        if obs.shape[-1] != OBS:
            raise ValueError(f'Expected calibrated observation {OBS}, got {obs.shape[-1]}')
        h, z, o = state
        h = self.gru(self.inp(torch.cat([z, a], -1)), h)
        pm, ps = self._stats(self.prior(h))
        encoded = self.enc(torch.cat([obs[:, :ROPE], obs[:, QUALITY_SLICE]], -1))
        qm, qs = self._stats(self.post(torch.cat([h, encoded], -1)))
        return (h, qm+qs*torch.randn_like(qs), self._obstacle_step(o, obs[:, MEAS_SLICE])), (pm, ps), (qm, qs)
