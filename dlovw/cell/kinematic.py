"""Known-robot endpoint propagation for learned rope imagination."""
import numpy as np
import torch
from .planner import CellPlanner


class KinematicRopeWorld:
    """Wrap a rope model while retaining its hidden dynamics and weight format."""
    def __init__(self, base, action_scale):
        self.base, self.action_scale = base, action_scale

    def __getattr__(self, name):
        return getattr(self.base, name)

    def img_step(self, state, action, meas=None, sample=True):
        latent = self.base.img_step(state[:3], action, meas, sample)
        return (*latent, state[3]+action*self.action_scale)

    def rope_out(self, state):
        raw = self.base.rope_out(state[:3])
        if len(state) == 3:
            return raw
        delta = state[3]-raw[...,117:123]
        fraction = torch.linspace(0,1,39,device=raw.device,dtype=raw.dtype)[:,None]
        shift = (1-fraction)*delta[...,:3,None].transpose(-1,-2)+fraction*delta[...,3:,None].transpose(-1,-2)
        curve = raw[...,:117].reshape(*raw.shape[:-1],39,3)+shift
        return torch.cat([curve.flatten(-2),state[3]],-1)


class KinematicCellPlanner(CellPlanner):
    def __init__(self, wm, fc, actor, critic, lim, *, action_scale, **kwargs):
        super().__init__(KinematicRopeWorld(wm, action_scale), fc, actor, critic, lim, **kwargs)

    def choose(self, state, beliefs, goal, grip, target, origin, previous, audit=None):
        known = torch.as_tensor((grip-np.tile(origin,2))*10,dtype=state[0].dtype,device=state[0].device)[None]
        return super().choose((*state[:3],known),beliefs,goal,grip,target,origin,previous,audit)
