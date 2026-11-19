"""Exact differentiable segment distance with sqrt only after nearest-segment selection."""
import torch


@torch.jit.script
def segment_distances(points: torch.Tensor, centres: torch.Tensor, protected: float):
    a = points[:, None, :-1]
    ab = (points[:, 1:]-points[:, :-1])[:, None]
    diff = centres[:, :, None]-a
    length2 = (ab*ab).sum(-1).clamp_min(1e-15)
    dot = (diff*ab).sum(-1)
    t = (dot/length2).clamp(0., 1.)
    squared = (diff*diff).sum(-1)-2*t*dot+t*t*length2
    return squared.min(-1).values.clamp_min(1e-15).sqrt()-protected
