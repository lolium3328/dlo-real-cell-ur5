"""Exact differentiable segment distance with sqrt only after nearest-segment selection."""
import torch


@torch.jit.script
def segment_distances(points: torch.Tensor, centres: torch.Tensor, protected: float):
    # Each dot product has three terms. Avoid allocating a coordinate axis
    # for every centre and segment pair and then reducing over that axis.
    ax, ay, az = points[:, None, :-1].unbind(-1)
    vx, vy, vz = (points[:, 1:]-points[:, :-1])[:, None].unbind(-1)
    cx, cy, cz = centres.unbind(-1)
    dx, dy, dz = cx[..., None]-ax, cy[..., None]-ay, cz[..., None]-az
    length2 = (vx*vx + vy*vy + vz*vz).clamp_min(1e-15)
    dot = dx*vx + dy*vy + dz*vz
    t = (dot/length2).clamp(0., 1.)
    squared = (dx*dx + dy*dy + dz*dz)-2*t*dot+t*t*length2
    return squared.min(-1).values.clamp_min(1e-15).sqrt()-protected
