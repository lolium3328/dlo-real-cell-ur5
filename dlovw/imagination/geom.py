"""Obstacle geometry for the G system (docs/notes/geom_tracker_prereg_20261106.md, section 2).

- Signed distance from points to an obstacle of known shape at a given position: sphere, capsule, box, convex
  polytope (half-spaces; exact inside, a lower bound outside, so it errs towards "closer").
- The rope is a chain of capsules: its points, densified, with ROPE_RADIUS_M; the clearance that counts is
  sdf - ROPE_RADIUS_M - CLEARANCE_M (for the 40 mm sphere this is |p - c| - PROTECTED_M, as everywhere else).
- TrajectoryRisk: crash probability of candidate rope rollouts under whole-trajectory obstacle particles. Each
  particle is one future of every obstacle, so a candidate either meets it or not over the whole horizon; the
  steps are not treated as independent (the per-step product of marginal risks is).
Shapes do not rotate (the scripted motions translate), so a shape's orientation is fixed and known.
Positions in metres; the caller picks the frame (the planner uses the episode frame, origin subtracted).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

ROPE_RADIUS_M, CLEARANCE_M = 0.010, 0.005
CONT_SCALE_M = 0.003                     # as actor_critic: soft crash sigmoid(-d / 3 mm)


@dataclass(frozen=True)
class Shape:
    kind: str                            # "sphere" | "capsule" | "box" | "convex"
    radius: float = 0.0                  # sphere, capsule
    half_length: float = 0.0             # capsule: half the length of its core segment
    axis: tuple = (1.0, 0.0, 0.0)        # capsule: direction of the core segment
    half_extents: tuple = (0.0, 0.0, 0.0)            # box
    rotation: tuple | None = None        # box / convex: 3x3 body-to-world, row-major (None = identity)
    planes: tuple | None = None          # convex: ((n_x, n_y, n_z, b), ...) body frame, inside where n.x <= b

    def bounding_radius(self):
        """Radius of the smallest sphere about the shape's position that contains it (the L arms' stand-in)."""
        if self.kind == "sphere":
            return self.radius
        if self.kind == "capsule":
            return self.half_length + self.radius
        if self.kind == "box":
            return float(np.linalg.norm(self.half_extents))
        raise ValueError("bounding radius of a convex polytope: pass its vertices to the caller")


def sphere(radius):
    return Shape("sphere", radius=float(radius))


def capsule(radius, half_length, axis):
    a = np.asarray(axis, float)
    return Shape("capsule", radius=float(radius), half_length=float(half_length), axis=tuple(a / np.linalg.norm(a)))


def box(half_extents, rotation=None):
    return Shape("box", half_extents=tuple(map(float, half_extents)),
                 rotation=None if rotation is None else tuple(map(float, np.asarray(rotation, float).ravel())))


def convex(planes, rotation=None):
    """planes: (F, 4) rows (n, b) with |n| = 1 after normalisation; the polytope is {x : n.x <= b for all rows}."""
    P = np.asarray(planes, float)
    norm = np.linalg.norm(P[:, :3], axis=1, keepdims=True)
    P = P / norm
    return Shape("convex", planes=tuple(map(tuple, P)),
                 rotation=None if rotation is None else tuple(map(float, np.asarray(rotation, float).ravel())))


def _to_body(shape, d):
    if shape.rotation is None:
        return d
    R = torch.as_tensor(shape.rotation, dtype=d.dtype, device=d.device).reshape(3, 3)
    return d @ R                                       # rows of d times R = (R^T d^T)^T: world offset -> body frame


def sdf(shape, d):
    """Signed distance of points at offsets d (..., 3) from the shape's position (torch, metres) -> (...)."""
    if shape.kind == "sphere":
        return d.norm(dim=-1) - shape.radius
    if shape.kind == "capsule":
        u = torch.as_tensor(shape.axis, dtype=d.dtype, device=d.device)
        t = (d * u).sum(-1, keepdim=True).clamp(-shape.half_length, shape.half_length)
        return (d - t * u).norm(dim=-1) - shape.radius
    x = _to_body(shape, d)
    if shape.kind == "box":
        e = torch.as_tensor(shape.half_extents, dtype=d.dtype, device=d.device)
        q = x.abs() - e
        outside = q.clamp_min(0.0).norm(dim=-1)
        inside = q.max(-1).values.clamp_max(0.0)
        return outside + inside
    if shape.kind == "convex":
        P = torch.as_tensor(shape.planes, dtype=d.dtype, device=d.device)
        return (x @ P[:, :3].T - P[:, 3]).max(-1).values
    raise ValueError(f"unknown shape {shape.kind}")


def densify(pts, k=2):
    """Rope points (..., N, 3) -> (..., (N - 1) * k + 1, 3) with k - 1 points added inside every segment."""
    a, b = pts[..., :-1, :], pts[..., 1:, :]
    f = torch.arange(k, dtype=pts.dtype, device=pts.device) / k
    mid = a[..., :, None, :] + (b - a)[..., :, None, :] * f[:, None]
    return torch.cat([mid.reshape(*pts.shape[:-2], -1, 3), pts[..., -1:, :]], -2)


def clearance(shape, rope_pts, centers):
    """Clearance between the rope and the shape at each centre: rope_pts (B, P, 3), centers (B or 1, M, 3)
    -> (B, M): min over the rope of sdf - ROPE_RADIUS_M - CLEARANCE_M (negative = violation)."""
    d = rope_pts[:, None, :, :] - centers[:, :, None, :]
    return sdf(shape, d).min(-1).values - ROPE_RADIUS_M - CLEARANCE_M


class TrajectoryRisk:
    """Survival of B candidate rollouts under M whole-trajectory particles of J obstacles.

    parts: list over obstacles of (M, H, 3) particle positions (the same M futures for every candidate) or (B, M, H, 3)
    (every row its own futures); shapes: list over obstacles; mask (B, J) or None: obstacles a row does not have.
    step(i, rope_pts) after the i-th imagined decision returns (p_crash, margin), each (B,): the probability of
    crashing at this step given not crashed before (so prod(1 - p_crash) is the survival, as the planners accumulate
    it) and the margin penalty expected over the surviving particles (plus `penetration` * depth, the actor-critic
    reward's extra term)."""

    def __init__(self, parts, shapes, B, scale=CONT_SCALE_M, k=2, mask=None, penetration=0.0):
        self.parts, self.shapes, self.scale, self.k, self.mask, self.pen = parts, shapes, scale, k, mask, penetration
        M = parts[0].shape[-3]
        self.surv = torch.ones(B, M, dtype=parts[0].dtype, device=parts[0].device)

    def step(self, i, rope_pts):
        pts = densify(rope_pts, self.k)
        keep, margin = torch.ones_like(self.surv), torch.zeros_like(self.surv)
        for j, (P, shape) in enumerate(zip(self.parts, self.shapes)):
            d = clearance(shape, pts, P[None, :, i] if P.dim() == 3 else P[:, :, i])     # (B, M)
            if self.mask is not None:
                d = torch.where(self.mask[:, j:j + 1] > 0, d, torch.ones_like(d))
            keep = keep * torch.sigmoid(d / self.scale)
            margin = margin + 4.0 * torch.sigmoid((0.010 - d) / 0.004)
            if self.pen:
                margin = margin + self.pen * torch.relu(-d)
        before = self.surv.mean(-1)
        w = self.surv / self.surv.sum(-1, keepdim=True).clamp_min(1e-12)
        self.surv = self.surv * keep
        after = self.surv.mean(-1)
        p_crash = (1.0 - after / before.clamp_min(1e-12)).clamp(0.0, 1.0)
        return p_crash, (w * margin).sum(-1)

    def survival(self):
        return self.surv.mean(-1)
