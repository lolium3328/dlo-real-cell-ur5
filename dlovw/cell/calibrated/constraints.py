"""Matched differentiable/runtime constraints for the non-rectangular task domain."""
from dataclasses import dataclass
import numpy as np
import torch

from .config import Config
from .actuator_solver import refine
from ..cbf import hildreth


def bounds_at(z, slices):
    zs = [r['z_m'] for r in slices]
    return (np.array([np.interp(z, zs, [r[k] for r in slices]) for k in ('x_low_m', 'y_low_m')]),
            np.array([np.interp(z, zs, [r[k] for r in slices]) for k in ('x_high_m', 'y_high_m')]))


def in_domain(p, slices, tol=1e-8):
    p = np.asarray(p, float)
    lo, hi = bounds_at(p[2], slices)
    return bool(np.isfinite(p).all() and slices[0]['z_m']-tol <= p[2] <= slices[-1]['z_m']+tol
                and np.all(p[:2] >= lo-tol) and np.all(p[:2] <= hi+tol))


def segment_ok(a, b, slices, tol=1e-8):
    if not in_domain(a, slices, tol) or not in_domain(b, slices, tol):
        return False
    for row in slices[1:-1]:
        if abs(b[2]-a[2]) > 1e-12:
            t = (row['z_m']-a[2])/(b[2]-a[2])
            if 0 < t < 1 and not in_domain(a+t*(b-a), slices, tol):
                return False
    return True


def clip_norm(x, radius):
    return x * min(1., radius/max(float(np.linalg.norm(x)), 1e-12))


@dataclass(frozen=True)
class CalibratedLimits:
    ws_low: tuple
    ws_high: tuple
    stretch_max: float
    cap: float
    slices: tuple
    fixed_right: bool = True
    acceleration: float = .15
    dt: float = .008
    speed: float = .05

    def limit_change_torch(self, a, previous, steps):
        delta = a-previous
        n = delta[:, :3].norm(dim=-1, keepdim=True).clamp_min(1e-12)
        # dt=.008, a_max=.15, v_max=.05; normalized delta over a decision.
        scale = torch.clamp((self.acceleration*self.dt*steps/self.speed)/n, max=1.)
        return previous+delta*scale

    @classmethod
    def from_config(cls, config=Config()):
        p, rows = config.material, config.slices
        fixed = tuple(p['fixed_end_m'])
        lo = (min(r['x_low_m'] for r in rows), min(r['y_low_m'] for r in rows), rows[0]['z_m'])+fixed
        hi = (max(r['x_high_m'] for r in rows), max(r['y_high_m'] for r in rows), rows[-1]['z_m'])+fixed
        return cls(lo, hi, .97*p['length_m'], config.cap, tuple(tuple(r[k] for k in
                   ('z_m', 'x_low_m', 'x_high_m', 'y_low_m', 'y_high_m')) for r in rows),
                   acceleration=config.acceleration, dt=config.dt, speed=config.speed)

    def project_torch(self, a, grip, steps=1, margin=1e-5):
        rows = torch.as_tensor(self.slices, dtype=a.dtype, device=a.device)
        p = grip[:, :3]
        raw = p + a[:, :3]*self.cap*steps
        z = raw[:, 2].clamp(rows[0, 0]+margin, rows[-1, 0]-margin)
        index = torch.searchsorted(rows[:, 0].contiguous(), z.contiguous()).clamp(1, len(rows)-1)
        r0, r1 = rows[index-1], rows[index]
        w = ((z-r0[:, 0])/(r1[:, 0]-r0[:, 0]))[:, None]
        bound = r0[:, 1:]+w*(r1[:, 1:]-r0[:, 1:])
        x = raw[:, 0].maximum(bound[:, 0]+margin).minimum(bound[:, 1]-margin)
        y = raw[:, 1].maximum(bound[:, 2]+margin).minimum(bound[:, 3]-margin)
        target = torch.stack([x, y, z], -1)
        d = target-grip[:, 3:]
        n = d.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        target = target-torch.relu(n-self.stretch_max)*d/n
        u = (target-p)/steps
        norm = u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        u = u*torch.clamp(self.cap/norm, max=1.)
        # Apply the same wall braking reserve used by the physical actuator.
        # Inputs to the world model must be realizable velocities, rather than
        # velocities the execution QP would immediately suppress near a wall.
        idx = torch.searchsorted(rows[:, 0].contiguous(), p[:, 2].contiguous()).clamp(1, len(rows)-1)
        r0, r1 = rows[idx-1], rows[idx]
        slope = (r1[:, 3]-r0[:, 3])/(r1[:, 0]-r0[:, 0])
        fixed_a = torch.tensor([[1.,0,0],[-1.,0,0],[0,0,1.],[0,0,-1.]], dtype=a.dtype, device=a.device)
        fixed_b = torch.stack([rows[0,1],-rows[0,2],rows[0,0],-rows[-1,0]])
        upper = (rows[1:,4]-rows[:-1,4])/(rows[1:,0]-rows[:-1,0])
        upper_a = torch.stack([torch.zeros_like(upper),-torch.ones_like(upper),upper], -1)
        upper_b = -(rows[:-1,4]-upper*rows[:-1,0])
        lower_a = torch.stack([torch.zeros_like(slope),torch.ones_like(slope),-slope], -1)[:,None]
        lower_b = (r0[:,3]-slope*r0[:,0])[:,None]
        A = torch.cat([fixed_a[None].expand(len(p),-1,-1),upper_a[None].expand(len(p),-1,-1),lower_a],1)
        b = torch.cat([fixed_b[None].expand(len(p),-1),upper_b[None].expand(len(p),-1),lower_b],1)
        norm = A.norm(dim=-1)
        h = ((A*p[:,None]).sum(-1)-b)/norm
        projection = (A*u[:,None]).sum(-1)/norm
        brake = self.acceleration*self.dt**2/A.shape[1]
        safe = torch.minimum((brake*brake+2*brake*h.clamp_min(0.)).sqrt()-brake, h.clamp_min(0.)/steps)
        factors = torch.where(projection < -1e-12, safe/(-projection).clamp_min(1e-12), torch.ones_like(projection))
        alpha = factors.min(-1).values.clamp(0.,1.)
        alpha = torch.where(h.min(-1).values >= -1e-8, alpha, torch.ones_like(alpha))
        u = u*alpha[:,None]
        return torch.cat([u/self.cap, torch.zeros_like(u)], -1)

    def project(self, action, grip, steps=1):
        rows = self.slices
        raw = np.asarray(grip[:3])+np.asarray(action[:3])*self.cap*steps
        z = np.clip(raw[2], rows[0][0]+1e-5, rows[-1][0]-1e-5)
        b = [np.interp(z, [r[0] for r in rows], [r[j] for r in rows]) for j in (1, 2, 3, 4)]
        target = np.array([np.clip(raw[0], b[0]+1e-5, b[1]-1e-5),
                           np.clip(raw[1], b[2]+1e-5, b[3]-1e-5), z])
        d = target-grip[3:]
        target -= max(0., np.linalg.norm(d)-self.stretch_max)*d/max(np.linalg.norm(d), 1e-9)
        u = clip_norm((target-grip[:3])/steps, self.cap)
        index = int(np.clip(np.searchsorted([r[0] for r in rows], grip[2]), 1, len(rows)-1))
        normals = [[1.,0,0],[-1.,0,0],[0,0,1.],[0,0,-1.]]
        offsets = [rows[0][1],-rows[0][2],rows[0][0],-rows[-1][0]]
        for i in range(len(rows)-1):
            r0,r1 = rows[i],rows[i+1]
            slope = (r1[4]-r0[4])/(r1[0]-r0[0])
            normals.append([0.,-1.,slope]);offsets.append(-(r0[4]-slope*r0[0]))
            if i == index-1:
                slope = (r1[3]-r0[3])/(r1[0]-r0[0])
                normals.append([0.,1.,-slope]);offsets.append(r0[3]-slope*r0[0])
        A,b = np.asarray(normals),np.asarray(offsets)
        norm = np.linalg.norm(A,axis=1)
        h,dot = (A@grip[:3]-b)/norm, (A@u)/norm
        if h.min() >= -1e-8:
            brake = self.acceleration*self.dt**2/len(A)
            safe = np.minimum(np.sqrt(brake*brake+2*brake*np.maximum(h,0.))-brake,np.maximum(h,0.)/steps)
            factors = np.where(dot < -1e-12,safe/np.maximum(-dot,1e-12),1.)
            u *= np.clip(factors.min(),0.,1.)
        return np.r_[u/self.cap, 0, 0, 0]


def execute_action(command, position, fixed, previous, config=Config(), gain=1.):
    """Apply after gain/delay/CBF. Reject paths crossing the curved slab boundary."""
    rows = config.slices
    index = int(np.clip(np.searchsorted([r['z_m'] for r in rows], position[2]), 1, len(rows)-1))
    A = [[1., 0, 0], [-1., 0, 0], [0, 0, 1.], [0, 0, -1.]]
    b = [rows[0]['x_low_m'], -rows[0]['x_high_m'], rows[0]['z_m'], -rows[-1]['z_m']]
    for i in range(len(rows)-1):
        a, z = rows[i], rows[i+1]
        dz = z['z_m']-a['z_m']
        slope = (z['y_high_m']-a['y_high_m'])/dz
        A.append([0., -1., slope]); b.append(-(a['y_high_m']-slope*a['z_m']))
        # The lower envelope is concave: only the active slab, plus an
        # approaching slab when the current point already satisfies it.
        slope = (z['y_low_m']-a['y_low_m'])/dz
        normal, offset = np.array([0., 1., -slope]), a['y_low_m']-slope*a['z_m']
        if i == index-1:
            A.append(normal); b.append(offset)
    A, b = np.asarray(A), np.asarray(b)
    norm = np.linalg.norm(A, axis=1)
    A, b = A/norm[:, None], b/norm
    h = np.maximum(A@position-b-1e-8, 0.)
    da = config.acceleration*config.dt**2
    # Reserve braking authority for simultaneous wall normals. This keeps the
    # feasible acceleration ball nonempty before reaching a workspace boundary.
    brake = da/len(A)
    lower = np.maximum(b-A@position+1e-7, -np.sqrt(brake*brake+2*brake*h)+brake)
    desired = clip_norm(np.asarray(command, float)*gain, config.cap)
    simple = previous+clip_norm(desired-previous, da)
    if (np.all(A@simple >= lower-1e-12) and np.linalg.norm(simple) <= config.cap+1e-12
            and segment_ok(position, position+simple, rows)):
        return simple
    u, corrections = desired.copy(), [np.zeros(3) for _ in range(3)]
    for _ in range(40):
        old = u.copy()
        for j in range(3):
            v = u+corrections[j]
            if j == 0:
                projected, _ = hildreth(v, A, lower, iters=30)
            elif j == 1:
                projected = clip_norm(v, config.cap)
            else:
                projected = previous+clip_norm(v-previous, da)
            corrections[j], u = v-projected, projected
        if np.linalg.norm(u-old) < 1e-12:
            break
    if (np.linalg.norm(u-previous) > da+1e-9 or np.linalg.norm(u) > config.cap+1e-9
            or (A@u-lower).min() < -1e-10 or not segment_ok(position, position+u, rows, tol=1e-8)):
        # Alternating projections converge slowly when the acceleration sphere
        # is almost tangent to a wall. Refine before declaring infeasibility.
        refined = refine(desired, previous, A, lower, config.cap, da)
        if refined is None or not segment_ok(position, position+refined, rows, tol=1e-8):
            raise RuntimeError('No verified acceleration-feasible workspace action')
        u = refined
    return u
