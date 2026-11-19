"""Frozen-material trajectory probes and full-cell kinematic visibility checks."""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from .physics import rope_model, initialize
from .refinement import set_rest_fraction
from .vision import cameras, resample3
from .visibility import curve_margin, ray_visibility
from ..cell.arms import Arm
from ..cell.scene import mat


class WorkspaceProbe:
    def __init__(self, profile, scene_binary, margin_px=20., uncertainty_m=.01):
        self.margin_px, self.uncertainty_m = margin_px, uncertainty_m
        self.p = json.loads(Path(profile).read_text())
        p = self.p
        self.cams = cameras(p['cameras'])
        self.m = rope_model(p['length_m'], p['mass_kg'], p['radius_m'],
                            p['joint_stiffness_nm_per_rad'], .003)
        self.d = initialize(self.m, np.asarray(p['initial_curve']))
        set_rest_fraction(self.m, self.d, p.get('initial_rest_fraction', 0.))
        self.m.dof_frictionloss[6:] = p.get('joint_frictionloss_nm', 0.)
        mujoco.mj_step(self.m, self.d, nstep=round(30/self.m.opt.timestep))
        self.m.dof_damping[6:] = p['joint_damping_nm_s_per_rad']
        self.d.qvel[:] = 0
        self.home = self.d.qpos.copy()
        self.start = np.asarray(p['initial_curve'][0])
        self.ids = [self.m.body(f'dlo_{i:02d}').id for i in range(25)]
        self.full = mujoco.MjModel.from_binary_path(str(scene_binary))
        self.fd = mujoco.MjData(self.full)
        self.arm = Arm(self.full, 'left')
        self.rotation = mat(p['left_tcp_quaternion_wxyz'])
        self.frope = [self.full.geom(f'dlo_geom_{i:02d}').id for i in range(25)]
        self.qmap = [(self.full.joint('dlo_root_free').qposadr[0], 0, 7)]
        self.qmap += [(self.full.joint(f'dlo_joint_{i:02d}').qposadr[0], self.m.jnt_qposadr[i], 4)
                      for i in range(1, 25)]
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:] = self.home
        self.d.mocap_pos[:] = np.asarray(self.p['initial_curve'])[[0, -1]]
        seed = initialize(self.m, np.asarray(self.p['initial_curve']))
        self.d.mocap_quat[:] = seed.mocap_quat
        mujoco.mj_forward(self.m, self.d)
        mujoco.mj_resetDataKeyframe(self.full, self.fd, self.full.key('home').id)
        mujoco.mj_forward(self.full, self.fd)
        self.qarm = self.fd.qpos[self.arm.qadr].copy()

    def inspect(self, kinematic=True, uncertainty_m=None):
        uncertainty_m = self.uncertainty_m if uncertainty_m is None else uncertainty_m
        mujoco.mj_forward(self.m, self.d)
        curve = resample3(self.d.xpos[self.ids], 39)
        # Check both endpoints of every capsule, including the terminal stub.
        axes = self.d.geom_xmat.reshape(-1, 3, 3)[:, :, 2]
        tips = np.concatenate([self.d.geom_xpos + sign*self.m.geom_size[:, 1, None]*axes
                               for sign in (-1, 1)])
        margins = curve_margin(tips, self.cams, self.p['radius_m'], margin_px=self.margin_px,
                               uncertainty_m=uncertainty_m)
        raw_margins = curve_margin(tips, self.cams, self.p['radius_m'], margin_px=0., uncertainty_m=0.)
        row = dict(t=float(self.d.time), tcp=self.d.mocap_pos[0].tolist(), curve=curve.tolist(),
                   margin_m=margins, fov_ok=bool(min(margins.values()) >= 0),
                   raw_margin_m=raw_margins, raw_fov_ok=bool(min(raw_margins.values()) >= 0),
                   speed_m_s=float(np.linalg.norm(self.d.cvel[self.ids, 3:], axis=1).max()))
        if not kinematic:
            return row
        q, ep, er = self.arm.ik(self.d.mocap_pos[0], self.rotation, self.qarm, iters=80)
        self.qarm = q
        self.arm.apply(self.fd, q)
        for dst, src, n in self.qmap:
            self.fd.qpos[dst:dst+n] = self.d.qpos[src:src+n]
        mujoco.mj_forward(self.full, self.fd)
        vis = ray_visibility(self.full, self.fd, curve, self.cams, self.frope, self.p['radius_m'])
        # Endpoint pixels may be covered by the actual clamp/probe.
        direct = np.logical_and.reduce([r['visible'] for r in vis.values()])
        row.update(ik_position_error_m=ep, ik_rotation_error_rad=er,
                   ik_ok=bool(ep <= .001 and er <= .01), visibility=vis,
                   stereo_internal_fraction=float(direct[1:-1].mean()),
                   direct_ok=bool(direct[1:-1].all()))
        return row

    def move(self, target, speed=.05, settle_s=2., kinematic=False, sample_dt=.1):
        """Minimum-jerk translation capped by its analytic peak speed (1.875 d/T)."""
        start = self.d.mocap_pos[0].copy()
        delta = np.asarray(target)-start
        duration = max(.5, 1.875*np.linalg.norm(delta)/speed)
        dt = self.m.opt.timestep
        steps = int(np.ceil((duration+settle_s)/dt))
        stride = max(1, round(sample_dt/dt))
        records = [self.inspect(kinematic)]
        for i in range(1, steps+1):
            s = min(1., i*dt/duration)
            self.d.mocap_pos[0] = start+(10*s**3-15*s**4+6*s**5)*delta
            mujoco.mj_step(self.m, self.d)
            if i % stride == 0 or i == steps:
                records.append(self.inspect(kinematic))
        return records


def summary(rows, require_direct=False):
    reasons = []
    if not all(r['fov_ok'] for r in rows): reasons.append('rope_fov')
    if not all(r.get('ik_ok', True) for r in rows): reasons.append('arm_ik')
    if require_direct and not all(r.get('direct_ok', False) for r in rows): reasons.append('occlusion')
    if not all(np.isfinite(r['curve']).all() for r in rows): reasons.append('nonfinite')
    return dict(accepted=not reasons, reasons=reasons,
                min_margin_m=min(min(r['margin_m'].values()) for r in rows),
                min_stereo_internal_fraction=min(r.get('stereo_internal_fraction', 1.) for r in rows))
