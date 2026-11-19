"""Refined material and frozen initial rest shape with one moving clamp."""
import copy
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

from ...real_rope.physics import rope_model, initialize
from ...real_rope.refinement import set_rest_fraction
from ..perception import resample
from .config import Config

_BASE = {}


def _base(config):
    key = config.profile
    if key not in _BASE:
        p = config.material
        model = rope_model(p['length_m'], p['mass_kg'], p['radius_m'], p['joint_stiffness_nm_per_rad'],
                           p['joint_damping_nm_s_per_rad'])
        data = initialize(model, np.asarray(p['initial_curve']))
        set_rest_fraction(model, data, p['initial_rest_fraction'])
        model.dof_frictionloss[6:] = p['joint_frictionloss_nm']
        mujoco.mj_step(model, data, nstep=round(30/model.opt.timestep))
        mujoco.mj_forward(model, data)
        _BASE[key] = (model, data.qpos.copy(), data.mocap_pos.copy(), data.mocap_quat.copy())
    return _BASE[key]


class RopeState:
    def __init__(self, start, config=Config(), seed=0, deploy=False, qpos=None):
        self.config, self.p = config, config.material
        model, q, pos, rot = _base(config)
        self.model = copy.copy(model)
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = q
        self.data.mocap_pos[:], self.data.mocap_quat[:] = pos, rot
        self.rng = np.random.default_rng(seed)
        self.fixed = np.asarray(self.p['fixed_end_m'])
        self.randomization = dict(stiffness=1., damping=1., mass=1., clamp_offset_m=[0.,0.,0.], rest_scale=1.)
        if deploy:
            self.randomization['stiffness'] = float(self.rng.uniform(.6, 1.6))
            self.randomization['damping'] = float(self.rng.uniform(.6, 1.6))
            self.model.jnt_stiffness *= self.randomization['stiffness']
            self.model.dof_damping[6:] *= self.randomization['damping']
            scale = self.rng.uniform(.85, 1.2)
            self.randomization['mass'] = float(scale)
            self.model.body_mass *= scale
            self.model.body_inertia *= scale
            direction = self.rng.normal(size=3)
            offset = direction/max(np.linalg.norm(direction), 1e-12)*self.rng.uniform(0., .004)
            self.randomization['clamp_offset_m'] = offset.tolist()
            self.data.mocap_pos[1] = self.fixed+offset
            rest_scale = self.rng.uniform(.8, 1.2)
            self.randomization['rest_scale'] = float(rest_scale)
            springs = self.model.qpos_spring[7:].reshape(-1, 4)
            rotation = Rotation.from_quat(springs[:, [1,2,3,0]])
            springs[:] = Rotation.from_rotvec(rotation.as_rotvec()*rest_scale).as_quat()[:, [3,0,1,2]]
            # mj_setConst uses MjData as scratch and overwrites mocap/state fields.
            # Recompute inertial constants without touching the episode state.
            mujoco.mj_setConst(self.model, mujoco.MjData(self.model))
        self.ids = np.array([self.model.body(f'dlo_{i:02d}').id for i in range(25)])
        if qpos is None:
            self.move(start)
        else:
            self.data.qpos[:] = qpos
            self.data.qvel[:] = 0.
            self.data.mocap_pos[0] = start
            mujoco.mj_forward(self.model, self.data)
            if deploy:
                self.move(start, settle=.4)
        self.data.time = 0.

    def curve(self):
        return resample(self.data.xpos[self.ids])

    def left(self):
        return self.data.mocap_pos[0].copy()

    def advance(self, displacement):
        before = self.left()
        sub = round(self.config.dt/self.model.opt.timestep)
        for k in range(1, sub+1):
            self.data.mocap_pos[0] = before+np.asarray(displacement)*k/sub
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def move(self, target, settle=4.):
        p0 = self.left()
        distance = np.linalg.norm(np.asarray(target)-p0)
        duration = max(.5, 1.875*distance/self.config.speed,
                       np.sqrt(5.774*distance/self.config.acceleration))
        steps = max(1, round(duration/self.config.dt))
        for k in range(1, steps+1):
            s = k/steps
            p = p0+(np.asarray(target)-p0)*(10*s**3-15*s**4+6*s**5)
            self.advance(p-self.left())
        # Same bounded settling budget for all candidates; report failures upstream.
        for _ in range(round(settle/self.config.dt)):
            self.advance(np.zeros(3))


def surface_distance(curve, centre, sphere_radius, rope_radius=0.):
    """Centre-to-polyline-segment distance, minus physical radii (metres)."""
    a, b = curve[:-1], curve[1:]
    ab = b-a
    t = np.clip(((centre-a)*ab).sum(-1)/np.maximum((ab*ab).sum(-1), 1e-15), 0, 1)
    return float(np.linalg.norm(centre-(a+t[:, None]*ab), axis=1).min()-sphere_radius-rope_radius)
