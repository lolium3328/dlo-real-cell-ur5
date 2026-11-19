"""Full measured scene, real image observations and kinematic checks for training."""
import json
import copy
from pathlib import Path
import mujoco
import numpy as np

from ...real_rope.scene import build_measured_scene, apply_spring_reference
from ...real_rope.vision import cameras
from ...real_rope.tracking import RecordedRopeTracker
from ..arms import Arm
from ..scene import mat
from .config import Config, ROOT, right_rotation, tcp_from_sphere, digest
from .observations import QualityGate, EvidenceMemory
from scipy.spatial.transform import Rotation


def build(config=Config()):
    config.freeze()
    path = config.path/'scene.xml'
    report = config.path/'scene.json'
    if path.exists() and report.exists():
        record = json.loads(report.read_text())
        if record['xml_sha256'] != digest(path):
            raise ValueError('Calibrated scene cache changed')
        return path
    record = build_measured_scene(ROOT/config.profile, path, sphere_offset=config.sphere_offset,
                                  right_quat=config.right_quat, sphere_radius=config.sphere_radius)
    record.update(xml_sha256=digest(path), sphere_offset=list(config.sphere_offset),
                  sphere_radius=config.sphere_radius, config=config.provenance())
    report.write_text(json.dumps(record, indent=2))
    return path


class SceneView:
    def __init__(self, config=Config(), render=True):
        self.config = config
        self.model = mujoco.MjModel.from_xml_path(str(config.path/'scene.xml'))
        apply_spring_reference(self.model)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key('home').id)
        self.left, self.right = Arm(self.model, 'left'), Arm(self.model, 'right')
        self.ql, self.qr = self.data.qpos[self.left.qadr].copy(), self.data.qpos[self.right.qadr].copy()
        self.Rl, self.Rr = mat(config.material['left_tcp_quaternion_wxyz']), right_rotation(config)
        self.mapping = [(self.model.joint('dlo_root_free').qposadr[0], 0, 7)]
        self.mapping += [(self.model.joint(f'dlo_joint_{i:02d}').qposadr[0], 7+4*(i-1), 4) for i in range(1, 25)]
        self.renderer = mujoco.Renderer(self.model, 480, 640) if render else None
        self.cams = cameras(ROOT/config.material['cameras'])
        self.last_ik = (0., 0.)
        self.frames = 0

    def sync(self, rope, sphere):
        d = self.data
        actual_fixed = rope.data.mocap_pos[1]
        self.model.body('fixed_clamp').pos = actual_fixed
        post = self.model.geom('fixture_post').id
        self.model.geom_pos[post] = [actual_fixed[0], actual_fixed[1], actual_fixed[2]/2]
        self.model.geom_size[post, 1] = actual_fixed[2]/2
        for dst, src, n in self.mapping:
            d.qpos[dst:dst+n] = rope.data.qpos[src:src+n]
        d.mocap_pos[0] = rope.left()
        self.ql, el, rl = self.left.ik(rope.left(), self.Rl, self.ql, iters=30)
        self.qr, er, rr = self.right.ik(tcp_from_sphere(sphere, self.Rr, self.config), self.Rr, self.qr, iters=40)
        self.left.apply(d, self.ql)
        self.right.apply(d, self.qr)
        self.last_ik = (max(el, er), max(rl, rr))
        self.frames += 1
        mujoco.mj_forward(self.model, d)

    def images(self):
        result = {}
        for role in self.cams:
            self.renderer.update_scene(self.data, camera=role+'_view')
            result[role] = self.renderer.render().copy()
        return result

    def close(self):
        if self.renderer is not None:
            self.renderer.close()


class RGBObservations:
    """RGB evidence only; no geometric visibility or simulation curve enters policy inputs."""
    def __init__(self, config, view, seed=0, deploy=False):
        self.config, self.view, self.deploy = config, view, deploy
        self.rng = np.random.default_rng(seed)
        cams = copy.deepcopy(view.cams)
        if deploy:
            for cam in cams.values():
                cam['T'][:3, 3] += self.rng.normal(0., .0015, 3)
                delta = Rotation.from_rotvec(self.rng.normal(0., np.deg2rad(.15), 3)).as_matrix()
                cam['T'][:3, :3] = delta@cam['T'][:3, :3]
        self.tracker = RecordedRopeTracker(cams, config.material)
        self.queue, self.next_frame, self.last = [], 0., None
        self.gate = QualityGate(config.material['length_m'], getattr(config, 'gate_mode', 'age'))
        self.evidence = EvidenceMemory()
        self.first = True
        self.boundary_mask = np.ones(39, bool)
        self.boundary_mask[:2] = self.boundary_mask[-2:] = False
        self.pause = False

    def update(self, t, rope, sphere, grip_velocity, *, stress=None):
        if t+1e-9 >= self.next_frame:
            self.view.sync(rope, sphere)
            images = self.view.images()
            if self.deploy:
                images = {k: np.clip(v+np.float32(3.)*self.rng.standard_normal(v.shape, dtype=np.float32), 0, 255).astype(np.uint8)
                          for k, v in images.items()}       # N(0, 3) pixel noise in float32 (v37; same distribution as v1)
            if stress == 'front_missing':
                images.pop('front')
            elif stress == 'both_missing':
                images = {}
            delay = self.config.image_latency_s if self.deploy and not self.first else 0.
            self.queue.append((t+delay, t, images))
            self.next_frame = t+1/self.config.image_hz
            self.first = False
        ready = [r for r in self.queue if r[0] <= t+1e-9]
        if ready:
            _, captured, images = ready[-1]
            self.queue = [r for r in self.queue if r[0] > t+1e-9]
        else:
            captured, images = t, {}
        grip = np.r_[rope.left(), rope.fixed]
        if self.last is None and not images:
            raise RuntimeError('RGB initialization requires an available camera frame')
        quality = self.tracker.observe(images, grip, t, view_t={r: captured for r in images},
                                       grip_vel=np.r_[grip_velocity, 0, 0, 0])
        quality = self.evidence.update(t, quality, {r:captured for r in images})
        # Stereo-age alone incorrectly stops on stable single-view self-occlusion.
        # Boundary clamps are constrained by kinematics; unsupported interior nodes
        # retain uncertainty and cause a stop after .3 s / 20%, or any 1 s.
        self.pause = self.gate.update(t, quality)
        quality['pause_requested'] = self.pause
        quality['gate_basis'] = 'any_image_support_age_except_two_clamp_nodes_each_end'
        self.last = quality
        return quality
