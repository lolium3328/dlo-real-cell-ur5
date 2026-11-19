"""Versioned measured-cell inputs, accepted hardware geometry and training defaults."""
from dataclasses import asdict, dataclass
from functools import cached_property
import hashlib
import json
from pathlib import Path
import subprocess
import copy

import numpy as np

from .. import layout
from ..scene import mat

ROOT = Path(__file__).resolve().parents[3]
VERSION = 'calibrated_cell_deploy_v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_revision():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return 'standalone_source_package'


def verify_source(expected):
    changed = [name for name, sha in expected.items() if not (ROOT/name).is_file() or digest(ROOT/name) != sha]
    if changed:
        raise ValueError(f'Run source changed: {changed}; preserve this run and use a new output directory')


@dataclass(frozen=True)
class Config:
    profile: str = 'outputs/real_rope_refine_20261113/profile.json'
    workspace: str = 'outputs/visibility_reference_20261113/recommended_workspace.json'
    out: str = 'outputs/cell_calibrated_20261113'
    sphere_radius: float = .04
    sphere_offset: tuple = (0., 0., -.025)
    right_quat: tuple = (-.109382, .640856, -.640856, -.408218)   # v41 (UR5): the v40 pose rolled 90 deg about the
    # probe (the ball sits on the probe axis, so it does not move): with the UR5's shorter wrist the v40 roll put the
    # flange plate into the forearm (test_reach_0087, -2.3 mm); rolls 30-120 deg clear it. v40: (-.365998, .906308,
    # 0, -.211309) = probe tilted 50 deg toward azimuth 300 deg (search_right_pose.py); v1-v39 (0, 1, 0, 0)
    dt: float = .008
    decision_steps: int = 10
    speed: float = .05
    acceleration: float = .15
    clearance: float = .005
    cbf_margin: float = .005
    horizon: int = 32
    plan_horizon: int = 15
    plan_commit: tuple = ()      # CellPlanner commitment (abs, rel, risk); () = r10 switch rule (0.3)
    image_hz: float = 30.
    image_latency_s: float = .048
    margin_px: float = 20.
    max_time_s: float = 30.        # v37: 16 -> 30 s, covers the slowed ball event (<= 5/ball_speed_scale s) + return + hold
    crossing_node_max: int = 20    # v40: 6..20 (pilot: crossings at nodes >= 20 cannot be dodged); v38-v39: 13; v1-v37: 20
    arm_clearance: float = .02     # v40: min left-arm / right-arm(+ball) distance along the oracle (mesh hulls); 0 = off
    max_oracle_plans: int = 30     # v40: screening stops after this many dodge plans (pilot: all accepted tasks <= 23; v38: 82)
    right_rope_clearance: float = .005   # v40: min right arm (links + probe, no ball) to rope distance along the oracle
    causal_oracle: bool = True     # v38: screening + demos only dodge after the ball starts moving (tasks.onset_time)
    headless: bool = True          # v37: synthetic tracking error instead of rendering + RGB tracking (headless.py)
    headless_model: str = 'occlusion'  # v39: vision-validated OcclusionTracking (headless_occlusion.py); 'plain' = v37/v38
    occlusion_error_scale: float = 1.  # occlusion model: error field x this (calibrated to a 2.6 mm mean node error)
    occlusion_quality: bool = True  # occlusion model: real quality features (False: training's constant ones)
    gate_mode: str = 'age'          # 'age' (v1-v39 QualityGate) or 'bound' (global-loss pause + per-node CBF bound)
    headless_pause: bool = True    # occlusion model only: apply its QualityGate pauses, as RGB tracking does
    ball_speed_scale: float = .05/.21875   # v37: ball time-scaled by arm speed ratio (0.05 m/s vs old cell 0.219 m/s); 1 = v1-v36
    hold_s: float = .4
    seed: int = 20261113
    rounds: int = 60               # v35 (r10 fix): 20 -> 60; v2 seed 0 was still improving at round 20
    wm_pretrain: int = 6000
    bc_pretrain: int = 2000
    train_steps: int = 200
    collect: int = 60
    batch: int = 32
    starts: int = 512
    deploy_fraction: float = .75
    cont_scale: float = .003       # training imagination soft-survival width (m); r10 value kept (10 mm did not help, v35 diagnostic)
    grad_truncate: int = 0         # detach the imagined state every N steps; 0 = r10. v35 used 8 (hurt avoidance, stopped r23)
    state_grad_clip: float = 1e-3  # per-step, per-row clip of the gradient through the imagined state; 0 = r10, v36 = 1e-3

    @property
    def path(self):
        return ROOT / self.out

    @cached_property
    def material(self):
        return json.loads((ROOT / self.profile).read_text())

    @cached_property
    def slices(self):
        return json.loads((ROOT / self.workspace).read_text())['height_slices']

    @property
    def protected(self):
        return self.sphere_radius + self.material['radius_m'] + self.clearance

    @property
    def cap(self):
        return self.speed * self.dt

    @property
    def decision_dt(self):
        return self.dt * self.decision_steps

    def provenance(self):
        if hasattr(self, '_provenance_snapshot'):
            return copy.deepcopy(self._provenance_snapshot)
        p = self.material
        sources = [self.profile, self.workspace, p['cameras']]
        code = sorted(Path(__file__).parent.glob('*.py'))
        code += [ROOT/name for name in ('dlovw/cell/cbf.py', 'dlovw/cell/cost.py', 'dlovw/cell/planner.py',
                 'dlovw/imagination/r10.py', 'dlovw/imagination/r10_exec.py', 'dlovw/imagination/feasible.py',
                 'dlovw/imagination/world_model.py', 'dlovw/imagination/episodes.py', 'dlovw/imagination/obstacle_model.py',
                 'dlovw/imagination/rgb_rope.py', 'dlovw/real_rope/tracking.py')]
        record = dict(protocol=VERSION, config=json.loads(json.dumps(asdict(self))),
                    inputs={s: digest(ROOT / s) for s in sources},
                    base_transform=layout.T_WORLD_FROM_RIGHT_BASE.tolist(),
                    source_sha256={str(s.relative_to(ROOT)): digest(s) for s in code},
                    code_commit=code_revision())
        object.__setattr__(self, '_provenance_snapshot', record)
        return copy.deepcopy(record)

    def freeze(self):
        self.path.mkdir(parents=True, exist_ok=True)
        record = self.provenance()
        target = self.path / 'config.json'
        if target.exists():
            previous = json.loads(target.read_text())
            for key in ('protocol', 'config', 'inputs', 'base_transform'):
                if previous[key] != record[key]:
                    raise ValueError(f'Configuration/input changed at {target}: {key}; use a new output directory')
        else:
            target.write_text(json.dumps(record, indent=2))
        return record


def sphere_from_tcp(position, rotation, config=Config(), *, frame='left_base'):
    """Rigid mounted centre. Rotation is world/base-from-tool, not a fixed world-z offset."""
    position, rotation = np.asarray(position, float), np.asarray(rotation, float)
    if position.shape != (3,) or rotation.shape != (3, 3):
        raise ValueError('Expected TCP xyz and 3x3 rotation')
    if not np.isfinite(position).all() or not np.isfinite(rotation).all():
        raise ValueError('Non-finite TCP pose')
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) < 0:
        raise ValueError('Invalid tool rotation')
    centre = position + rotation @ np.asarray(config.sphere_offset)
    if frame == 'right_base':
        T = layout.T_WORLD_FROM_RIGHT_BASE
        centre = T[:3, :3] @ centre + T[:3, 3]
    elif frame != 'left_base':
        raise ValueError(f'Unknown TCP frame: {frame}')
    return centre


def tcp_from_sphere(centre, rotation, config=Config()):
    return np.asarray(centre, float) - np.asarray(rotation) @ np.asarray(config.sphere_offset)


def right_rotation(config=Config()):
    return mat(config.right_quat)


def require_cuda():
    """Device isolation is an error to resolve, never permission to train on CPU."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required: check sandbox device isolation, permissions and CUDA; CPU fallback disabled')
    probe = torch.ones(32, device='cuda')
    if float(probe.sum().item()) != 32:
        raise RuntimeError('CUDA arithmetic check failed')
    return torch.device('cuda')
