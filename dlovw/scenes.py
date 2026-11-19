"""Scene generation v1: independent start/target tasks with bounded, mid-rope crossing obstacles.

Design rules come from docs/notes/scene_audit_and_step0_20261020.md:
- one XML per obstacle count (no unused spheres left in the workspace);
- start/target span ratio within [0.70, 0.90] (legacy hard limit 0.96);
- every scene has its own endpoints (no reuse across obstacle counts);
- each obstacle crosses the obstacle-blind reference rope at an interior material point
  (arc fraction 0.3-0.7) at a chosen time, approaching from 15-30 cm away;
- motion is bounded by a position envelope above the table and in front of the backdrop.
"""
from __future__ import annotations

import numpy as np

from . import legacy  # noqa: F401  (sets sys.path)
from control.mpc_observation import state_poses  # noqa: E402
from control.mpc_scenarios import Scenario  # noqa: E402
from envs.mpc_scene import initialize_scene, make_env  # noqa: E402
from envs.translation_action import clipped_delta, execute_translation_action  # noqa: E402
from .geometry_audit import DT, L_ROPE, T_STEPS, replay

REF_STEPS = 220                       # legacy target_generation_steps (2 mm/step straight moves)
ENVELOPE_LOW = np.array([0.18, -0.34, 0.12])
ENVELOPE_HIGH = np.array([1.18, 0.48, 0.72])
SPAN_RANGE = (0.70, 0.90)


def _inside(p, lo, hi, margin):
    return bool(np.all(p >= np.asarray(lo) + margin) and np.all(p <= np.asarray(hi) - margin))


def sample_endpoints(rng, workspace, margin=0.015, tries=2000, move_range=(0.08, 0.22)):
    """Independent start and target gripper positions (no orientation change).
    `move_range` bounds the midpoint transport distance (v1 default; gates need longer moves)."""
    for _ in range(tries):
        mid = np.array([rng.uniform(0.56, 0.80), rng.uniform(-0.10, 0.10), rng.uniform(0.34, 0.42)])
        span = rng.uniform(*SPAN_RANGE) * L_ROPE
        yaw = np.deg2rad(rng.uniform(-35, 35))
        dz = rng.uniform(-0.04, 0.04)
        d = np.array([np.cos(yaw), np.sin(yaw), 0.0]) * span
        d[2] = dz
        d *= span / np.linalg.norm(d)
        il, ir = mid - d / 2, mid + d / 2
        move = rng.normal(size=3) * np.array([1.0, 1.0, 0.3])
        move *= rng.uniform(*move_range) / np.linalg.norm(move)
        span2 = np.clip(span + rng.uniform(-0.08, 0.06) * L_ROPE, SPAN_RANGE[0] * L_ROPE, SPAN_RANGE[1] * L_ROPE)
        yaw2 = yaw + np.deg2rad(rng.uniform(-25, 25))
        d2 = np.array([np.cos(yaw2), np.sin(yaw2), 0.0]) * span2
        d2[2] = dz + rng.uniform(-0.03, 0.03)
        d2 *= span2 / np.linalg.norm(d2)
        tl, tr = mid + move - d2 / 2, mid + move + d2 / 2
        if not (_inside(il, workspace["left_low"], workspace["left_high"], margin)
                and _inside(tl, workspace["left_low"], workspace["left_high"], margin)
                and _inside(ir, workspace["right_low"], workspace["right_high"], margin)
                and _inside(tr, workspace["right_low"], workspace["right_high"], margin)):
            continue
        if max(np.linalg.norm(tl - il), np.linalg.norm(tr - ir)) > 0.002 * REF_STEPS * 0.9:
            continue
        return dict(initial_left=il, initial_right=ir, target_left=tl, target_right=tr)
    raise RuntimeError("Could not sample endpoints")


def reference_trajectory(sim_cfg, ends, seed):
    """Obstacle-blind nominal trajectory, computed exactly like ClosedLoopSession._generate_target."""
    env, left_arm, right_arm = make_env(sim_cfg, seed)
    try:
        scenario = Scenario(name="dlovw", **{k: np.asarray(v, float) for k, v in ends.items()})
        initialize_scene(env, left_arm, right_arm, scenario, sim_cfg, seed)
        lp, rp, c = state_poses(env, 39)
        cl, ll, rr = [c], [lp], [rp]
        for _ in range(REF_STEPS):
            cur = env.data.mocap_pos[[env.left_mocap_id, env.right_mocap_id]]
            act = np.r_[clipped_delta(scenario.target_left, cur[0], .002),
                        clipped_delta(scenario.target_right, cur[1], .002)]
            execute_translation_action(env, left_arm, right_arm, act, sim_cfg)
            lp, rp, c = state_poses(env, 39)
            cl.append(c); ll.append(lp); rr.append(rp)
        return dict(centerlines=np.asarray(cl), left=np.asarray(ll), right=np.asarray(rr))
    finally:
        env.close()


def point_at_fraction(line, s):
    seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
    cum = np.r_[0, np.cumsum(seg)] / seg.sum()
    i = int(np.clip(np.searchsorted(cum, s) - 1, 0, len(seg) - 1))
    f = (s - cum[i]) / max(cum[i + 1] - cum[i], 1e-12)
    p = line[i] + f * (line[i + 1] - line[i])
    tangent = (line[i + 1] - line[i]) / max(seg[i], 1e-12)
    return p, tangent


def crossing_profile(rng, ref_centerlines, *, t_cross, fraction, speed, approach, seed,
                     direction_hint=None, max_angle_deg=None):
    """Constant-velocity obstacle that sits still, then crosses the reference rope at
    (t_cross, fraction) after travelling `approach` metres, braking at the envelope.
    With `direction_hint`, only directions within `max_angle_deg` of it are admissible
    (the RNG stream is unchanged when no hint is given)."""
    k = int(round(t_cross / DT))
    line = ref_centerlines[min(k, len(ref_centerlines) - 1)]
    p_c, tangent = point_at_fraction(line, fraction)
    a = np.cross(tangent, [0.0, 0.0, 1.0])
    if np.linalg.norm(a) < 1e-6:
        a = np.array([0.0, 1.0, 0.0])
    a /= np.linalg.norm(a)
    b = np.cross(tangent, a)
    for _ in range(200 if direction_hint is None else 2000):
        theta = rng.uniform(0, 2 * np.pi)
        u = np.cos(theta) * a + np.sin(theta) * b + rng.uniform(-0.3, 0.3) * tangent
        u /= np.linalg.norm(u)
        if direction_hint is not None and np.degrees(np.arccos(np.clip(
                u @ np.asarray(direction_hint) / np.linalg.norm(direction_hint), -1, 1))) > max_angle_deg:
            continue
        start = p_c - u * approach
        # start and the stretch just past the crossing must lie inside the envelope with margin,
        # otherwise the envelope brake bends the path before it reaches the rope
        if _inside(start, ENVELOPE_LOW, ENVELOPE_HIGH, 0.04) and _inside(p_c + u * 0.10, ENVELOPE_LOW, ENVELOPE_HIGH, 0.03):
            break
    else:
        raise RuntimeError("No admissible approach direction")
    delay = max(t_cross - approach / speed, 0.0)
    prof = dict(profile="constant_velocity", speed_mps=float(speed), direction=[float(x) for x in u],
                a_max_mps2=1.5, v_max_mps=1.0, delta_c_m=0.005, delta_v_mps=0.02, latency_steps=0.0,
                seed=int(seed), start_delay_s=float(delay), start_position_m=[float(x) for x in start],
                position_low_m=[float(x) for x in ENVELOPE_LOW], position_high_m=[float(x) for x in ENVELOPE_HIGH])
    design = dict(t_cross=float(t_cross), fraction=float(fraction), speed=float(speed), approach=float(approach),
                  crossing_point=[float(x) for x in p_c], direction=[float(x) for x in u])
    return prof, design


def obstacle_tracks(profiles):
    return np.stack([replay(p, T_STEPS) for p in profiles], axis=1)
