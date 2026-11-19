"""Episodes at decision resolution (docs/notes/imagination_ac_design_20261022.md, section 4).

One decision covers K control steps (80 ms). Positions are relative to the midpoint of the two target
gripper positions and scaled by POS (0.1 m -> 1); velocities by VEL (0.5 m/s -> 1).

obs (OBS)     rope centerline 39x3, grippers 2x3, obstacle measurement (position, velocity, visible);
              zeros while the obstacle is hidden. The only input the world model's encoder gets.
goal (GOAL)   target centerline and target grippers (actor, critic and reward).
truth (TRUTH) true obstacle position and velocity: world-model supervision only.
track (M, 3)  true obstacle position at every decision step, also past the episode end; imagined
              rewards use it (the obstacle's motion does not depend on our actions).
track_meas (M, 7)  what the scene's sensor reports at every decision step along that track (same
              occluder, noise, latency and bias model as the episode): imagination replays it into the
              obstacle filter, so the imagined actor sees the obstacle exactly when it would be visible.
act (ACT)     executed gripper translation per control step, averaged over the decision, / CAP.
"""
from __future__ import annotations

import numpy as np

K, CAP, N_PTS = 10, 0.00175, 39
POS, VEL = 10.0, 2.0
ROPE = 3 * N_PTS + 6                 # leading block of obs that the world model reconstructs
OBS, GOAL, TRUTH, ACT = ROPE + 7, ROPE, 6, 6
PROTECTED_M = 0.055                  # 40 mm sphere + 10 mm rope radius + 5 mm clearance


def origin_of(target):
    return 0.5 * (np.asarray(target["left_pose"], float)[:3] + np.asarray(target["right_pose"], float)[:3])


def encode_meas(meas, origin):
    """(position, velocity) measurement or None (hidden) -> 7 numbers."""
    return (np.zeros(7) if meas is None else
            np.r_[(np.asarray(meas[0], float) - origin) * POS, np.asarray(meas[1], float) * VEL, 1.0]).astype(np.float32)


def encode_obs(center, left, right, meas, origin):
    center = np.asarray(center, float)
    if center.shape != (N_PTS, 3):
        raise ValueError(f"centerline must be {N_PTS}x3, got {center.shape}")
    grip = np.r_[np.asarray(left, float)[:3] - origin, np.asarray(right, float)[:3] - origin]
    return np.r_[(center - origin).ravel() * POS, grip * POS, encode_meas(meas, origin)].astype(np.float32)


def measurement_track(track, rec, origin, offset, seed):
    """The scene's sensor (stage-0 Sensor: occluder, noise, latency, bias) run along the whole true track,
    sampled at the decision steps offset, offset + K, ... -> (M, 7)."""
    from ..belief import NOMINAL, Sensor
    from ..motion import velocities
    track = np.asarray(track, float)
    vel = velocities(track)
    occluder = (rec.get("designs") or [{}])[0].get("occluder")
    sensor = Sensor((rec.get("knob") or {}).get("perception") or NOMINAL, occluder, np.random.default_rng(seed))
    out = []
    for n in range(offset, len(track), K):
        r = sensor.read(track[:n + 1], vel[:n + 1])
        out.append(encode_meas(None if r is None else (r[1], r[2]), origin))
    return np.stack(out)


def encode_goal(target, origin):
    return encode_obs(target["centerline"], target["left_pose"], target["right_pose"], None, origin)[:GOAL]


def build_episode(center, left, right, meas, obst_p, obst_v, track, offset, target, info, rec, seed=0):
    """Per-control-step sequences (index i <-> control step offset + i) -> one decision-level episode.
    `track` is the full true obstacle track indexed by control step; `rec` the scene (sensor model)."""
    origin = origin_of(target)
    n = len(center)
    idx = list(range(0, n, K))
    if idx[-1] != n - 1:                       # last, shorter decision up to the final state
        idx.append(n - 1)
    grip = np.stack([np.r_[np.asarray(left[i], float)[:3], np.asarray(right[i], float)[:3]] for i in range(n)])
    obs = np.stack([encode_obs(center[i], left[i], right[i], meas[i], origin) for i in idx])
    truth = np.stack([np.r_[(np.asarray(obst_p[i], float) - origin) * POS, np.asarray(obst_v[i], float) * VEL]
                      for i in idx]).astype(np.float32)
    act = np.stack([(grip[b] - grip[a]) / ((b - a) * CAP) for a, b in zip(idx[:-1], idx[1:])]).astype(np.float32)
    trk = ((np.asarray(track, float)[offset::K] - origin) * POS).astype(np.float32)
    return dict(obs=obs, act=act, truth=truth, goal=encode_goal(target, origin), track=trk, info=dict(info),
                origin=np.asarray(origin, np.float32),
                track_meas=measurement_track(track, rec, origin, offset, seed + 1))


def from_stage0(traj, rec, target, seed):
    """A stage-0 run (outputs/baseline/<tag>/traj/<id>.npz: state after each control step, true obstacle
    centres) -> episode. Obstacle measurements are replayed with the stage-0 Sensor on the true track."""
    from ..belief import NOMINAL, Sensor
    from ..motion import track as motion_track, velocities
    motion = rec["profiles"][0]["motion"]
    tr = motion_track(motion, 1300)
    vel = velocities(tr)
    center, left, right = traj["centerline"], traj["left"], traj["right"]
    n = len(center)                            # traj[t] is the state after control step t + 1
    if not np.allclose(traj["obstacles"][:, 0], tr[1:n + 1], atol=1e-6):
        raise ValueError("recorded obstacle does not match the scripted track")
    occluder = (rec.get("designs") or [{}])[0].get("occluder")
    sensor = Sensor((rec.get("knob") or {}).get("perception") or NOMINAL, occluder, np.random.default_rng(seed))
    meas = []
    for t in range(n):
        r = sensor.read(tr[:t + 2], vel[:t + 2])
        meas.append(None if r is None else (r[1], r[2]))
    return build_episode(center, left, right, meas, tr[1:n + 1], vel[1:n + 1], tr, 1, target, {}, rec, seed)


def future_index(T, H, M):
    """(T, H) indices into a length-M track: decisions t + 1 ... t + H after each step t (clipped)."""
    return np.minimum(np.arange(T)[:, None] + 1 + np.arange(H)[None], M - 1)


def batch(episodes, H):
    """Pad whole episodes to the longest one. Returns numpy arrays:
    obs (B, T, OBS), act_prev (B, T, ACT) (the action executed before each obs; zero at the start),
    truth (B, T, TRUTH), mask (B, T), goal (B, GOAL), future (B, T, H, 3) true obstacle positions and
    future_meas (B, T, H, 7) sensor readings for the H decisions after each step."""
    T = max(len(e["obs"]) for e in episodes)
    B = len(episodes)
    out = dict(obs=np.zeros((B, T, OBS), np.float32), act_prev=np.zeros((B, T, ACT), np.float32),
               truth=np.zeros((B, T, TRUTH), np.float32), mask=np.zeros((B, T), np.float32),
               goal=np.stack([e["goal"] for e in episodes]), future=np.zeros((B, T, H, 3), np.float32),
               future_meas=np.zeros((B, T, H, 7), np.float32), origin=np.stack([e["origin"] for e in episodes]),
               has_obst=np.array([float(e.get("has_obst", True)) for e in episodes], np.float32))
    for b, e in enumerate(episodes):
        n = len(e["obs"])
        out["obs"][b, :n], out["obs"][b, n:] = e["obs"], e["obs"][-1]
        out["truth"][b, :n], out["truth"][b, n:] = e["truth"], e["truth"][-1]
        out["act_prev"][b, 1:n] = e["act"]
        out["mask"][b, :n] = 1.0
        out["future"][b] = e["track"][future_index(T, H, len(e["track"]))]
        out["future_meas"][b] = e["track_meas"][future_index(T, H, len(e["track_meas"]))]
    return out
