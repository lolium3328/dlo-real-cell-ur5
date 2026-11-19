"""Episodes of the cell's single-obstacle task for the r10 machinery (docs/notes/cell_policy_20261113.md).

The r10 code (world model, heads, critic, planner) is reused unchanged: the fixed clamp is the "right gripper"
(it never moves; feasible.Limits.fixed_right), the obstacle reading is the right arm's sphere centre (session.Sensor).

- build_episode: per-control-step records -> one decision-level episode as episodes.build_episode, with the
  perceived centerline in obs (what the robot sees) and the true one in `rope_truth` (the world model's
  reconstruction target, world_model.wm_loss), and the sphere readings replayed along the whole track (track_meas).
- run_controller: the CBF baseline ("go to the target", the safety filter dodges) or an oracle plan (tasks.py).
- run_actor: collection with an r10 actor (one head per episode + noise).
- run_planner: closed-loop evaluation with the r10 HeadsPlanner (heads, optionally the hand-written candidates).
"""
from __future__ import annotations

import time

import numpy as np
import torch

from ..belief import Sensor
from ..imagination.episodes import CAP, K, POS, VEL, encode_goal, encode_meas, encode_obs, origin_of
from ..imagination.feasible import feasible_np
from .session import CellSession, limits
from .tasks import oracle_plans, plan_action

ROPE_TRUTH = 3 * 39 + 6


def _measurement_track(track, cfg, origin, offset, seed):
    from ..motion import velocities
    track = np.asarray(track, float)
    vel = velocities(track)
    o = cfg or dict(delta_c_m=0.0, delta_v_mps=0.0, latency_steps=0, bias_m=[0.0, 0.0, 0.0])
    sensor = Sensor(o, None, np.random.default_rng(seed))
    out = []
    for n in range(offset, len(track), K):
        r = sensor.read(track[:n + 1], vel[:n + 1])
        out.append(encode_meas(None if r is None else (r[1], r[2]), origin))
    return np.stack(out)


class Recorder:
    def __init__(self, s):
        self.s, self.seq = s, dict(center=[], perceived=[], left=[], right=[], meas=[], op=[], ov=[])
        self.record()

    def record(self):
        s, f = self.s, self.s.frames[-1]
        d = s.dynamic.drivers[0]
        for k, v in (("center", f.centerline), ("perceived", s.perceived_centerline()), ("left", f.left_pose[:3]),
                     ("right", f.right_pose[:3]), ("op", d.position), ("ov", d.velocity)):
            self.seq[k].append(np.array(v, float))
        self.seq["meas"].append(s.measurement(0))

    def episode(self, info, seed):
        s, q = self.s, self.seq
        origin = origin_of(s.target)
        n = len(q["center"])
        idx = list(range(0, n, K))
        if idx[-1] != n - 1:
            idx.append(n - 1)
        grip = np.stack([np.r_[q["left"][i], q["right"][i]] for i in range(n)])
        obs = np.stack([encode_obs(q["perceived"][i], q["left"][i], q["right"][i], q["meas"][i], origin) for i in idx])
        truth_rope = np.stack([encode_obs(q["center"][i], q["left"][i], q["right"][i], None, origin)[:ROPE_TRUTH] for i in idx])
        truth = np.stack([np.r_[(q["op"][i] - origin) * POS, q["ov"][i] * VEL] for i in idx]).astype(np.float32)
        act = np.stack([(grip[b] - grip[a]) / ((b - a) * CAP) for a, b in zip(idx[:-1], idx[1:])]).astype(np.float32)
        track = np.asarray(s.dynamic.drivers[0].track, float)
        return dict(obs=obs, rope_truth=truth_rope.astype(np.float32), act=act, truth=truth, goal=encode_goal(s.target, origin),
                    track=((track[0::K] - origin) * POS).astype(np.float32), info=dict(info), origin=np.asarray(origin, np.float32),
                    track_meas=_measurement_track(track, (s.noise or {}).get("obstacle"), origin, 0, seed + 1),
                    target={k: np.array(v) for k, v in s.target.items()})


def _finish(s, rec, extra):
    return dict(id=rec["id"], version=rec["version"], task=rec["task"], terminal=s.progress.terminal, steps=int(s.progress.steps),
                min_surface_m=float(min(np.linalg.norm(f.centerline - f.obstacle, axis=1).min() for f in s.frames) - 0.040),
                final_rmse=float(s.errors()["centerline_rmse_m"]), **extra)


def run_controller(rec, kind="cbf", noise=None, seed=0, plan=None, record=True):
    """kind 'cbf': straight to the target (step cap), the safety filter dodges; 'plan': oracle plan `plan` (index into
    tasks.oracle_plans, or the plan itself), no filter. Returns (row, episode or None)."""
    s = CellSession(rec, noise=noise, seed=seed)
    tgt = np.asarray(rec["target_left"], float)
    if kind == "plan" and isinstance(plan, (int, np.integer)):
        plan = oracle_plans(rec)[int(plan)]
    rc = Recorder(s) if record else None
    iv = 0
    while s.progress.terminal == "running":
        if kind == "cbf":
            d = tgt - s.left()
            l1 = np.abs(d).sum()
            u = d if l1 <= CAP else d * CAP / l1
            tr, _ = s.step(np.r_[u, 0, 0, 0], cbf=True)
        else:
            tr, _ = s.step(np.r_[plan_action(plan, s, tgt), 0, 0, 0], cbf=False)
        iv += tr["cbf_intervention"] > 1e-6
        if rc:
            rc.record()
    row = _finish(s, rec, dict(arm=kind, cbf_active_frac=iv / max(s.n, 1)))
    ep = rc.episode(dict(row, source=f"cell_{kind}"), seed) if rc else None
    return row, ep


def run_actor(rec, wm, fc, actor, noise=0.3, seed=0, head=None, deploy=None):
    """Collection: one head per episode (random unless given) plus Gaussian action noise; deploy = deployment noise
    dict (noise.sample_noise) or None. Returns the decision-level episode."""
    from ..imagination.r10 import obstacle_inputs
    s = CellSession(rec, noise=deploy, seed=seed)
    rng = np.random.default_rng(seed)
    head = int(rng.integers(actor.heads)) if head is None else head
    lim, rc = limits(), Recorder(s)
    origin = origin_of(s.target)
    goal = torch.as_tensor(encode_goal(s.target, origin))[None]
    state, o = wm.initial(1, "cpu"), fc.initial(1)
    a_prev, n, iv, start = torch.zeros(1, 6), 0, 0, None
    with torch.no_grad():
        while s.progress.terminal == "running":
            q = rc.seq
            if n % K == 0:
                g = np.r_[q["left"][-1], q["right"][-1]]
                if n:
                    a_prev = torch.as_tensor((g - start) / (K * CAP), dtype=torch.float32)[None]
                start = g
                ob = encode_obs(q["perceived"][-1], q["left"][-1], q["right"][-1], q["meas"][-1], origin)
                state = wm.obs_step(state, a_prev, torch.as_tensor(ob)[None])[0]
                o = fc.step(o, torch.as_tensor(encode_meas(q["meas"][-1], origin))[None], None)
                rope = wm.rope_out(state)
                a = actor(torch.cat(state[:2], -1), goal, obstacle_inputs(fc, o[:, None], rope), torch.ones(1, 1))[0, head].numpy()
                a = np.clip(a.astype(float) + noise * rng.normal(size=6), -1.0, 1.0) if noise else a.astype(float)
            tr, _ = s.step(feasible_np(a, np.r_[q["left"][-1], q["right"][-1]], lim) * CAP)
            iv += tr["cbf_intervention"] > 1e-6
            n += 1
            rc.record()
    row = _finish(s, rec, dict(arm="actor", head=head, noise=float(noise), cbf_active_frac=iv / max(n, 1)))
    return rc.episode(dict(row, source="actor_cell", index=int(rec["index"])), seed)


def run_planner(rec, wm, fc, planner, seed=0, deploy=None, max_time_s=7.2):
    """Closed-loop evaluation with an r10 HeadsPlanner (planner.lim is set here). Returns the row."""
    from ..imagination.r10_exec import HeadsPlanner  # noqa: F401  (type of planner)
    torch.manual_seed(seed)
    s = CellSession(rec, noise=deploy, seed=seed, max_time_s=max_time_s)
    planner.lim, planner.prev = limits(), None
    origin = origin_of(s.target)
    goal = torch.as_tensor(encode_goal(s.target, origin))[None]
    target_grip = np.r_[s.target["left_pose"][:3], s.target["right_pose"][:3]]
    state, o = wm.initial(1, "cpu"), [fc.initial(1)]
    a_prev, n, start, iv = torch.zeros(1, 6), 0, None, 0
    choices, dec_ms = {}, []
    with torch.no_grad():
        while s.progress.terminal == "running":
            f = s.frames[-1]
            if n % K == 0:
                t0 = time.perf_counter()
                g = np.r_[f.left_pose[:3], f.right_pose[:3]]
                if n:
                    a_prev = torch.as_tensor((g - start) / (K * CAP), dtype=torch.float32)[None]
                start = g
                ob = encode_obs(s.perceived_centerline(), f.left_pose, f.right_pose, s.measurement(0), origin)
                state = wm.obs_step(state, a_prev, torch.as_tensor(ob)[None])[0]
                o[0] = fc.step(o[0], torch.as_tensor(encode_meas(s.measurement(0), origin))[None], None)
                name, act, _ = planner.choose(state, o, goal, start, target_grip, origin, a_prev)
                choices[name] = choices.get(name, 0) + 1
                dec_ms.append(1000 * (time.perf_counter() - t0))
            tr, _ = s.step(feasible_np(act, np.r_[f.left_pose[:3], f.right_pose[:3]], planner.lim) * CAP)
            iv += tr["cbf_intervention"] > 1e-6
            n += 1
    return _finish(s, rec, dict(arm="r10", choices=choices, cbf_active_frac=iv / max(n, 1), decision_ms_mean=float(np.mean(dec_ms))))
