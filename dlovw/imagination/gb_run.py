"""G-B (gb.py) in the simulator: collection episodes, the planner and evaluation episodes, checkpoints.

The live tracker (g_system.TRACKER_CONFIG, gb.PRIOR_BOX) is updated with the scene readings at every decision, in the
episode frame. The heads read gb.obstacle_feats on its particles; the planner scores the heads' imagined rollouts
with geom.TrajectoryRisk on the same particles (as G-A, without hand-written candidates).
"""
from __future__ import annotations

import zlib

import numpy as np
import torch

from .actor_critic import CRASH_PENALTY, GAMMA, action_cost, progress
from .episodes import CAP, K, N_PTS, POS, ROPE, encode_goal, encode_obs, origin_of
from .feasible import feasible_np, feasible_torch, limits_of
from .g_system import SPHERE, TRACKER_CONFIG
from .gb import LOOK, M_TRAIN, O_IN_G, PRIOR_BOX, add_parts, obstacle_feats
from .geom import TrajectoryRisk
from .r10 import GroupedSetActor, SetCritic, extra_obstacles
from .tracker import MultiModelTracker


def load_gb(path):
    """(wm, actor, critic) of a G-B checkpoint, CPU, eval mode."""
    from . import actor_critic as ac_mod
    from .world_model import FactoredWM
    st = torch.load(path, map_location="cpu")
    ac_mod.PROGRESS = st.get("progress", "exp")
    wm = FactoredWM(**st["wm_kw"])
    wm.load_state_dict(st["wm"])
    rope_dim = wm.h_dim + wm.z_dim
    actor, critic = GroupedSetActor(rope_dim, st["groups"], st["cap"], o_in=O_IN_G), SetCritic(rope_dim, o_in=O_IN_G)
    actor.load_state_dict(st["actor"])
    critic.load_state_dict(st["critic"])
    return wm.eval(), actor.eval(), critic.eval()


def _trackers(rec, J):
    return [MultiModelTracker(seed=zlib.crc32(f"{rec['id']}_{j}".encode()), prior_box=PRIOR_BOX, config=TRACKER_CONFIG) for j in range(J)]


def _pts(rope):
    return rope[:, :3 * N_PTS].reshape(len(rope), N_PTS, 3) / POS


# ---------------------------------------------------------------------------------------------- collection
def run_episode_gb(rec, wm, actor, *, noise=0.0, seed=0, max_time_s=7.2, target=None, head=None):
    """As r10.run_episode_r10 with the G-B actor (training scenes, one obstacle): one head per episode (random unless
    given) plus Gaussian noise. The episode also gets its two synthetic obstacles and every obstacle's belief particles
    (gb.add_parts), so the trainer can use it as it is."""
    from .episodes import build_episode
    from .rollout import open_actor_session
    s = open_actor_session(rec, None, max_time_s=max_time_s, target=target)
    rng = np.random.default_rng(seed)
    head = int(rng.integers(actor.heads)) if head is None else head
    seq = dict(center=[], left=[], right=[], meas=[], op=[], ov=[])
    try:
        origin = origin_of(s.target)
        goal = torch.as_tensor(encode_goal(s.target, origin))[None]
        drv = s.dynamic.drivers[0]
        lim = limits_of(s)
        trk = _trackers(rec, 1)[0]
        mask = torch.ones(1, 1)

        def record():
            f = s.frames[-1]
            for k_, v in (("center", f.centerline), ("left", f.left_pose[:3]), ("right", f.right_pose[:3]),
                          ("op", drv.position), ("ov", drv.velocity)):
                seq[k_].append(np.array(v, float))
            seq["meas"].append(s.measurement(0))

        record()
        state = wm.initial(1, "cpu")
        a_prev, n, interventions = torch.zeros(1, 6), 0, 0
        with torch.no_grad():
            while s.progress.terminal == "running":
                if n % K == 0:
                    if n:
                        a_prev = torch.as_tensor((np.r_[seq["left"][-1], seq["right"][-1]] - start) / (K * CAP), dtype=torch.float32)[None]
                    start = np.r_[seq["left"][-1], seq["right"][-1]]
                    m = seq["meas"][-1]
                    ob = encode_obs(seq["center"][-1], seq["left"][-1], seq["right"][-1], m, origin)
                    state = wm.obs_step(state, a_prev, torch.as_tensor(ob)[None])[0]
                    trk.update(n, None if m is None else (np.asarray(m[0]) - origin, np.asarray(m[1])))
                    parts = torch.as_tensor(trk.sample(n, max(LOOK) + 1, K, M_TRAIN), dtype=torch.float32)[None]
                    ox = obstacle_feats(_pts(wm.rope_out(state)), [parts], 0, mask)
                    a = actor(torch.cat(state[:2], -1), goal, ox, mask)[0, head].numpy().astype(float)
                    if noise:
                        a = np.clip(a + noise * rng.normal(size=6), -1.0, 1.0)
                tr, _ = s.step(feasible_np(a, np.r_[seq["left"][-1], seq["right"][-1]], lim) * CAP)
                interventions += float(tr["cbf_intervention"]) > 1e-6
                n += 1
                record()
        info = dict(id=rec["id"], terminal=s.progress.terminal, steps=int(s.progress.steps),
                    cbf_active_frac=interventions / max(n, 1), noise=float(noise), head=head)
        track = np.asarray(drv.track)
    finally:
        s.close()
    ep = build_episode(seq["center"], seq["left"], seq["right"], seq["meas"], seq["op"], seq["ov"], track, 0, s.target, info, rec, seed)
    ep["target"] = {k_: np.array(v) for k_, v in s.target.items()}
    ep["extra_track"], ep["extra_meas"] = extra_obstacles(ep, rng, 2)
    return add_parts(ep, seed)


# ---------------------------------------------------------------------------------------------- execution
class GBPlanner:
    """The G-B heads as the only candidates, each imagined as the policy of its own rollout; crash risk from
    geom.TrajectoryRisk on the tracker's particles. Score as r10.HeadsPlanner; switching needs `switch` more."""

    def __init__(self, wm, actor, critic, lim, hold=6, horizon=15, switch=0.3):
        self.wm, self.actor, self.critic, self.lim = wm, actor, critic, lim
        self.horizon, self.switch = horizon, switch
        self.prev, self.last_audit = None, None

    def names(self):
        return [f"head{k}" for k in range(self.actor.heads)]

    def choose(self, state, parts, goal, origin, a_prev, shapes, audit=None):
        """parts: list over obstacles of (M, horizon + max(LOOK) + 1, 3) tensors (metres, episode frame)."""
        from .planner import sampled_risk
        wm, Kh, J = self.wm, self.actor.heads, len(parts)
        B = Kh
        st = tuple(x.expand(B, -1).clone() for x in state)
        g = goal.expand(B, -1)
        mask = torch.ones(B, J)
        off = torch.as_tensor(np.r_[origin, origin], dtype=torch.float32)
        plist = [P[None] for P in parts]                                         # shared by every row
        risk = TrajectoryRisk(parts, shapes, B)
        rope, ap = wm.rope_out(st), a_prev.expand(B, -1)
        total, alive, disc, first, alive_true = torch.zeros(B), torch.ones(B), 1.0, None, torch.ones(B)
        for i in range(self.horizon):
            rf = torch.cat(st[:2], -1)
            a = self.actor(rf, g, obstacle_feats(_pts(rope), plist, i, mask, shapes), mask)[torch.arange(B), torch.arange(Kh)]
            if first is None:
                first = a.clone()
            a = feasible_torch(a, rope[:, 3 * N_PTS:ROPE] / POS + off, self.lim, steps=K)
            st = wm.img_step(st, a, None, sample=False)
            rope = wm.rope_out(st)
            prog, _ = progress(rope, g)
            p_crash, margin = risk.step(i, _pts(rope))
            total += disc * alive * (prog - margin - CRASH_PENALTY * p_crash - action_cost(a, ap))
            if audit is not None:
                zero, one = torch.zeros(B, 2, 3), torch.tensor([1.0, 0.0]).expand(B, 2)
                for j in range(len(audit)):
                    alive_true = alive_true * (1.0 - sampled_risk(rope, one, audit[j][i].reshape(1, 1, 3).expand(B, 2, 3), zero, 2)[0])
            alive, disc, ap = alive * (1.0 - p_crash), disc * GAMMA, a
        crit = disc * alive * self.critic(torch.cat(st[:2], -1), g, obstacle_feats(_pts(rope), plist, self.horizon, mask, shapes), mask)
        total += crit
        if audit is not None:
            self.last_audit = dict(pred=(1.0 - alive).numpy(), true=(1.0 - alive_true).numpy(), crit=crit.numpy(), total=total.numpy())
        pick = int(total.argmax())
        if self.prev is not None and float(total[self.prev]) >= float(total[pick]) - self.switch:
            pick = self.prev
        self.prev = pick
        return self.names()[pick], first[pick].numpy().astype(float), total.numpy()


def episode_gb(rec, wm, planner, *, max_time_s=7.2, seed=0, particles=64, audit=False):
    """Closed-loop G-B episode (C0 setting, any number of obstacles). Row fields as r10.episode_r10 where they apply."""
    from .rollout import open_actor_session
    torch.manual_seed(seed)
    s = open_actor_session(rec, max_time_s=max_time_s)
    try:
        origin = origin_of(s.target)
        goal = torch.as_tensor(encode_goal(s.target, origin))[None]
        drivers = s.dynamic.drivers
        J = len(drivers)
        tracks = [np.asarray(d.track) - origin for d in drivers]
        planner.lim, planner.prev = limits_of(s), None
        trackers = _trackers(rec, J)
        state = wm.initial(1, "cpu")
        a_prev, n, start = torch.zeros(1, 6), 0, None
        choices, steps, diff, audit_log = {}, [], [], []
        ph = planner.horizon + max(LOOK) + 1
        with torch.no_grad():
            while s.progress.terminal == "running":
                if n % K == 0:
                    f = s.frames[-1]
                    if n:
                        a_prev = torch.as_tensor((np.r_[f.left_pose[:3], f.right_pose[:3]] - start) / (K * CAP), dtype=torch.float32)[None]
                    start = np.r_[f.left_pose[:3], f.right_pose[:3]]
                    state = wm.obs_step(state, a_prev, torch.as_tensor(encode_obs(f.centerline, f.left_pose, f.right_pose,
                                                                                    s.measurement(0), origin))[None])[0]
                    for j in range(J):
                        m = s.measurement(j)
                        trackers[j].update(n, None if m is None else (np.asarray(m[0]) - origin, np.asarray(m[1])))
                    parts = [torch.as_tensor(t.sample(n, ph, K, particles), dtype=torch.float32) for t in trackers]
                    fut = (torch.stack([torch.stack([torch.as_tensor(tr[min(n + K * (i + 1), len(tr) - 1)] * POS, dtype=torch.float32)
                                                     for i in range(planner.horizon)]) for tr in tracks]) if audit else None)
                    name, act, _ = planner.choose(state, parts, goal, origin, a_prev, [SPHERE] * J, audit=fut)
                    if audit:
                        aud, rope_now = planner.last_audit, np.asarray(f.centerline)
                        audit_log.append(dict(n=n, pick=int(planner.prev), act=[round(float(x), 3) for x in act],
                                              clear=round(min(float(np.linalg.norm(rope_now - d.position, axis=1).min()) for d in drivers), 4),
                                              **{k: [round(float(x), 3) for x in v] for k, v in aud.items()}))
                    choices[name] = choices.get(name, 0) + 1
                    al, ar = np.abs(act[:3]).sum(), np.abs(act[3:]).sum()
                    diff.append(float(np.abs(act[:3] - act[3:]).sum() / max(al + ar, 1e-9)))
                f = s.frames[-1]
                tr_, _ = s.step(feasible_np(act, np.r_[f.left_pose[:3], f.right_pose[:3]], planner.lim) * CAP)
                n += 1
                steps.append(([s.measurement(j) is not None for j in range(J)], float(tr_["cbf_intervention"])))
        return dict(id=rec["id"], version=rec["version"], terminal=s.progress.terminal, steps=int(s.progress.steps), n_obstacles=J,
                    visible_frac=[float(np.mean([x[0][j] for x in steps])) for j in range(J)],
                    cbf_active=float(np.mean([x[1] > 1e-6 for x in steps])), choices=choices, shape_share=float(np.mean(diff)),
                    obstacle_prediction="tracker", **(dict(audit=audit_log) if audit else {}))
    finally:
        s.close()
