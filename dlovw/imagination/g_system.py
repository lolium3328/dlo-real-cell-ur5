"""G system step 3, G-A (docs/notes/geom_tracker_prereg_20261106.md): the r10 / r11 heads planner with its crash
risk from the multi-model tracker's whole-trajectory particles and the obstacles' signed distance
(geom.TrajectoryRisk) instead of the GRU forecaster's per-step marginals. The heads and the critic still read the
GRU forecaster's states (their training inputs); only the planner's risk term changes. G-A is for finding bugs,
not a decision arm (the policy was trained against the GRU's risk).

Frames: the tracker works in the episode frame (origin subtracted), metres, time in control steps.
"""
from __future__ import annotations

import zlib

import numpy as np
import torch

from .actor_critic import CRASH_PENALTY, GAMMA, action_cost, progress
from .episodes import CAP, K, N_PTS, POS, ROPE
from .feasible import feasible_torch
from .geom import TrajectoryRisk, sphere
from .r10 import obstacle_inputs
from .r10_exec import HeadsPlanner
from .tracker import MultiModelTracker

SPHERE = sphere(0.04)
# Tracker settings frozen after the G-A offline debugging (geom_tracker_prereg section 8.3-8.4): the constant-velocity
# models' velocity decays (0.5 s), the bounded models keep at least 0.3 weight, no constant acceleration, particles
# stay within the seen extent + 0.2 m.
TRACKER_CONFIG = dict(cv_tau=0.5, floor=0.3, prior_w=dict(cv=0.25, ca=0.0, periodic=0.25, patrol=0.25, wide=0.10), extent=0.20)


class GeomHeadsPlanner(HeadsPlanner):
    """As HeadsPlanner; choose() takes the obstacles' particles (list of (M, horizon, 3) tensors, metres, episode
    frame) and shapes. With `audit`, last_audit also holds, on the same rollouts: the GRU's per-step risk (pred_gru)
    and the tracker particles' risk with the steps treated as independent (pred_marg), next to the trajectory
    risk that decides (pred) and the true one (true)."""

    def choose(self, state, o_list, goal, grip, target_grip, origin, a_prev, parts, shapes, audit=None):
        from .planner import candidate_commands, sampled_risk
        wm, fc, Kh, J = self.wm, self.fc, self.actor.heads, len(o_list)
        fixed = (torch.as_tensor(candidate_commands(np.asarray(grip, float), np.asarray(target_grip, float)), dtype=torch.float32)
                 if self.hand else torch.zeros(0, 6))
        B = Kh + len(fixed)
        st = tuple(x.expand(B, -1).clone() for x in state)
        g = goal.expand(B, -1)
        o_im = torch.cat(o_list, 0)[None]
        mask = torch.ones(B, J)
        off = torch.as_tensor(np.r_[origin, origin], dtype=torch.float32)
        rows = torch.arange(B)
        head = torch.cat([torch.arange(Kh), torch.zeros(len(fixed), dtype=torch.long)])
        rope, ap = wm.rope_out(st), a_prev.expand(B, -1)
        risk = TrajectoryRisk(parts, shapes, B)
        total, alive, disc, first = torch.zeros(B), torch.ones(B), 1.0, None
        alive_true, alive_gru, alive_marg = torch.ones(B), torch.ones(B), torch.ones(B)
        for i in range(self.horizon):
            rf = torch.cat(st[:2], -1)
            a = self.actor(rf, g, obstacle_inputs(fc, o_im.expand(B, -1, -1), rope), mask)[rows, head]
            if i < self.hold and len(fixed):
                a = torch.cat([a[:Kh], fixed], 0)
            if first is None:
                first = a.clone()
            a = feasible_torch(a, rope[:, 3 * N_PTS:ROPE] / POS + off, self.lim, steps=K)
            st = wm.img_step(st, a, None, sample=False)
            o_im = torch.stack([fc.step(o_im[:, j], torch.zeros(1, 7), None) for j in range(J)], 1)
            rope = wm.rope_out(st)
            prog, _ = progress(rope, g)
            pts = rope[:, :3 * N_PTS].reshape(B, N_PTS, 3) / POS
            p_crash, margin = risk.step(i, pts)
            total += disc * alive * (prog - margin - CRASH_PENALTY * p_crash - action_cost(a, ap))
            if audit is not None:
                zero, one = torch.zeros(B, 2, 3), torch.tensor([1.0, 0.0]).expand(B, 2)
                w, mu, sd = fc.dist(o_im)
                for j in range(J):
                    alive_true = alive_true * (1.0 - sampled_risk(rope, one, audit[j][i].reshape(1, 1, 3).expand(B, 2, 3), zero, 2)[0])
                    alive_gru = alive_gru * (1.0 - sampled_risk(rope, w[:, j].expand(B, -1), mu[:, j].expand(B, -1, -1),
                                                                sd[:, j].expand(B, -1, -1), self.samples, self.sd_scale)[0])
                marg = TrajectoryRisk([P[:, i:i + 1] for P in parts], shapes, B)
                alive_marg = alive_marg * (1.0 - marg.step(0, pts)[0])
            alive, disc, ap = alive * (1.0 - p_crash), disc * GAMMA, a
        crit = disc * alive * self.critic(torch.cat(st[:2], -1), g, obstacle_inputs(fc, o_im.expand(B, -1, -1), rope), mask)
        total += crit
        if audit is not None:
            self.last_audit = dict(pred=(1.0 - alive).numpy(), true=(1.0 - alive_true).numpy(), pred_gru=(1.0 - alive_gru).numpy(),
                                   pred_marg=(1.0 - alive_marg).numpy(), crit=crit.numpy(), total=total.numpy())
        pick = int(total.argmax())
        if self.prev is not None and float(total[self.prev]) >= float(total[pick]) - self.switch:
            pick = self.prev
        self.prev = pick
        return self.names()[pick], first[pick].numpy().astype(float), total.numpy()


def episode_g(rec, wm, fc, planner, *, max_time_s=7.2, seed=0, particles=64, audit=False, tracker_config=None):
    """episode_r10 with GeomHeadsPlanner: one MultiModelTracker per obstacle, updated with the same readings as the
    GRU forecaster at every decision. Row fields as episode_r10, plus per-decision tracker errors when auditing:
    the particle mean's error one decision and `horizon` decisions ahead against the true track, with the
    number of decisions since the obstacle was last seen."""
    from .episodes import encode_goal, encode_meas, encode_obs, origin_of
    from .feasible import feasible_np, limits_of
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
        target_grip = np.r_[s.target["left_pose"][:3], s.target["right_pose"][:3]]
        lo = np.minimum(target_grip[:3], target_grip[3:]) - origin - 0.25
        hi = np.maximum(target_grip[:3], target_grip[3:]) - origin + 0.25
        cfg = TRACKER_CONFIG if tracker_config is None else tracker_config
        trackers = [MultiModelTracker(seed=zlib.crc32(f"{rec['id']}_{j}".encode()), prior_box=(lo, hi), config=cfg) for j in range(J)]
        state, ofs = wm.initial(1, "cpu"), [fc.initial(1) for _ in range(J)]
        a_prev, n, start = torch.zeros(1, 6), 0, None
        choices, steps, diff, audit_log, unseen = {}, [], [], [], [0] * J
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
                        ofs[j] = fc.step(ofs[j], torch.as_tensor(encode_meas(m, origin))[None], None)
                        trackers[j].update(n, None if m is None else (np.asarray(m[0]) - origin, np.asarray(m[1])))
                        unseen[j] = 0 if m is not None else unseen[j] + 1
                    X = [trk.sample(n, planner.horizon, K, particles) for trk in trackers]
                    parts = [torch.as_tensor(x, dtype=torch.float32) for x in X]
                    fut = (torch.stack([torch.stack([torch.as_tensor(tr[min(n + K * (i + 1), len(tr) - 1)] * POS, dtype=torch.float32)
                                                     for i in range(planner.horizon)]) for tr in tracks]) if audit else None)
                    name, act, _ = planner.choose(state, ofs, goal, start, target_grip, origin, a_prev, parts, [SPHERE] * J, audit=fut)
                    if audit:
                        aud = planner.last_audit
                        err = [[round(float(np.linalg.norm(X[j][:, i].mean(0) - tracks[j][min(n + K * (i + 1), len(tracks[j]) - 1)])), 4)
                                for i in (0, planner.horizon - 1)] for j in range(J)]
                        audit_log.append(dict(n=n, pick=int(planner.prev), act=[round(float(x), 3) for x in act], track_err=err,
                                              unseen=list(unseen), models=[{k: round(v, 2) for k, v in t.active.items()} for t in trackers] if
                                              all(t.steps for t in trackers) else None,
                                              **{k: [round(float(x), 3) for x in v] for k, v in aud.items()}))
                    choices[name] = choices.get(name, 0) + 1
                    al, ar = np.abs(act[:3]).sum(), np.abs(act[3:]).sum()
                    diff.append(float(np.abs(act[:3] - act[3:]).sum() / max(al + ar, 1e-9)))
                f = s.frames[-1]
                tr_, _ = s.step(feasible_np(act, np.r_[f.left_pose[:3], f.right_pose[:3]], planner.lim) * CAP)
                n += 1
                steps.append(([s.measurement(j) is not None for j in range(J)], float(tr_["cbf_intervention"])))
        out = dict(id=rec["id"], version=rec["version"], terminal=s.progress.terminal, steps=int(s.progress.steps), n_obstacles=J,
                   visible_frac=[float(np.mean([x[0][j] for x in steps])) for j in range(J)],
                   cbf_active=float(np.mean([x[1] > 1e-6 for x in steps])), choices=choices, shape_share=float(np.mean(diff)),
                   obstacle_prediction="tracker", **(dict(audit=audit_log) if audit else {}))
        return out
    finally:
        s.close()
