"""r10 / r11 at execution (moved from r10.py to keep files short; behaviour unchanged): HeadsPlanner, the heads as
the planner's candidates, and episode_r10, the closed-loop evaluation episode."""
from __future__ import annotations

import time

import numpy as np
import torch

from .actor_critic import CRASH_PENALTY, GAMMA, action_cost, progress
from .episodes import CAP, K, N_PTS, POS, ROPE, encode_goal, encode_meas, encode_obs, origin_of
from .feasible import feasible_np, feasible_torch, limits_of
from .r10 import obstacle_inputs

SPHERE_R = 0.040                                   # the scenes' obstacle radius (sessions.EVAL_GEOMETRY: 55 mm protected)


class HeadsPlanner:
    """Candidate scoring with the r10 heads: each head is imagined as the policy of its own rollout for the whole
    horizon (as in training); with hand=True the six hand-written candidates are added (held for `hold`
    decisions, then head 0). Crash risk: planner.sampled_risk under each obstacle's blind forecast, combined
    over the obstacles. Score as planner.Planner; switching away from the previous choice needs `switch` more."""

    def __init__(self, wm, fc, actor, critic, lim, hold=6, horizon=15, switch=0.3, samples=64, sd_scale=1.0, hand=False, risk=True):
        """risk=False (ablation d): the score has no crash term and no margin; the heads and the critic decide alone."""
        self.risk = risk
        self.wm, self.fc, self.actor, self.critic, self.lim = wm, fc, actor, critic, lim
        self.hold, self.horizon, self.switch, self.samples, self.sd_scale, self.hand = hold, horizon, switch, samples, sd_scale, hand
        self.prev, self.last_audit = None, None

    def names(self):
        from .planner import NAMES
        return [f"head{k}" for k in range(self.actor.heads)] + ([f"hand_{x}" for x in NAMES[1:]] if self.hand else [])

    def choose(self, state, o_list, goal, grip, target_grip, origin, a_prev, audit=None):
        """`audit` (n_obstacles, horizon, 3), the true obstacle future: on the same imagined rope rollouts, also the
        crash probability against the true obstacles (as planner.Planner); stored in self.last_audit with the
        predicted one, the critic's share of the score and the score. Decisions are unchanged."""
        from .planner import candidate_commands, sampled_risk
        wm, fc, Kh, J = self.wm, self.fc, self.actor.heads, len(o_list)
        fixed = (torch.as_tensor(candidate_commands(np.asarray(grip, float), np.asarray(target_grip, float)), dtype=torch.float32)
                 if self.hand else torch.zeros(0, 6))
        B = Kh + len(fixed)
        st = tuple(x.expand(B, -1).clone() for x in state)
        g = goal.expand(B, -1)
        o_now = torch.cat(o_list, 0)[None]                                          # (1, J, 128)
        mask = torch.ones(B, J)
        off = torch.as_tensor(np.r_[origin, origin], dtype=torch.float32)
        rows = torch.arange(B)
        head = torch.cat([torch.arange(Kh), torch.zeros(len(fixed), dtype=torch.long)])
        rope, ap, o_im = wm.rope_out(st), a_prev.expand(B, -1), o_now
        total, alive, disc, first, alive_true = torch.zeros(B), torch.ones(B), 1.0, None, torch.ones(B)
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
            if i == 0 and audit is not None:
                self.last_rope1 = rope.detach().clone()                              # one decision ahead, every row
            prog, _ = progress(rope, g)
            w, mu, sd = fc.dist(o_im)                                               # blind forecast, the same for all rows
            keep, margin = torch.ones(B), torch.zeros(B)
            for j in range(J):
                p, mg = sampled_risk(rope, w[:, j].expand(B, -1), mu[:, j].expand(B, -1, -1), sd[:, j].expand(B, -1, -1),
                                     self.samples, self.sd_scale)
                keep, margin = keep * (1.0 - p), margin + mg
            p_crash = 1.0 - keep if J > 1 else p
            if not self.risk:
                p_crash, margin = torch.zeros_like(p_crash), torch.zeros_like(margin)
            total += disc * alive * (prog - margin - CRASH_PENALTY * p_crash - action_cost(a, ap))
            if audit is not None:
                zero, one = torch.zeros(B, 2, 3), torch.tensor([1.0, 0.0]).expand(B, 2)
                for j in range(len(audit)):
                    alive_true = alive_true * (1.0 - sampled_risk(rope, one, audit[j][i].reshape(1, 1, 3).expand(B, 2, 3), zero, 2)[0])
            alive, disc, ap = alive * (1.0 - p_crash), disc * GAMMA, a
        crit = disc * alive * self.critic(torch.cat(st[:2], -1), g, obstacle_inputs(fc, o_im.expand(B, -1, -1), rope), mask)
        total += crit
        if audit is not None:
            self.last_audit = dict(pred=(1.0 - alive).numpy(), true=(1.0 - alive_true).numpy(), crit=crit.numpy(), total=total.numpy())
        pick = int(total.argmax())
        if self.prev is not None and float(total[self.prev]) >= float(total[pick]) - self.switch:
            pick = self.prev
        self.prev = pick
        return self.names()[pick], first[pick].numpy().astype(float), total.numpy()


def episode_r10(rec, wm, fc, planner, *, max_time_s=7.2, seed=0, audit=False, perception=None):
    """Closed-loop episode with the r10 planner, any number of obstacles (C0 setting: true rope, scene sensors).
    Row fields as multi.episode_multi where they apply, plus the share of the executed command that changes the
    rope's shape: |a_left - a_right| / (|a_left| + |a_right|). audit=True adds the per-decision risk audit
    (HeadsPlanner.choose) and the true clearance to the nearest obstacle; decisions are unchanged.
    Also (ablation_prereg_20261106.md, metrics 4-5): min_surface_m, the smallest distance over the episode from any
    centerline point to any obstacle's surface (centre distance - SPHERE_R), and the wall time of each decision
    (readings in, forecasters stepped, planner chosen) in ms; auditing inflates the latter.
    perception: a perception.PRESETS entry (e.g. "V2"); the world model then sees the perceived centerline and
    the obstacle readings come from that perception; the row gains `vision` (MultiVisionPerception.summary)."""
    from .rollout import open_actor_session
    torch.manual_seed(seed)
    s = open_actor_session(rec, max_time_s=max_time_s, perception=perception)
    try:
        origin = origin_of(s.target)
        goal = torch.as_tensor(encode_goal(s.target, origin))[None]
        drivers = s.dynamic.drivers
        J = len(drivers)
        tracks = [torch.as_tensor((np.asarray(d.track) - origin) * POS, dtype=torch.float32) for d in drivers]
        audit_log, prev_r1 = [], None
        planner.lim, planner.prev = limits_of(s), None
        target_grip = np.r_[s.target["left_pose"][:3], s.target["right_pose"][:3]]
        state, ofs = wm.initial(1, "cpu"), [fc.initial(1) for _ in range(J)]
        a_prev, n, start = torch.zeros(1, 6), 0, None
        choices, steps, diff, dec_ms = {}, [], [], []
        min_d = min(float(np.linalg.norm(np.asarray(s.frames[-1].centerline) - d.position, axis=1).min()) for d in drivers)
        with torch.no_grad():
            while s.progress.terminal == "running":
                if n % K == 0:
                    t_dec = time.perf_counter()
                    f = s.frames[-1]
                    if n:
                        a_prev = torch.as_tensor((np.r_[f.left_pose[:3], f.right_pose[:3]] - start) / (K * CAP), dtype=torch.float32)[None]
                    start = np.r_[f.left_pose[:3], f.right_pose[:3]]
                    ob = encode_obs(s.perceived_centerline() if perception else f.centerline, f.left_pose, f.right_pose, s.measurement(0), origin)
                    state = wm.obs_step(state, a_prev, torch.as_tensor(ob)[None])[0]
                    if audit and prev_r1 is not None:              # the world model's rope one decision ahead vs the real one
                        audit_log[-1]["wm_err_mm"] = round(float(np.linalg.norm((ob[:3 * N_PTS] - prev_r1).reshape(N_PTS, 3), axis=1).mean()
                                                                 / POS * 1000), 2)
                    for j in range(J):
                        ofs[j] = fc.step(ofs[j], torch.as_tensor(encode_meas(s.measurement(j), origin))[None], None)
                    fut = (torch.stack([torch.stack([tr[min(n + K * (i + 1), len(tr) - 1)] for i in range(planner.horizon)])
                                        for tr in tracks]) if audit else None)
                    name, act, _ = planner.choose(state, ofs, goal, start, target_grip, origin, a_prev, audit=fut)
                    dec_ms.append(1000.0 * (time.perf_counter() - t_dec))
                    if audit:
                        aud, rope_now = planner.last_audit, np.asarray(f.centerline)
                        prev_r1 = planner.last_rope1[planner.prev, :3 * N_PTS].numpy()
                        audit_log.append(dict(n=n, pick=int(planner.prev), act=[round(float(x), 3) for x in act],
                                              clear=round(min(float(np.linalg.norm(rope_now - d.position, axis=1).min()) for d in drivers), 4),
                                              **{k: [round(float(x), 3) for x in v] for k, v in aud.items()}))
                    choices[name] = choices.get(name, 0) + 1
                    al, ar = np.abs(act[:3]).sum(), np.abs(act[3:]).sum()
                    diff.append(float(np.abs(act[:3] - act[3:]).sum() / max(al + ar, 1e-9)))
                f = s.frames[-1]
                tr, _ = s.step(feasible_np(act, np.r_[f.left_pose[:3], f.right_pose[:3]], planner.lim) * CAP)
                n += 1
                min_d = min(min_d, min(float(np.linalg.norm(np.asarray(s.frames[-1].centerline) - d.position, axis=1).min()) for d in drivers))
                steps.append(([s.measurement(j) is not None for j in range(J)], float(tr["cbf_intervention"]),
                              float(tr["errors"]["centerline_rmse_m"])))
        out = dict(id=rec["id"], version=rec["version"], terminal=s.progress.terminal, steps=int(s.progress.steps), n_obstacles=J,
                   visible_frac=[float(np.mean([x[0][j] for x in steps])) for j in range(J)],
                   cbf_active=float(np.mean([x[1] > 1e-6 for x in steps])), final_rmse=steps[-1][2], best_rmse=min(x[2] for x in steps),
                   choices=choices, shape_share=float(np.mean(diff)), obstacle_prediction="r10",
                   min_surface_m=min_d - SPHERE_R, decision_ms_mean=float(np.mean(dec_ms)), decision_ms_p95=float(np.percentile(dec_ms, 95)),
                   decision_ms_max=float(np.max(dec_ms)),
                   **(dict(audit=audit_log, shape_log=[round(x, 3) for x in diff]) if audit else {}))
        perc = getattr(s, "_perc", None)
        if perception and perc is not None and hasattr(perc, "summary"):
            out.update(vision=perc.summary(K))
        if out["terminal"] == "clearance_violation":
            rope = np.asarray(s.frames[-1].centerline)
            dist = [float(np.linalg.norm(rope - d.position, axis=1).min()) for d in drivers]
            out.update(crash_obstacle=int(np.argmin(dist)))
        return out
    finally:
        s.close()
