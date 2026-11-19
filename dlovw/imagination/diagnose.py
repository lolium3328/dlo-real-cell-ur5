"""Why does the actor crash? (docs/notes/imagination_ac_design_20261022.md, section 10)

Runs the actor without noise and, at every decision, also asks the world model what it believes:
- `belief_err`: decoded obstacle position vs the true one (m);
- `imag_clear_belief`: smallest clearance over the next H decisions of an imagined rollout with the
  actor, measured against the obstacle the model itself decodes along the rollout;
- `imag_clear_true`: the same rollout measured against the true obstacle future (what training uses).
Each collision is then put in one class (thresholds are judgement calls, stated in the note):
- "unseen":   the obstacle was hidden in the last 0.5 s and the belief was off by > 30 mm
              -> memory / perception limit;
- "ignored":  0.5 s before the crash the model itself predicted a violation (belief clearance < 0)
              -> the actor went anyway: objective / reward problem;
- "unforeseen": belief roughly right but the imagined rollout stayed clear while reality crashed
              -> the world model's rope prediction is the limit.
"""
from __future__ import annotations

import numpy as np
import torch

from .episodes import CAP, K, N_PTS, POS, PROTECTED_M, encode_goal, encode_obs, origin_of
from .corridor import CorridorForecast
from .feasible import feasible_np, limits_of
from .rollout import open_actor_session, oracle_now, servo_action, servo_mode

H_DIAG, LOOKBACK_DEC, HIDDEN_WINDOW, BELIEF_TOL_M = 16, 6, 62, 0.030


def _clearance(rope_dec, obstacle):
    pts = rope_dec[..., :3 * N_PTS].reshape(*rope_dec.shape[:-1], N_PTS, 3) / POS
    return (pts - obstacle[..., None, :] / POS).norm(dim=-1).min(-1).values - PROTECTED_M


def episode(rec, wm, actor, *, target=None, max_time_s=7.2, seed=0, servo=False, planner=None, trace=False, perception=None,
            trigger=None):
    """`planner` (planner.Planner): choose among discrete candidates instead of executing the actor.
    `trace`: also return per-step goal errors, the success hold counter, the obstacle's clearance from
    the target rope, and the planner's choice per decision (timeout analysis).
    `trigger` (docs/notes/perception_robustness_20261023.md, section 10): also re-decide between the 80 ms
    decision boundaries when a new obstacle reading arrives - "reappear": the first two readings after the
    obstacle was not seen (the second one is the first with a fitted velocity for the cameras); "frame": every
    new reading. An event decision starts from the states saved before the last boundary's observe step, observes
    the current rope, grippers and reading instead, and scores the candidates from the current gripper position;
    the new command applies at once. The recurrent states carried to the next boundary are left unchanged."""
    torch.manual_seed(seed)
    s = open_actor_session(rec, max_time_s=max_time_s, target=target, perception=perception)
    choices, tr_rows, tr_choice, tr_dec = {}, [], [], []
    try:
        origin = origin_of(s.target)
        goal = torch.as_tensor(encode_goal(s.target, origin))[None]
        drv = s.dynamic.drivers[0]
        lim = limits_of(s)
        if planner is not None:
            planner.lim, planner.prev = lim, None
        target_grip = np.r_[s.target["left_pose"][:3], s.target["right_pose"][:3]]
        track = torch.as_tensor((np.asarray(drv.track) - origin) * POS, dtype=torch.float32)
        cf = (CorridorForecast(seed=seed) if planner is not None and (planner.corridor or planner.obst_model is not None)
              else None)
        if cf is not None:
            cf.update(0, s.measurement(0))
        of = planner.obst_model.initial(1) if planner is not None and planner.obst_model is not None else None
        state = wm.initial(1, "cpu")
        a_prev, n, start, servoing, servo_steps = torch.zeros(1, 6), 0, None, False, 0
        steps, decisions = [], []
        pre_state = pre_of = None
        last_rid, fresh, dec_fresh, event, n_events, n_switch = None, 0, 0, False, 0, 0

        def plan_inputs(n, of_in):
            fut = (torch.stack([track[min(n + K * (i + 1), len(track) - 1)] for i in range(planner.horizon)])
                   if planner.oracle else None)
            parts = (torch.as_tensor((cf.forecast(n, planner.horizon, K) - origin) * POS, dtype=torch.float32)
                     if planner.corridor and cf.ready() else None)
            obst, of_out = None, of_in
            if planner.obst_model is not None:        # standalone obstacle forecaster (its state may be falsy)
                from .obstacle_data import corridor_features
                from .episodes import encode_meas
                fe = torch.as_tensor(corridor_features(cf, n, origin))[None]
                of_out = planner.obst_model.step(of_in, torch.as_tensor(encode_meas(s.measurement(0), origin))[None], fe)
                obst = planner.obst_model.blind(of_out, fe, planner.horizon)
            return fut, parts, obst, of_out

        with torch.no_grad():
            while s.progress.terminal == "running":
                if n % K and event and not servoing:   # event decision between boundaries
                    f = s.frames[-1]
                    grip_now = np.r_[f.left_pose[:3], f.right_pose[:3]]
                    seen = s.perceived_centerline() if perception else f.centerline
                    o = encode_obs(seen, f.left_pose, f.right_pose, s.measurement(0), origin)
                    st_ev = wm.obs_step(pre_state, a_prev, torch.as_tensor(o)[None])[0]
                    fut, parts, obst, _ = plan_inputs(n, pre_of)
                    before = planner.prev
                    name, act, scores = planner.choose(st_ev, goal, grip_now, target_grip, origin, a_prev, fut, parts, obst)
                    n_events += 1
                    n_switch += planner.prev != before
                    dec_fresh = fresh
                    if trace:
                        tr_dec.append(dict(n=n, event=True, choice=name, scores=[round(float(x), 2) for x in scores],
                                           obst=[round(float(x), 4) for x in drv.position], visible=s.measurement(0) is not None))
                event = False
                if n % K == 0:
                    f = s.frames[-1]
                    if n:
                        a_prev = torch.as_tensor((np.r_[f.left_pose[:3], f.right_pose[:3]] - start) / (K * CAP), dtype=torch.float32)[None]
                    start = np.r_[f.left_pose[:3], f.right_pose[:3]]
                    seen = s.perceived_centerline() if perception else f.centerline
                    o = encode_obs(seen, f.left_pose, f.right_pose, s.measurement(0), origin)
                    pre_state, pre_of, dec_fresh = state, of, fresh
                    state = wm.obs_step(state, a_prev, torch.as_tensor(o)[None])[0]
                    a = actor(wm.feat(state), goal, oracle_now(actor, drv, n, origin))
                    point, spread = wm.obstacle_out(state)
                    belief = point[0, :3] / POS
                    true = torch.as_tensor(drv.position - origin, dtype=torch.float32)
                    st, aa, cb, ct = state, a, [], []
                    for i in range(H_DIAG):
                        st = wm.img_step(st, aa)
                        rope = wm.rope_out(st)
                        cb.append(float(_clearance(rope, wm.obstacle_out(st)[0][:, :3])[0]))
                        ct.append(float(_clearance(rope, track[min(n + K * (i + 1), len(track) - 1)][None])[0]))
                        aa = actor(wm.feat(st), goal, oracle_now(actor, drv, n + K * (i + 1), origin))
                    decisions.append(dict(n=n, belief_err=float((belief - true).norm()), belief_spread=float(spread[0]) * 3 ** 0.5 / POS,
                                          imag_clear_belief=min(cb),
                                          imag_clear_true=min(ct)))
                    act = a[0].numpy().astype(float)
                    if planner is not None:
                        fut, parts, obst, of = plan_inputs(n, of)
                        name, act, scores = planner.choose(state, goal, start, target_grip, origin, a_prev, fut, parts, obst)
                        choices[name] = choices.get(name, 0) + 1
                        tr_choice.append(name)
                        if trace:
                            tr_dec.append(dict(n=n, choice=name, scores=[round(float(x), 2) for x in scores],
                                               grip=[round(float(x), 4) for x in start], obst=[round(float(x), 4) for x in drv.position],
                                               rmse=round(float(np.sqrt(((np.asarray(f.centerline) - np.asarray(s.target["centerline"])) ** 2).sum(-1).mean())), 4),
                                               visible=s.measurement(0) is not None))
                    servoing = servo and servo_mode(s, servoing)
                f = s.frames[-1]
                u = servo_action(s) if servoing else feasible_np(act, np.r_[f.left_pose[:3], f.right_pose[:3]], lim)
                tr, _ = s.step(u * CAP)
                servo_steps += servoing
                n += 1
                if cf is not None:
                    cf.update(n, s.measurement(0))
                if trigger and planner is not None:
                    rid = s.reading_id(0)
                    new_reading = rid is not None and rid != last_rid
                    fresh = 0 if rid is None else fresh + new_reading
                    last_rid = rid
                    event = new_reading and (trigger == "frame" or dec_fresh < 2)
                steps.append((s.measurement(0) is not None, float(tr["min_h"]), float(tr["cbf_intervention"]),
                              float(tr["errors"]["centerline_rmse_m"])))
                if trace:
                    e = tr["errors"]
                    tgt_clear = float(np.linalg.norm(np.asarray(s.target["centerline"]) - drv.position, axis=1).min()) - PROTECTED_M
                    tr_rows.append((e["centerline_rmse_m"], e["point_error_p95_m"], e["endpoint_rmse_m"], int(s.progress.hold_count),
                                    tgt_clear, float(tr["min_h"]), s.measurement(0) is not None))
        out = dict(id=rec["id"], version=rec["version"], terminal=s.progress.terminal, steps=int(s.progress.steps),
                   visible_frac=float(np.mean([x[0] for x in steps])), cbf_active=float(np.mean([x[2] > 1e-6 for x in steps])),
                   final_rmse=steps[-1][3], best_rmse=min(x[3] for x in steps), servo_frac=servo_steps / max(n, 1),
                   choices=choices, events=n_events, event_switches=n_switch,
                   obstacle_prediction=(type(planner.obst_model).__name__ if planner is not None and planner.obst_model is not None
                                        else "oracle" if planner is not None and planner.oracle else "world_model" if planner is not None else None),
                   perceived_rmse=float(np.mean(s._perc.err)) if perception and s._perc is not None else None,
                   **(dict(vision_obst_err=float(np.mean(s._perc.obst_err)) if s._perc.obst_err else None,
                           vision_detect_frac=s._perc.frames_detected / max(s._perc.frames_seen, 1),
                           vision_frame_ms=float(np.median(s._perc.frame_ms)) if s._perc.frame_ms else None,
                           perceived_p95=float(np.percentile(s._perc.err, 95)))
                      if perception and getattr(s._perc, "frame_ms", None) is not None else {}),
                   **(dict(trace=dict(rows=tr_rows, choices=tr_choice, decisions=tr_dec)) if trace else {}),
                   belief_err_visible=[d["belief_err"] for d, ok in zip(decisions, [steps[max(d["n"] - 1, 0)][0] for d in decisions]) if ok],
                   belief_err_hidden=[d["belief_err"] for d, ok in zip(decisions, [steps[max(d["n"] - 1, 0)][0] for d in decisions]) if not ok])
        if out["terminal"] == "clearance_violation":
            nc = len(steps)
            last, before = decisions[-1], decisions[max(len(decisions) - 1 - LOOKBACK_DEC, 0)]
            hidden_recent = not all(x[0] for x in steps[max(nc - HIDDEN_WINDOW, 0):])
            if hidden_recent and last["belief_err"] > BELIEF_TOL_M:
                cls = "unseen"
            elif before["imag_clear_belief"] < 0:
                cls = "ignored"
            else:
                cls = "unforeseen"
            lead = 0                                  # decisions before the crash with a predicted violation, unbroken
            for d in reversed(decisions):
                if d["imag_clear_belief"] >= 0:
                    break
                lead += 1
            out.update(crash_class=cls, hidden_recent=hidden_recent, belief_err_at_crash=last["belief_err"], crash_lead_s=lead * K * 0.008,
                       belief_spread_at_crash=last["belief_spread"], belief_spread_before=before["belief_spread"],
                       imag_clear_belief_before=before["imag_clear_belief"], imag_clear_true_before=before["imag_clear_true"])
        return out
    finally:
        s.close()


def summary(rows):
    """Flat numbers for the log: outcomes, collision causes, obstacle belief errors (mm)."""
    ok = [r for r in rows if "error" not in r]
    n = max(len(ok), 1)
    term = [r["terminal"] for r in ok]
    crashes = [r for r in ok if r["terminal"] == "clearance_violation"]
    mm = lambda xs: float(1000 * np.median(xs)) if len(xs) else None  # noqa: E731
    out = dict(n=len(ok), errors=len(rows) - len(ok), success=term.count("success") / n,
               collision=len(crashes) / n, timeout=term.count("task_timeout") / n,
               success_steps_median=float(np.median([r["steps"] for r in ok if r["terminal"] == "success"])) if "success" in term else None,
               cbf_active=float(np.mean([r["cbf_active"] for r in ok])) if ok else None,
               perceived_rmse_mm=float(1000 * np.mean([r["perceived_rmse"] for r in ok if r.get("perceived_rmse") is not None]))
               if any(r.get("perceived_rmse") is not None for r in ok) else None,
               belief_err_visible_mm=mm([e for r in ok for e in r["belief_err_visible"]]),
               belief_err_hidden_mm=mm([e for r in ok for e in r["belief_err_hidden"]]))
    late = [r for r in ok if r["terminal"] == "task_timeout" and "final_rmse" in r]
    if late:                                   # did timeouts get close (precision) or not (waiting / stuck)?
        out.update(timeout_final_rmse_mm=mm([r["final_rmse"] for r in late]), timeout_best_rmse_mm=mm([r["best_rmse"] for r in late]))
    for cls in ("unseen", "ignored", "unforeseen"):
        out[f"crash_{cls}"] = sum(r["crash_class"] == cls for r in crashes)
    if crashes:
        leads = [r["crash_lead_s"] for r in crashes if "crash_lead_s" in r]
        if leads:
            out.update(crash_lead_s_median=float(np.median(leads)), crash_lead_over_1s=float(np.mean(np.asarray(leads) > 1.0)))
        out.update(crash_hidden_recent=float(np.mean([r["hidden_recent"] for r in crashes])),
                   crash_belief_err_mm=mm([r["belief_err_at_crash"] for r in crashes]),
                   crash_belief_spread_mm=mm([r.get("belief_spread_at_crash") for r in crashes if r.get("belief_spread_at_crash") is not None]),
                   crash_model_clear_before_mm=mm([r["imag_clear_belief_before"] for r in crashes]),
                   crash_true_clear_before_mm=mm([r["imag_clear_true_before"] for r in crashes]))
    for v in sorted({r["version"] for r in ok}):
        sub = [r["terminal"] for r in ok if r["version"] == v]
        out[f"{v}_success"] = sub.count("success") / len(sub)
        out[f"{v}_collision"] = sub.count("clearance_violation") / len(sub)
    return out
