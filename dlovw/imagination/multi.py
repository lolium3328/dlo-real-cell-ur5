"""Episodes with several obstacles, without retraining (docs/notes/multi_obstacle_prereg_20261030.md).

The world model, actor and critic were trained with one obstacle. Here:
- every obstacle gets its own standalone forecaster state and its own world-model obstacle filter state
  (`wm._obstacle_step`; the rope part is observed once per decision, as in diagnose.episode);
- the planner's crash risk covers all obstacles (planner.choose with a list of per-obstacle forecasts);
- the actor and critic, which take one obstacle, see the most threatening one: the obstacle whose predicted
  path over the planning horizon comes closest to the current rope.
With one obstacle this reproduces diagnose.episode step for step (same calls in the same order, including the
random draws of its diagnostic imagination), which is checked before any multi-obstacle run.
"""
from __future__ import annotations

import numpy as np
import torch

from .diagnose import H_DIAG, HIDDEN_WINDOW, _clearance
from .episodes import CAP, K, POS, encode_goal, encode_meas, encode_obs, origin_of
from .corridor import CorridorForecast
from .feasible import feasible_np, limits_of
from .obstacle_data import corridor_features
from .rollout import open_actor_session, oracle_now


def _closest(paths, rope):
    """Smallest distance (m) from each obstacle's predicted path (J, H, 3) to the rope (N, 3)."""
    return np.linalg.norm(paths[:, :, None] - rope[None, None], axis=-1).min(axis=(1, 2))


def episode_multi(rec, wm, actor, *, max_time_s=7.2, seed=0, planner=None, risk_all=True, audit=False):
    """risk_all=False (ablation): the planner's crash risk only covers the obstacle the actor and critic see."""
    if planner is None or planner.corridor:
        raise ValueError("episode_multi needs the candidate planner (without corridor particles)")
    torch.manual_seed(seed)
    s = open_actor_session(rec, max_time_s=max_time_s)
    try:
        origin = origin_of(s.target)
        goal = torch.as_tensor(encode_goal(s.target, origin))[None]
        drivers = s.dynamic.drivers
        J = len(drivers)
        lim = limits_of(s)
        planner.lim, planner.prev = lim, None
        target_grip = np.r_[s.target["left_pose"][:3], s.target["right_pose"][:3]]
        tracks = [torch.as_tensor((np.asarray(d.track) - origin) * POS, dtype=torch.float32) for d in drivers]
        use_fc = planner.obst_model is not None
        cfs = [CorridorForecast(seed=seed) for _ in range(J)] if use_fc else None
        if use_fc:
            for j in range(J):
                cfs[j].update(0, s.measurement(j))
        ofs = [planner.obst_model.initial(1) for _ in range(J)] if use_fc else None
        state = wm.initial(1, "cpu")
        o_rest = [state[2].clone() for _ in range(J)]          # filter states of obstacles 1..J-1 (0 lives in state)
        a_prev, n, start = torch.zeros(1, 6), 0, None
        choices, steps, picked, threat_log, audit_log, shape = {}, [], [], [], [], []
        with torch.no_grad():
            while s.progress.terminal == "running":
                if n % K == 0:
                    f = s.frames[-1]
                    if n:
                        a_prev = torch.as_tensor((np.r_[f.left_pose[:3], f.right_pose[:3]] - start) / (K * CAP), dtype=torch.float32)[None]
                    start = np.r_[f.left_pose[:3], f.right_pose[:3]]
                    o = encode_obs(f.centerline, f.left_pose, f.right_pose, s.measurement(0), origin)
                    state = wm.obs_step(state, a_prev, torch.as_tensor(o)[None])[0]
                    for j in range(1, J):
                        o_rest[j] = wm._obstacle_step(o_rest[j], torch.as_tensor(encode_meas(s.measurement(j), origin))[None])
                    o_all = [state[2]] + o_rest[1:]
                    preds = []
                    if use_fc:
                        for j in range(J):
                            fe = torch.as_tensor(corridor_features(cfs[j], n, origin))[None]
                            ofs[j] = planner.obst_model.step(ofs[j], torch.as_tensor(encode_meas(s.measurement(j), origin))[None], fe)
                            preds.append(planner.obst_model.blind(ofs[j], fe, planner.horizon))
                    fut = (torch.stack([torch.stack([tr[min(n + K * (i + 1), len(tr) - 1)] for i in range(planner.horizon)])
                                        for tr in tracks]) if planner.oracle else None)
                    if fut is not None:
                        paths = fut.numpy() / POS + origin
                    elif use_fc:
                        paths = np.stack([np.stack([((p[0][0][..., None] * p[1][0]).sum(0)).numpy() for p in pr]) for pr in preds]) / POS + origin
                    else:
                        paths = np.stack([wm.obstacle_out((state[0], state[1], oj))[0][:, :3].numpy() for oj in o_all]) / POS + origin
                    close = _closest(paths, np.asarray(f.centerline))
                    sel = int(np.argmin(close))
                    picked.append(sel)
                    threat_log.append(close.tolist())
                    state_sel = (state[0], state[1], o_all[sel])
                    a = actor(wm.feat(state_sel), goal, oracle_now(actor, drivers[sel], n, origin))
                    st, aa = state_sel, a                          # diagnostic imagination as in diagnose.episode
                    for i in range(H_DIAG):                        # (keeps its random draws, so one obstacle reproduces it)
                        st = wm.img_step(st, aa)
                        _clearance(wm.rope_out(st), wm.obstacle_out(st)[0][:, :3])
                        aa = actor(wm.feat(st), goal, oracle_now(actor, drivers[sel], n + K * (i + 1), origin))
                    if not risk_all:
                        fut = None if fut is None else fut[sel:sel + 1]
                        preds = preds[sel:sel + 1]
                    true_fut = (torch.stack([torch.stack([tr[min(n + K * (i + 1), len(tr) - 1)] for i in range(planner.horizon)])
                                             for tr in tracks]) if audit else None)
                    name, act, _ = planner.choose(state_sel, goal, start, target_grip, origin, a_prev, fut, None,
                                                  preds if use_fc else None, audit=true_fut)
                    if audit:
                        aud = planner.last_audit
                        audit_log.append(dict(n=n, pick=int(planner.prev), pred=[round(float(x), 3) for x in aud["pred"]],
                                              true=[round(float(x), 3) for x in aud["true"]]))
                    choices[name] = choices.get(name, 0) + 1
                    shape.append(float(np.abs(act[:3] - act[3:]).sum() / max(np.abs(act[:3]).sum() + np.abs(act[3:]).sum(), 1e-9)))
                f = s.frames[-1]
                tr, _ = s.step(feasible_np(act, np.r_[f.left_pose[:3], f.right_pose[:3]], lim) * CAP)
                n += 1
                if use_fc:
                    for j in range(J):
                        cfs[j].update(n, s.measurement(j))
                steps.append(([s.measurement(j) is not None for j in range(J)], float(tr["cbf_intervention"]),
                              float(tr["errors"]["centerline_rmse_m"])))
        out = dict(id=rec["id"], version=rec["version"], terminal=s.progress.terminal, steps=int(s.progress.steps), n_obstacles=J,
                   visible_frac=[float(np.mean([x[0][j] for x in steps])) for j in range(J)],
                   cbf_active=float(np.mean([x[1] > 1e-6 for x in steps])), final_rmse=steps[-1][2], best_rmse=min(x[2] for x in steps),
                   choices=choices, picked_frac=[float(np.mean([p == j for p in picked])) for j in range(J)],
                   shape_share=float(np.mean(shape)) if shape else None,
                   pick_switch_frac=float(np.mean([a != b for a, b in zip(picked, picked[1:])])) if len(picked) > 1 else 0.0,
                   obstacle_prediction="oracle" if planner.oracle else type(planner.obst_model).__name__ if use_fc else "world_model",
                   **(dict(audit=audit_log) if audit else {}))
        if out["terminal"] == "clearance_violation":
            rope = np.asarray(s.frames[-1].centerline)
            dist = [float(np.linalg.norm(rope - d.position, axis=1).min()) for d in drivers]
            j = int(np.argmin(dist))
            out.update(crash_obstacle=j, crash_obstacle_was_picked=picked[-1] == j,
                       crash_hidden_recent=not all(x[0][j] for x in steps[-HIDDEN_WINDOW:]),
                       crash_predicted_closest_mm=[round(1000 * c, 1) for c in threat_log[-1]])
        return out
    finally:
        s.close()
