"""Scene geometry checks shared by the legacy-scene audit and the new scene generator.

All quantities are in metres / seconds; the protected radius matches the legacy safety layer
(40 mm obstacle + 10 mm rope + 5 mm clearance).
"""
import numpy as np

from . import legacy  # noqa: F401  (sets sys.path)
from envs.dynamic_obstacle import DynamicObstacleConfig, DynamicObstacleDriver  # noqa: E402

R_OBS, R_ROPE, CLR = 0.040, 0.010, 0.005
RPROT = R_OBS + R_ROPE + CLR          # protected-clearance radius used by the repo's safety layer
L_ROPE, DT, T_STEPS, HOLD_S = 0.532, 0.008, 500, 0.4
GRIP_R = 0.05                         # crude gripper envelope radius (arm agent does exact geometry)
TABLE_TOP, BACKDROP_Y = -0.04, 0.62 - 0.012
MAX_GRIP_SPEED = 0.0035 / DT          # per-step translation cap -> 0.4375 m/s


def poly_dist(centers, line):
    """Distance from centers (...,3) to polyline (N,3)."""
    centers, line = np.asarray(centers, float), np.asarray(line, float)
    d = np.diff(line, axis=0)
    off = centers[..., None, :] - line[:-1]
    f = np.clip((off * d).sum(-1) / np.maximum((d * d).sum(-1), 1e-24), 0, 1)
    return np.linalg.norm(centers[..., None, :] - (line[:-1] + f[..., None] * d), axis=-1).min(-1)


def replay(profile, steps):
    if profile.get("profile") == "scripted":              # stage-0 scripted motions (dlovw/motion.py)
        from .motion import track
        return track(profile["motion"], steps, DT)
    cfg = DynamicObstacleConfig(**{k: v for k, v in profile.items() if k in DynamicObstacleConfig.__dataclass_fields__})
    drv = DynamicObstacleDriver(None, None, cfg, DT, np.asarray(profile["start_position_m"], float))
    tr = [drv.position.copy()]
    for _ in range(steps):
        tr.append(drv.advance()[0].copy())
    return np.asarray(tr)


EPISODE_STEPS = 900                   # closed-loop episode limit: 7.2 s of 8 ms decisions
CONTACT_SURFACE_M = RPROT - R_OBS     # 15 mm: obstacle surface this close to the target centerline blocks the rope


def target_contact(profiles, centerlines, steps=EPISODE_STEPS):
    """Closest obstacle-surface -> target-centerline distance (m) from the nominal arrival (last step of the
    obstacle-blind reference) to the episode limit, replaying each profile over the whole episode."""
    obs = np.stack([replay(p, steps) for p in profiles], 1)                      # (steps+1, k, 3)
    ta = len(centerlines) - 1
    return float(poly_dist(obs[ta:], centerlines[-1]).min() - R_OBS)


def pad_to(arr, n):
    arr = np.asarray(arr)
    if len(arr) >= n:
        return arr[:n]
    return np.concatenate([arr, np.repeat(arr[-1:], n - len(arr), axis=0)], axis=0)


def longest_true(mask):
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


def in_box(p, lo, hi, tol=1e-3):
    return bool(np.all(np.asarray(p) >= np.asarray(lo) - tol) and np.all(np.asarray(p) <= np.asarray(hi) + tol))


def evaluate(sid, set_name, obs, ref, target, gl, gr, ends, t_arr, workspace, extra):
    """obs (T+1,k,3); ref (T+1,39,3) obstacle-blind nominal rope; target (39,3); gl/gr (T+1,3)."""
    n = T_STEPS + 1
    r = dict(id=sid, set=set_name, n_obs=int(obs.shape[1]) if obs is not None else 0, **extra)
    il, ir, tl, tr_ = (np.asarray(ends[k]) for k in ("initial_left", "initial_right", "target_left", "target_right"))
    r["span_init"] = float(np.linalg.norm(ir - il) / L_ROPE)
    r["span_target"] = float(np.linalg.norm(tr_ - tl) / L_ROPE)
    r["poly_len_target"] = float(np.linalg.norm(np.diff(target, axis=0), axis=1).sum())
    r["ws_ok"] = bool(in_box(il, workspace["left_low"], workspace["left_high"]) and in_box(tl, workspace["left_low"], workspace["left_high"])
                      and in_box(ir, workspace["right_low"], workspace["right_high"]) and in_box(tr_, workspace["right_low"], workspace["right_high"]))
    r["rope_zmin"] = float(min(ref[:, :, 2].min(), target[:, 2].min()))
    r["ends_mm"] = [int(round(v * 1000)) for k in ("initial_left", "initial_right", "target_left", "target_right") for v in ends[k]]
    flags = list(extra.pop("pre_flags", []))
    if r["span_init"] > 0.96 or r["span_target"] > 0.96: flags.append("CRIT_span_over_0.96")
    elif r["span_init"] > 0.94 or r["span_target"] > 0.94: flags.append("WARN_span_tight_0.94")
    if not r["ws_ok"]: flags.append("CRIT_endpoint_outside_workspace")
    if r["n_obs"] == 0:
        flags.append("INFO_no_obstacle")
        r["flags"] = flags
        return r
    obs = pad_to(obs, n); ref = pad_to(ref, n); gl = pad_to(gl, n); gr = pad_to(gr, n)
    k = obs.shape[1]
    h_ref = np.stack([poly_dist(obs[t], ref[t]) for t in range(n)]) - RPROT        # (n,k)
    h_ref_min = h_ref.min(1)
    h_tgt = poly_dist(obs, target).min(1) - RPROT                                    # (n,)
    d_grip = np.minimum(np.linalg.norm(obs - gl[:, None], axis=-1), np.linalg.norm(obs - gr[:, None], axis=-1)).min(1) - (R_OBS + GRIP_R)
    sub_o, sub_r = obs[::5].reshape(-1, 3), ref[::5]
    spatial = min(float(poly_dist(sub_o, sub_r[j]).min()) for j in range(len(sub_r))) - RPROT
    vel = np.linalg.norm(np.diff(obs, axis=0), axis=-1) / DT
    r.update(h0=float(h_ref_min[0]), min_h_ref=float(h_ref_min.min()), t_min_h_ref=float(np.argmin(h_ref_min) * DT),
             first_violation_s=(float(np.flatnonzero(h_ref_min <= 0)[0] * DT) if (h_ref_min <= 0).any() else None),
             spatial_min_h=float(spatial), min_grip=float(d_grip.min()),
             grip_hit_s=(float(np.flatnonzero(d_grip < 0)[0] * DT) if (d_grip < 0).any() else None),
             max_speed=float(vel.max()), obs_zmin=float(obs[..., 2].min()), obs_ymax=float(obs[..., 1].max()),
             t_arrive_s=float(t_arr * DT))
    win = h_tgt[t_arr:]
    r["target_blocked_frac"] = float((win <= 0).mean()) if len(win) else 0.0
    r["target_longest_clear_s"] = float(longest_true(win > 0) * DT)
    r["target_blocked_at_end"] = bool((h_tgt[-int(HOLD_S / DT):] <= 0).any())
    if r["h0"] <= 0: flags.append("CRIT_initial_overlap")
    elif r["h0"] < 0.02: flags.append("WARN_initial_close")
    if r["target_longest_clear_s"] < HOLD_S + 0.08: flags.append("CRIT_target_unholdable")
    elif r["target_blocked_frac"] > 0: flags.append("WARN_target_swept_after_arrival")
    if r["min_h_ref"] > 0.03 and spatial > 0.03: flags.append("CRIT_no_threat_to_nominal_path")
    elif r["min_h_ref"] > 0.03: flags.append("WARN_threat_only_at_wrong_time")
    if r["grip_hit_s"] is not None: flags.append("WARN_obstacle_hits_gripper_zone")
    close = np.flatnonzero(h_ref_min < 0.03)
    last_threat = int(close[-1]) if len(close) else -1
    below = np.flatnonzero(obs[..., 2].min(1) < 0.0)
    r["table_entry_s"] = float(below[0] * DT) if len(below) else None
    r["last_threat_s"] = float(last_threat * DT) if last_threat >= 0 else None
    if len(below):
        flags.append("WARN_obstacle_into_table" + ("_tail_only" if last_threat >= 0 and below[0] > last_threat else ""))
    if abs(r["min_h_ref"]) < 0.01: flags.append("WARN_graze_only_within_1cm")
    if r["obs_ymax"] > BACKDROP_Y - R_OBS:
        yb = np.flatnonzero(obs[..., 1].max(1) > BACKDROP_Y - R_OBS)
        flags.append("WARN_obstacle_into_backdrop" + ("_tail_only" if last_threat >= 0 and yb[0] > last_threat else ""))
    if r["max_speed"] > MAX_GRIP_SPEED: flags.append("WARN_obstacle_faster_than_gripper")
    r["flags"] = flags
    return r


