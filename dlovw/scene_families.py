"""Obstacle designs for scene set v2 (docs/notes/hard_scene_design_20261020.md).

Each family returns (profiles, designs, knob): legacy driver dicts, per-obstacle design records,
and the family's difficulty knob(s), sampled uniformly over the whole range (easy end included).
Obstacles start moving `lead_s` before their crossing, so the threat is visible well in advance.
"""
from __future__ import annotations

import numpy as np

from . import scenes

MIN_OBSTACLE_GAP_M = 0.09            # centre distance: two 40 mm spheres plus 10 mm


def _crossing(rng, ref, *, t_cross, fraction, speed, lead, hint=None, max_angle_deg=None):
    prof, des = scenes.crossing_profile(rng, ref["centerlines"], t_cross=t_cross, fraction=fraction,
                                        speed=speed, approach=speed * lead, seed=int(rng.integers(1, 2**31 - 1)),
                                        direction_hint=hint, max_angle_deg=max_angle_deg)
    des["lead_s"] = float(lead)
    return prof, des


def sweep(rng, ref):
    """F1: one fast obstacle through the rope mid-span. Knob: speed (0.20-0.43 m/s, below the
    gripper cap of 0.4375 m/s), so the grippers can match it but the mid-span cannot react late."""
    speed = rng.uniform(0.20, 0.43)
    lead = rng.uniform(0.7, 1.2)
    t_cross = rng.uniform(max(0.9, lead), 1.8)
    prof, des = _crossing(rng, ref, t_cross=t_cross, fraction=rng.uniform(0.4, 0.6), speed=speed, lead=lead)
    return [prof], [des], dict(speed=float(speed), lead_s=float(lead), t_cross=float(t_cross))


def dense(rng, ref):
    """F2: two obstacles cross two interior points (~0.3 and ~0.7 of the rope, about 21 cm apart)
    within a short gap, from the same side on near-parallel paths (within 30 deg of the first), so
    where the rope dodges the first decides whether it meets the second. Knob: the crossing gap.
    (A three-obstacle version did not fit the workspace: most candidates had obstacles passing
    through each other or the grippers; see the design note.)"""
    t0 = rng.uniform(0.75, 1.1)
    gap = rng.uniform(0.15, 0.5)
    fractions = rng.permutation([0.30, 0.70]) + rng.uniform(-0.03, 0.03, size=2)
    profiles, designs, hint = [], [], None
    for t, f in zip((t0, t0 + gap), fractions):
        speed = rng.uniform(0.20, 0.35)
        approach = min(rng.uniform(0.15, 0.30), speed * t)     # as v1; start no earlier than t = 0
        prof, des = _crossing(rng, ref, t_cross=t, fraction=float(f), speed=speed, lead=approach / speed,
                              hint=hint, max_angle_deg=None if hint is None else 30.0)
        hint = des["direction"] if hint is None else hint
        profiles.append(prof); designs.append(des)
    return profiles, designs, dict(gap_s=float(gap))


def _arrival_s(ref):
    """Time the obstacle-blind reference grippers reach their final positions (1 mm)."""
    d = np.maximum(np.linalg.norm(ref["left"][:, :3] - ref["left"][-1, :3], axis=1),
                   np.linalg.norm(ref["right"][:, :3] - ref["right"][-1, :3], axis=1))
    return float(np.flatnonzero(d <= 0.001)[0]) * scenes.DT


def _sweep_frame(ref, t, fraction):
    """Reference rope point at (t, fraction), its transport velocity and the sweep-surface normal."""
    cl = ref["centerlines"]
    k = min(int(round(t / scenes.DT)), len(cl) - 1)
    k0, k1 = max(k - 5, 0), min(k + 5, len(cl) - 1)
    p, tangent = scenes.point_at_fraction(cl[k], fraction)
    v = (scenes.point_at_fraction(cl[k1], fraction)[0] - scenes.point_at_fraction(cl[k0], fraction)[0]) / (scenes.DT * max(k1 - k0, 1))
    if np.linalg.norm(v) < 0.03:
        raise RuntimeError("Reference rope nearly still at the gate")
    n = np.cross(tangent, v)
    return p, v, n / np.linalg.norm(n)


def periodic_gate(rng, ref):
    """F3: one sphere oscillates (sine) across the rope's sweep surface, centred on the reference
    crossing point, so the nominal path is blocked or open depending on the phase when the rope
    arrives. Knobs: period (1.6-3.0 s, peak acceleration <= 1.5 m/s^2 so the tube assumption holds)
    and amplitude (8-10 cm, gate open while |A sin| > 55 mm)."""
    t_mid = rng.uniform(0.4, 0.6) * _arrival_s(ref)
    p, v, n = _sweep_frame(ref, t_mid, rng.uniform(0.4, 0.6))
    amp = rng.uniform(0.08, 0.10)
    period = rng.uniform(max(1.6, 2 * np.pi * np.sqrt(amp / 1.5)), 3.0)
    sign = float(rng.choice([-1.0, 1.0]))
    omega = 2 * np.pi / period
    prof = dict(profile="sine", amplitude_m=float(amp), frequency_hz=float(1 / period), direction=[float(x) for x in sign * n],
                speed_mps=float(amp * omega), a_max_mps2=1.5, v_max_mps=1.0, delta_c_m=0.005, delta_v_mps=0.02,
                latency_steps=0.0, seed=int(rng.integers(1, 2**31 - 1)), start_delay_s=0.0,
                start_position_m=[float(x) for x in p])
    phase = (omega * t_mid + (np.pi if sign < 0 else 0.0)) % (2 * np.pi)
    des = dict(t_cross=float(t_mid), fraction=None, speed=float(amp * omega), crossing_point=[float(x) for x in p],
               direction=[float(x) for x in n], transport_dir=[float(x) for x in v / np.linalg.norm(v)])
    open_frac = 1 - (2 / np.pi) * np.arcsin(min(0.055 / amp, 1.0))
    return [prof], [des], dict(period_s=float(period), amplitude_m=float(amp), phase_at_nominal=float(phase),
                               open_fraction=float(open_frac))


def static_gate(rng, ref):
    """F4: two static spheres form a slot across the rope's sweep surface; the slot is offset from the
    reference crossing point so the nominal path hits one sphere. The rope must shift into the slot
    (or go around both) and can get caught on a sphere. Knobs: centre separation 11-20 cm (55 mm
    protected radius each side: 11 cm leaves no margin) and the slot offset (3-8 cm)."""
    t_mid = rng.uniform(0.4, 0.6) * _arrival_s(ref)
    p, v, n = _sweep_frame(ref, t_mid, rng.uniform(0.4, 0.6))
    gap = rng.uniform(0.11, 0.20)
    offset = float(rng.choice([-1.0, 1.0])) * rng.uniform(0.03, 0.08)
    centre = p + offset * n
    profiles, designs = [], []
    for side in (1.0, -1.0):
        c = centre + side * gap / 2 * n
        profiles.append(dict(profile="static", speed_mps=0.0, v_max_mps=0.0, a_max_mps2=0.0, direction=[float(x) for x in n],
                             delta_c_m=0.005, delta_v_mps=0.0, latency_steps=0.0, seed=int(rng.integers(1, 2**31 - 1)),
                             start_delay_s=0.0, start_position_m=[float(x) for x in c]))
        designs.append(dict(t_cross=float(t_mid), fraction=None, speed=0.0, crossing_point=[float(x) for x in c],
                            direction=[float(x) for x in n],
                            transport_dir=[float(x) for x in v / np.linalg.norm(v)]))
    return profiles, designs, dict(gap_m=float(gap), offset_m=float(abs(offset)))



# ---- stage 0 (docs/notes/stage0_prereg_20261022.md, section 2) -------------------------------------
V_PEAK_MAX = 0.45                     # peak speed of the scripted motions (m/s)


def _perception(rng):
    """Per-scene perception level: nominal (as v1/v2) or degraded (noise, latency, unknown bias)."""
    if rng.uniform() < 0.5:
        return dict(level="nominal", delta_c_m=0.005, delta_v_mps=0.02, latency_steps=0, bias_m=[0.0, 0.0, 0.0])
    d = rng.normal(size=3)
    return dict(level="degraded", delta_c_m=0.015, delta_v_mps=0.05, latency_steps=3,
                bias_m=[float(x) for x in 0.010 * d / np.linalg.norm(d)])


def _scripted(rng, motion, n):
    from . import motion as motion_mod
    start = motion_mod.track(motion, 0)[0]
    return dict(profile="scripted", motion=motion, speed_mps=0.0, direction=[float(x) for x in n], a_max_mps2=1.5,
                v_max_mps=1.0, delta_c_m=0.005, delta_v_mps=0.02, latency_steps=0.0,
                seed=int(rng.integers(1, 2**31 - 1)), start_delay_s=0.0, start_position_m=[float(x) for x in start])


def _gate_frame(rng, ref):
    t_mid = rng.uniform(0.4, 0.6) * _arrival_s(ref)
    p, v, n = _sweep_frame(ref, t_mid, rng.uniform(0.4, 0.6))
    return t_mid, p, v, n


def _end_occluder(rng, p, n, amp):
    """Hide one end of the sweep: the fraction f of a sine cycle with A sin > A cos(pi f)."""
    f = rng.uniform(0.0, 0.5)
    side = float(rng.choice([-1.0, 1.0]))
    return f, dict(center=[float(x) for x in p], normal=[float(x) for x in side * n], lo=float(amp * np.cos(np.pi * f)), hi=1e9)


def s0_quasi_periodic(rng, ref):
    """G1: a sphere sweeps across the rope's path (sine, half periods jittered +-15% each). Amplitude
    14-17 cm and base period 3-5 s leave 1-1.7 s open windows (|offset| > 75 mm) per half period; the
    first smoke test (10-13 cm, 2.2-3.5 s) left ~0.8 s, about the time the slow rope needs to cross."""
    t_mid, p, v, n = _gate_frame(rng, ref)
    amp, base = rng.uniform(0.14, 0.17), rng.uniform(3.0, 5.0)
    halves = np.maximum(base / 2 * (1 + rng.uniform(-0.15, 0.15, size=16)), np.pi * amp / V_PEAK_MAX)
    motion = dict(kind="quasi_periodic", center=[float(x) for x in p], direction=[float(x) for x in n],
                  amplitude_m=float(amp), half_periods_s=[float(h) for h in halves], phase0=float(rng.uniform(0, 2 * np.pi)))
    f, occluder = _end_occluder(rng, p, n, amp)
    des = dict(t_cross=float(t_mid), fraction=None, speed=float(amp * 2 * np.pi / base), crossing_point=[float(x) for x in p],
               direction=[float(x) for x in n], transport_dir=[float(x) for x in v / np.linalg.norm(v)], occluder=occluder)
    return [_scripted(rng, motion, n)], [des], dict(period_s=float(base), amplitude_m=float(amp), occluded_fraction=float(f),
                                                     perception=_perception(rng))


def s0_patrol(rng, ref):
    """G2: back and forth at a cruise speed, dwelling 0.3-1.0 s at each end (each dwell jittered +-15%)."""
    t_mid, p, v, n = _gate_frame(rng, ref)
    amp, speed, dwell = rng.uniform(0.14, 0.17), rng.uniform(0.20, 0.35), rng.uniform(0.3, 1.0)
    cycle = 2 * (2 * amp / speed + speed / 1.5 + dwell)
    motion = dict(kind="patrol", center=[float(x) for x in p], direction=[float(x) for x in n], amplitude_m=float(amp),
                  speed_mps=float(speed), accel_mps2=1.5, dwells_s=[float(d) for d in dwell * (1 + rng.uniform(-0.15, 0.15, size=16))],
                  start_offset_s=float(rng.uniform(0, cycle)))
    f, occluder = _end_occluder(rng, p, n, amp)
    des = dict(t_cross=float(t_mid), fraction=None, speed=float(speed), crossing_point=[float(x) for x in p],
               direction=[float(x) for x in n], transport_dir=[float(x) for x in v / np.linalg.norm(v)], occluder=occluder)
    return [_scripted(rng, motion, n)], [des], dict(period_s=float(cycle), amplitude_m=float(amp), dwell_s=float(dwell),
                                                     speed=float(speed), occluded_fraction=float(f), perception=_perception(rng))


def s0_cv_occluded(rng, ref):
    """G0 (control): a sweep crossing that is hidden for 0.2-0.8 s, reappearing 0.1-0.3 s before the
    nominal crossing (a slab across its straight path)."""
    profiles, designs, knob = sweep(rng, ref)
    d = designs[0]
    t_occ, t_end = rng.uniform(0.2, 0.8), rng.uniform(0.1, 0.3)
    d["occluder"] = dict(center=d["crossing_point"], normal=d["direction"], lo=float(-d["speed"] * (t_end + t_occ)),
                         hi=float(-d["speed"] * t_end))
    knob.update(occluded_s=float(t_occ), reappear_s=float(t_end), perception=_perception(rng))
    return profiles, designs, knob


def s0_rope(rng, ref):
    """G3: a fully visible sweep crossing with unknown rope stiffness and damping (x0.25-x4 each, log-uniform)."""
    profiles, designs, knob = sweep(rng, ref)
    knob.update(rope=dict(stiffness_scale=float(np.exp(rng.uniform(np.log(0.25), np.log(4.0)))),
                          damping_scale=float(np.exp(rng.uniform(np.log(0.25), np.log(4.0))))))
    return profiles, designs, knob

# ---- near-rope occlusion (docs/notes/occlusion_near_prereg_20261023.md) -------------------------------
def _near_occluder(rng, p, n):
    """Hide one side of the sweep from 55 mm + g (g 10-50 mm) beyond the crossing point outwards: the
    obstacle vanishes on its way out, turns unseen and reappears just outside the protected radius,
    heading for the rope."""
    gap = rng.uniform(0.010, 0.050)
    side = float(rng.choice([-1.0, 1.0]))
    return gap, dict(center=[float(x) for x in p], normal=[float(x) for x in side * n], lo=float(0.055 + gap), hi=1e9)


def _hidden_fraction(motion, occluder, steps=1300):
    from . import motion as motion_mod
    tr = motion_mod.track(motion, steps)
    return float(np.mean([motion_mod.hidden(x, occluder) for x in tr]))


def s1n_quasi_periodic(rng, ref):
    """As G1 (s0_quasi_periodic) with the occluder next to the rope instead of at the far end."""
    t_mid, p, v, n = _gate_frame(rng, ref)
    amp, base = rng.uniform(0.14, 0.17), rng.uniform(3.0, 5.0)
    halves = np.maximum(base / 2 * (1 + rng.uniform(-0.15, 0.15, size=16)), np.pi * amp / V_PEAK_MAX)
    motion = dict(kind="quasi_periodic", center=[float(x) for x in p], direction=[float(x) for x in n],
                  amplitude_m=float(amp), half_periods_s=[float(h) for h in halves], phase0=float(rng.uniform(0, 2 * np.pi)))
    gap, occluder = _near_occluder(rng, p, n)
    des = dict(t_cross=float(t_mid), fraction=None, speed=float(amp * 2 * np.pi / base), crossing_point=[float(x) for x in p],
               direction=[float(x) for x in n], transport_dir=[float(x) for x in v / np.linalg.norm(v)], occluder=occluder)
    return [_scripted(rng, motion, n)], [des], dict(period_s=float(base), amplitude_m=float(amp), reappear_gap_m=float(gap),
                                                     occluded_fraction=_hidden_fraction(motion, occluder), perception=_perception(rng))


def s1n_patrol(rng, ref):
    """As G2 (s0_patrol) with the occluder next to the rope instead of at the far end."""
    t_mid, p, v, n = _gate_frame(rng, ref)
    amp, speed, dwell = rng.uniform(0.14, 0.17), rng.uniform(0.20, 0.35), rng.uniform(0.3, 1.0)
    cycle = 2 * (2 * amp / speed + speed / 1.5 + dwell)
    motion = dict(kind="patrol", center=[float(x) for x in p], direction=[float(x) for x in n], amplitude_m=float(amp),
                  speed_mps=float(speed), accel_mps2=1.5, dwells_s=[float(d) for d in dwell * (1 + rng.uniform(-0.15, 0.15, size=16))],
                  start_offset_s=float(rng.uniform(0, cycle)))
    gap, occluder = _near_occluder(rng, p, n)
    des = dict(t_cross=float(t_mid), fraction=None, speed=float(speed), crossing_point=[float(x) for x in p],
               direction=[float(x) for x in n], transport_dir=[float(x) for x in v / np.linalg.norm(v)], occluder=occluder)
    return [_scripted(rng, motion, n)], [des], dict(period_s=float(cycle), amplitude_m=float(amp), dwell_s=float(dwell),
                                                     speed=float(speed), reappear_gap_m=float(gap),
                                                     occluded_fraction=_hidden_fraction(motion, occluder), perception=_perception(rng))


# ---- two obstacles (docs/notes/multi_obstacle_prereg_20261030.md) ----------------------------------------
def _m2(rng, ref, kind):
    """Two independent obstacles crossing the rope's path at about 0.3 and 0.7 of its length (about 21 cm
    apart), each moving as in G1 (kind "qp": quasi-periodic sine) or G2 ("patrol": dwelling at the ends), each
    with its own occluder: next to the rope (as s1n) or at the far end (as stage 0), half and half."""
    t_mid = rng.uniform(0.4, 0.6) * _arrival_s(ref)
    profiles, designs, occ_kinds, hidden_fracs = [], [], [], []
    for f in rng.permutation([0.30, 0.70]) + rng.uniform(-0.03, 0.03, size=2):
        p, v, n = _sweep_frame(ref, t_mid, float(f))
        amp = rng.uniform(0.14, 0.17)
        if kind == "qp":
            base = rng.uniform(3.0, 5.0)
            halves = np.maximum(base / 2 * (1 + rng.uniform(-0.15, 0.15, size=16)), np.pi * amp / V_PEAK_MAX)
            motion = dict(kind="quasi_periodic", center=[float(x) for x in p], direction=[float(x) for x in n], amplitude_m=float(amp),
                          half_periods_s=[float(h) for h in halves], phase0=float(rng.uniform(0, 2 * np.pi)))
            speed = amp * 2 * np.pi / base
        else:
            speed, dwell = rng.uniform(0.20, 0.35), rng.uniform(0.3, 1.0)
            cycle = 2 * (2 * amp / speed + speed / 1.5 + dwell)
            motion = dict(kind="patrol", center=[float(x) for x in p], direction=[float(x) for x in n], amplitude_m=float(amp),
                          speed_mps=float(speed), accel_mps2=1.5,
                          dwells_s=[float(d) for d in dwell * (1 + rng.uniform(-0.15, 0.15, size=16))],
                          start_offset_s=float(rng.uniform(0, cycle)))
        if rng.uniform() < 0.5:
            occluder = _near_occluder(rng, p, n)[1]
            occ_kinds.append("near")
        else:
            occluder = _end_occluder(rng, p, n, amp)[1]
            occ_kinds.append("end")
        hidden_fracs.append(_hidden_fraction(motion, occluder))
        profiles.append(_scripted(rng, motion, n))
        designs.append(dict(t_cross=float(t_mid), fraction=float(f), speed=float(speed), crossing_point=[float(x) for x in p],
                            direction=[float(x) for x in n], transport_dir=[float(x) for x in v / np.linalg.norm(v)], occluder=occluder))
    return profiles, designs, dict(occluder_kinds=occ_kinds, occluded_fraction=hidden_fracs, perception=_perception(rng))


def m2_quasi_periodic(rng, ref):
    return _m2(rng, ref, "qp")


def m2_patrol(rng, ref):
    return _m2(rng, ref, "patrol")


# ---- generalisation set (docs/notes/geom_tracker_prereg_20261106.md, section 9) -----------------------------
def _occluder_half(rng, p, n, amp):
    if rng.uniform() < 0.5:
        return "near", _near_occluder(rng, p, n)[1]
    return "end", _end_occluder(rng, p, n, amp)[1]


def _one_moving(rng, ref, motion_fn):
    """One sphere crossing the rope's path as G1/G2 with a motion from motion_fn(rng, p, n, v) -> (motion, amp,
    speed, knob) and an occluder next to the rope or at the far end (half and half)."""
    t_mid, p, v, n = _gate_frame(rng, ref)
    motion, amp, speed, knob = motion_fn(rng, p, n, v / np.linalg.norm(v))
    kind, occluder = _occluder_half(rng, p, n, amp)
    des = dict(t_cross=float(t_mid), fraction=None, speed=float(speed), crossing_point=[float(x) for x in p],
               direction=[float(x) for x in n], transport_dir=[float(x) for x in v / np.linalg.norm(v)], occluder=occluder)
    knob = dict(knob, occluder_kinds=[kind], occluded_fraction=[_hidden_fraction(motion, occluder)], perception=_perception(rng))
    return [_scripted(rng, motion, n)], [des], knob


def _varspeed_motion(rng, p, n, u):
    amp = rng.uniform(0.14, 0.17)
    speeds, dwells = rng.uniform(0.10, 0.45, size=40), rng.uniform(0.0, 0.8, size=40)
    cycle = 2 * (2 * amp / speeds[:2].mean() + dwells[:2].mean())
    motion = dict(kind="varspeed", center=[float(x) for x in p], direction=[float(x) for x in n], amplitude_m=float(amp),
                  leg_speeds_mps=[float(x) for x in speeds], accel_mps2=1.5, dwells_s=[float(d) for d in dwells],
                  start_offset_s=float(rng.uniform(0, cycle)))
    return motion, amp, float(speeds.mean()), dict(motion="varspeed", amplitude_m=float(amp))


def _walk_motion(rng, p, n, u):
    a2 = np.cross(n, u)
    a2 = a2 / np.linalg.norm(a2)
    periods = rng.uniform(2.5, 5.0, size=2)
    motion = dict(kind="random_walk", center=[float(x) for x in p], direction=[float(x) for x in n],
                  axes=[[float(x) for x in n], [float(x) for x in a2]], natural_periods_s=[float(x) for x in periods],
                  damping_ratio=0.3, sd_m=[0.08, 0.04], seed=int(rng.integers(1, 2**31 - 1)))
    return motion, 0.16, float(0.08 * 2 * np.pi / periods[0]), dict(motion="random_walk", natural_periods_s=[float(x) for x in periods])


def _period_motion(rng, p, n, u):
    fast = rng.uniform() < 0.5
    base, amp = (rng.uniform(1.5, 2.5), rng.uniform(0.10, 0.13)) if fast else (rng.uniform(6.0, 9.0), rng.uniform(0.14, 0.17))
    halves = np.maximum(base / 2 * (1 + rng.uniform(-0.15, 0.15, size=40)), np.pi * amp / V_PEAK_MAX)
    motion = dict(kind="quasi_periodic", center=[float(x) for x in p], direction=[float(x) for x in n], amplitude_m=float(amp),
                  half_periods_s=[float(h) for h in halves], phase0=float(rng.uniform(0, 2 * np.pi)))
    return motion, amp, float(amp * 2 * np.pi / base), dict(motion="new_period", period_s=float(base), amplitude_m=float(amp),
                                                             fast=bool(fast))


def gm_varspeed(rng, ref):
    return _one_moving(rng, ref, _varspeed_motion)


def gm_walk(rng, ref):
    return _one_moving(rng, ref, _walk_motion)


def gm_period(rng, ref):
    return _one_moving(rng, ref, _period_motion)


def gc_m3(rng, ref):
    """Three independent obstacles crossing the rope's path at about 0.25, 0.5 and 0.75 of its length, each moving
    as in training (quasi-periodic or patrol, half and half), each with no occluder / one next to the rope / one
    at the far end (a third each)."""
    t_mid = rng.uniform(0.4, 0.6) * _arrival_s(ref)
    profiles, designs, occ_kinds, hidden_fracs, kinds = [], [], [], [], []
    for f in rng.permutation([0.25, 0.50, 0.75]) + rng.uniform(-0.02, 0.02, size=3):
        p, v, n = _sweep_frame(ref, t_mid, float(f))
        amp = rng.uniform(0.14, 0.17)
        if rng.uniform() < 0.5:
            base = rng.uniform(3.0, 5.0)
            halves = np.maximum(base / 2 * (1 + rng.uniform(-0.15, 0.15, size=16)), np.pi * amp / V_PEAK_MAX)
            motion = dict(kind="quasi_periodic", center=[float(x) for x in p], direction=[float(x) for x in n], amplitude_m=float(amp),
                          half_periods_s=[float(h) for h in halves], phase0=float(rng.uniform(0, 2 * np.pi)))
            speed = amp * 2 * np.pi / base
        else:
            speed, dwell = rng.uniform(0.20, 0.35), rng.uniform(0.3, 1.0)
            cycle = 2 * (2 * amp / speed + speed / 1.5 + dwell)
            motion = dict(kind="patrol", center=[float(x) for x in p], direction=[float(x) for x in n], amplitude_m=float(amp),
                          speed_mps=float(speed), accel_mps2=1.5,
                          dwells_s=[float(d) for d in dwell * (1 + rng.uniform(-0.15, 0.15, size=16))],
                          start_offset_s=float(rng.uniform(0, cycle)))
        kinds.append(motion["kind"])
        u = rng.uniform()
        if u < 1 / 3:
            occ_kind, occluder = "none", None
        elif u < 2 / 3:
            occ_kind, occluder = "near", _near_occluder(rng, p, n)[1]
        else:
            occ_kind, occluder = "end", _end_occluder(rng, p, n, amp)[1]
        occ_kinds.append(occ_kind)
        hidden_fracs.append(_hidden_fraction(motion, occluder) if occluder else 0.0)
        profiles.append(_scripted(rng, motion, n))
        designs.append(dict(t_cross=float(t_mid), fraction=float(f), speed=float(speed), crossing_point=[float(x) for x in p],
                            direction=[float(x) for x in n], transport_dir=[float(x) for x in v / np.linalg.norm(v)], occluder=occluder))
    return profiles, designs, dict(motions=kinds, occluder_kinds=occ_kinds, occluded_fraction=hidden_fracs, perception=_perception(rng))


# name -> (n_obstacles, sampler, rng code, piercing rule: every obstacle / at least one / none must pierce the
# reference, midpoint transport range in m). Gates need long transports so the gate sits >= 11 cm from
# both the start and the target rope (a 40 mm sphere plus the 55 mm protected radius).
FAMILIES = {"sweep": (1, sweep, 1, "all", (0.08, 0.22)), "dense": (2, dense, 3, "all", (0.08, 0.22)),
            "pgate": (1, periodic_gate, 4, "none", (0.28, 0.36)), "sgate": (2, static_gate, 5, "any", (0.28, 0.36)),
            "s0_cvocc": (1, s0_cv_occluded, 6, "all", (0.08, 0.22)), "s0_qp": (1, s0_quasi_periodic, 7, "none", (0.28, 0.36)),
            "s0_patrol": (1, s0_patrol, 8, "none", (0.28, 0.36)), "s0_rope": (1, s0_rope, 9, "all", (0.08, 0.22)),
            "s1n_qp": (1, s1n_quasi_periodic, 10, "none", (0.28, 0.36)), "s1n_patrol": (1, s1n_patrol, 11, "none", (0.28, 0.36)),
            "m2_qp": (2, m2_quasi_periodic, 12, "none", (0.28, 0.36)), "m2_patrol": (2, m2_patrol, 13, "none", (0.28, 0.36)),
            "gm_varspeed": (1, gm_varspeed, 14, "none", (0.28, 0.36)), "gm_walk": (1, gm_walk, 15, "none", (0.28, 0.36)),
            "gm_period": (1, gm_period, 16, "none", (0.28, 0.36)), "gc_m3": (3, gc_m3, 17, "none", (0.28, 0.36))}
# pgate uses "none": its sphere crosses the reference path every half period by construction, and
# filtering on whether the nominal timing collides would keep only some phases (the phase contrast is
# one of the pre-registered tests).


def min_pair_distance(obs):
    """Smallest centre distance between any two obstacles over the episode (inf for one obstacle)."""
    n = obs.shape[1]
    d = [np.linalg.norm(obs[:, a] - obs[:, b], axis=1).min() for a in range(n) for b in range(a + 1, n)]
    return float(min(d)) if d else float("inf")
