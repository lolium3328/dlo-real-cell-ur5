"""Read-only per-scene planning and feasibility checking (no robot motion).

Right arm: the scripted ball track -> TCP poses -> controller IK chained with qnear -> a joint trajectory, checked
against the controller's safety limits, joint step / speed limits and capsule clearances. Own-arm limits, self-collision and the environment are left to the UR controller (safety limits, IK).
Only the two-arm (and ball-vs-left-arm) clearance uses capsules. Left arm: IK + safety limits
on the start, target and a grid over the allowed workspace (the policy's path is not known in advance), and the
right-arm trajectory is checked against *every* such left pose (policy-independent lower bound).
"""
import numpy as np

import geometry as G
from robots import PROBE_TCP, T_RIGHT, Rp_to_pose, pose_to_Rp, world_to_right_base, wrap
from dlovw.cell.calibrated.config import tcp_from_sphere
from dlovw.motion import track


def _gap_report(info, fail, key, gap, min_gap, label=''):
    g = float(np.min(gap))
    info[key] = round(g*1e3, 1)
    if g < min_gap:
        fail.append(f'{key} {g*1e3:.0f} mm < {min_gap*1e3:.0f} mm {label}')


def make_plan(rec, cfg, kin, left, right, args):
    fail, warn, info = [], [], {}
    ql, qr, pl, pr = left.q(), right.q(), left.tcp(), right.tcp()
    Rl = pose_to_Rp(pl)[0]
    # right tool orientation: the one the scenes were checked with (Config.right_quat), not whatever the arm holds now;
    # homing (moveJ to the planned joints) turns the tool there
    from dlovw.cell.calibrated.config import right_rotation
    Rr_w = right_rotation(cfg)
    plan = dict(Rl=Rl, Rr_w=Rr_w, fail=fail, warn=warn, info=info, ik_dt=args.ik_dt, ok=False)
    # ---- geometry model vs the controller (the capsule checks are only valid if these agree) ----
    for name, arm, q, T in (('left', left, ql, np.eye(4)), ('right', right, qr, T_RIGHT)):
        off = arm.tcp_offset()
        if np.abs(off[:3]-np.array(PROBE_TCP[:3])).max() > .003:
            fail.append(f'{name} TCP offset {np.round(off[:3], 4)} is not the probe tip {PROBE_TCP[:3]}')
        if not args.no_capsule:
            actual = T[:3, :3] @ arm.tcp()[:3]+T[:3, 3]
            err = float(np.linalg.norm(kin.tcp(name, q)[0]-actual))
            info[f'model_vs_robot_tcp_mm_{name}'] = round(err*1e3, 1)
            if err > .01:
                fail.append(f'{name}: simulation kinematics differ from the robot by {err*1e3:.0f} mm; capsule checks invalid')

    def pose_l(p):
        return Rp_to_pose(Rl, np.asarray(p, float))

    # ---- left arm: start, target, workspace grid ----
    qs = left.ik(pose_l(rec['start_left']), ql)
    qt = None if qs is None else left.ik(pose_l(rec['target_left']), qs)
    if qs is None or qt is None:
        fail.append('left IK has no solution at the start or the target')
        return plan
    Lq, miss, unsafe = [qs, qt], 0, 0
    for row in cfg.slices:
        for x in np.linspace(row['x_low_m']+.002, row['x_high_m']-.002, 3):
            for y in np.linspace(row['y_low_m']+.002, row['y_high_m']-.002, 3):
                pose = pose_l([x, y, row['z_m']])
                q = left.ik(pose, qs)
                if q is None:
                    miss += 1
                    continue
                unsafe += not (left.pose_ok(pose) and left.joints_ok(q))
                Lq.append(q)
    info['left_grid_ik_missing'], info['left_grid_unsafe'] = miss, unsafe
    if miss or unsafe:
        fail.append(f'left workspace grid: {miss} without IK, {unsafe} outside the controller safety limits')
    PL = np.array([kin.points('left', q) for q in Lq]) if not args.no_capsule else None
    # ---- left homing path (straight line, right arm still at its current joints) ----
    qprev, PLh = ql, []
    for a in np.linspace(0, 1, 25):
        q = left.ik(pose_l(pl[:3]+(np.asarray(rec['start_left'])-pl[:3])*a), qprev)
        if q is None:
            fail.append('left homing path: IK failed')
            return plan
        PLh.append(kin.points('left', q) if not args.no_capsule else None)
        qprev = q
    if not args.no_capsule:
        _gap_report(info, fail, 'gap_left_homing_vs_right_mm', G.arm_arm_gap(np.array(PLh), kin.points('right', qr)[None])[0], args.min_gap_m)
    # ---- right arm trajectory ----
    steps = int(np.ceil((cfg.max_time_s+2.)/args.ik_dt))
    tr = track(rec['motion'], steps, args.ik_dt)
    Q, qprev, unsafe = [], qr, 0
    for k, s in enumerate(tr):
        Rb, pb = world_to_right_base(Rr_w, tcp_from_sphere(s, Rr_w, cfg))
        pose = Rp_to_pose(Rb, pb)
        q = right.ik(pose, qprev)
        if q is None:
            fail.append(f'right IK has no solution at t = {k*args.ik_dt:.2f} s (ball {np.round(s, 3)})')
            return plan
        unsafe += not (right.pose_ok(pose) and right.joints_ok(q))
        Q.append(q)
        qprev = q
    Q = np.array(Q)
    info['right_unsafe_samples'] = int(unsafe)
    if unsafe:
        fail.append(f'right trajectory: {unsafe} samples outside the controller safety limits')
    dq = np.abs(np.diff(Q, axis=0)).max()
    info['right_max_joint_step_rad'], info['right_max_joint_speed_rad_s'] = round(float(dq), 4), round(float(dq/args.ik_dt), 2)
    if dq/args.ik_dt > args.max_joint_speed or dq > .1:
        fail.append(f'right joint speed {dq/args.ik_dt:.2f} rad/s (step {dq:.3f} rad) exceeds the limit')
    PR = np.array([kin.points('right', q) for q in Q])
    elbow, wrist = np.abs(np.sin(Q[:, 2])).min(), np.abs(np.sin(Q[:, 4])).min()
    shoulder = np.linalg.norm(PR[:, 4, :2]-PR[:, 1, :2], axis=1).min()
    info.update(right_min_sin_elbow=round(float(elbow), 2), right_min_sin_wrist2=round(float(wrist), 2),
                right_shoulder_offset_min_mm=round(float(shoulder)*1e3))
    if elbow < .15 or wrist < .15 or shoulder < .05:
        warn.append(f'right arm near a singularity (sin elbow {elbow:.2f}, sin wrist2 {wrist:.2f}, shoulder offset {shoulder*1e3:.0f} mm)')
    info['right_q_range'] = [np.round(Q.min(0), 2).tolist(), np.round(Q.max(0), 2).tolist()]
    # ---- clearances over the whole trajectory ----
    if not args.no_capsule:
        gap, pair = G.arm_arm_gap(PL, PR)
        k = int(np.argmin(gap.min(1)))
        _gap_report(info, fail, 'gap_arm_arm_mm', gap, args.min_gap_m, f'(any left workspace pose, {pair}, t = {k*args.ik_dt:.1f} s)')
        _gap_report(info, fail, 'gap_ball_left_arm_mm', G.ball_arm_gap(tr[:len(Q)], PL), args.min_gap_m)
        # ---- right homing (moveJ interpolation, left arm already at its start) ----
        PRh = np.array([kin.points('right', qr+(Q[0]-qr)*a) for a in np.linspace(0, 1, 40)])
        _gap_report(info, fail, 'gap_right_homing_vs_left_mm', G.arm_arm_gap(PL[:1], PRh)[0], args.min_gap_m)
    info['right_homing_max_joint_move_rad'] = round(float(np.abs(Q[0]-qr).max()), 2)
    if np.abs(Q[0]-qr).max() > 2.:
        warn.append('right homing moves a joint by more than 2 rad; check the arm is near the sim home configuration')
    # ---- branch vs the simulation's IK (informational + used after homing) ----
    ref_l = kin.arms['left'].ik(np.asarray(rec['start_left'], float), Rl, kin.home['left'], iters=200)[0]
    pw = tcp_from_sphere(tr[0], Rr_w, cfg)
    ref_r = kin.arms['right'].ik(pw, Rr_w, kin.home['right'], iters=200)[0]
    dev_l, dev_r = float(np.abs(wrap(qs-ref_l)).max()), float(np.abs(wrap(Q[0]-ref_r)).max())
    info['branch_dev_vs_sim_rad'] = dict(left=round(dev_l, 2), right=round(dev_r, 2))
    for name, dev in (('left', dev_l), ('right', dev_r)):
        if dev > args.branch_tol:
            warn.append(f'{name} planned start joints differ from the simulated IK by {dev:.2f} rad (other branch?)')
    plan.update(qs=qs, ref_l=ref_l, Q=Q, tr=tr, left_home_pose=pose_l(rec['start_left']), ok=not fail)
    return plan


def right_q_at(plan, s):
    Q, dt = plan['Q'], plan['ik_dt']
    k = float(np.clip(s/dt, 0, len(Q)-1.001))
    i = int(k)
    return Q[i]+(k-i)*(Q[i+1]-Q[i])


def sphere_at(plan, s):
    tr, dt = plan['tr'], plan['ik_dt']
    k = float(np.clip(s/dt, 0, len(tr)-1.001))
    i = int(k)
    return tr[i]+(k-i)*(tr[i+1]-tr[i])


def print_report(rec, plan):
    print(f"  --- feasibility check: {rec['id']} ---")
    for k, v in plan['info'].items():
        print(f'    {k}: {v}')
    for w in plan['warn']:
        print('    WARN:', w)
    for f in plan['fail']:
        print('    FAIL:', f)
    print('    =>', 'OK' if plan['ok'] else 'NOT FEASIBLE')
