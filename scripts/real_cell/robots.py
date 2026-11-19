"""Robot / camera interfaces for the real cell runner: ur_rtde wrapper, a MuJoCo-backed mock, RealSense cameras.

Both robot classes expose the same API. Poses are 6-vectors [x, y, z, rotvec] in the robot's own base frame; joint
vectors are the 6 UR joint angles. MuJoCo (the simulation scene) is used here only as a link-position kernel
(`Kin`); IK, FK and limits on the real robot come from the UR controller itself.
"""
import sys
import threading
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'cell'))
from dlovw.cell import layout
from dlovw.cell.arms import Arm

PERIOD = .008                       # = Config.dt, the simulated control step (125 Hz)
T_RIGHT = layout.T_WORLD_FROM_RIGHT_BASE
T_INV = np.linalg.inv(T_RIGHT)
PROBE_TCP = [0., 0., .165, 0., 0., 0.]
LINKS = ('base', 'shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link')


def pose_to_Rp(pose):
    return Rot.from_rotvec(np.asarray(pose[3:], float)).as_matrix(), np.asarray(pose[:3], float)


def Rp_to_pose(R, p):
    return list(map(float, np.r_[p, Rot.from_matrix(R).as_rotvec()]))


def tool_rotation_error(arm_pose, R_ref):
    """rad between an arm pose's tool rotation and R_ref (same base frame)."""
    return float(np.linalg.norm(Rot.from_matrix(pose_to_Rp(arm_pose)[0].T @ R_ref).as_rotvec()))


def world_to_right_base(R, p):
    return T_INV[:3, :3] @ R, T_INV[:3, :3] @ p + T_INV[:3, 3]


def wrap(a):
    return (np.asarray(a, float)+np.pi) % (2*np.pi)-np.pi


class Kin:
    """Link origins (world frame = left base) of both arms from the simulation scene's kinematic tree."""
    def __init__(self, scene_xml):
        self.model = mujoco.MjModel.from_xml_path(str(scene_xml))
        d = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, d, self.model.key('home').id)
        self.arms = {s: Arm(self.model, s) for s in ('left', 'right')}
        self.home = {s: d.qpos[a.qadr].copy() for s, a in self.arms.items()}
        self.bodies = {s: [self.model.body(f'{s}_ur5e_{b}').id for b in LINKS] for s in self.arms}
        post = self.model.geom('fixture_post')
        self.post_radius = float(post.size[0])

    def points(self, side, q):
        """(8, 3) world points: 7 link origins + TCP."""
        arm = self.arms[side]
        p, _ = arm.fk(np.asarray(q, float))
        return np.vstack([arm.scratch.xpos[self.bodies[side]], p])

    def tcp(self, side, q):
        return self.arms[side].fk(np.asarray(q, float))


class UR:
    """ur_rtde wrapper."""
    def __init__(self, ip, side):
        import rtde_control, rtde_receive
        self.side, self.ip, self.n_fault_checks = side, ip, 0
        try:
            self.r = rtde_receive.RTDEReceiveInterface(ip)
        except Exception as e:                       # noqa: BLE001
            raise ConnectionError(f'{side} arm {ip}: RTDE receive connection failed ({e})') from e
        try:
            self.c = rtde_control.RTDEControlInterface(ip)
        except Exception as e:                       # noqa: BLE001
            raise ConnectionError(f'{side} arm {ip}: RTDE control connection failed ({e}); stop any program on the '
                                  'pendant, close pop-ups, and disconnect other RTDE / External Control clients (CB3 has no '
                                  'Remote Control switch; on e-Series enable Remote Control)') from e

    def status(self):
        r, c = self.r, self.c
        return dict(robot_mode=int(r.getRobotMode()), safety_mode=int(r.getSafetyMode()),
                    runtime_state=int(r.getRuntimeState()), protective=bool(r.isProtectiveStopped()),
                    emergency=bool(r.isEmergencyStopped()), speed_scaling=round(float(r.getSpeedScaling()), 3),
                    payload_kg=round(float(r.getPayload()), 3), receive_connected=bool(r.isConnected()),
                    control_connected=bool(c.isConnected()), program_running=bool(c.isProgramRunning()))

    def q(self):
        return np.asarray(self.r.getActualQ(), float)

    def tcp(self):
        return np.asarray(self.r.getActualTCPPose(), float)

    def tcp_offset(self):
        return np.asarray(self.c.getTCPOffset(), float)

    def ik(self, pose, qnear):
        """Controller IK nearest to qnear, or None. URScript get_inverse_kin raises a runtime error (stopping the
        control script) when there is no solution, so ask get_inverse_kin_has_solution first (verified on URSim)."""
        x, qn = list(map(float, pose)), list(map(float, qnear))
        if not self.c.getInverseKinematicsHasSolution(x, qn):
            return None
        q = self.c.getInverseKinematics(x, qn)
        return np.asarray(q, float) if len(q) == 6 else None

    def fk(self, q, tcp_offset=None):
        return np.asarray(self.c.getForwardKinematics(list(map(float, q)), tcp_offset or []), float)

    def pose_ok(self, pose):
        return bool(self.c.isPoseWithinSafetyLimits(list(map(float, pose))))

    def joints_ok(self, q):
        return bool(self.c.isJointsWithinSafetyLimits(list(map(float, q))))

    def faulted(self):
        """Protective / emergency stop every call; every 10th call also a non-normal safety mode, a lost connection
        or a stopped ur_rtde control script (servo commands would then be ignored silently)."""
        if self.r.isProtectiveStopped() or self.r.isEmergencyStopped():
            return True
        self.n_fault_checks += 1
        if self.n_fault_checks % 10:
            return False
        return bool(self.r.getSafetyMode() not in (1, 2) or not self.r.isConnected() or not self.c.isConnected()
                    or not self.c.isProgramRunning())

    def move_l(self, pose, speed, acc):
        if not self.c.moveL(list(map(float, pose)), speed, acc):
            raise RuntimeError(f'{self.side} moveL 失败（控制器拒绝或被中断）')

    def move_j(self, q, speed, acc):
        if not self.c.moveJ(list(map(float, q)), speed, acc):
            raise RuntimeError(f'{self.side} moveJ 失败（控制器拒绝或被中断）')

    def servo_l(self, pose, lookahead, gain):
        self.c.servoL(list(map(float, pose)), 0., 0., PERIOD, lookahead, gain)

    def servo_j(self, q, lookahead, gain):
        self.c.servoJ(list(map(float, q)), 0., 0., PERIOD, lookahead, gain)

    def stop(self):
        try:
            self.c.servoStop()
            self.c.stopL(1.0)
        except Exception as e:                       # noqa: BLE001 - last-resort stop must never raise
            print('stop error:', e)

    def close(self):
        for x in (self.c, self.r):
            try:
                x.disconnect()
            except Exception:                        # noqa: BLE001
                pass


class MockUR:
    """Same API, backed by the simulation arm (damped-LS IK). Tests the script logic only, not the real robot."""
    def __init__(self, kin, side):
        self.kin, self.side, self.arm = kin, side, kin.arms[side]
        self.T = np.eye(4) if side == 'left' else T_RIGHT
        self._q = kin.home[side].copy()

    def _to_world(self, pose):
        R, p = pose_to_Rp(pose)
        return self.T[:3, :3] @ R, self.T[:3, :3] @ p + self.T[:3, 3]

    def q(self):
        return self._q.copy()

    def tcp(self):
        return self.fk(self._q)

    def tcp_offset(self):
        return np.array(PROBE_TCP)

    def fk(self, q, tcp_offset=None):
        p, R = self.arm.fk(np.asarray(q, float))
        return np.array(Rp_to_pose(self.T[:3, :3].T @ R, self.T[:3, :3].T @ (p-self.T[:3, 3])))

    def ik(self, pose, qnear):
        R, p = self._to_world(pose)
        q, e, er = self.arm.ik(p, R, np.asarray(qnear, float), iters=200)
        return q if e < 1e-3 and er < 1e-2 else None

    def pose_ok(self, pose):
        return True

    def joints_ok(self, q):
        return bool((np.abs(q) < 2*np.pi).all() and abs(q[2]) < np.pi)

    def faulted(self):
        return False

    def status(self):
        return dict(robot_mode=7, safety_mode=1, runtime_state=2, protective=False, emergency=False, speed_scaling=1.,
                    payload_kg=0., receive_connected=True, control_connected=True, program_running=True, mock=True)

    def move_l(self, pose, speed, acc):
        q = self.ik(pose, self._q)
        if q is None:
            raise RuntimeError('mock moveL: no IK')
        self._q = q

    def move_j(self, q, speed, acc):
        self._q = np.asarray(q, float)

    def servo_l(self, pose, lookahead, gain):
        self.move_l(pose, 0, 0)

    def servo_j(self, q, lookahead, gain):
        self._q = np.asarray(q, float)

    def stop(self):
        pass

    def close(self):
        pass


class RealSense:
    """front/top colour 640x480@30; keeps the newest frame of each with its host arrival time."""
    def __init__(self, latency_s=0.):
        import pyrealsense2 as rs
        self.latency, self.lock, self.frames, self.stopped = latency_s, threading.Lock(), {}, False
        self.pipes = {}
        for role, cam in layout.CAMERAS.items():
            cfg = rs.config()
            cfg.enable_device(cam['serial'])
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
            pipe = rs.pipeline()
            pipe.start(cfg)
            self.pipes[role] = pipe
        for r, p in self.pipes.items():
            threading.Thread(target=self._loop, args=(r, p), daemon=True).start()

    def _loop(self, role, pipe):
        while not self.stopped:
            try:
                f = pipe.wait_for_frames(1000).get_color_frame()
                with self.lock:
                    self.frames[role] = (np.asanyarray(f.get_data()).copy(), time.monotonic()-self.latency)
            except Exception:                        # noqa: BLE001 - timeouts: keep the old frame, tracker ages it out
                continue

    def latest(self):
        with self.lock:
            return dict(self.frames)

    def close(self):
        self.stopped = True
        for p in self.pipes.values():
            try:
                p.stop()
            except Exception:                        # noqa: BLE001
                pass


class NoCameras:
    def latest(self):
        return {}

    def close(self):
        pass
