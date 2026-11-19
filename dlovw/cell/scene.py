"""MuJoCo scene of the 2026-11-12 cell (layout.py), built from the legacy 532 mm scene through MjSpec.

Kept from the legacy XML (envs/xml/bimanual_dlo_ur5e_sim2real_532mm_yellow_dyn_sphere_deform.xml): both UR5e arms
(meshes, joints, position actuators), the rope (25 bodies dlo_00..dlo_24, 0.532 m between the welded body origins,
same capsule / ball-joint parameters as every legacy dataset), solver options and defaults.

Changed:
- right base at the calibrated T_world_from_right_base, on a riser to the table;
- Robotiq grippers removed; each flange carries a flat plate and a 165 mm probe whose tip is the TCP (site
  `<side>_tcp`), as on the real arms; arm collision geoms do not collide (the arms are driven kinematically);
- rope end dlo_00 welded to the mocap `left_tcp_mocap` (the left TCP), leaving the probe tip along the probe axis;
  rope end dlo_24 welded to the world body `fixed_clamp` on a post (layout.FIXED_END_POS), entering it horizontally;
- obstacle sphere `obstacle_sphere` fixed to the right probe tip (non-colliding, as the legacy obstacles: clearance
  is measured, not felt);
- cameras `front_view` / `top_view` at the calibrated poses with the RealSense intrinsics (focalpixel /
  principalpixel), 640 x 480; black table and backdrop as the real black cloth;
- keyframe `home`: arms at the home TCPs, rope settled under gravity.
"""
from __future__ import annotations

import os

import mujoco
import numpy as np

from . import layout as L

LEGACY_XML = "envs/xml/bimanual_dlo_ur5e_sim2real_532mm_yellow_dyn_sphere_deform.xml"
VERSION = "cell_20261119_ur5"
N_ROPE = 25
WELD_SOLREF = (0.006, 1.0)
WELD_SOLIMP = (0.98, 0.995, 0.001, 0.5, 2.0)
TOOL_RGBA = (0.72, 0.72, 0.74, 1.0)
FIXTURE_RGBA = (0.16, 0.16, 0.17, 1.0)
BLACK = (0.10, 0.10, 0.10, 1.0)
OBSTACLE_RGBA = (0.85, 0.12, 0.12, 1.0)
PROBE_RADIUS_M = 0.004
PLATE_X_M = (-0.025, 0.105)                       # plate along +x of the TCP frame (sim_model.py, front images)
OPT_FROM_MJ = np.diag([1.0, -1.0, -1.0])           # MuJoCo cameras look along -z with +y up
ARM_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint",
              "wrist_3_joint")
# rope body frame: +x along the rope (from dlo_00 towards dlo_24). At the left end it leaves the probe tip along the
# probe (+z of the TCP); y_rope = y_tcp, z_rope = -x_tcp.
ROPE_IN_LEFT_TCP = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])   # columns: x_r, y_r, z_r in TCP


def legacy_root():
    root = os.environ.get("DLOVW_LEGACY_ROOT")
    if root:
        return os.path.expanduser(root)
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "..", "ImaginationDLO"))


def quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, float).reshape(-1))
    return q


def mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, float))
    return m.reshape(3, 3)


def _orthonormal(R):
    u, _, vt = np.linalg.svd(R)
    return u @ vt


def _look_at(eye, target):
    """MuJoCo camera rotation (looks along -z, +y up) from eye towards target, world +z up."""
    back = (eye - target) / np.linalg.norm(eye - target)
    right = np.cross([0.0, 0.0, 1.0], back)
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(back, right), back], 1)


def fixed_clamp_rotation():
    """Rope frame at dlo_24: x along FIXED_END_TANGENT, z as close to world +z as possible."""
    x = L.FIXED_END_TANGENT / np.linalg.norm(L.FIXED_END_TANGENT)
    z = np.array([0.0, 0.0, 1.0]) - x[2] * x
    z /= np.linalg.norm(z)
    return np.stack([x, np.cross(z, x), z], 1)


def _strip_legacy(sp):
    for e in list(sp.equalities):
        sp.delete(e)
    for e in list(sp.excludes):
        sp.delete(e)
    for name in ("left_gripper", "right_gripper", "obstacle_mocap"):
        sp.delete(sp.body(name))
    for side in ("left", "right"):
        wrist3 = sp.body(f"{side}_ur5e_wrist_3_link")
        for child in list(wrist3.bodies):
            sp.delete(child)                             # the Robotiq mount and its subtree
    for g in list(sp.worldbody.geoms):
        sp.delete(g)                                     # floor, table, backdrop, trajectory markers
    for c in list(sp.cameras):
        sp.delete(c)
    for lt in list(sp.worldbody.lights):
        sp.delete(lt)
    for s in list(sp.sites):
        if s.name.startswith(("left_ur5e_attachment", "right_ur5e_attachment")):
            sp.delete(s)
    for k in list(sp.keys):
        sp.delete(k)
    for t in list(sp.tendons):
        sp.delete(t)
    for a in list(sp.actuators):
        if "robotiq" in a.name or "gripper" in a.name:
            sp.delete(a)


def _add_tool(sp, side, flange=0.1):
    """Plate + 165 mm probe on the flange; TCP site at the probe tip. The flange frame is the menagerie
    attachment frame (wrist_3 + (0, flange, 0), quat (-1, 1, 0, 0)): +z out of the flange (UR5e 0.1, UR5 0.0823)."""
    wrist3 = sp.body(f"{side}_ur5e_wrist_3_link")
    q = np.array([-1.0, 1.0, 0.0, 0.0]) / np.sqrt(2.0)
    tool = wrist3.add_body(name=f"{side}_tool", pos=[0.0, flange, 0.0], quat=q)
    tool.add_geom(name=f"{side}_plate", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[0.5 * (PLATE_X_M[1] - PLATE_X_M[0]), 0.015, 0.0015],
                  pos=[0.5 * (PLATE_X_M[0] + PLATE_X_M[1]), 0.0, 0.0015], rgba=TOOL_RGBA, contype=0, conaffinity=0,
                  group=2, mass=0.05)
    tool.add_geom(name=f"{side}_probe", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[PROBE_RADIUS_M, 0, 0],
                  fromto=[0, 0, 0.004, 0, 0, L.TCP_FROM_FLANGE_M - PROBE_RADIUS_M], rgba=TOOL_RGBA, contype=0,
                  conaffinity=0, group=2, mass=0.05)
    tool.add_site(name=f"{side}_tcp", pos=[0, 0, L.TCP_FROM_FLANGE_M], size=[0.004, 0, 0], group=4,
                  rgba=(0.1, 0.8, 0.1, 0.6))
    return tool


def _add_cameras(sp):
    for role, cam in L.CAMERAS.items():
        T = cam["T"]
        fx, fy, cx, cy = cam["K"]
        R = _orthonormal(T[:3, :3]) @ OPT_FROM_MJ
        c = sp.worldbody.add_camera(name=f"{role}_view", pos=T[:3, 3].copy(), quat=quat(R))
        c.resolution = [L.IMAGE_W, L.IMAGE_H]
        c.focal_pixel = [fx, fy]
        # MuJoCo's principal offset runs opposite to OpenCV's (cx, cy), from the centre between the middle pixels;
        # checked against the analytic projection on markers in both cameras (scripts/cell/check_scene.py)
        c.principal_pixel = [0.5 * (L.IMAGE_W - 1) - cx, 0.5 * (L.IMAGE_H - 1) - cy]
        c.sensor_size = [L.IMAGE_W * 1e-6, L.IMAGE_H * 1e-6]               # nonzero: use the intrinsics, not fovy
    eye, target = np.array([-1.05, -1.95, 1.05]), np.array([0.15, -0.55, 0.30])
    sp.worldbody.add_camera(name="overview", pos=eye, quat=quat(_look_at(eye, target)), fovy=55)


def _add_world(sp):
    wb = sp.worldbody
    wb.add_light(name="key", pos=[0.2, -0.6, 2.4], dir=[0, 0, -1], diffuse=[0.85, 0.85, 0.8])
    wb.add_light(name="fill", pos=[-1.2, -0.6, 1.0], dir=[1, 0, -0.4], diffuse=[0.35, 0.35, 0.35])
    wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.9, 0.75, 0.02],
                pos=[0.15, -0.55, L.TABLE_TOP_Z - 0.02], rgba=BLACK)
    wb.add_geom(name="backdrop", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.01, 1.2, 0.9], pos=[1.05, -0.55, 0.6],
                rgba=BLACK, contype=0, conaffinity=0)
    Rb = L.T_WORLD_FROM_RIGHT_BASE
    riser_h = float(Rb[2, 3]) - L.TABLE_TOP_Z
    wb.add_geom(name="right_riser", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.09, 0.09, 0.5 * riser_h],
                pos=[Rb[0, 3], Rb[1, 3], L.TABLE_TOP_Z + 0.5 * riser_h], rgba=FIXTURE_RGBA, contype=0, conaffinity=0)
    # fixed end: a post from the table and a clamp block around the last 3 cm of rope
    F = L.FIXED_END_POS
    Rc = fixed_clamp_rotation()
    post_xy = F[:2] + 0.03 * Rc[:2, 0]                    # behind the clamp point, outside the free rope
    wb.add_geom(name="fixture_post", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.012, 0.5 * (F[2] - 0.015)],
                pos=[post_xy[0], post_xy[1], 0.5 * (F[2] - 0.015)], rgba=FIXTURE_RGBA, contype=0, conaffinity=0)
    clamp = wb.add_body(name="fixed_clamp", pos=F.copy(), quat=quat(Rc))
    clamp.add_geom(name="fixture_clamp", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.018, 0.015],
                   pos=[0.02, 0, 0], rgba=FIXTURE_RGBA, contype=0, conaffinity=0)
    clamp.add_site(name="fixed_end", size=[0.004, 0, 0], group=4, rgba=(0.1, 0.8, 0.1, 0.6))


def _place_rope_straight(sp):
    """qpos0: the rope straight from its fixed end back towards the left arm (fixed weld satisfied exactly)."""
    Rc = fixed_clamp_rotation()
    root = sp.body("dlo_00")
    root.pos = L.FIXED_END_POS - L.ROPE_LENGTH_M * Rc[:, 0]
    root.quat = quat(Rc)
    return root.pos.copy(), Rc


def build_spec():
    sp = mujoco.MjSpec.from_file(os.path.join(legacy_root(), LEGACY_XML))
    _strip_legacy(sp)
    Rb = L.T_WORLD_FROM_RIGHT_BASE
    flange = 0.1
    if L.ARM_MODEL == "ur5":
        from . import ur5
        for side in ("left", "right"):
            flange = ur5.convert(sp, side)
    Rz = np.diag([-1.0, -1.0, 1.0]) if L.CONTROLLER_BASE_FRAME else np.eye(3)   # base_link -> controller base
    lb, rb = sp.body("left_ur5e_base"), sp.body("right_ur5e_base")
    for b in (lb, rb):          # the legacy XML gives the bases euler="0 0 0", which overrides .quat (until v40
        b.alt.type = mujoco.mjtOrientation.mjORIENTATION_QUAT      # the right base rotation was not applied)
    lb.quat = quat(Rz)
    rb.pos, rb.quat = Rb[:3, 3].copy(), quat(_orthonormal(Rb[:3, :3]) @ Rz)
    for g in sp.geoms:
        if g.name.startswith(("left_ur5e", "right_ur5e")) or (g.classname and "ur5e" in g.classname.name):
            g.contype, g.conaffinity = 0, 0
    _add_tool(sp, "left", flange)
    right_tool = _add_tool(sp, "right", flange)
    right_tool.add_geom(name="obstacle_sphere", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[L.OBSTACLE_RADIUS_M, 0, 0],
                        pos=[0, 0, L.TCP_FROM_FLANGE_M] + L.OBSTACLE_CENTRE_IN_TCP, rgba=OBSTACLE_RGBA, contype=0,
                        conaffinity=0, group=0, mass=0.05)
    _add_world(sp)
    _add_cameras(sp)
    p0, Rc = _place_rope_straight(sp)
    # left TCP mocap at the rope's free end, oriented so that the rope frame relative to it is ROPE_IN_LEFT_TCP
    R_tcp0 = Rc @ ROPE_IN_LEFT_TCP.T
    sp.worldbody.add_body(name="left_tcp_mocap", mocap=True, pos=p0, quat=quat(R_tcp0))
    sp.add_equality(name="left_tcp_rope", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                    name1="left_tcp_mocap", name2="dlo_00", solref=WELD_SOLREF, solimp=WELD_SOLIMP,
                    data=_weld_data(quat(ROPE_IN_LEFT_TCP)))
    sp.add_equality(name="fixed_clamp_rope", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                    name1="fixed_clamp", name2=f"dlo_{N_ROPE - 1:02d}", solref=WELD_SOLREF, solimp=WELD_SOLIMP,
                    data=_weld_data([1.0, 0.0, 0.0, 0.0]))
    sp.visual.global_.offwidth, sp.visual.global_.offheight = 1280, 960
    sp.meshdir = os.path.join(legacy_root(), "envs", "xml", sp.meshdir or "")
    sp.texturedir = os.path.join(legacy_root(), "envs", "xml", sp.texturedir or "")
    return sp


def _weld_data(relquat):
    """eq_data of a weld: anchor (3), relpose position (3) and quaternion (4) of body2 in body1, torquescale."""
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, *map(float, relquat), 1.0]


def repo_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def scene_xml(rebuild=False):
    """Absolute path of the scene XML with the settled `home` keyframe (outputs/cell/<VERSION>.xml)."""
    out = os.path.join(repo_root(), "outputs", "cell", f"{VERSION}.xml")
    if rebuild or not os.path.exists(out):
        from .settle import add_home_keyframe
        sp = build_spec()
        add_home_keyframe(sp)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = f"{out}.{os.getpid()}.tmp"
        with open(tmp, "w") as fh:
            fh.write(sp.to_xml())
        os.replace(tmp, out)
    return out
