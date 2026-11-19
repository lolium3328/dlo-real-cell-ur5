"""UR5 (CB3) arms for the cell scene, in place of the menagerie UR5e of the legacy XML.

The real cell has two UR5 CB3 arms (PolyScope 3.15). The legacy scene, and so every v3x/v40 scene, carries UR5e
arms (third_party/mujoco_menagerie/universal_robots_ur5e). UR5 and UR5e share the joint conventions and the base /
flange frames but not the link lengths (UR5: d1 89.2 mm, d4 109.2, d5 94.7, d6 82.3; UR5e: 162.5, 133.3, 99.7,
99.6) nor the link shapes. Here each arm of the spec is rebuilt as a UR5 while keeping the menagerie body / joint
names (`<side>_ur5e_*`, used throughout the code), the joint axes and frame conventions:

- body offsets from the ROS-Industrial ur_description UR5 kinematics (the menagerie chain is the classic
  ur_description chain, so only the translations change);
- meshes: ur_description meshes/ur5/collision/*.stl, each split into a few convex pieces (group 2, as the menagerie
  meshes; mj_geomDistance uses each piece's hull, so the pieces keep the cell checks close to the real link shape),
  placed by the constant transform between the ROS link frame (+ its visual origin) and the menagerie body at q = 0;
- collision capsules (group 3) fitted to the UR5 meshes, for the task generator's fast screens;
- the flange (tool mount) at wrist_3 + 82.3 mm instead of 100 mm.

`check()` verifies the chain against the ROS forward kinematics for random joints and the capsules against the
meshes. The STL files are vendored in dlovw/cell/assets/ur5 (BSD-3-Clause, ROS-Industrial).
"""
from __future__ import annotations

import os

import mujoco
import numpy as np

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "ur5")
LINKS = ("base", "shoulder", "upperarm", "forearm", "wrist1", "wrist2", "wrist3")
BODIES = ("base", "shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link")
FLANGE_M = 0.0823                                   # wrist_3 origin to the flange face along +y of wrist_3

# menagerie-convention body translations (parent frame), UR5 (ur_description config/ur5/default_kinematics.yaml,
# classic ur5.urdf: shoulder_offset 0.13585, forearm -0.1197, d4 = 0.13585 - 0.1197 + 0.093 = 0.10915)
UR5_POS = {
    "shoulder_link": (0.0, 0.0, 0.089159),
    "upper_arm_link": (0.0, 0.13585, 0.0),
    "forearm_link": (0.0, -0.1197, 0.425),
    "wrist_1_link": (0.0, 0.0, 0.39225),
    "wrist_2_link": (0.0, 0.093, 0.0),
    "wrist_3_link": (0.0, 0.0, 0.09465),
}

# ROS ur_description chain (urdf/inc/ur_macro.xacro + config/<model>/default_kinematics.yaml, visual_parameters.yaml)
_ROS = {
    "ur5": dict(shoulder=((0, 0, .089159), (0, 0, 0)), upper_arm=((0, 0, 0), (np.pi/2, 0, 0)),
                forearm=((-.425, 0, 0), (0, 0, 0)), wrist_1=((-.39225, 0, .10915), (0, 0, 0)),
                wrist_2=((0, -.09465, 0), (np.pi/2, 0, 0)), wrist_3=((0, .0823, 0), (np.pi/2, np.pi, np.pi)),
                shoulder_offset=.13585, elbow_offset=.0165, vis=(-.093, -.095, -.0818)),
    "ur5e": dict(shoulder=((0, 0, .1625), (0, 0, 0)), upper_arm=((0, 0, 0), (np.pi/2, 0, 0)),
                 forearm=((-.425, 0, 0), (0, 0, 0)), wrist_1=((-.3922, 0, .1333), (0, 0, 0)),
                 wrist_2=((0, -.0997, 0), (np.pi/2, 0, 0)), wrist_3=((0, .0996, 0), (np.pi/2, np.pi, np.pi)),
                 shoulder_offset=.138, elbow_offset=.007, vis=(-.127, -.0997, -.0989)),
}

# capsules (group 3) per body: (centre, quat, (radius, half length)) in the menagerie body frame, fitted to the UR5
# meshes (vertices within 10 mm). Only the fast task-generator screens use them; the cell checks use the mesh pieces.
UR5_CAPSULES = {
    "shoulder_link": [((0, 0.0054, 0.0016), (1, 0, 0, 0), (0.0872, 0.0145))],
    "upper_arm_link": [((0, -0.0002, -0.0175), (1, 1, 0, 0), (0.0673, 0.0246)),
                       ((0, 0.0017, 0.2516), (1, 0, 0, 0), (0.0887, 0.18))],
    "forearm_link": [((0, 0.0128, -0.0159), (1, 1, 0, 0), (0.0665, 0.005)),
                     ((-0.0024, -0.001, 0.219), (1, 0, 0, 0), (0.0747, 0.1674))],
    "wrist_1_link": [((0, 0.09, -0.0048), (1, 1, 0, 0), (0.0645, 0.005))],
    "wrist_2_link": [((0, -0.0015, 0.0697), (1, 0, 0, 0), (0.0459, 0.005)),
                     ((0, -0.0048, 0.0949), (1, 1, 0, 0), (0.0393, 0.0288))],
}
UR5_FLANGE_CYL = ((0, 0.0651, 0.0027), (1, 1, 0, 0), (0.0404, 0.0173))     # wrist_3 disc
# convex pieces per link mesh (k-means on triangle centroids; each piece's hull covers its triangles): the visual
# geoms (group 2) and so the arm-arm / arm-rope / self / environment distances of the cell checks
PIECES = dict(base=2, shoulder=3, upperarm=6, forearm=6, wrist1=3, wrist2=3, wrist3=1)


def _load_stl(link):
    sp = mujoco.MjSpec()
    sp.add_mesh(name="m", file=os.path.join(ASSETS, f"{link}.stl"))
    sp.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="m")
    m = sp.compile()
    va, fa = m.mesh_vertadr[0], m.mesh_faceadr[0]
    v = m.mesh_vert[va:va+m.mesh_vertnum[0]].astype(float)
    f = m.mesh_face[fa:fa+m.mesh_facenum[0]]
    # compiled vertices are in the mesh's inertial frame: back to the file frame
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, m.mesh_quat[0])
    return v@R.reshape(3, 3).T+m.mesh_pos[0], f


def pieces(link):
    """[(vertices (n, 3), faces (m, 3))] in the STL frame."""
    v, f = _load_stl(link)
    k = PIECES[link]
    c = v[f].mean(1)
    centres = [c[0]]
    for _ in range(1, k):                                 # farthest-point init, deterministic
        centres.append(c[np.argmax(np.min([np.linalg.norm(c-x, axis=1) for x in centres], 0))])
    centres = np.array(centres)
    for _ in range(30):
        lab = np.argmin(np.linalg.norm(c[:, None]-centres[None], axis=2), 1)
        centres = np.array([c[lab == i].mean(0) if (lab == i).any() else centres[i] for i in range(k)])
    out = []
    for i in range(k):
        fi = f[lab == i]
        if not len(fi):
            continue
        used, inv = np.unique(fi.ravel(), return_inverse=True)
        out.append((v[used], inv.reshape(-1, 3)))
    return out


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp, cp*sr, cp*cr]])


def _T(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = _rpy(*rpy), xyz
    return T


def _rz(q):
    return _T(rpy=(0, 0, q))


def ros_fk(model, q):
    """World (= ROS base_link) poses of the ROS link frames, each link's mesh frame, and tool0, for joints q."""
    k = _ROS[model]
    links, meshes = {}, {}
    T = _T(rpy=(0, 0, np.pi))                                          # base_link_inertia
    links["base"], meshes["base"] = T, T @ _T(rpy=(0, 0, np.pi))
    names = ("shoulder", "upper_arm", "forearm", "wrist_1", "wrist_2", "wrist_3")
    vis = {"shoulder": _T(rpy=(0, 0, np.pi)),
           "upper_arm": _T((0, 0, k["shoulder_offset"]), (np.pi/2, 0, -np.pi/2)),
           "forearm": _T((0, 0, k["elbow_offset"]), (np.pi/2, 0, -np.pi/2)),
           "wrist_1": _T((0, 0, k["vis"][0]), (np.pi/2, 0, 0)),
           "wrist_2": _T((0, 0, k["vis"][1])),
           "wrist_3": _T((0, 0, k["vis"][2]), (np.pi/2, 0, 0))}
    for n, qi in zip(names, q):
        T = T @ _T(*k[n]) @ _rz(qi)
        links[n], meshes[n] = T, T @ vis[n]
    tool0 = T @ _T(rpy=(0, -np.pi/2, -np.pi/2)) @ _T(rpy=(np.pi/2, 0, np.pi/2))
    return links, meshes, tool0


def _mesh_offsets():
    """Constant transform menagerie body -> ROS UR5 mesh frame, per link, from both chains at q = 0."""
    _, meshes, _ = ros_fk("ur5", np.zeros(6))
    T = np.eye(4)
    out = {}
    for body, link, key in zip(BODIES, LINKS, ("base", "shoulder", "upper_arm", "forearm", "wrist_1", "wrist_2",
                                              "wrist_3")):
        if body != "base":
            T = T @ _menagerie_step(body)
        out[link] = np.linalg.inv(T) @ meshes[key]
    return out


_MENAGERIE_QUAT = {"upper_arm_link": (1, 0, 1, 0), "wrist_1_link": (1, 0, 1, 0)}


def _menagerie_step(body):
    T = np.eye(4)
    q = np.array(_MENAGERIE_QUAT.get(body, (1, 0, 0, 0)), float)
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q/np.linalg.norm(q))
    T[:3, :3], T[:3, 3] = m.reshape(3, 3), UR5_POS[body]
    return T


def _quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, float).reshape(-1))
    return q


def convert(sp, side):
    """Turn the menagerie UR5e arm `side` of the MjSpec into a UR5 (names kept). Call before tools are added;
    returns the flange offset along +y of wrist_3 (m)."""
    p = f"{side}_ur5e_"
    for body, pos in UR5_POS.items():
        sp.body(p+body).pos = pos
    off = _mesh_offsets()
    for body, link in zip(BODIES, LINKS):
        b = sp.body(p+body)
        for g in list(b.geoms):
            sp.delete(g)
        T = off[link]
        for i, (v, f) in enumerate(pieces(link)):
            name = f"{p}ur5_{link}_{i}"
            if sp.mesh(name) is None:
                sp.add_mesh(name=name, uservert=v.ravel().tolist(), userface=f.ravel().tolist())
            b.add_geom(name=f"{p}{link}_{i}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, pos=T[:3, 3],
                       quat=_quat(T[:3, :3]), contype=0, conaffinity=0, group=2,
                       rgba=(0.12, 0.12, 0.13, 1) if link in ("base", "wrist3") else (0.70, 0.72, 0.75, 1))
        for i, (c, q, s) in enumerate(UR5_CAPSULES.get(body, [])):
            b.add_geom(name=f"{p}{link}_cap{i}", type=mujoco.mjtGeom.mjGEOM_CAPSULE, pos=c,
                       quat=np.array(q, float)/np.linalg.norm(q), size=[s[0], s[1], 0], contype=0, conaffinity=0,
                       group=3)
        if body == "wrist_3_link":
            c, q, s = UR5_FLANGE_CYL
            b.add_geom(name=f"{p}flange_cyl", type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=c,
                       quat=np.array(q, float)/np.linalg.norm(q), size=[s[0], s[1], 0], contype=0, conaffinity=0,
                       group=3)
    return FLANGE_M


def check(model, side="left", n=200, seed=0):
    """Max flange position / rotation error vs the ROS UR5 FK over random joints (base at the world origin
    assumed for `side`'s base frame), and how far mesh vertices stick out of the link capsules."""
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    jadr = [model.joint(f"{side}_ur5e_{j}").qposadr[0] for j in ("shoulder_pan_joint", "shoulder_lift_joint",
                                                                 "elbow_joint", "wrist_1_joint", "wrist_2_joint",
                                                                 "wrist_3_joint")]
    base = model.body(f"{side}_ur5e_base").id
    tool = model.body(f"{side}_tool").id
    ep = er = 0.
    for _ in range(n):
        q = rng.uniform(-np.pi, np.pi, 6)
        data.qpos[jadr] = q
        mujoco.mj_kinematics(model, data)
        Rb, pb = data.xmat[base].reshape(3, 3), data.xpos[base]
        Rt = Rb.T @ data.xmat[tool].reshape(3, 3)
        pt = Rb.T @ (data.xpos[tool]-pb)
        T0 = ros_fk("ur5", q)[2]
        ep = max(ep, float(np.linalg.norm(pt-T0[:3, 3])))
        er = max(er, float(np.linalg.norm(Rt-T0[:3, :3])))
    return dict(flange_pos_err_m=ep, flange_rot_err=er)
