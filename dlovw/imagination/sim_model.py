"""The simulated cell corrected to the real one (perception_rgb_20261110.md, section 10), applied to the legacy scene
XMLs at load time through MjSpec; the legacy repository itself is not changed.

Measured on the real cell (configs/real_world/sim2real_532mm_hardware.yaml and WINDOWS_AGENT_HANDOFF.md on the USB
drive, and the 34 recorded frames):
- right arm base: T_world_from_base (1.332, 0.008, 0.133), yaw 2.8 deg; the legacy scene has (1.36, 0, 0);
- end effector: no gripper; a flat plate on the flange and a 165 mm probe whose tip is the TCP and the rope's clamp
  (both arms' active TCP is [0, 0, 0.165]); the legacy scene has a Robotiq 2F-85 with its site 149 mm from the flange.
  The Robotiq bodies and joints stay (the legacy code sets their joints), their geoms are hidden and do not collide;
  the probe and plate are added to the Robotiq base body (its axes are the TCP axes; visual only: the rope is clamped
  at the probe tip and would otherwise press on it), and the arm's attachment site
  (the arm IK's target, which the simulator puts at the rope end) moves to the probe tip.
- rope clamp: the rope leaves the probe tip along the probe axis (front images: it hangs straight on from the tip);
  the legacy welds hold the rope's end tangent horizontal, along the gripper's x axis.
Geometry of probe and plate from the front images (1.5-1.7 mm per px at the TCPs): plate about 130 x 30 x 3 mm along
+x of the TCP frame, from -25 to +105 mm of the probe axis; probe about 8 mm thick.
"""
from __future__ import annotations

import os

import mujoco
import numpy as np

T_WORLD_FROM_RIGHT_BASE = np.array([
    [0.998648893, -0.048108372, 0.019646173, 1.331810271],
    [0.048401884, 0.998719058, -0.014747901, 0.007710044],
    [-0.018911510, 0.015678886, 0.999698218, 0.132684864],
])
TCP_FROM_FLANGE_M = 0.165
PROBE_RADIUS_M = 0.004
PLATE_X_M = (-0.025, 0.105)
PLATE_HALF_Y_M, PLATE_HALF_Z_M = 0.015, 0.0015
TOOL_RGBA = (0.72, 0.72, 0.74, 1.0)
HIDDEN_GROUP = 5
CLAMP_RELQUAT = {"left_grip_dlo": lambda c: (c, 0.0, -c, 0.0), "right_grip_dlo": lambda c: (c, 0.0, c, 0.0)}


def _quat_from_matrix(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(-1))
    return q


def corrected_spec(xml_path, base=True, tool=True, clamp=True):
    sp = mujoco.MjSpec.from_file(str(xml_path))
    if clamp:
        # the rope leaves the probe tip along the probe (the TCP's +z, down), not sideways: the rope bodies' +x is
        # the tangent; left end: x_rope = +z_tcp (rotation -90 deg about y); right end: x_rope = -z_tcp (+90 deg)
        c = np.sqrt(0.5)
        for e in sp.equalities:
            if e.name in CLAMP_RELQUAT:
                d = np.array(e.data, float)
                d[3:6], d[6:10] = 0.0, CLAMP_RELQUAT[e.name](c)
                e.data = d
    if base:
        b = sp.body("right_ur5e_base")
        R = T_WORLD_FROM_RIGHT_BASE[:, :3]
        U, _, Vt = np.linalg.svd(R)
        b.pos = T_WORLD_FROM_RIGHT_BASE[:, 3].copy()
        b.quat = _quat_from_matrix(U @ Vt)
    if tool:
        for side in ("left", "right"):
            rb = sp.body(f"{side}_robotiq_base")
            flange_z = -float(rb.pos[2])                       # the mount is the flange; the Robotiq base sits 3.8 mm out
            for g in sp.geoms:
                if g.name.startswith(f"{side}_robotiq") or _in_subtree(g.parent, rb):
                    g.group, g.contype, g.conaffinity = HIDDEN_GROUP, 0, 0
            tip = flange_z + TCP_FROM_FLANGE_M
            rb.add_geom(name=f"{side}_probe", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[PROBE_RADIUS_M, 0, 0],
                        fromto=[0, 0, flange_z + 0.004, 0, 0, tip - PROBE_RADIUS_M], rgba=TOOL_RGBA, group=2,
                        contype=0, conaffinity=0)
            rb.add_geom(name=f"{side}_plate", type=mujoco.mjtGeom.mjGEOM_BOX,
                        size=[0.5 * (PLATE_X_M[1] - PLATE_X_M[0]), PLATE_HALF_Y_M, PLATE_HALF_Z_M],
                        pos=[0.5 * (PLATE_X_M[0] + PLATE_X_M[1]), 0, flange_z + PLATE_HALF_Z_M], rgba=TOOL_RGBA, group=2,
                        contype=0, conaffinity=0)
            sp.site(f"{side}_ur5e_attachment_site").pos = [0, 0, tip]
    return sp


def _in_subtree(body, root):
    for _ in range(64):                                   # the world body is its own end; never walk past it
        if body is None or body.name in ("", "world"):
            return False
        if body.name == root.name:
            return True
        body = body.parent
    return False


def corrected_model(xml_path, **kw):
    return corrected_spec(xml_path, **kw).compile()


def write_corrected_xml(xml_path, out_path, **kw):
    """The corrected scene as an XML file (asset paths made absolute so it loads from anywhere)."""
    sp = corrected_spec(xml_path, **kw)
    sp.meshdir = os.path.join(os.path.dirname(os.path.abspath(xml_path)), sp.meshdir or "")
    sp.texturedir = os.path.join(os.path.dirname(os.path.abspath(xml_path)), sp.texturedir or "")
    sp.compile()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(sp.to_xml())
    return out_path


VERSION = "cell_v1"        # bump when the corrections change: the cached XMLs are regenerated under the new name


def scene_xml(n_obstacles):
    """Absolute path of the corrected scene with n obstacle spheres (generated into outputs/sim_model on first use)."""
    from .. import legacy
    src = os.path.join(legacy.LEGACY_ROOT, legacy.XML_BY_OBSTACLES[n_obstacles])
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = os.path.join(repo, "outputs", "sim_model", f"{VERSION}_{n_obstacles}.xml")
    if not os.path.exists(out):
        tmp = f"{out}.{os.getpid()}.tmp"
        write_corrected_xml(src, tmp)
        os.replace(tmp, out)
    return out
