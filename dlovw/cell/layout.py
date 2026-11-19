"""Layout of the 2026-11-12 robot cell: one rope end on the left arm, the other end clamped to a fixed post, the
obstacle sphere carried by the right arm.

World = left UR base (as in all earlier simulation and calibration). Sources (archive
calibration_data_2026-11-12.zip, outputs/real_calibration/):

- T_WORLD_FROM_RIGHT_BASE: dual_ur_bases/dual_ur_base_calibration.json (12 common TCP points, RMSE 2.3 mm). The
  archive's validation run (dual_ur_bases_validation, "passed": false, 1.17 m error) still used the previous cell's
  transform (1.332, 0.008, 0.133); re-scored with this transform the same 8 validation points give 2.2 mm RMSE,
  2.5 mm max (thresholds 3 / 5 / 8 mm): the base calibration passes.
- cameras: left_base_cameras/left_base_camera_calibration.json (Shah hand-eye, calibration set 1.0 / 1.3 px), and
  the RealSense native colour intrinsics (zero distortion) of left_base_camera_observations.json. The independent
  validation of these extrinsics did not pass (about 19 mm / 6 deg on 7 of 8 poses, consistent with the board
  moving on the TCP); the poses are used for simulation only and must be re-checked before real deployment.

The fixed end and the workspaces are chosen here (not measured); see docs/notes/cell_20261112_scene.md.
"""
from __future__ import annotations

import numpy as np

ROPE_LENGTH_M = 0.532                      # between the two welded points (legacy 532 mm rope)
TCP_FROM_FLANGE_M = 0.165                  # both arms' active TCP [0, 0, 0.165] (probe tip)

# The real arms are UR5 CB3 (checked over RTDE 2026-11-19); "ur5e" restores the legacy menagerie arms.
ARM_MODEL = "ur5"
# Base frames are the UR controllers' base frames (TCP readings, touch check). The menagerie base body is the ROS
# base_link, turned 180 deg about z from the controller base, so the sim joint angles differed from the real ones
# by pi on the shoulder pan. True: the base bodies are turned so that sim and controller share joint angles.
CONTROLLER_BASE_FRAME = True

T_WORLD_FROM_LEFT_BASE = np.eye(4)
T_WORLD_FROM_RIGHT_BASE = np.array([
    [0.9999439483143366, 0.009387264134953313, 0.004896886929065748, 0.4002537395814177],
    [-0.00937688500994682, 0.99995374998131, -0.0021382062170568427, -0.7375284228653249],
    [-0.004916732354488277, 0.00209216882135357, 0.9999857241843898, 0.14456221911014072],
    [0.0, 0.0, 0.0, 1.0],
])

# OpenCV optical frame (+x image right, +y image down, +z forward), 640 x 480.
CAMERAS = {
    "front": dict(serial="001622072105", K=(610.5870361328125, 609.24267578125, 317.1688232421875, 252.54904174804688),
                  T=np.array([
                      [0.046153652546155485, -0.008023915396802417, 0.9989021259054129, -0.6512376754939058],
                      [-0.998279322298455, -0.03657743372421294, 0.04583105948467543, -0.6186864693446348],
                      [0.03616953176342973, -0.9992986080866199, -0.00969829099163759, 0.4547523155023202],
                      [0.0, 0.0, 0.0, 1.0]])),
    "top": dict(serial="837212070312", K=(613.5178833007812, 613.4658813476562, 320.3066711425781, 236.6110076904297),
                T=np.array([
                    [0.0016722986896389597, 0.9943795877990481, 0.10586046847472547, -0.07655748612884095],
                    [0.9999937770403373, -0.0019917326506804778, 0.0029118519277245153, -0.649105029963346],
                    [0.0031063318711000987, 0.10585494022313707, -0.9943767305868851, 1.020575053338447],
                    [0.0, 0.0, 0.0, 1.0]])),
}
IMAGE_W, IMAGE_H = 640, 480

# TCP orientations (both probes pointing down, world -z), as in the hardware config:
# left TCP axes x=-world_x, y=+world_y, z=-world_z; right TCP axes x=+world_x, y=-world_y, z=-world_z.
LEFT_TCP_QUAT = np.array([0.0, 0.0, 1.0, 0.0])
RIGHT_TCP_QUAT = np.array([0.0, 1.0, 0.0, 0.0])

# --- chosen layout --------------------------------------------------------------------------------------------
# The rope runs along world y, square to the front camera's view (+x), centred on both cameras' image centre
# (y about -0.62). The clamp is on the far side so the left wrist only covers the moving end in the top view; the
# right arm (base at x = 0.40, level with the rope's middle) brings the sphere in from +x.
FIXED_END_POS = np.array([0.0, -0.82, 0.45])
FIXED_END_TANGENT = np.array([0.0, -1.0, 0.0])     # rope enters the clamp horizontally, pointing away from the left arm
LEFT_HOME_POS = np.array([0.0, -0.42, 0.45])       # 0.40 m from the clamp (75 % of the rope length)

# workspaces: every corner in both images (left TCP: >= 12 px margin; sphere: whole sphere, >= 5 px margin;
# the top view narrows with height). Below x = -0.10 the right forearm would also reach over the rope.
LEFT_WORKSPACE = (np.array([-0.12, -0.54, 0.34]), np.array([0.12, -0.40, 0.52]))
# the policy's box for the left TCP: larger than the camera box above (the RGB tracker takes the ends from
# kinematics, the TCP need not be in view; the rope stays mostly in view). docs/notes/cell_policy_20261113.md
POLICY_LEFT_WORKSPACE = (np.array([-0.15, -0.56, 0.30]), np.array([0.15, -0.36, 0.55]))
MIN_END_DISTANCE_M = 0.18
MAX_END_DISTANCE_M = 0.90 * ROPE_LENGTH_M

OBSTACLE_RADIUS_M = 0.04
OBSTACLE_CENTRE_IN_TCP = np.array([0.0, 0.0, OBSTACLE_RADIUS_M + 0.005])   # sphere on the probe tip, 5 mm socket
OBSTACLE_HOME_POS = np.array([0.16, -0.62, 0.38])  # parked 0.16 m off the rope's plane
OBSTACLE_WORKSPACE = (np.array([-0.10, -0.76, 0.31]), np.array([0.16, -0.48, 0.48]))
WORK_CENTRE_XY = np.array([0.0, -0.62])            # IK prefers elbows away from here (less occlusion of the rope)

TABLE_TOP_Z = 0.0                                  # left base stands on the table; right base on a 0.145 m riser


def project(points, role):
    """Pixels (N, 2) and optical depth (N,) of world points in camera `role`."""
    cam = CAMERAS[role]
    T = cam["T"]
    q = (np.atleast_2d(np.asarray(points, float)) - T[:3, 3]) @ T[:3, :3]
    fx, fy, cx, cy = cam["K"]
    return np.stack([fx * q[:, 0] / q[:, 2] + cx, fy * q[:, 1] / q[:, 2] + cy], 1), q[:, 2]
