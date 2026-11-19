"""Rendering and ground truth for the RGB rope study in the 2026-11-12 cell (docs/notes/cell_rgb_rope_20261112.md).

The scene's cameras carry the calibrated poses and the RealSense intrinsics directly (scene.py), so frames need no
remap. Realism as measured on the real cell (perception_rgb_20261110.md): sensor noise (grey levels), the top frame
taken a little after the front one, and a grey level for the black table / backdrop.

Truth: the 39-node centerline resampled by arc length along the 25 rope body origins (left TCP -> fixed end), and per
camera and node what the node's pixel shows: rope (a rope pixel within 2 px), robot, sphere, background, outside.
"""
from __future__ import annotations

import mujoco
import numpy as np

from . import layout as L

N_PTS = 39
BACKGROUND = ("table", "backdrop")
CAM_NAMES = {"front": "front_view", "top": "top_view"}


def resample(points, n=N_PTS):
    p = np.asarray(points, float)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    t = np.linspace(0.0, s[-1], n)
    out = np.stack([np.interp(t, s, p[:, k]) for k in range(3)], 1)
    out[0], out[-1] = p[0], p[-1]
    return out


def cams():
    """Camera dict in the form the RGB methods take (role -> K, T)."""
    return {r: dict(K=c["K"], T=c["T"]) for r, c in L.CAMERAS.items()}


class CellRenderer:
    def __init__(self, xml_path, noise_sd=0.0, background=0.0, seed=0):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        m = self.model
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(m.ngeom)]
        self.rope_ids = np.array([g for g, n in enumerate(names) if n.startswith("dlo_geom")])
        self.sphere_ids = np.array([g for g, n in enumerate(names) if n == "obstacle_sphere"])
        self.bg_ids = np.array([g for g, n in enumerate(names) if n in BACKGROUND])
        if background:
            for g in self.bg_ids:
                m.geom_rgba[g, :3] = background / 255.0
        self.noise_sd, self.rng = float(noise_sd), np.random.default_rng(seed)
        self.rgb = mujoco.Renderer(m, L.IMAGE_H, L.IMAGE_W)
        self.seg = mujoco.Renderer(m, L.IMAGE_H, L.IMAGE_W)
        self.seg.enable_segmentation_rendering()

    def set_state(self, st, i):
        d = self.data
        d.qpos[:], d.mocap_pos[:], d.mocap_quat[:] = st["qpos"][i], st["mocap_pos"][i], st["mocap_quat"][i]
        mujoco.mj_forward(self.model, d)

    def render(self, role, want_seg=True):
        self.rgb.update_scene(self.data, camera=CAM_NAMES[role])
        img = self.rgb.render()
        if self.noise_sd:
            img = np.clip(img.astype(np.float32) + self.rng.normal(0, self.noise_sd, img.shape), 0, 255).astype(np.uint8)
        seg = None
        if want_seg:
            self.seg.update_scene(self.data, camera=CAM_NAMES[role])
            s = self.seg.render()
            seg = np.where(s[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), s[..., 0], -1)
        return img, seg

    def frame(self, st, i, offset=0, want_seg=True):
        """Front image of record i, top image of record i + offset (clipped). Returns rgbs, segs, the record index
        each view shows."""
        n = len(st["t"])
        rgbs, segs, idx = {}, {}, {}
        for role, j in (("front", i), ("top", min(i + offset, n - 1))):
            self.set_state(st, j)
            rgbs[role], segs[role] = self.render(role, want_seg)
            idx[role] = j
        return rgbs, segs, idx

    def visibility(self, centerline, seg, role):
        px, z = L.project(centerline, role)
        out = []
        for (u, v), zz in zip(px, z):
            ui, vi = int(round(u)), int(round(v))
            if zz <= 0 or not (0 <= ui < L.IMAGE_W and 0 <= vi < L.IMAGE_H):
                out.append("outside")
                continue
            win = seg[max(vi - 2, 0):vi + 3, max(ui - 2, 0):ui + 3]
            if np.isin(win, self.rope_ids).any():
                out.append("rope")
            elif np.isin(seg[vi, ui], self.sphere_ids):
                out.append("sphere")
            elif seg[vi, ui] < 0 or np.isin(seg[vi, ui], self.bg_ids):
                out.append("background")
            else:
                out.append("robot")
        return out

    def close(self):
        for r in (self.rgb, self.seg):
            r.close()


def load_states(path):
    z = np.load(path)
    eps = {}
    for key in z.files:
        e, f = key.split("/")
        eps.setdefault(e, {})[f] = z[key]
    return eps
