"""Rendered RGB frames of recorded simulator states (scripts/perception_rgb/record_states.py) under a camera model
(camera_model.py), with the realism settings measured on the real cell (docs/notes/perception_rgb_20261110.md):

- noise_sd: per-pixel Gaussian sensor noise in grey levels (real D435i colour images: 4.7-5.1, rendered: 0);
- offset_steps: the top camera's frame is taken this many control steps after the front one (real pair: 14-20 ms
  apart, not synchronised; 2 steps = 16 ms);
- background: grey level added to the black table / backdrop / floor (real dark cloth: 24-29, rendered: 9-13).

Ground truth for evaluation only: per camera and rope node, whether the node's pixel shows the rope (rope geoms in the
segmentation within 2 px), and what covers it otherwise (robot, obstacle sphere, outside the image).
"""
from __future__ import annotations

import os

import mujoco
import numpy as np

from . import camera_model as cm

BACKGROUND_GEOMS = ("table", "front_depth_backdrop", "floor")


class FrameRenderer:
    def __init__(self, xml_path, preset="real", noise_sd=0.0, offset_steps=0, background=0.0, seed=0, corrected=False):
        if corrected:                                       # the cell as measured (sim_model.py)
            from .sim_model import corrected_model
            self.model = corrected_model(xml_path)
        else:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.cams = cm.PRESETS[preset] if isinstance(preset, str) else preset
        cm.apply(self.model, self.cams)
        self.noise_sd, self.offset, self.rng = float(noise_sd), int(offset_steps), np.random.default_rng(seed)
        m = self.model
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(m.ngeom)]
        self.rope_ids = np.array([g for g, n in enumerate(names) if n.startswith("dlo_geom")])
        self.sphere_ids = np.array([g for g, n in enumerate(names) if n.startswith("obstacle_sphere")])
        self.bg_ids = np.array([g for g, n in enumerate(names) if n in BACKGROUND_GEOMS])
        if background:
            for g in self.bg_ids:
                if m.geom_matid[g] >= 0:
                    m.mat_rgba[m.geom_matid[g], :3] = background / 255.0
                m.geom_rgba[g, :3] = background / 255.0
        self.rgb = mujoco.Renderer(m, cm.H, cm.W)
        self.seg = mujoco.Renderer(m, cm.H, cm.W)
        self.seg.enable_segmentation_rendering()
        self.hidden_group = 5

    def set_state(self, st, i, hidden=None):
        d = self.data
        d.qpos[:], d.mocap_pos[:], d.mocap_quat[:] = st["qpos"][i], st["mocap_pos"][i], st["mocap_quat"][i]
        mujoco.mj_forward(self.model, d)
        groups = self.model.geom_group[self.sphere_ids].copy()
        h = st["hidden"][i] if hidden is None else hidden
        self.model.geom_group[self.sphere_ids] = np.where(h[:len(self.sphere_ids)], self.hidden_group, groups)
        return groups

    def _one(self, role, want_seg):
        cam = self.cams[role]
        self.rgb.update_scene(self.data, camera=cm.MJ_NAMES[role])
        img = cm.warp(self.rgb.render(), cam)
        if self.noise_sd:
            img = np.clip(img.astype(np.float32) + self.rng.normal(0, self.noise_sd, img.shape), 0, 255).astype(np.uint8)
        seg = None
        if want_seg:
            self.seg.update_scene(self.data, camera=cm.MJ_NAMES[role])
            s = self.seg.render()
            ids = cm.warp(s[..., 0].astype(np.float32), cam, nearest=True).astype(np.int32)
            typ = cm.warp(s[..., 1].astype(np.float32), cam, nearest=True).astype(np.int32)
            seg = np.where(typ == int(mujoco.mjtObj.mjOBJ_GEOM), ids, -1)
        return img, seg

    def frame(self, st, i, want_seg=True):
        """Images of state i (front) and i + offset (top, clipped to the episode). Returns dict role -> rgb, dict
        role -> segmentation (geom id, -1 elsewhere) and the truth of each camera's state."""
        n = len(st["k"])
        rgbs, segs, truth = {}, {}, {}
        for role, j in (("front", i), ("top", min(i + self.offset, n - 1))):
            groups = self.set_state(st, j)
            try:
                rgbs[role], segs[role] = self._one(role, want_seg)
            finally:
                self.model.geom_group[self.sphere_ids] = groups
            truth[role] = dict(centerline=st["centerline"][j], grip=st["grip"][j], obstacles=st["obstacles"][j], hidden=st["hidden"][j])
        return rgbs, segs, truth

    def visibility(self, centerline, seg, role):
        """Per node: 'rope' if a rope pixel lies within 2 px of its projection, else what is there."""
        px, z = cm.project(centerline, self.cams[role])
        out = []
        for (u, v), zz in zip(px, z):
            ui, vi = int(round(u)), int(round(v))
            if zz <= 0 or not (0 <= ui < cm.W and 0 <= vi < cm.H):
                out.append("outside")
                continue
            win = seg[max(vi - 2, 0):vi + 3, max(ui - 2, 0):ui + 3]
            if np.isin(win, self.rope_ids).any():
                out.append("rope")
            elif np.isin(seg[vi, ui], self.sphere_ids):
                out.append("sphere")
            elif np.isin(seg[vi, ui], self.bg_ids) or seg[vi, ui] < 0:
                out.append("background")
            else:
                out.append("robot")
        return out

    def close(self):
        for r in (self.rgb, self.seg):
            r.close()


def load_states(path):
    """dict episode id -> dict of arrays (record_states.py output)."""
    z = np.load(path)
    eps = {}
    for key in z.files:
        e, f = key.split("/")
        eps.setdefault(e, {})[f] = z[key]
    return eps


def xml_for(n_obstacles):
    from .. import legacy
    return os.path.join(legacy.LEGACY_ROOT, legacy.XML_BY_OBSTACLES[n_obstacles])
