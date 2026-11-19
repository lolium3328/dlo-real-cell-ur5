"""Backup shield: keep an escape route open (docs/notes/backup_shield_20261020.md).

Every control step, after the SQP planner proposes its first action, the shield predicts with the
RSSM rollout (the planner's own `_predict_model`, anchored like the planner) what happens if that
action is taken and then one of a few fixed backup manoeuvres is held for `steps` steps. Obstacles are
extrapolated at constant estimated velocity with a tube of delta_c + delta_v t. If no backup keeps the
predicted clearance >= `margin_m`, the shield instead starts the backup that scores best from the
current state. The chosen action still goes through the legacy CBF filter.

This is a non-learned "score of the current decision's future (safety) impact": the best backup's
predicted minimum clearance after the candidate action.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from .sessions import Session
from control.task_value_protocol import sphere_clearances  # noqa: E402

STEP_M = 0.003


class ShieldSession(Session):
    cache_latent_gain0 = False   # the shield rolls the RSSM out from the current latent
    shield_steps = 50            # backup horizon, 0.4 s
    shield_margin_m = 0.002      # predicted clearance the best backup must keep
    shield_gate_m = 0.15         # skip the check when every obstacle stays this far from the still rope
    shield_latch = 1             # once triggered, hold the chosen backup this many steps

    def _shield_init(self):
        if not hasattr(self, "shield_log"):
            self.shield_log = dict(checks=0, triggers=0, ms=[], trace=[])

    def _backups(self, x, obstacles):
        """Fixed backup manoeuvres as constant 6-d actions [left xyz, right xyz]."""
        left, right = x[0], x[-1]
        axis = (right - left) / max(np.linalg.norm(right - left), 1e-9)
        lateral = np.cross([0.0, 0.0, 1.0], axis)
        lateral /= max(np.linalg.norm(lateral), 1e-9)
        out = {"up": np.r_[0, 0, 1, 0, 0, 1] * STEP_M, "down": np.r_[0, 0, -1, 0, 0, -1] * STEP_M,
               "contract": np.r_[axis, -axis] * STEP_M * 0.8,
               "lat_plus": np.r_[lateral, lateral] * STEP_M, "lat_minus": -np.r_[lateral, lateral] * STEP_M}
        # away: common translation from the nearest obstacle centre towards the nearest rope point
        best, away = np.inf, None
        for o in obstacles:
            d = np.linalg.norm(x - o.position, axis=1)
            i = int(d.argmin())
            if d[i] < best:
                best, away = d[i], (x[i] - o.position) / max(d[i], 1e-9)
        if away is not None:
            out["away"] = np.r_[away, away] * STEP_M
        return out

    def _min_clearance(self, worlds, obstacles, radii):
        """worlds: (B, K, N, 3) predicted centerlines for steps 1..K. Returns (B,) min protected clearance."""
        dt = self.protocol.dt
        out = np.full(len(worlds), np.inf)
        for k in range(worlds.shape[1]):
            t = (k + 1) * dt
            centers = np.stack([o.position + o.velocity * t for o in obstacles])
            grow = 0.005 + 0.02 * t
            for b in range(len(worlds)):
                h = sphere_clearances(worlds[b, k], centers, radii + grow, 0.010, 0.005, 0.0)[1].min()
                out[b] = min(out[b], h)
        return out

    def _rollout(self, sequences, x, h, z, frame):
        import torch
        p = self.planner
        saved = (p.horizon, getattr(p, "enforce_endpoint_kinematics", False))
        p.horizon, p.enforce_endpoint_kinematics = sequences.shape[1], True
        try:
            p.set_current_orientations(np.asarray(frame.left_pose)[3:7], np.asarray(frame.right_pose)[3:7])
            acts = torch.as_tensor(np.concatenate([sequences, np.zeros((1, *sequences.shape[1:]))]),
                                   device=p.device, dtype=torch.float32)
            lt = torch.as_tensor(np.asarray(frame.left_pose)[:3], device=p.device, dtype=torch.float32)
            rt = torch.as_tensor(np.asarray(frame.right_pose)[:3], device=p.device, dtype=torch.float32)
            with torch.no_grad():
                world, _, _ = p._predict_model(acts, SimpleNamespace(x_k=x, num_points=len(x)), h, z, lt, rt)
            world = world.cpu().numpy().astype(float)
        finally:
            p.horizon, p.enforce_endpoint_kinematics = saved
        return world[:-1] + (x - world[-1, 0])          # zero-first anchor, as the planner does

    def step(self, nominal=None, **kw):
        self._shield_init()
        if nominal is not None or kw:
            return super().step(nominal, **kw)
        if getattr(self, "_latched", None) and self._latched[1] > 0:
            action, left = self._latched
            self._latched = (action, left - 1)
            self.plan()                                   # keep the planner's warm start in sync
            return super().step(action)
        nominal = np.asarray(self.plan()[0], float)
        obstacles = list(self.current_obstacles.obstacles) if self.current_obstacles is not None else []
        if not obstacles:
            return super().step(nominal)
        x = np.asarray(self.state().x_k, float)
        horizon_t = self.shield_steps * self.protocol.dt
        reach = min(np.min(np.linalg.norm(x[:, None] - (o.position + o.velocity * s)[None], axis=-1))
                    - o.size[0] for o in obstacles for s in np.linspace(0, horizon_t, 6))
        if reach > self.shield_gate_m:
            return super().step(nominal)
        started = time.perf_counter()
        self.shield_log["checks"] += 1
        backups = self._backups(x, obstacles)
        names = list(backups)
        n = self.shield_steps
        seqs = [np.vstack([nominal[None], np.tile(backups[b], (n, 1))]) for b in names]      # candidate, then backup
        seqs += [np.tile(backups[b], (n + 1, 1)) for b in names]                            # backup from now
        h, z = self.latent()
        worlds = self._rollout(np.stack(seqs), x, h, z, self.frames[-1])
        radii = np.asarray([o.size[0] for o in obstacles])
        score = self._min_clearance(worlds, obstacles, radii)
        after, now = score[:len(names)], score[len(names):]
        action = nominal
        if after.max() < self.shield_margin_m:
            action = backups[names[int(now.argmax())]]
            self.shield_log["triggers"] += 1
            self._latched = (action, self.shield_latch - 1)
        self.shield_log["ms"].append(1000 * (time.perf_counter() - started))
        self.shield_log["trace"].append((int(self.progress.steps), float(after.max()), float(now.max()),
                                         names[int(after.argmax())], names[int(now.argmax())], bool(action is not nominal)))
        return super().step(action)
