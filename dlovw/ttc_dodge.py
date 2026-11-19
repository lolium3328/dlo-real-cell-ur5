"""Hand-written "when to dodge early" rule (docs/notes/backup_shield_and_speed_20261020.md, section 4).

Every control step, each obstacle's estimated straight-line path over the next `lookahead_s` is checked
against the current (assumed still) rope. If some obstacle is predicted to come within the protected
radius plus `margin_m` within `trigger_s`, both grippers translate away from that obstacle's path
(perpendicular to its velocity; straight up if the rope point is on the path) at `speed_frac` of the
gripper cap, instead of following the planner, until no obstacle triggers. The action still goes
through the CBF. No learning and no rope model: this is the baseline a learned decision value has to beat.

With `reinit_after_dodge`, the planner and local Jacobian are re-initialised (`canonical_initialize`)
when a dodge ends, as `run_to_end` does after a forced opening; otherwise the planner resumes from a
warm start that assumed its own actions were executed.

With `choose_direction`, the escape direction is no longer "away from the earliest threat" but the
best of {up, down, lateral +-, away from each obstacle}, scored against *all* obstacles: the whole rope
is assumed to translate rigidly along the direction (stopping where either gripper would leave the
workspace) and the score is the smallest predicted margin to any obstacle's straight-line path over
`lookahead_s`. The previous direction is kept unless another scores `switch_m` better.
`fixed_direction` ("up", "down", "lat+", "lat-") forces one of those candidates instead.

With `after_mode = "hold"`, once a dodge has happened the grippers stay still (zero action) while any
obstacle is still predicted to enter the inflated radius within `lookahead_s`, instead of handing
control straight back to the planner, which tends to pull the rope back into the obstacle's path and
re-trigger the dodge (docs/notes/speed_headroom_20261021.md, section 1).
"""
from __future__ import annotations

import numpy as np

from .sessions import Session

PROTECTED_M = 0.055


class TTCDodgeSession(Session):
    trigger_s = 0.6
    margin_m = 0.02
    lookahead_s = 1.5
    reinit_after_dodge = False
    choose_direction = False
    switch_m = 0.005
    speed_frac = 0.85
    fixed_direction = None
    after_mode = "plan"

    def _ttc_init(self):
        if not hasattr(self, "dodge_log"):
            self.dodge_log = dict(steps=0, episodes=0, hold_steps=0, gate_steps=0)
            self._dodging = False
            self._holding = False
            self._direction = None

    def _path(self, j, o, ts):
        """Predicted centres (T, 3) of obstacle j at times ts ahead, and the extra radius (T,) on top
        of o.size[0]. Default: straight line at the estimated velocity (stage-0 beliefs override)."""
        return o.position[None] + ts[:, None] * np.asarray(o.velocity, float)[None], np.zeros(len(ts))

    def _threat(self, x, within=None):
        """Earliest predicted entry (time, obstacle, rope point) into the inflated radius within
        `within` seconds (default `trigger_s`), or None."""
        within = self.trigger_s if within is None else within
        best = None
        for j, o in enumerate(self.current_obstacles.obstacles):
            ts = self._times()
            path, grow = self._path(j, o, ts)                                     # (T, 3), (T,)
            d = np.linalg.norm(x[None, :, :] - path[:, None, :], axis=-1)         # (T, N)
            hit = np.flatnonzero(d.min(1) < PROTECTED_M + self.margin_m - 0.040 + o.size[0] + grow)
            if len(hit) and ts[hit[0]] <= within and (best is None or ts[hit[0]] < best[0]):
                k = hit[0]
                best = (ts[k], o, x[int(d[k].argmin())])
        return best

    def _times(self):
        return np.arange(0.0, self.lookahead_s + 1e-9, self.protocol.dt * 4)

    def _away(self, o, p):
        """Unit vector from obstacle o's path towards rope point p, perpendicular to its velocity."""
        v = np.asarray(o.velocity, float)
        rel = p - o.position
        if np.linalg.norm(v) > 1e-6:
            vh = v / np.linalg.norm(v)
            rel = rel - (rel @ vh) * vh
        return rel / np.linalg.norm(rel) if np.linalg.norm(rel) > 1e-3 else np.array([0.0, 0.0, 1.0])

    def _travel(self, x, d):
        """Largest distance both grippers can translate along d inside the workspace box."""
        ws, out = self.sim_cfg["workspace"], np.inf
        for p, lo, hi in ((x[0], ws["left_low"], ws["left_high"]), (x[-1], ws["right_low"], ws["right_high"])):
            for k in range(3):
                if d[k] > 1e-9:
                    out = min(out, (hi[k] - p[k]) / d[k])
                elif d[k] < -1e-9:
                    out = min(out, (lo[k] - p[k]) / d[k])
        return max(out, 0.0)

    def _escape_score(self, x, d, speed):
        ts = self._times()
        shift = np.minimum(ts * speed, self._travel(x, d))[:, None, None] * d       # (T, 1, 3)
        rope = x[None] + shift                                                     # (T, N, 3)
        worst = np.inf
        for j, o in enumerate(self.current_obstacles.obstacles):
            path, grow = self._path(j, o, ts)
            dist = np.linalg.norm(rope - path[:, None, :], axis=-1).min(1)
            worst = min(worst, float((dist - (PROTECTED_M + self.margin_m - 0.040 + o.size[0] + grow)).min()))
        return worst

    def _candidates(self, x):
        axis = x[-1] - x[0]
        axis[2] = 0.0
        lat = np.cross([0.0, 0.0, 1.0], axis / max(np.linalg.norm(axis), 1e-9))
        dirs = {"up": np.array([0.0, 0.0, 1.0]), "down": np.array([0.0, 0.0, -1.0]), "lat+": lat, "lat-": -lat}
        ts = self._times()
        for j, o in enumerate(self.current_obstacles.obstacles):
            path, _ = self._path(j, o, ts)
            d = np.linalg.norm(x[None] - path[:, None, :], axis=-1)
            k, i = np.unravel_index(int(d.argmin()), d.shape)
            dirs[f"away{j}"] = self._away(o, x[i])
        return dirs

    def _choose(self, x, speed):
        dirs = self._candidates(x)
        score = {n: self._escape_score(x, d, speed) for n, d in dirs.items()}
        best = max(score, key=score.get)
        if self._direction in score and score[self._direction] >= score[best] - self.switch_m:
            best = self._direction
        self._direction = best
        return dirs[best]

    def _gate_blocked(self, x):
        """Go/wait gate (stage-0 belief arms override); the plain rule never waits."""
        return False

    def _release(self, planned):
        """Hand control back to the planner after a dodge or hold (planning now if not done yet)."""
        if (self._dodging or self._holding) and self.reinit_after_dodge:
            self.canonical_initialize()
            planned = None
        if planned is None:
            planned = np.asarray(self.plan()[0], float)
        self._dodging = self._holding = False
        self._direction = None
        return planned

    def step(self, nominal=None, **kw):
        self._ttc_init()
        if nominal is not None or kw or self.current_obstacles is None:
            return super().step(nominal, **kw)
        # With reinit_after_dodge the planner is rebuilt whenever a dodge or hold ends, so a plan made
        # while dodging or holding is never used: plan lazily (same actions, far fewer QP solves).
        planned = None if self.reinit_after_dodge else np.asarray(self.plan()[0], float)
        x = np.asarray(self.state().x_k, float)
        threat = self._threat(x)
        if threat is None:
            if self._gate_blocked(x):                 # wait until the way to the target is predicted clear
                self._dodging, self._holding = False, True
                self.dodge_log["gate_steps"] += 1
                return super().step(np.zeros(6))
            if (self.after_mode == "hold" and (self._dodging or self._holding)
                    and self._threat(x, self.lookahead_s) is not None):
                self._dodging, self._holding = False, True
                self.dodge_log["hold_steps"] += 1
                return super().step(np.zeros(6))
            return super().step(self._release(planned))
        self.dodge_log["episodes"] += not self._dodging
        self._dodging, self._holding = True, False
        _, o, p = threat
        step = self.speed_frac * float(self.sim_cfg["max_translation_step_m"])
        if self.fixed_direction:
            away = self._candidates(x)[self.fixed_direction]
        elif self.choose_direction:
            away = self._choose(x, step / self.protocol.dt)
        else:
            away = self._away(o, p)
        self.dodge_log["steps"] += 1
        return super().step(np.r_[away, away] * step)
