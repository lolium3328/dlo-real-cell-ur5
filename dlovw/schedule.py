"""Stage-0′ decision layer (docs/notes/stage0p_prereg_20261023.md, section 3).

`ScheduleSession` is the B2 dodge rule on exact, unoccluded measurements (the EV arm) with the go/wait
gate replaced by a fixed wait schedule: during the given step intervals the grippers hold still (the
dodge reflex still takes priority), otherwise the planner drives. The search tries a fixed family of
schedules per scene and keeps the best outcome.
"""
from __future__ import annotations

from .belief import BeliefSession

DT = 0.008
EXACT = dict(level="exact", delta_c_m=0.0, delta_v_mps=0.0, latency_steps=0, bias_m=[0.0, 0.0, 0.0])
START_WAITS_S = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0)
GO_THEN_WAIT_S = tuple((g, w) for g in (0.5, 1.0) for w in (0.25, 0.5, 1.0))


def steps(t):
    return int(round(t / DT))


def candidates():
    """name -> ((start_step, end_step), ...) hold intervals, in the pre-registered order."""
    out = {f"w{w:g}": ((0, steps(w)),) if w > 0 else () for w in START_WAITS_S}
    out.update({f"g{g:g}w{w:g}": ((steps(g), steps(g + w)),) for g, w in GO_THEN_WAIT_S})
    return out


def exact_unoccluded(cfg):
    """The EV information setting: exact measurements and no occluders (in place)."""
    cfg["dlovw"]["perception"] = dict(EXACT)
    cfg["dlovw"]["occluders"] = []
    return cfg


class ScheduleSession(BeliefSession):
    arm = "H0"
    waits = ()

    def _gate_blocked(self, x):
        k = self.progress.steps
        return any(a <= k < b for a, b in self.waits)
