"""Collection workers that run alongside training (docs/notes/imagination_ac_design_20261022.md, section 8).

Each worker is a spawned CPU process (one thread, no CUDA, no RSSM bundle) that takes jobs from a queue
and returns finished episodes. The pool grows by one worker per `maintain()` call while the machine
has at least RAMP_MB available, up to `max_workers`, and drops one worker whenever less than MIN_MB is
available. Workers exit after `jobs_per_worker` episodes (legacy sessions leak a little memory) and
are replaced.
"""
from __future__ import annotations

import queue
from multiprocessing import get_context

MIN_MB, RAMP_MB = 8000, 10000


def mem_available_mb():
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    return 0


_NETS = {}


def _nets(ckpt):
    if _NETS.get("ckpt") != ckpt:
        import torch
        from .actor_critic import Actor
        from .world_model import FactoredWM
        from . import actor_critic as ac_mod
        state = torch.load(ckpt, map_location="cpu")
        ac_mod.PROGRESS = state.get("progress", "relu")        # the reward the checkpoint was trained with
        wm = FactoredWM(**state["wm_kw"])
        wm.load_state_dict(state["wm"])
        actor = Actor(wm.feat_dim, extra=state.get("actor_extra", 0))
        actor.load_state_dict(state["actor"])
        _NETS.update(ckpt=ckpt, wm=wm.eval(), actor=actor.eval())
    return _NETS["wm"], _NETS["actor"]


def run_job(job):
    """job: dict(kind, tag, rec, ckpt, noise, seed, max_time, target). Returns the episode or an error."""
    from .diagnose import episode as diagnose_episode
    from .rollout import run_episode
    rec = job["rec"]
    try:
        if job.get("gb"):                         # G-B checkpoint (dlovw/imagination/gb.py): collection only
            from .gb_run import load_gb, run_episode_gb
            if _NETS.get("ckpt") != job["ckpt"]:
                _NETS.update(ckpt=job["ckpt"], gb=load_gb(job["ckpt"]))
            wm, actor, _ = _NETS["gb"]
            ep = run_episode_gb(rec, wm, actor, noise=job["noise"], seed=job["seed"], max_time_s=job["max_time"], target=job.get("target"))
            ep["info"].update(version=rec["version"], index=int(rec["index"]), source="actor_gb")
            ep.update(kind=job["kind"], tag=job["tag"])
            return ep
        if job.get("cell"):                       # cell 2026-11-12 (dlovw/cell/rollout.py): r10 actor, deployment noise
            from .r10 import load_r10
            from ..cell.noise import sample_noise
            from ..cell.rollout import run_actor
            import numpy as np
            if _NETS.get("ckpt") != job["ckpt"]:
                _NETS.update(ckpt=job["ckpt"], r10=load_r10(job["ckpt"]))
            wm, fc, actor, _ = _NETS["r10"]
            rng = np.random.default_rng(job["seed"] + 17)
            deploy = sample_noise(rng) if rng.uniform() < job.get("deploy_frac", 0.75) else None
            ep = run_actor(rec, wm, fc, actor, noise=job["noise"], seed=job["seed"], deploy=deploy)
            ep["info"].update(version=rec["version"], index=int(rec["index"]), deploy=deploy is not None)
            ep.update(kind=job["kind"], tag=job["tag"])
            return ep
        if job.get("r10"):                        # r10 checkpoint (dlovw/imagination/r10.py): collection only
            from .r10 import load_r10, run_episode_r10
            if _NETS.get("ckpt") != job["ckpt"]:
                _NETS.update(ckpt=job["ckpt"], r10=load_r10(job["ckpt"]))
            wm, fc, actor, _ = _NETS["r10"]
            ep = run_episode_r10(rec, wm, fc, actor, noise=job["noise"], seed=job["seed"], max_time_s=job["max_time"],
                                 target=job.get("target"))
            ep["info"].update(version=rec["version"], index=int(rec["index"]), source="actor_r10")
            ep.update(kind=job["kind"], tag=job["tag"])
            return ep
        wm, actor = _nets(job["ckpt"])
        if job["kind"] == "eval":                 # no noise, with the failure diagnosis
            row = diagnose_episode(rec, wm, actor, target=job.get("target"), max_time_s=job["max_time"], seed=job["seed"])
            row.update(kind="eval", tag=job["tag"])
            return row
        ep = run_episode(rec, wm, actor, noise=job["noise"], seed=job["seed"], max_time_s=job["max_time"], target=job.get("target"))
        ep["info"].update(version=rec["version"], index=int(rec["index"]), source="actor")
        ep.update(kind=job["kind"], tag=job["tag"])
        return ep
    except Exception as exc:
        return dict(kind=job["kind"], tag=job["tag"], error=f"{rec['id']}: {exc!r}")


def _main(jobs, results, max_jobs):
    import os
    import torch
    torch.set_num_threads(1)
    parent, done = os.getppid(), 0
    while done < max_jobs and os.getppid() == parent:      # leave when the trainer is gone
        try:
            job = jobs.get(timeout=10)
        except queue.Empty:
            continue
        if job is None:
            break
        results.put(run_job(job))
        done += 1
    if os.getppid() != parent:
        results.cancel_join_thread()          # nobody will read them: do not hang at exit flushing results


class Workers:
    def __init__(self, start, max_workers, jobs_per_worker=40):
        self.ctx = get_context("spawn")
        self.jobs, self.results = self.ctx.Queue(), self.ctx.Queue()
        self.target, self.max_workers, self.jobs_per_worker = start, max_workers, jobs_per_worker
        self.procs, self.inflight, self.lost = [], 0, 0
        self.maintain(ramp=False)

    def maintain(self, ramp=True):
        """Replace exited workers, then ramp up or shed one worker according to available memory."""
        alive = []
        for p in self.procs:
            if p.is_alive():
                alive.append(p)
            elif p.exitcode not in (0, None):             # died holding a job: count it as lost
                self.lost += 1
                self.inflight = max(self.inflight - 1, 0)
        self.procs = alive
        avail = mem_available_mb()
        if avail < MIN_MB and self.target > 1:
            self.target -= 1
            self.jobs.put(None)                           # one worker leaves after its current job
        elif ramp and avail >= RAMP_MB and self.target < self.max_workers:
            self.target += 1
        while len(self.procs) < self.target:
            p = self.ctx.Process(target=_main, args=(self.jobs, self.results, self.jobs_per_worker), daemon=True)
            p.start()
            self.procs.append(p)
        return dict(workers=len(self.procs), target_workers=self.target, mem_available_mb=avail, lost_jobs=self.lost)

    def submit(self, job):
        self.jobs.put(job)
        self.inflight += 1

    def get(self, timeout=None):
        """One finished job (blocking up to `timeout` s; None if nothing came)."""
        try:
            out = self.results.get(timeout=timeout) if timeout else self.results.get_nowait()
        except queue.Empty:
            return None
        self.inflight -= 1
        return out

    def close(self):
        for _ in self.procs:
            self.jobs.put(None)
        for p in self.procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
