"""Frozen-input multi-session trajectory evaluation for model refinement."""
from __future__ import annotations
import json
from pathlib import Path
import time
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from .physics import rope_model,initialize
from .replay import score_curve
from .session import sessions
from .vision import cameras

_DATA={}
_SESSIONS={}
_CAMS=None
_BASE=None


def worker_init(root):
    global _DATA,_SESSIONS,_CAMS,_BASE
    root=Path(root)
    _BASE=json.loads((root/'baseline_profile.json').read_text())
    _CAMS=cameras(_BASE['cameras'])
    _SESSIONS={s.sid:s for s in sessions('/media/anilam/SSD_DISK/rope_motion') if s.sid in ['161441','170118','163526','163618','162231','170844']}
    _DATA={p.stem.split('_')[1]:json.loads(p.read_text()) for p in root.glob('cache_*.json')}
    for c in _DATA.values():
        for r in c['rows']:r['xy']=np.array(r['xy'])


def set_rest_fraction(model,seed,fraction):
    """Shared scalar initial-shape memory, frozen throughout the trajectory."""
    for j in range(1,model.njnt):
        qadr=model.jnt_qposadr[j];q=seed.qpos[qadr:qadr+4]
        rot=Rotation.from_quat(np.r_[q[1:],q[0]])
        qnew=Rotation.from_rotvec(fraction*rot.as_rotvec()).as_quat()
        model.qpos_spring[qadr:qadr+4]=np.r_[qnew[3],qnew[:3]]


def evaluate_session(sid,parameters,stride=30,dt=.002,save_frames=False,stop_after=None):
    cache=_DATA[sid];s=_SESSIONS[sid];initial=cache['initial'];x=np.array(initial['xyz']);start=initial['t']
    rows=[r for r in cache['rows'] if r['index']%stride==0 and (stop_after is None or r['t']<=start+stop_after)]
    length,k,damping,friction,rest=parameters
    m=rope_model(length,_BASE['mass_kg'],_BASE['radius_m'],k,.003)
    m.opt.timestep=dt;m.dof_frictionloss[6:]=friction
    d=initialize(m,x);set_rest_fraction(m,d,rest)
    ids=np.array([m.body(f'dlo_{i:02d}').id for i in range(len(x))])
    for _ in range(round(30/dt)):mujoco.mj_step(m,d)
    mujoco.mj_forward(m,d)
    settled=dict(shape_rmse_m=float(np.sqrt(np.mean(np.sum((d.xpos[ids]-x)**2,axis=1)))),
                 max_speed_m_s=float(np.linalg.norm(d.cvel[ids,3:],axis=1).max()))
    m.dof_damping[6:]=damping;d.qvel[:]=0;d.time=0
    offset=x[0]-s.position(start)
    count=int(np.ceil((rows[-1]['t']-start)/dt))+2
    times=start+np.arange(count+1)*dt
    targets=s.position(times)+offset
    records=[];step=0;max_endpoint=0.
    for row in rows:
        needed=max(step,int(round((row['t']-start)/dt)))
        while step<needed:
            step+=1;d.mocap_pos[0]=targets[step];mujoco.mj_step(m,d)
        mujoco.mj_forward(m,d)
        if not np.isfinite(d.qpos).all() or abs(d.time-step*dt)>.01:
            return dict(session=sid,failed=True,reason='unstable simulation',mean_error_px=1e4)
        p=d.xpos[ids].copy();max_endpoint=max(max_endpoint,float(np.linalg.norm(p[[0,-1]]-d.mocap_pos,axis=1).max()))
        record=dict(t=row['t'],role=row['role'],index=row['index'],moving=row['moving'],
                    error_px=score_curve(p,_CAMS[row['role']],row['xy']))
        if save_frames:record['xyz']=p.tolist()
        records.append(record)
    def stats(v):
        e=np.array([r['error_px'] for r in v])
        return dict(n=len(e),mean=float(e.mean()),p95=float(np.percentile(e,95))) if len(e) else dict(n=0)
    total=stats(records)
    return dict(session=sid,failed=False,mean_error_px=total['mean'],p95_error_px=total['p95'],n=total['n'],
                by_role={r:stats([v for v in records if v['role']==r]) for r in ['front','top']},
                by_phase={r:stats([v for v in records if v['moving']==b]) for r,b in [('moving',True),('stationary',False)]},
                max_endpoint_error_m=max_endpoint,settled=settled,frames=records if save_frames else None)


def evaluate_job(job):
    name,parameters,sids,stride,dt,save,stop_after=job
    begin=time.monotonic()
    reports=[evaluate_session(s,parameters,stride,dt,save,stop_after) for s in sids]
    # Equal weight per session, regardless of duration or image count.
    return dict(name=name,parameters=list(map(float,parameters)),sessions=reports,
                score=float(np.mean([r['mean_error_px'] for r in reports])),seconds=time.monotonic()-begin,
                stride=stride,dt=dt,stop_after=stop_after)
