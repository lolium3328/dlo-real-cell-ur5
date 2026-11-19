"""Replay observed endpoint translations and score visible image centerlines."""
import cv2
import mujoco
import numpy as np
from scipy.spatial import cKDTree
from ..imagination.camera_model import project
from ..imagination.rgb_usb import thin
from .physics import rope_model,initialize
from .vision import image,segment


def prepare(s,start,cams,stride=6):
    samples=[]
    for role in cams:
        for i in range(0,len(s.frames[role]),stride):
            t=s.ft[role][i]
            if t<start or t>s.t[-1]:continue
            im=image(s,role,i);mask,info=segment(im)
            if info['area']<200:continue
            yy,xx=np.where(mask);crop=mask[yy.min():yy.max()+1,xx.min():xx.max()+1]
            sk=thin(np.pad(crop,2).astype(np.uint8)*255)
            y,x=np.where(sk);xy=np.column_stack([x+xx.min()-2,y+yy.min()-2])[::3]
            if len(xy)<20:continue
            samples.append(dict(t=float(t),role=role,index=i,xy=xy))
    return sorted(samples,key=lambda x:x['t'])


def score_curve(x,camera,observed):
    dense=np.concatenate([np.linspace(a,b,8,endpoint=False) for a,b in zip(x[:-1],x[1:])]+[x[-1:]])
    uv,z=project(dense,camera)
    valid=(z>0)&(uv[:,0]>=0)&(uv[:,0]<640)&(uv[:,1]>=0)&(uv[:,1]<480)
    uv=uv[valid]
    if len(uv)<10:return 100.
    a=cKDTree(uv).query(observed)[0];b=cKDTree(observed).query(uv)[0]
    return float(.5*(np.mean(np.minimum(a,50))+np.mean(np.minimum(b,50))))


def replay(s,initial,profile,cams,samples,damping):
    x=np.array(initial['xyz']);start=initial['t']
    m=rope_model(profile['length_m'],profile['mass_kg'],profile['radius_m'],profile['joint_stiffness_nm_per_rad'],.003)
    d=initialize(m,x);ids=[m.body(f'dlo_{i:02d}').id for i in range(len(x))]
    # Same equilibrium state for every candidate; damping changes only after settling.
    for _ in range(15000):mujoco.mj_step(m,d)
    m.dof_damping[6:]=damping
    d.qvel[:]=0;d.time=0
    offset=x[0]-s.position(start)
    out=[];dt=m.opt.timestep
    for row in samples:
        while d.time<row['t']-start-dt/2:
            d.mocap_pos[0]=s.position(start+d.time+dt)+offset
            mujoco.mj_step(m,d)
        mujoco.mj_forward(m,d);p=d.xpos[ids].copy()
        out.append(dict(t=row['t'],role=row['role'],index=row['index'],error_px=score_curve(p,cams[row['role']],row['xy']),xyz=p.tolist()))
    return out
