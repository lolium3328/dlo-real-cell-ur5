"""Conditional discrete-rod identification. Mass is an explicit scale convention."""
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from scipy.optimize import least_squares
from ..cell.scene import quat,mat


def frame(t):
    x=np.asarray(t,float);x=x/np.linalg.norm(x)
    ref=np.array([0.,0.,1.]) if abs(x[2])<.95 else np.array([0.,1.,0.])
    y=np.cross(ref,x);y/=np.linalg.norm(y)
    return np.column_stack([x,y,np.cross(x,y)])


def numbers(x):return ' '.join(map(str,np.asarray(x).ravel()))


def rope_model(length,mass,radius,stiffness,damping,n=25):
    root=ET.Element('mujoco',model='identified_rope')
    ET.SubElement(root,'option',timestep='.002',gravity='0 0 -9.81',integrator='implicitfast',iterations='50')
    world=ET.SubElement(root,'worldbody');parent=world;seg=length/(n-1)
    for i in range(n):
        body=ET.SubElement(parent,'body',name=f'dlo_{i:02d}',pos=numbers([0,0,0] if i==0 else [seg,0,0]))
        if i==0:ET.SubElement(body,'freejoint',name='root')
        else:ET.SubElement(body,'joint',name=f'joint_{i}',type='ball',stiffness=str(stiffness),damping=str(damping),armature='0')
        ET.SubElement(body,'geom',type='capsule',fromto=numbers([0,0,0,seg if i<n-1 else seg*.2,0,0]),size=str(radius),mass=str(mass/n),contype='0',conaffinity='0')
        parent=body
    for end in ['left','right']:ET.SubElement(world,'body',name=end,mocap='true')
    eq=ET.SubElement(root,'equality')
    for end,i in [('left',0),('right',n-1)]:
        ET.SubElement(eq,'weld',body1=end,body2=f'dlo_{i:02d}',relpose='0 0 0 1 0 0 0',solref='.006 1',solimp='.98 .995 .001 .5 2')
    return mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))


def initialize(model,curve):
    d=mujoco.MjData(model);n=len(curve)
    tangent=np.diff(curve,axis=0);tangent/=np.linalg.norm(tangent,axis=1)[:,None]
    rot=[frame(tangent[0])]
    for direction in tangent[1:]:
        a=rot[-1][:,0];v=np.cross(a,direction);sn=np.linalg.norm(v);cs=np.clip(a@direction,-1,1)
        if sn<1e-10:
            if cs<0:raise ValueError('Reversed adjacent tangent')
            R=np.eye(3)
        else:
            u=v/sn;K=np.array([[0,-u[2],u[1]],[u[2],0,-u[0]],[-u[1],u[0],0]])
            R=np.eye(3)+sn*K+(1-cs)*K@K
        rot.append(R@rot[-1])
    rot.append(rot[-1])
    d.qpos[:3]=curve[0];d.qpos[3:7]=quat(rot[0])
    for i in range(1,n):d.qpos[model.jnt_qposadr[i]:model.jnt_qposadr[i]+4]=quat(rot[i-1].T@rot[i])
    d.mocap_pos[:]=curve[[0,-1]];d.mocap_quat[:]=[quat(rot[0]),quat(rot[-1])]
    mujoco.mj_forward(model,d)
    return d


def simulate_static(curve,length,mass,radius,k,damping=.003,seconds=30.):
    m=rope_model(length,mass,radius,k,damping,len(curve));d=initialize(m,curve)
    for _ in range(round(seconds/m.opt.timestep)):mujoco.mj_step(m,d)
    mujoco.mj_forward(m,d);ids=[m.body(f'dlo_{i:02d}').id for i in range(len(curve))]
    x=d.xpos[ids].copy()
    return x,dict(rmse_m=float(np.sqrt(np.mean(np.sum((x-curve)**2,axis=1)))),max_speed_m_s=float(np.linalg.norm(d.cvel[ids,3:],axis=1).max()),
        endpoint_error_m=float(np.linalg.norm(x[[0,-1]]-curve[[0,-1]],axis=1).max()))


def fit_decay(records):
    good=[r for r in records if r['valid'] and r['t']>=0]
    if len(good)<40:return dict(accepted=False,reason='insufficient_visible_frames',n=len(good))
    t=np.array([r['t'] for r in good]);t-=t[0];Y=np.array([r['centroid'] for r in good]);Y-=Y.mean(0)
    _,_,v=np.linalg.svd(Y,full_matrices=False);y=Y@v[0]
    noise=max(.05,float(np.median(abs(np.diff(y,2)))/1.652))
    def pred(p):
        b,a,c,decay,f=p
        return b+np.exp(-decay*t)*(a*np.cos(2*np.pi*f*t)+c*np.sin(2*np.pi*f*t))
    best=None
    for f in [.5,1,2,4,7]:
        p=least_squares(lambda p:pred(p)-y,[0,float(np.std(y)),0,1,f],bounds=([-10,-20,-20,.01,.15],[10,20,20,20,10]))
        if best is None or np.sum(p.fun**2)<np.sum(best.fun**2):best=p
    p=best.x;r2=1-np.sum(best.fun**2)/max(np.sum(y*y),1e-12)
    amp=float(np.hypot(p[1],p[2]));snr=amp/noise
    accepted=bool(r2>.5 and snr>4 and .03<p[3]<15 and p[4]*t[-1]>=2)
    return dict(accepted=accepted,reason='ok' if accepted else 'no_resolved_decaying_mode',n=len(t),r2=float(r2),
        amplitude_px=amp,noise_px=noise,snr=snr,decay_per_s=float(p[3]),frequency_hz=float(p[4]),
        damping_ratio=float(p[3]/np.hypot(p[3],2*np.pi*p[4])))
