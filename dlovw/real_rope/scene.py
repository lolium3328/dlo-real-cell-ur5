"""Explicit measured-rope variant of the new robot cell; no global layout mutation."""
import json
from pathlib import Path
import mujoco
import numpy as np
from ..cell import layout as L
from ..cell.scene import build_spec,quat,mat,OPT_FROM_MJ
from ..cell.arms import Arm
from ..cell.env import CellEnv,_rate_limit
from .physics import rope_model,initialize
from .vision import cameras
from .refinement import set_rest_fraction


def apply_spring_reference(model):
    """Restore optional ball-joint spring references carried by the XML numeric data."""
    idx=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_NUMERIC,'rope_spring_reference')
    if idx<0:return
    start=model.numeric_adr[idx];size=model.numeric_size[idx]
    values=model.numeric_data[start:start+size].reshape(-1,4)
    for i,q in enumerate(values,1):
        adr=model.joint(f'dlo_joint_{i:02d}').qposadr[0]
        model.qpos_spring[adr:adr+4]=q/np.linalg.norm(q)


def build_measured_scene(profile_path,output,*,sphere_offset=None,right_quat=None,sphere_radius=None):
    p=json.loads(Path(profile_path).read_text());x=np.array(p['initial_curve']);n=len(x)
    sp=build_spec();seg=p['length_m']/(n-1)
    sphere_offset=np.asarray(L.OBSTACLE_CENTRE_IN_TCP if sphere_offset is None else sphere_offset,float)
    right_quat=L.RIGHT_TCP_QUAT if right_quat is None else right_quat
    obstacle=sp.geom('obstacle_sphere')
    obstacle.pos=np.array([0.,0.,L.TCP_FROM_FLANGE_M])+sphere_offset
    if sphere_radius is not None:obstacle.size=[sphere_radius,0,0]
    core=rope_model(p['length_m'],p['mass_kg'],p['radius_m'],p['joint_stiffness_nm_per_rad'],p['joint_damping_nm_s_per_rad'],n)
    seed=initialize(core,x)
    rest=p.get('initial_rest_fraction',0.)
    set_rest_fraction(core,seed,rest)
    if rest:
        sp.add_numeric(name='rope_spring_reference',data=core.qpos_spring[7:].tolist())
    R0=mat(seed.mocap_quat[0]);Rf=mat(seed.mocap_quat[1])
    Rtcp=mat(p['left_tcp_quaternion_wxyz'])
    for i in range(n):
        b=sp.body(f'dlo_{i:02d}')
        if i:b.pos=[seg,0,0]
        g=sp.geom(f'dlo_geom_{i:02d}');g.type=mujoco.mjtGeom.mjGEOM_CAPSULE
        g.size=[p['radius_m'],0,0];g.fromto=[0,0,0,seg if i<n-1 else seg*.2,0,0]
        g.mass=p['mass_kg']/n;g.contype=0;g.conaffinity=0
        if i:
            j=sp.joint(f'dlo_joint_{i:02d}');j.stiffness=[p['joint_stiffness_nm_per_rad'],0,0];j.damping=[p['joint_damping_nm_s_per_rad'],0,0];j.armature=0
            j.frictionloss=p.get('joint_frictionloss_nm',0.)
    sp.body('dlo_00').pos=x[0];sp.body('dlo_00').quat=quat(R0)
    clamp=sp.body('fixed_clamp');clamp.pos=x[-1];clamp.quat=quat(Rf)
    post=sp.geom('fixture_post');post.pos=[x[-1,0],x[-1,1],x[-1,2]/2];post.size=[.012,x[-1,2]/2,0]
    mc=sp.body('left_tcp_mocap');mc.pos=x[0];mc.quat=quat(Rtcp)
    eq=next(e for e in sp.equalities if e.name=='left_tcp_rope')
    eq.data=[0,0,0,0,0,0,*quat(Rtcp.T@R0),1]
    for role,c in cameras(p['cameras']).items():
        cam=sp.camera(role+'_view');cam.pos=c['T'][:3,3];cam.quat=quat(c['T'][:3,:3]@OPT_FROM_MJ)
    model=sp.compile();data=mujoco.MjData(model)
    apply_spring_reference(model)
    root=model.joint('dlo_root_free').qposadr[0];data.qpos[root:root+7]=seed.qpos[:7]
    for i in range(1,n):
        qa=model.joint(f'dlo_joint_{i:02d}').qposadr[0];sa=core.jnt_qposadr[i]
        data.qpos[qa:qa+4]=seed.qpos[sa:sa+4]
    left,right=Arm(model,'left'),Arm(model,'right')
    ql,el,_=left.ik_global(x[0],Rtcp)
    qr,er,_=right.ik_global(L.OBSTACLE_HOME_POS-mat(right_quat)@sphere_offset,mat(right_quat))
    for _ in range(round(30/model.opt.timestep)):
        left.apply(data,ql);right.apply(data,qr);mujoco.mj_step(model,data)
    mujoco.mj_forward(model,data)
    sp.add_key(name='home',qpos=data.qpos.tolist(),mpos=data.mocap_pos.ravel().tolist(),mquat=data.mocap_quat.ravel().tolist(),ctrl=data.ctrl.tolist())
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(sp.to_xml())
    binary=sp.compile();apply_spring_reference(binary)
    mujoco.mj_saveModel(binary,str(output.with_suffix('.mjb')))
    ids=[model.body(f'dlo_{i:02d}').id for i in range(n)];actual=data.xpos[ids]
    return dict(xml=str(output),mjb=str(output.with_suffix('.mjb')),initial_rest_fraction=rest,
        left_ik_error_m=el,right_ik_error_m=er,rope_mass_kg=float(model.body_mass[ids].sum()),
        endpoint_error_m=np.linalg.norm(actual[[0,-1]]-x[[0,-1]],axis=1).tolist(),
        initial_shape_rmse_m=float(np.sqrt(np.mean(np.sum((actual-x)**2,axis=1)))),max_speed_m_s=float(np.linalg.norm(data.cvel[ids,3:],axis=1).max()))


class MeasuredCellEnv(CellEnv):
    """New cell with explicit fitted profile and its own endpoint/workspace definitions."""
    def __init__(self,profile_path,xml_path,**kwargs):
        self.profile=json.loads(Path(profile_path).read_text())
        super().__init__(xml_path=str(xml_path),**kwargs)
        apply_spring_reference(self.model)
        self.R_left=mat(self.profile['left_tcp_quaternion_wxyz'])
        self.fixed_point=np.array(self.profile['initial_curve'][-1])

    def observe(self):
        obs=super().observe();obs['end_distance']=float(np.linalg.norm(self.rope()[0]-self.fixed_point))
        return obs

    def clip_left(self,target,current):
        p=np.clip(target,self.profile['workspace_low'],self.profile['workspace_high'])
        v=p-self.fixed_point;dist=np.linalg.norm(v)
        p=self.fixed_point+v/max(dist,1e-9)*np.clip(dist,.15,.97*self.profile['length_m'])
        return _rate_limit(p,current,self.max_step_m)
