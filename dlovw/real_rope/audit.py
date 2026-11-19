"""Fast data-quality checks; missing depth is a limitation, not an RGB rejection."""
import cv2
import numpy as np
from ..imagination.rgb_rope import masks
from .session import sha256
from .vision import segment


def audit(s, stride=120):
    issues=[]; views={}; hashes={}
    def flag(code,severity,detail):issues.append(dict(code=code,severity=severity,detail=detail))
    for name in ('session.json','left_tcp.csv','events.jsonl','front/camera.json','top/camera.json','front/frames.csv','top/frames.csv'):
        p=s.path/name
        if p.exists():hashes[name]=sha256(p)
    n=sum(map(len,s.frames.values()))
    if not n:flag('no_recording','exclude','No camera frames; home-only is not a dataset')
    if s.meta.get('status')!='complete':flag('failed_session','exclude',s.meta.get('status'))
    camera_intervals=[]
    for role,rows in s.frames.items():
        if not rows:
            flag('missing_camera','exclude',role);continue
        t=s.ft[role]; d=np.diff(t)
        camera_intervals.append([t[0],t[-1]])
        missing=sum(not (s.path/role/r['rgb_path']).is_file() for r in rows)
        gaps=sum(max(0,int(r['frame_number_gap'])) for r in rows)
        if missing:flag('missing_png','exclude',dict(role=role,count=missing))
        if gaps:flag('camera_frame_drop','warn',dict(role=role,count=gaps))
        if (d<=0).any():flag('camera_time_nonmonotonic','exclude',role)
        stats=[]
        for r in rows[::stride]:
            im=cv2.imread(str(s.path/role/r['rgb_path']))
            if im is None:continue
            rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
            raw_mask,_,_,_=masks(rgb)
            mask,_=segment(rgb)
            cc,lab,st,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8))
            area=int(mask.sum()); largest=int(st[1:,4].max()) if cc>1 else 0
            border=bool(mask[:2].any() or mask[-2:].any() or mask[:,:2].any() or mask[:,-2:].any())
            stats.append(dict(index=int(r['index']),legacy_area=int(raw_mask.sum()),area=area,largest_share=largest/max(area,1),
                border=border,components=cc-1,brightness=float(im.mean()),saturated_share=float((im.max(2)>=250).mean())))
        if any(r['border'] for r in stats):flag('rope_out_of_frame','warn',dict(role=role,sampled=sum(r['border'] for r in stats)))
        if any(r['largest_share']<.9 for r in stats):flag('fragmented_rope','warn',role)
        views[role]=dict(frames=len(rows),fps=float(1/np.median(d)) if len(d) else None,
            frame_drops=gaps,missing_png=missing,duration_s=float(t[-1]-t[0]),samples=stats,
            host_camera_delay_ms_median=float(np.median([int(r['host_unix_ns'])*1e-6-float(r['camera_timestamp_ms']) for r in rows])))
    tcp={}
    if len(s.t)>1:
        dt=np.diff(s.t); rt=np.array([float(r['robot_timestamp_s']) for r in s.tcp]); dr=np.diff(rt)
        overlap=[(float(s.t[i]),float(s.t[i+1])) for i in np.where(dt>.1)[0] if any(s.t[i]<b and s.t[i+1]>a for a,b in camera_intervals)]
        if overlap:flag('tcp_gap_during_camera','warn',overlap)
        if (dt<=0).any():flag('tcp_monotonic_clock_invalid','exclude','nonincreasing')
        wall=np.array([int(r['host_unix_ns']) for r in s.tcp],dtype=np.int64)
        tcp=dict(samples=len(s.raw_tcp),unique_robot_samples=len(s.tcp),effective_hz=float(1/np.median(dr[dr>0])),
            duplicate_robot_share=1-len(s.tcp)/len(s.raw_tcp),duplicate_wall_timestamps=int((np.diff(wall)==0).sum()),
            max_gap_s=float(dt.max()),overlap_gaps=overlap)
        if tcp['effective_hz']<100:flag('tcp_rate_below_125','info',tcp['effective_hz'])
    elif n:flag('no_tcp','exclude','No usable TCP stream')
    if n and not any(e['event']=='recording_end' for e in s.events):flag('missing_recording_end','warn','No end event')
    if n:flag('exposure_settings_unverified','info','Exposure/white-balance lock not established from recorder metadata')
    if not s.meta.get('cameras',{}).get('camera_extrinsics_calibrated',False):flag('unvalidated_extrinsics','warn','Use explicit external candidate, not session extrinsics')
    if n and not any(r.get('depth_path') for rows in s.frames.values() for r in rows):flag('rgb_only','info','No depth; use stereo with quality gates')
    flag('material_metadata_unverified','info','Length, mass, diameter and boundary conditions require measurement or explicit assumptions')
    return dict(session=s.path.name,mode=s.meta.get('mode'),status=s.meta.get('status'),
        usable_recording=not any(i['severity']=='exclude' for i in issues),tcp=tcp,views=views,holds=s.holds(),issues=issues,input_sha256=hashes)
