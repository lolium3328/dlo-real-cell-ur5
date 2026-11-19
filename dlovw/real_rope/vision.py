"""Recorded-cell bright-yellow segmentation and gated, model-free stereo curves."""
import json
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from ..imagination import rgb_usb as usb, camera_model as cm

HSV_LO=(17,90,170)
HSV_HI=(43,255,255)


def cameras(path):
    raw=json.load(open(path))['cameras'];out={}
    for role,c in raw.items():
        K=np.array(c['K']);out[role]=dict(T=np.array(c['T_world_from_camera_optical']),K=(K[0,0],K[1,1],K[0,2],K[1,2]))
    return out


def segment(rgb):
    h=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    m=cv2.inRange(h,np.array(HSV_LO,dtype=np.uint8),np.array(HSV_HI,dtype=np.uint8))
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    n,lab,st,_=cv2.connectedComponentsWithStats(m)
    good=np.flatnonzero(st[1:,4]>=100)+1
    mask=np.isin(lab,good)
    largest=int(st[good,4].max()) if len(good) else 0
    info=dict(area=int(mask.sum()),components=len(good),largest_share=largest/max(int(mask.sum()),1),
        border=bool(mask[:2].any() or mask[-2:].any() or mask[:,:2].any() or mask[:,-2:].any()))
    return mask,info


def path(mask):
    yy,xx=np.where(mask)
    if len(xx)<100:raise ValueError('too_few_pixels')
    crop=mask[yy.min():yy.max()+1,xx.min():xx.max()+1]
    sk=usb.thin(np.pad(crop,2).astype(np.uint8)*255)
    xy=usb.longest_path(sk)+[xx.min()-2,yy.min()-2]
    xy=gaussian_filter1d(xy,2,axis=0)
    return xy


def resample3(x,n=25):
    t=np.r_[0,np.cumsum(np.linalg.norm(np.diff(x,axis=0),axis=1))]
    return np.array([np.interp(np.linspace(0,t[-1],n),t,x[:,j]) for j in range(3)]).T


def reconstruct(images,cams,tcp):
    paths={};info={}
    for role,img in images.items():
        m,inf=segment(img);info[role]=inf
        if inf['area']<200 or inf['largest_share']<.95:raise ValueError(role+':fragmented')
        if inf['border']:raise ValueError(role+':out_of_frame')
        paths[role]=path(m)
    f=usb.resample(paths['front'],39)
    # In this cell the moving end is left in front and right in top.
    if f[0,0]>f[-1,0]:f=f[::-1]
    top=usb.resample(paths['top'],400)
    if top[0,0]<top[-1,0]:top=top[::-1]
    prm=dict(usb.DEFAULTS,match_prior_weight=30.)
    matched,cost=usb.epipolar_match(cams,f,top,prm)
    x,gap=usb.triangulate(cams,f,matched)
    if np.percentile(gap,95)>.012:raise ValueError('stereo_ray_gap')
    if np.linalg.norm(x[0]-tcp)>.035:raise ValueError('moving_endpoint_mismatch')
    for role in cams:
        if (cm.project(x,cams[role])[1]<=0).any():raise ValueError('negative_depth')
    x=gaussian_filter1d(x,.8,axis=0)
    length=float(np.linalg.norm(np.diff(x,axis=0),axis=1).sum())
    if not .25<length<1.2:raise ValueError('implausible_length')
    return resample3(x),dict(length_m=length,ray_gap_p95_m=float(np.percentile(gap,95)),views=info)


def image(s,role,index):
    p=s.path/role/s.frames[role][index]['rgb_path'];im=cv2.imread(str(p))
    if im is None:raise ValueError('unreadable_image')
    return cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
