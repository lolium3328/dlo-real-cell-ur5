"""Recorder session IO. Interpolate on monotonic time, not quantized wall time."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def csv_rows(path):
    if not Path(path).exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Session:
    def __init__(self, path):
        self.path = Path(path)
        self.meta = json.loads((self.path/'session.json').read_text())
        self.sid = self.path.name.split('_')[1]
        self.anchor = self.meta['host_clock_anchor']
        self.raw_tcp = csv_rows(self.path/'left_tcp.csv')
        self.tcp = [r for r in self.raw_tcp if r['same_robot_timestamp']=='0']
        self.t = np.array([self.mono(r['host_monotonic_ns']) for r in self.tcp])
        self.xyz = np.array([[float(r[k+'_m']) for k in 'xyz'] for r in self.tcp]).reshape(-1,3)
        self.velocity = np.array([[float(r['v'+k+'_m_s']) for k in 'xyz'] for r in self.tcp]).reshape(-1,3)
        self.frames = {role:csv_rows(self.path/role/'frames.csv') for role in ('front','top')}
        self.ft = {role:np.array([float(r['camera_timestamp_ms'])*.001-self.anchor['unix_ns']*1e-9 for r in rows]) for role,rows in self.frames.items()}
        p = self.path/'events.jsonl'
        self.events = [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []

    def mono(self, ns):
        return (int(ns)-self.anchor['monotonic_ns'])*1e-9

    def position(self,t):
        return np.array([np.interp(t,self.t,self.xyz[:,i]) for i in range(3)]).T

    def stationary(self,t,window=.15):
        if not len(self.t) or t-window<self.t[0] or t+window>self.t[-1]:return False
        a,b=np.searchsorted(self.t,[t-window,t+window])
        return b>a and np.max(np.linalg.norm(self.velocity[a:b],axis=1))<.003

    def holds(self):
        out=[]; start=None
        for e in self.events:
            if e['event']=='hold_start':start=e
            elif e['event']=='hold_end' and start is not None:
                out.append(dict(start=self.mono(start['host_monotonic_ns']),end=self.mono(e['host_monotonic_ns']),phase=start.get('phase'),trial=start.get('trial')))
                start=None
        return out


def sessions(root):
    return [Session(p.parent) for p in sorted(Path(root).glob('*/session.json'))]
