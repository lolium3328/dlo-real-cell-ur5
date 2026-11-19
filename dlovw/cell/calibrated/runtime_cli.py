"""JSONL real-input command adapter; no robot IO or simulator state inputs."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from .deployment import Controller


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--policy', type=Path, required=True)
    ap.add_argument('--goal', type=Path, required=True)
    ap.add_argument('--input', type=Path)
    ap.add_argument('--commit', default='', help='planner commitment abs,rel,risk (v39 demo: 3,0.1,0.2)')
    a = ap.parse_args()
    goal = json.loads(a.goal.read_text())
    controller = Controller(a.policy, goal['centerline'], goal['target_left'],
                            commit=tuple(map(float, a.commit.split(','))) if a.commit else None)
    stream = a.input.open() if a.input else sys.stdin
    for line in stream:
        try:
            record = json.loads(line)
            images = {}
            for role, path in record.get('images', {}).items():
                frame = cv2.imread(path)
                if frame is None:
                    continue
                images[role] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = controller.step(record['t'], record['left'], record['right_tcp'], record['right_rotation'],
                                     images, record.get('image_times'), right_frame=record.get('right_frame', 'left_base'),
                                     right_timestamp=record.get('right_timestamp'), left_timestamp=record.get('left_timestamp'))
            output = {k:v for k,v in result.items() if k != 'quality'}
            print(json.dumps(output, default=lambda v:np.asarray(v).tolist()), flush=True)
        except (ValueError, KeyError, RuntimeError, TypeError, OverflowError) as exc:
            print(json.dumps(dict(command_xyz=[0., 0., 0.], status='input_fault', stop_required=True, error=str(exc))), flush=True)
