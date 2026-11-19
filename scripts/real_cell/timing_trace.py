"""In-memory scene events on the same host clock as the controller."""
import json
import threading
import time
from itertools import chain


class TimingTrace:
    MAX_EVENTS = 65_536

    def __init__(self, origin):
        self.origin = origin
        self.anchor_unix_ns = time.time_ns()
        self.anchor_scene_s = self.now()
        self.lock = threading.Lock()
        self.events = []
        self.dropped = 0
        self.closed = False

    def now(self):
        return time.perf_counter()-self.origin

    def record(self, event, t_s=None, **fields):
        row = dict(event=event, t_s=self.now() if t_s is None else t_s, **fields)
        with self.lock:
            if self.closed:
                return None
            if len(self.events) >= self.MAX_EVENTS and event != 'scene_terminal':
                self.dropped += 1
                return None
            row['event_id'] = event_id = len(self.events)+1
            self.events.append(row)
            return event_id

    def save(self, path):
        with self.lock:
            self.closed = True
            header = dict(event='trace_header', version=1, clock='host_perf_counter_relative_s',
                          anchor_unix_ns=self.anchor_unix_ns, anchor_scene_s=self.anchor_scene_s,
                          events=len(self.events), dropped_events=self.dropped)
        with path.open('w', encoding='utf-8') as f:
            for row in chain((header,), self.events):
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
