"""Front / top camera videos of a scene run, for review and demos.

A background thread takes the newest frame of each view fps times per second and writes it to <scene>_front.avi /
<scene>_top.avi (MJPG; .mp4 / mp4v if MJPG is unavailable), played back in real time. <scene>_video.json lists each
written frame's camera time in wall seconds since the scene start (the clock of the *_s fields in the other logs),
so a frame can be matched to the control log. Recording never stops the scene: an error ends the recording and is
written to the JSON.

Cost (2026-11-19, 640x480): MJPG 1.7 ms CPU per frame on one thread, mp4v 3.4 ms spread over all cores. At the
default 10 fps (enough at --time-scale 6: motion per frame as at 60 fps unscaled) both views take ~3 % of one core;
OpenCV releases the GIL while encoding, so tracker and planner threads are not blocked.
"""
import json
import threading
import time
import traceback

ROLES = ('front', 'top')


class VideoRecorder:
    def __init__(self, cams, out_dir, sid, T0, fps=10.):
        self.cams, self.out_dir, self.sid, self.T0, self.fps = cams, out_dir, sid, T0, fps
        self.writers, self.paths, self.times = {}, {}, {r: [] for r in ROLES}
        self.error, self.stopped, self.thread = None, False, None

    def start(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def _writer(self, role, shape):
        import cv2
        h, w = shape[:2]
        for ext, code in (('avi', 'MJPG'), ('mp4', 'mp4v')):
            path = self.out_dir/f'{self.sid}_{role}.{ext}'
            wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*code), self.fps, (w, h))
            if wr.isOpened():
                self.paths[role] = path
                return wr
            wr.release()
        raise RuntimeError(f'no video codec available for {role}')

    def _loop(self):
        import cv2
        seen, nxt = {}, time.perf_counter()
        try:
            while not self.stopped:
                nxt += 1./self.fps
                time.sleep(max(nxt-time.perf_counter(), 0.))
                frames = self.cams.latest()
                for role in ROLES:
                    if role not in frames:
                        continue
                    im, stamp = frames[role]
                    if stamp <= seen.get(role, -float('inf')):
                        continue
                    seen[role] = stamp
                    if role not in self.writers:
                        self.writers[role] = self._writer(role, im.shape)
                    self.writers[role].write(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
                    self.times[role].append(round(stamp-self.T0, 4))
        except Exception as e:                       # noqa: BLE001 - recording is optional, the scene goes on
            traceback.print_exc()
            self.error = repr(e)

    def stop(self):
        """Finish the files and write <scene>_video.json; returns the summary."""
        self.stopped = True
        if self.thread is not None:
            self.thread.join(2.)
        for wr in self.writers.values():
            wr.release()
        summary = dict(fps=self.fps, error=self.error,
                       files={r: p.name for r, p in self.paths.items()},
                       frames={r: len(t) for r, t in self.times.items()},
                       frame_s=self.times)
        try:
            (self.out_dir/f'{self.sid}_video.json').write_text(json.dumps(summary))
        except OSError as e:
            print(f'  video log error: {e}', flush=True)
        return summary
