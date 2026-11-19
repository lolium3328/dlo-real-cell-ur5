"""Headless observations (v37): the rope centreline as the RGB tracker would report it, without rendering.

Policy training does not need real time or images: the observation is the true centreline plus a synthetic
tracking error fitted to the v1 RGB demonstrations (400 episodes, 27133 decisions; tracked - true):
  - per-node standard deviation SD_MM (mean of the three axes; 0.2 mm at the kinematic left end, 2-4 mm inside),
  - temporal AR(1), correlation 0.937 per 80 ms decision,
  - smooth along the rope (measured node lag-2 / lag-6 correlation 0.91 / 0.64): noise drawn on 10 knots and
    interpolated,
  - the 48 ms image latency under deployment noise.
Not reproduced: the heavy tail under ball occlusion (measured p99 19 mm, Gaussian here) and occlusion pauses.
Quality features are a constant "supported by both views, fresh" in training and evaluation alike; the RGB
segmentation / tracking is validated separately against this error model.
"""
import numpy as np

SD_MM = [0.23, 1.48, 2.38, 2.82, 3.05, 3.2, 3.32, 3.42, 3.55, 3.64, 3.81, 3.85, 3.9, 4.0, 3.95, 3.66, 3.34, 3.09, 2.93,
         2.8, 3.04, 3.56, 3.51, 3.45, 3.39, 3.29, 3.11, 2.94, 2.79, 2.68, 2.57, 2.45, 2.3, 2.15, 2.0, 1.83, 1.62, 1.25,
         1.14]
MEDIAN_MATCH = .61     # SD_MM is inflated by the heavy tail; .61 makes the 3-D node error median 2.6 mm as measured
RHO_DECISION = .937
KNOTS = 10
QUALITY = np.tile(np.array([1., 1., 0., .002], np.float32), (39, 1))    # both views, age 0 s, 2 mm radius


class HeadlessTracking:
    def __init__(self, config, seed, deploy):
        self.rng = np.random.default_rng(seed)
        self.rho = RHO_DECISION**(1/config.decision_steps)                # per 8 ms control step
        s = np.linspace(0, KNOTS-1, 39)
        i = np.minimum(s.astype(int), KNOTS-2)
        w = s-i
        self.basis = np.zeros((39, KNOTS))
        self.basis[np.arange(39), i], self.basis[np.arange(39), i+1] = 1-w, w
        self.basis /= np.linalg.norm(self.basis, axis=1, keepdims=True)    # unit variance per node
        self.sd = MEDIAN_MATCH*np.asarray(SD_MM)[:, None]/1000.
        self.state = self.rng.standard_normal((KNOTS, 3))
        self.lag = round(config.image_latency_s/config.dt) if deploy else 0
        self.history = []

    def update(self, curve):
        self.state = self.rho*self.state+np.sqrt(1-self.rho**2)*self.rng.standard_normal(self.state.shape)
        self.history.append(np.asarray(curve, float).copy())
        if len(self.history) > self.lag+1:
            self.history.pop(0)
        seen = self.history[0]+self.sd*(self.basis@self.state)
        return dict(centerline=seen, quality_features=QUALITY.copy(), pause_requested=False, headless=True)
