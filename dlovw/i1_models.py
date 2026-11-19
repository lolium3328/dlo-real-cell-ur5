"""I1 learned perception (docs/notes/i1_prereg_20261025.md, section 4).

Shared by training (`scripts/experiments/train_i1.py`) and closed-loop inference (`dlovw/image_session.py`):
the input/target encodings and three models.
- `LRSSM`: release VisualEncoder + RSSMCore (same shapes, release-initialised), a new 12 -> 128 condition
  encoder (gripper positions and inter-frame gripper motion), heads on (h, z), front RGB-D reconstruction.
- `LCNN`: the same encoder and heads per frame, no recurrence (obstacle velocity comes from the hand Kalman
  filter at inference).
- `LGRUHand`: a GRU over the hand pipeline's per-frame outputs.
Heads give the rope as a residual on the straight line between the two rope ends (grippers plus the offsets
read from the frame-0 calibration), the obstacle centre and velocity in world coordinates, and a
visibility logit.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from . import legacy  # noqa: F401  (sets sys.path)
from wm.decoder import ImageDecoder  # noqa: E402
from wm.encoder import VisualEncoder  # noqa: E402
from wm.rssm import RSSMCore  # noqa: E402

N_NODES = 39
DEPTH_LO, DEPTH_HI = 0.4, 1.6
WS_CENTRE = np.array([0.68, 0.0, 0.25])
POS_SCALE, ROPE_SCALE, VEL_SCALE, ACT_SCALE = 0.3, 0.1, 0.5, 1 / 0.007
HAND_DIM = 3 + 1 + 6 + 3 * N_NODES + 6 + 6
OUT_DIM = 3 * N_NODES + 3 + 3 + 1
RELEASE_CKPT = "pretrained/sim_532mm_rssm.pt"


# ---- encodings (numpy, per frame) ------------------------------------------------------------------
def image_array(rgb, depth):
    """(4, 64, 64) float32: RGB in [0, 1], depth mapped to [0, 1], invalid depth -> 0."""
    rgb = np.asarray(rgb, np.float32)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    d = np.asarray(depth, np.float32)
    d = np.where(np.isfinite(d), np.clip((d - DEPTH_LO) / (DEPTH_HI - DEPTH_LO), 0.0, 1.0), 0.0)
    return np.concatenate([np.transpose(rgb, (2, 0, 1)), d[None]], axis=0).astype(np.float32)


def cond_array(grip, action):
    grip = np.asarray(grip, float).reshape(6)
    return np.r_[(grip - np.r_[WS_CENTRE, WS_CENTRE]) / POS_SCALE, np.asarray(action, float).reshape(6) * ACT_SCALE].astype(np.float32)


def ends_line(grip, offsets, n=N_NODES):
    grip = np.asarray(grip, float).reshape(2, 3)
    a, b = grip[0] + offsets[0], grip[1] + offsets[1]
    s = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - s) * a[None] + s * b[None]


def hand_array(meas, seen, kf, rope, grip, action, offsets):
    line = ends_line(grip, offsets)
    meas = np.where(np.isfinite(meas), (np.asarray(meas, float) - WS_CENTRE) / POS_SCALE, 0.0)
    kf = np.asarray(kf, float)
    kf = np.r_[(kf[:3] - WS_CENTRE) / POS_SCALE, kf[3:] / VEL_SCALE] if np.all(np.isfinite(kf)) else np.zeros(6)
    return np.r_[meas, float(seen), kf, ((np.asarray(rope, float) - line) / ROPE_SCALE).reshape(-1),
                 cond_array(grip, action)].astype(np.float32)


def target_array(rope, centre, vel, visible, grip, offsets):
    line = ends_line(grip, offsets)
    return np.r_[((np.asarray(rope, float) - line) / ROPE_SCALE).reshape(-1), (np.asarray(centre, float) - WS_CENTRE) / POS_SCALE,
                 np.asarray(vel, float) / VEL_SCALE, float(visible)].astype(np.float32)


def decode(out, grip, offsets):
    """Network output (OUT_DIM,) -> rope (39, 3), centre (3,), velocity (3,), visibility probability."""
    out = np.asarray(out, float)
    rope = ends_line(grip, offsets) + ROPE_SCALE * out[:3 * N_NODES].reshape(N_NODES, 3)
    c = WS_CENTRE + POS_SCALE * out[3 * N_NODES:3 * N_NODES + 3]
    v = VEL_SCALE * out[3 * N_NODES + 3:3 * N_NODES + 6]
    return rope, c, v, float(1 / (1 + np.exp(-out[-1])))


# ---- models ----------------------------------------------------------------------------------------
def mlp(i, o, hidden=256):
    return nn.Sequential(nn.Linear(i, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, o))


class LRSSM(nn.Module):
    kind = "rssm"

    def __init__(self):
        super().__init__()
        self.encoder = VisualEncoder(4, 256, (4, 4))
        self.cond = nn.Sequential(nn.Linear(12, 128), nn.SiLU(), nn.Linear(128, 128))
        self.rssm = RSSMCore(64, 256, 128, 256, 256)
        self.heads = mlp(320, OUT_DIM)
        self.decoder = ImageDecoder(320, 4, (64, 64))

    def init_state(self, b, device):
        return self.rssm.init_state(b, device)

    def step(self, state, img, cond, sample=True):
        """One filtering step. Returns new state, prior (mean, std), posterior (mean, std), feature."""
        h, z = state
        h, pm, ps = self.rssm.transition(h, z, self.cond(cond))
        z, qm, qs = self.rssm.posterior(h, self.encoder(img), sample=sample)
        return (h, z), (pm, ps), (qm, qs), torch.cat([h, z], -1)

    def imagine(self, state, cond):
        h, z = state
        h, pm, _ = self.rssm.transition(h, z, self.cond(cond))
        return (h, pm), torch.cat([h, pm], -1)


class LCNN(nn.Module):
    kind = "cnn"

    def __init__(self):
        super().__init__()
        self.encoder = VisualEncoder(4, 256, (4, 4))
        self.heads = mlp(256 + 12, OUT_DIM)

    def forward(self, img, cond):
        return self.heads(torch.cat([self.encoder(img), cond], -1))


class LGRUHand(nn.Module):
    kind = "gruhand"

    def __init__(self):
        super().__init__()
        self.gru = nn.GRUCell(HAND_DIM, 256)
        self.heads = mlp(256 + HAND_DIM, OUT_DIM)

    def step(self, h, x):
        h = self.gru(x, h)
        return h, self.heads(torch.cat([h, x], -1))


MODELS = {"rssm": LRSSM, "cnn": LCNN, "gruhand": LGRUHand}


def release_init(model, path):
    """Copy every release tensor whose name and shape match (encoder.*, rssm.*)."""
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and not any(hasattr(v, "shape") for v in sd.values()):
        sd = sd.get("model", sd.get("state_dict", sd))
    own = model.state_dict()
    copied = {k: v for k, v in sd.items() if k in own and hasattr(v, "shape") and tuple(v.shape) == tuple(own[k].shape)}
    own.update(copied)
    model.load_state_dict(own)
    return sorted(copied)


# ---- closed-loop wrapper ---------------------------------------------------------------------------
class Estimator:
    """Per-episode inference for one trained model (one call per camera frame)."""

    def __init__(self, ckpt, device="cuda"):
        blob = torch.load(ckpt, map_location="cpu")
        self.kind = blob["kind"]
        self.model = MODELS[self.kind]()
        self.model.load_state_dict(blob["state_dict"])
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.dc = float(blob["metrics"]["dc"])

    def reset(self, offsets):
        self.offsets, self.state, self.h = offsets, None, None

    @torch.no_grad()
    def step(self, rgb, depth, grip, action, hand=None):
        dev = self.device
        cond = torch.as_tensor(cond_array(grip, action))[None].to(dev)
        if self.kind == "gruhand":
            x = torch.as_tensor(hand_array(*hand, grip, action, self.offsets))[None].to(dev)
            self.h = torch.zeros(1, 256, device=dev) if self.h is None else self.h
            self.h, out = self.model.step(self.h, x)
        else:
            img = torch.as_tensor(image_array(rgb, depth))[None].to(dev)
            if self.kind == "cnn":
                out = self.model(img, cond)
            else:
                self.state = self.model.init_state(1, dev) if self.state is None else self.state
                self.state, _, _, feat = self.model.step(self.state, img, cond, sample=False)
                out = self.model.heads(feat)
        return decode(out[0].cpu().numpy(), grip, self.offsets)
