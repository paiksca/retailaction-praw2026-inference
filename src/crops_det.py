"""Hand crops computed the same way on every CPU architecture.

PyAV decodes H.264 to raw yuv420p, which the standard defines bit-exactly. The BT.601 limited-range color
conversion, zero padding and area resize then run in float64 with numpy and torch, so no SIMD path in
OpenCV or swscale can change a pixel.
"""
import os

import av
import numpy as np
import torch

from data import JI

CROP = 0.2  # crop side as a fraction of the frame
RES = 112
HAND_J = [(JI["right_hand"], JI["right_wrist"]), (JI["left_hand"], JI["left_wrist"])]
MEAN = np.array([0.485, 0.456, 0.406]); STD = np.array([0.229, 0.224, 0.225])


def yuv420_to_rgb(a, h, w):
    """a: (h*3/2, w) uint8 planar yuv420p. Returns (h, w, 3) float64 RGB in [0, 1]."""
    y = a[:h].astype(np.float64)
    u = a[h:h + h // 4].reshape(h // 2, w // 2).astype(np.float64)
    v = a[h + h // 4:].reshape(h // 2, w // 2).astype(np.float64)
    u = u.repeat(2, 0).repeat(2, 1); v = v.repeat(2, 0).repeat(2, 1)
    yy = (y - 16.0) * (255.0 / 219.0); cb = (u - 128.0) * (255.0 / 224.0); cr = (v - 128.0) * (255.0 / 224.0)
    r = yy + 1.402 * cr; g = yy - 0.344136 * cb - 0.714136 * cr; b = yy + 1.772 * cb
    return np.clip(np.stack([r, g, b], -1), 0, 255) / 255.0


def read_frames(path, n=32):
    out = []
    with av.open(path) as c:
        for f in c.decode(video=0):
            if f.format.name != "yuv420p" or f.width % 2 or f.height % 2:
                f = f.reformat(format="yuv420p")
            out.append(yuv420_to_rgb(f.to_ndarray(), f.height, f.width))
            if len(out) >= n:
                break
    return out


def hand_xy(pose, f, h):
    """pose (F, J, 3) stored as (row, col). Returns (x, y) of hand h at frame f, or None."""
    if f >= len(pose):
        return None
    for j in HAND_J[h]:
        r, c = pose[f, j, 0], pose[f, j, 1]
        if np.isfinite(r) and np.isfinite(c):
            return float(c), float(r)
    return None


def crops_for(rec, video_dir=None):
    """Returns normalized crops (32, 2, 2, 3, RES, RES) float64 and a (32, 2, 2) validity mask."""
    d = video_dir or rec["dir"]
    out = np.zeros((32, 2, 2, 3, RES, RES), np.float64)
    ok = np.zeros((32, 2, 2), bool)
    for v, r in enumerate(("rank0", "rank1")):
        path = os.path.join(d, f"{r}_video.mp4")
        if not os.path.exists(path):
            continue
        pose = rec[f"{r}_pose"]
        for f, img in enumerate(read_frames(path)):
            H, W = img.shape[:2]
            s = int(round(CROP * max(H, W)))
            pad = np.zeros((H + 2 * s, W + 2 * s, 3))
            pad[s:s + H, s:s + W] = img
            for h in range(2):
                p = hand_xy(pose, f, h)
                if p is None:
                    continue
                cx, cy = int(round(p[0] * W)), int(round(p[1] * H))
                x0, y0 = cx - s // 2 + s, cy - s // 2 + s
                crop = torch.from_numpy(np.ascontiguousarray(pad[y0:y0 + s, x0:x0 + s].transpose(2, 0, 1)))
                crop = torch.nn.functional.adaptive_avg_pool2d(crop[None], RES)[0].numpy()
                out[f, v, h] = (crop - MEAN[:, None, None]) / STD[:, None, None]
                ok[f, v, h] = True
    return out, ok


def load_dino(dino_dir, weights, dtype):
    m = torch.hub.load(dino_dir, "dinov2_vits14", source="local", pretrained=False)
    m.load_state_dict(torch.load(weights, map_location="cpu"))
    return m.eval().to(dtype)


@torch.no_grad()
def embed_rec(model, rec, dtype, video_dir=None):
    """(32, 4*384) float16 embeddings and (32, 4) mask for one sample."""
    cr, ok = crops_for(rec, video_dir)
    E = np.zeros((32 * 4, 384), np.float16)
    idx = np.nonzero(ok.reshape(-1))[0]
    if len(idx):
        x = torch.from_numpy(cr.reshape(-1, 3, RES, RES)[idx]).to(dtype)
        E[idx] = model(x).double().numpy().astype(np.float16)
    return E.reshape(32, -1), ok.reshape(32, -1)
