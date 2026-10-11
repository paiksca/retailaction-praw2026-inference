"""Embeds crops around both hands in every sampled frame with a frozen DINOv2 ViT-S/14.

usage: python handcrops.py --comp <competition dir> --meta <packed meta dir> --out <dir> [--splits ...]
Writes <out>/<split>.npy (N, F=32, V=2, H=2, 384) float16 and <out>/<split>_dirs.txt.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

cv2.setNumThreads(1)
sys.path.insert(0, os.path.dirname(__file__))
from data import JI, load_split  # noqa: E402

CROP = 0.2  # crop side as a fraction of the frame
RES = 112
HAND_J = [(JI["right_hand"], JI["right_wrist"]), (JI["left_hand"], JI["left_wrist"])]
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)


def hand_xy(pose, f, h):
    """pose (F, J, 3) stored as (row, col). Returns (x, y) of hand h at frame f, or None."""
    if f >= len(pose):
        return None
    for j in HAND_J[h]:
        r, c = pose[f, j, 0], pose[f, j, 1]
        if np.isfinite(r) and np.isfinite(c):
            return float(c), float(r)
    return None


def read_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    return frames


def crops_for(rec, comp_root):
    """Returns (32, 2, 2, RES, RES, 3) uint8 crops and a (32, 2, 2) validity mask."""
    out = np.zeros((32, 2, 2, RES, RES, 3), np.uint8)
    ok = np.zeros((32, 2, 2), bool)
    d = rec["dir"]
    if comp_root and "/kaggle/input/" in d and not os.path.exists(d):
        d = os.path.join(comp_root, d.split("retail-action-challenge-praw-workshop-wacv-2027/")[1])
    for v, r in enumerate(("rank0", "rank1")):
        frames = read_frames(os.path.join(d, f"{r}_video.mp4"))
        pose = rec[f"{r}_pose"]
        for f, img in enumerate(frames[:32]):
            H, W = img.shape[:2]
            for h in range(2):
                p = hand_xy(pose, f, h)
                if p is None:
                    continue
                s = int(round(CROP * max(H, W)))
                cx, cy = int(round(p[0] * W)), int(round(p[1] * H))
                x0, y0 = cx - s // 2, cy - s // 2
                pad = cv2.copyMakeBorder(img, s, s, s, s, cv2.BORDER_CONSTANT, value=0)
                crop = pad[y0 + s:y0 + 2 * s, x0 + s:x0 + 2 * s]
                out[f, v, h] = cv2.resize(crop, (RES, RES), interpolation=cv2.INTER_AREA)[:, :, ::-1]
                ok[f, v, h] = True
    return out, ok


def load_dino(weights=None):
    m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=weights is None)
    if weights:
        m.load_state_dict(torch.load(weights, map_location="cpu"))
    return m.eval()


@torch.no_grad()
def embed(model, crops, dev, half):
    x = torch.from_numpy(crops).to(dev).float().div(255)
    x = (x - torch.tensor(MEAN, device=dev)) / torch.tensor(STD, device=dev)
    x = x.permute(0, 3, 1, 2)
    if half:
        x = x.half()
    return model(x).float().cpu().numpy().astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", default="")
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="challenge_test_set,validation,test,train")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk", default="", help="i/n: process the i-th of n contiguous chunks of each split")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    half = dev == "cuda"
    model = load_dino(args.weights).to(dev)
    if half:
        model = model.half()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    for split in args.splits.split(","):
        recs = load_split(split, args.meta)
        if args.limit:
            recs = recs[: args.limit]
        tag = split
        if args.chunk:
            ci, cn = map(int, args.chunk.split("/"))
            step = (len(recs) + cn - 1) // cn
            recs = recs[ci * step:(ci + 1) * step]
            tag = f"{split}.part{ci}"
        E = np.zeros((len(recs), 32, 2, 2, 384), np.float16)
        M = np.zeros((len(recs), 32, 2, 2), bool)
        with ThreadPoolExecutor(args.threads) as ex:
            for i, (cr, ok) in enumerate(ex.map(lambda r: crops_for(r, args.comp), recs)):
                idx = np.nonzero(ok.reshape(-1))[0]
                if len(idx):
                    e = embed(model, cr.reshape(-1, RES, RES, 3)[idx], dev, half)
                    E[i].reshape(-1, 384)[idx] = e
                M[i] = ok
                if i % 1000 == 0:
                    print(split, i, len(recs), round(time.time() - t0), flush=True)
        np.save(os.path.join(args.out, f"{tag}.npy"), E)
        np.save(os.path.join(args.out, f"{tag}_mask.npy"), M)
        open(os.path.join(args.out, f"{tag}_dirs.txt"), "w").write("\n".join(r["dir"] for r in recs))
        print(split, "done", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
