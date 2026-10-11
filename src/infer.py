"""Inference: sample directories -> submission CSV.

usage: python infer.py --data_dir <dir holding one folder per sample> --models a.pt [b.pt ...] --out submission.csv
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from features import sample_arrays  # noqa: E402
from metric import write_submission  # noqa: E402
from model import PoseNet, decode  # noqa: E402

KEYS = ("X", "valid", "tn", "hands", "hmask")
EKEYS = ("E", "EM")


def embed_samples(recs, dino_dir, dino_weights, threads):
    """Hand-crop DINOv2 embeddings per sample, (32, 4*384) float16 plus a (32, 4) mask, as in training.
    Runs in float64 so results do not depend on the CPU architecture or thread count."""
    from handcrops import MEAN, RES, STD, crops_for
    torch.set_num_threads(threads)
    m = torch.hub.load(dino_dir, "dinov2_vits14", source="local", pretrained=False)
    m.load_state_dict(torch.load(dino_weights, map_location="cpu")); m.eval().double()
    mean = torch.tensor(MEAN, dtype=torch.float64); std = torch.tensor(STD, dtype=torch.float64)
    Es, Ms = [], []
    for n, rec in enumerate(recs):
        cr, ok = crops_for(rec, "")
        E = np.zeros((32 * 4, 384), np.float16)
        idx = np.nonzero(ok.reshape(-1))[0]
        if len(idx):
            x = torch.from_numpy(cr.reshape(-1, RES, RES, 3)[idx]).double().div(255)
            x = ((x - mean) / std).permute(0, 3, 1, 2)
            with torch.no_grad():
                E[idx] = m(x).numpy().astype(np.float16)
        Es.append(E.reshape(32, -1)); Ms.append(ok.reshape(32, -1))
        if n % 250 == 0:
            print("embedded", n, len(recs), flush=True)
    return Es, Ms


def run_models(paths, arrays, threads=2, bs=256):
    torch.set_num_threads(threads)
    D = {k: np.stack([a[k] for a in arrays]) for k in KEYS + EKEYS if k in arrays[0]}
    outs = []
    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        a = ck["args"]
        m = PoseNet(ck["d_in"], d=a["d"], layers=a["layers"], d_emb=ck.get("d_emb", 0))
        m.load_state_dict(ck["state"]); m.eval()
        m.double()  # float64 makes the printed digits identical across CPU architectures
        res = {"center": [], "se": [], "pt": []}
        with torch.no_grad():
            for i in range(0, len(D["X"]), bs):
                ks = KEYS + (EKEYS if ck.get("d_emb", 0) else ())
                o = m(*[torch.tensor(D[k][i:i + bs]).double() if D[k].dtype.kind == "f" else torch.tensor(D[k][i:i + bs]) for k in ks])
                res["center"].append(torch.sigmoid(o["center"]).numpy())
                res["se"].append(o["se"].numpy()); res["pt"].append(o["pt"].numpy())
        outs.append({k: np.concatenate(v) for k, v in res.items()})
    return {k: np.mean([o[k] for o in outs], 0) for k in outs[0]}, D


def to_json(preds, nd=6):
    acts = []
    for s, st, en, pts in preds:
        acts.append({"score": round(float(s), nd), "start": round(float(st), nd), "end": round(float(en), nd),
                     "spatial": {"action_cam": {r: {"x": round(float(p[0]), nd), "y": round(float(p[1]), nd)}
                                                for r, p in zip(("rank0", "rank1"), pts)}}})
    return {"actions": acts}


def finalize(preds):
    """Clips values into the valid ranges. Scores stay below 1.0 because the metric drops recall gained at 1.0."""
    out = []
    for s, st, en, pts in preds:
        st = float(np.clip(st, 0, 1)); en = float(np.clip(en, 0, 1))
        if en - st < 1e-3:
            st, en = max(0.0, min(st, 1 - 1e-3)), min(1.0, max(st, 1 - 1e-3) + 1e-3)
        pts = [tuple(float(np.clip(v, 0, 1)) for v in p) for p in pts]
        out.append((float(np.clip(s, 1e-4, 0.999)), st, en, pts))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--packed", default=None, help="packed metadata pickle, used instead of --data_dir")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--nms_iou", type=float, default=0.6)
    ap.add_argument("--threads", type=int, default=0, help="0: all cores")
    ap.add_argument("--dino_dir", default="/opt/dinov2")
    ap.add_argument("--dino_weights", default="/opt/dinov2_vits14_pretrain.pth")
    args = ap.parse_args()
    if args.packed:
        import gzip, pickle
        recs = sorted(pickle.load(gzip.open(args.packed, "rb")), key=lambda r: os.path.basename(r["dir"]))
    else:
        from loader import load
        dirs = sorted(d for d in glob.glob(os.path.join(args.data_dir, "*")) if os.path.isfile(os.path.join(d, "metadata.json")))
        recs = [load(d) for d in dirs]
    print("samples", len(recs), flush=True)
    ids = [os.path.basename(r["dir"].rstrip("/")) for r in recs]
    threads = args.threads or os.cpu_count()
    arrays = [sample_arrays(r) for r in recs]
    if any(torch.load(p, map_location="cpu", weights_only=False).get("d_emb", 0) for p in args.models):
        Es, Ms = embed_samples(recs, args.dino_dir, args.dino_weights, threads)
        for a, e, mk in zip(arrays, Es, Ms):
            a["E"], a["EM"] = e, mk
    P, D = run_models(args.models, arrays, threads)
    preds = []
    for i in range(len(ids)):
        p = decode(P["center"][i], P["se"][i], P["pt"][i], D["tn"][i], D["valid"][i], k=args.k, nms_iou=args.nms_iou)
        preds.append(to_json(finalize(p)))
    write_submission(args.out, ids, preds)
    print("wrote", args.out, len(ids), flush=True)


if __name__ == "__main__":
    main()
