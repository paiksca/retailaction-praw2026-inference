"""Loads the packed metadata pickles exported by kernels/export_meta."""
import gzip
import os
import pickle

import numpy as np

JOINTS = ["top_of_head", "nose", "neck", "right_shoulder", "right_elbow", "right_wrist", "right_hand",
          "left_shoulder", "left_elbow", "left_wrist", "left_hand", "middle_of_waist", "right_hip", "right_knee",
          "right_ankle", "right_foot", "left_hip", "left_knee", "left_ankle", "left_foot",
          "left_eye", "left_ear", "right_eye", "right_ear"]
JI = {j: i for i, j in enumerate(JOINTS)}
RANKS = ("rank0", "rank1")
META_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "meta")


def load_split(name, meta_dir=META_DIR):
    with gzip.open(os.path.join(meta_dir, f"{name}.pkl.gz"), "rb") as f:
        return pickle.load(f)


def sample_id(rec):
    return os.path.basename(rec["dir"].rstrip("/"))


def gt_actions(rec):
    """Returns list of (start, end, [(x0, y0), (x1, y1)], label)."""
    lab = rec.get("labels")
    if lab is None:
        return []
    acts = lab.get("action", lab) if isinstance(lab, dict) else lab
    out = []
    for a in acts or []:
        sp = (a.get("spatial") or {}).get("action_cam", {})
        pts = [(sp[r]["x"], sp[r]["y"]) if r in sp and sp[r] else (np.nan, np.nan) for r in RANKS]
        out.append((a["start"], a["end"], pts, a.get("label", a.get("action_class"))))
    return out
