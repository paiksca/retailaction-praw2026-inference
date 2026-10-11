"""Reads one sample directory (metadata.json) into the record format used by features.py.
Matches kernels/export_meta2/export_meta.py:load for the fields the model uses."""
import json
import os
from datetime import datetime

import numpy as np

from data import JI, JOINTS

EPOCH = datetime.fromisoformat("1970-01-01T00:00:00+00:00")


def tsec(s):
    if s is None:
        return np.nan
    d = datetime.fromisoformat(s if ("+" in s[10:] or s.endswith("Z")) else s + "+00:00")
    return (d - EPOCH).total_seconds()


def pose_arr(p):
    a = np.full((len(JOINTS), 3), np.nan, np.float32)
    if p is None:
        return a
    js = p.get("joint_scores", {})
    for k, v in p.get("pose", {}).items():
        if k in JI and v is not None:
            a[JI[k], :2] = v
            a[JI[k], 2] = js.get(k, np.nan)
    return a


def load(d):
    rec = {"dir": d}
    m = json.load(open(os.path.join(d, "metadata.json")))
    c = m.get("content", m)
    si = c.get("segment_info", {})
    rec["seg_start"] = tsec(si.get("sampled_at_start"))
    rec["seg_end"] = tsec(si.get("sampled_at_end"))
    rec["labels"] = c.get("labels", m.get("labels"))
    for r in ("rank0", "rank1"):
        rr = c["action_cam"].get(r, {})
        rec[r + "_ts"] = np.array([tsec(t) for t in (rr.get("frame_timestamps") or [])], np.float32)
        ss = rr.get("sampling_scores") or []
        rec[r + "_ss"] = np.array([[tsec(t), v if v is not None else np.nan] for t, v in ss if t is not None], np.float32).reshape(-1, 2)
        fp = rr.get("face_positions") or []
        rec[r + "_face"] = np.array([[f["col"], f["row"], tsec(f.get("sampled_at"))] if f else [np.nan] * 3 for f in fp], np.float32).reshape(-1, 3)
        rec[r + "_pose"] = np.stack([pose_arr(p) for p in rr.get("poses")]) if rr.get("poses") else np.zeros((0, len(JOINTS), 3), np.float32)
    return rec
