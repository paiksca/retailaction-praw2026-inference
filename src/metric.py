"""Fast re-implementation of the official RetailAction mAPglobal metric (davidemaz/map-global-metric-code).

Greedy matching per video does not depend on the confidence threshold beyond which predictions are
included, and the included set is a prefix of the score-sorted list. So one greedy pass per
(spatial, tIoU) pair gives the TP flag of every prediction, and cumulative sums give all 101 thresholds.
"""
import json
import math

import numpy as np

CONF_T = np.linspace(1, 0, 101)
TIOU_T = np.linspace(0.05, 0.5, 10)
SPAT_T = np.linspace(1, 0.05, 20)
EPS = 1e-4
M_PX_FACTOR_AVG = 3.07
BONES = {
    ("neck", "nose"): 0.19354, ("left_shoulder", "left_elbow"): 0.27096, ("left_elbow", "left_wrist"): 0.21228,
    ("right_shoulder", "right_elbow"): 0.27210, ("right_elbow", "right_wrist"): 0.21316,
    ("left_hip", "left_knee"): 0.39204, ("left_knee", "left_ankle"): 0.39530, ("right_hip", "right_knee"): 0.39266,
    ("right_knee", "right_ankle"): 0.39322, ("left_shoulder", "right_shoulder"): 0.35484,
    ("left_hip", "right_hip"): 0.17150, ("neck", "left_shoulder"): 0.18136, ("neck", "right_shoulder"): 0.18081,
    ("left_shoulder", "left_hip"): 0.51375, ("right_shoulder", "right_hip"): 0.51226,
}
RANKS = ("rank0", "rank1")


def m_px_factor(pose_arr, joints):
    """pose_arr: (F, J, 3) with x, y, score (NaN where the joint is absent). Mirrors the official parser."""
    ji = {j: i for i, j in enumerate(joints)}
    vals = []
    for (a, b), m in BONES.items():
        if a not in ji or b not in ji:
            continue
        pa, pb = pose_arr[:, ji[a]], pose_arr[:, ji[b]]
        sa = np.nan_to_num(pa[:, 2]); sb = np.nan_to_num(pb[:, 2])
        okA = ~np.isnan(pa[:, 0]) & ((sa > 0) | (pa[:, 0] != 0) | (pa[:, 1] != 0))
        okB = ~np.isnan(pb[:, 0]) & ((sb > 0) | (pb[:, 0] != 0) | (pb[:, 1] != 0))
        ok = okA & okB
        L = np.sqrt(((pa[ok, :2] - pb[ok, :2]) ** 2).sum(-1))
        L = L[L > 0]
        vals.extend((m / L).tolist())
    if not vals:
        return M_PX_FACTOR_AVG
    v = float(np.nanmedian(vals))
    return M_PX_FACTOR_AVG if math.isnan(v) else v


class Accum:
    def __init__(self):
        S, T, C = len(SPAT_T), len(TIOU_T), len(CONF_T)
        self.tp = np.zeros((S, T, C)); self.fp = np.zeros((S, T, C)); self.fn = np.zeros((S, T, C))

    def add(self, preds, gts, fac):
        """preds: list of (score, start, end, [(x0,y0),(x1,y1)]). gts: list of (start, end, [(x0,y0),(x1,y1)]).
        fac: (f0, f1)."""
        if not gts:
            return
        preds = sorted(preds, key=lambda p: p[0], reverse=True)  # stable, like the official sort
        G = len(gts)
        if not preds:
            self.fn += G
            return
        P = len(preds)
        ps = np.array([[p[1], p[2]] for p in preds]); gs = np.array([[g[0], g[1]] for g in gts])
        inter = np.minimum(ps[:, None, 1], gs[None, :, 1]) - np.maximum(ps[:, None, 0], gs[None, :, 0])
        union = np.maximum(ps[:, None, 1], gs[None, :, 1]) - np.minimum(ps[:, None, 0], gs[None, :, 0])
        tiou = np.maximum(0, inter / (union + EPS))  # (P, G)
        pxy = np.array([p[3] for p in preds], float)  # (P, 2, 2)
        gxy = np.array([g[2] for g in gts], float)  # (G, 2, 2)
        d = np.sqrt(((pxy[:, None] - gxy[None]) ** 2).sum(-1)) * np.asarray(fac)[None, None, :]  # (P, G, 2)
        dmin = d.min(-1)
        M = (tiou[None, None] >= TIOU_T[None, :, None, None]) & (tiou[None, None] > 0) & \
            (dmin[None, None] <= SPAT_T[:, None, None, None])  # (S, T, P, G)
        S, T = len(SPAT_T), len(TIOU_T)
        matched = np.zeros((S, T, G), bool)
        flag = np.zeros((S, T, P), bool)
        for i in range(P):
            cand = M[:, :, i, :] & ~matched
            w = np.where(cand, tiou[i][None, None, :], -1.0)
            g = w.argmax(-1)
            has = cand.any(-1)
            flag[:, :, i] = has
            idx = np.nonzero(has)
            matched[idx[0], idx[1], g[idx]] = True
        scores = np.array([p[0] for p in preds])
        inc = scores[:, None] >= CONF_T[None, :]  # (P, C)
        tpc = np.einsum("stp,pc->stc", flag.astype(float), inc.astype(float))
        n = inc.sum(0)[None, None, :]
        self.tp += tpc; self.fp += n - tpc; self.fn += G - tpc

    def ap(self):
        prec = self.tp / (self.tp + self.fp + EPS)
        rec = self.tp / (self.tp + self.fn + EPS)
        prec = np.flip(np.maximum.accumulate(np.flip(prec, -1), -1), -1)
        return np.sum((rec[..., 1:] - rec[..., :-1]) * prec[..., :-1], -1)  # (S, T)

    def summary(self):
        a = self.ap()
        return {"mAP": float(a.mean()), "mAPs": float(a[:, 0].mean()), "mAPt": float(a[0].mean())}


def pred_to_tuple(a):
    sp = a["spatial"]["action_cam"]
    return (float(a["score"]), float(a["start"]), float(a["end"]),
            [(sp[r]["x"], sp[r]["y"]) for r in RANKS])


def validate_prediction(pred):
    """Rejects anything that could look like use of the spatial-check bug: every action needs finite
    action_cam rank0 and rank1 points in [0, 1]."""
    assert set(pred) == {"actions"}, pred.keys()
    for a in pred["actions"]:
        assert set(a) == {"score", "start", "end", "spatial"}, a.keys()
        assert set(a["spatial"]) == {"action_cam"}, a["spatial"].keys()
        assert set(a["spatial"]["action_cam"]) == set(RANKS), a["spatial"]["action_cam"].keys()
        for r in RANKS:
            pt = a["spatial"]["action_cam"][r]
            assert set(pt) == {"x", "y"}, pt.keys()
            for k in ("x", "y"):
                assert isinstance(pt[k], float) and math.isfinite(pt[k]) and 0.0 <= pt[k] <= 1.0, pt
        for k in ("score", "start", "end"):
            assert isinstance(a[k], float) and math.isfinite(a[k]) and 0.0 <= a[k] <= 1.0, a
        assert a["start"] < a["end"], a


def write_submission(path, ids, preds):
    import csv
    assert len(ids) == len(set(ids)) == len(preds)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "prediction"])
        for i, p in zip(ids, preds):
            validate_prediction(p)
            w.writerow([i, json.dumps(p)])
