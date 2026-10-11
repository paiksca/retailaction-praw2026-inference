"""Per-frame features and targets for the pose model. One token per sampled frame, both views concatenated."""
import numpy as np

from data import JI, JOINTS, RANKS, gt_actions
from metric import m_px_factor

NF = 32
HANDS = [JI[j] for j in ("right_wrist", "right_hand", "left_wrist", "left_hand")]


def duration(rec):
    d = rec["seg_end"] - rec["seg_start"]
    if not np.isfinite(d) or d <= 0:
        ts = rec["rank0_ts"]
        d = float(np.nanmax(ts)) if len(ts) else 1.0
    return max(float(d), 1e-3)


def _pad(a, n=NF):
    out = np.full((n,) + a.shape[1:], np.nan, np.float32)
    out[: min(n, len(a))] = a[:n]
    return out


def view_feats(rec, r, tn, dur):
    pose = _pad(rec[f"{r}_pose"]) if len(rec[f"{r}_pose"]) else np.full((NF, len(JOINTS), 3), np.nan, np.float32)
    xy = pose[:, :, [1, 0]]  # stored as (row, col)
    pres = ~np.isnan(xy[:, :, 0])
    sc = np.nan_to_num(pose[:, :, 2])
    neck = xy[:, JI["neck"]]
    waist = xy[:, JI["middle_of_waist"]]
    center = np.where(np.isnan(neck), waist, neck)
    rel = xy - center[:, None, :]
    # Hand-joint velocity, in normalized units per second, by central differences over valid frames.
    t = tn * dur
    hv = np.zeros((NF, len(HANDS), 2), np.float32)
    for k, j in enumerate(HANDS):
        p = xy[:, j]
        ok = ~np.isnan(p[:, 0])
        if ok.sum() >= 2:
            pi = np.stack([np.interp(t, t[ok], p[ok, c]) for c in range(2)], -1)
            hv[:, k] = np.gradient(pi, t + np.arange(NF) * 1e-4, axis=0)
    # Arm extension: hand distance from the body center.
    ext = np.linalg.norm(rel[:, HANDS], axis=-1)
    ss = rec[f"{r}_ss"]
    if len(ss):
        ssv = np.interp(t, ss[:, 0] - rec["seg_start"], np.nan_to_num(ss[:, 1]))
    else:
        ssv = np.zeros(NF)
    face = _pad(rec[f"{r}_face"])[:, :2]
    fac = m_px_factor(rec[f"{r}_pose"], JOINTS) if len(rec[f"{r}_pose"]) else 3.07
    vis = pres.any(1).astype(np.float32)
    f = np.concatenate([
        np.nan_to_num(xy).reshape(NF, -1), np.nan_to_num(rel).reshape(NF, -1), pres.astype(np.float32), sc,
        hv.reshape(NF, -1).clip(-5, 5), np.nan_to_num(ext, nan=0.0), ssv[:, None], np.nan_to_num(face),
        np.full((NF, 1), fac / 3.0), vis[:, None],
    ], 1).astype(np.float32)
    return f, xy, fac


def sample_arrays(rec):
    dur = duration(rec)
    ts = _pad(rec["rank0_ts"])
    if np.isnan(ts).all():
        ts = _pad(rec["rank1_ts"])
    tn = (ts - rec["seg_start"]) / dur
    valid = ~np.isnan(tn)
    tn = np.where(valid, tn, 1.0).astype(np.float32)
    fs, xys, facs = [], [], []
    for r in RANKS:
        f, xy, fac = view_feats(rec, r, tn, dur)
        fs.append(f); xys.append(xy); facs.append(fac)
    dtp = np.diff(tn, prepend=0.0); dtn = np.diff(tn, append=1.0)
    glob = np.stack([tn, dtp, dtn, np.full(NF, np.log(dur) / 4), np.arange(NF) / NF], 1).astype(np.float32)
    X = np.concatenate([glob] + fs, 1)
    X[~valid] = 0
    xy = np.stack(xys, 1)  # (F, 2, J, 2)
    hands = xy[:, :, HANDS]
    hmask = ~np.isnan(hands[..., 0])
    return {"X": X, "tn": tn, "valid": valid, "dur": dur, "xy": xy, "fac": np.array(facs, np.float32),
            "hands": np.nan_to_num(hands).astype(np.float32), "hmask": hmask}


def targets(rec, tn):
    gts = gt_actions(rec)
    T = {"inside": np.zeros(NF, np.float32), "center": np.zeros(NF, np.float32),
         "se": np.zeros((NF, 2), np.float32), "pt": np.full((NF, 2, 2), np.nan, np.float32),
         "has": np.zeros(NF, np.float32)}
    if not gts:
        return T, gts
    cs = np.array([(g[0] + g[1]) / 2 for g in gts])
    near = np.abs(tn[:, None] - cs[None]).argmin(1)
    for i in range(NF):
        g = gts[near[i]]
        s, e = g[0], g[1]
        w = max(e - s, 1e-3)
        T["inside"][i] = float(any(gg[0] <= tn[i] <= gg[1] for gg in gts))
        T["center"][i] = float(np.max([np.exp(-0.5 * (((tn[i] - (gg[0] + gg[1]) / 2) / (0.25 * max(gg[1] - gg[0], 0.02))) ** 2)) for gg in gts]))
        T["se"][i] = (tn[i] - s, e - tn[i])
        T["pt"][i] = np.array(g[2], np.float32)
        T["has"][i] = 1.0
    return T, gts
