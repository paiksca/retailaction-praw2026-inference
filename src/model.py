"""Temporal transformer over the 32 sampled frames, with heads for action centers, boundaries and touch points."""
import math

import numpy as np
import torch
import torch.nn as nn


class PoseNet(nn.Module):
    def __init__(self, d_in, d=192, layers=4, heads=6, drop=0.1, d_emb=0):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d))
        self.d_emb = d_emb
        if d_emb:  # hand-crop embeddings: 2 views x 2 hands per frame, plus their presence mask
            self.emb = nn.Sequential(nn.LayerNorm(4 * d_emb), nn.Dropout(0.2), nn.Linear(4 * d_emb, d), nn.GELU(), nn.Linear(d, d))
        self.tfreq = nn.Parameter(torch.randn(d // 2) * 6, requires_grad=False)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, drop, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(d)
        self.center = nn.Linear(d, 1)
        self.inside = nn.Linear(d, 1)
        self.se = nn.Linear(d, 2)
        self.pt = nn.Linear(d, 4)  # free point per view
        self.dk = 32
        self.q = nn.Linear(d, 2 * self.dk)  # per view query
        self.k = nn.Linear(d, 2 * 4 * self.dk)  # per view, per hand joint key at each frame
        self.kfree = nn.Linear(d, 2)  # logit of the free point
        self.off = nn.Linear(d, 4)  # per-view offset added to the selected anchor

    def forward(self, X, valid, tn, hands, hmask, E=None, EM=None):
        h = self.inp(X)
        if self.d_emb:
            h = h + self.emb(E.to(h.dtype) * EM.to(h.dtype).repeat_interleave(self.d_emb, -1))
        ang = tn[..., None] * self.tfreq * 2 * math.pi
        h = h + torch.cat([ang.sin(), ang.cos()], -1)
        h = self.enc(h, src_key_padding_mask=~valid)
        h = self.norm(h)
        return {"center": self.center(h)[..., 0], "inside": self.inside(h)[..., 0],
                "se": nn.functional.softplus(self.se(h)), "pt": self.point(h, hands, hmask)}

    def point(self, h, hands, hmask):
        """Pointer over the hand joints of all frames, plus a free point, per view."""
        B, F = h.shape[:2]
        free = torch.sigmoid(self.pt(h)).view(B, F, 2, 1, 2)
        q = self.q(h).view(B, F, 2, self.dk)
        k = self.k(h).view(B, F, 2, 4, self.dk).permute(0, 2, 1, 3, 4).reshape(B, 2, F * 4, self.dk)
        logit = torch.einsum("bfvd,bvnd->bfvn", q, k) / self.dk ** 0.5  # (B, F, 2, F*4)
        km = hmask.permute(0, 2, 1, 3).reshape(B, 2, F * 4)
        logit = logit.masked_fill(~km[:, None], -1e4)
        logit = torch.cat([logit, self.kfree(h).view(B, F, 2, 1)], -1)
        w = torch.softmax(logit, -1)
        hc = hands.permute(0, 2, 1, 3, 4).reshape(B, 2, F * 4, 2)
        cand = torch.cat([hc[:, None].expand(B, F, 2, F * 4, 2), free], 3)
        anchor = (w[..., None] * cand).sum(3)
        return (anchor + 0.1 * torch.tanh(self.off(h)).view(B, F, 2, 2)).clamp(0, 1)


def decode(center, se, pt, tn, valid, k=5, nms_iou=0.6, min_score=0.0):
    """center: (F,) probabilities; se: (F, 2); pt: (F, 2, 2). Returns list of (score, start, end, pts)."""
    F = len(center)
    c = np.where(valid, center, -1)
    peaks = [i for i in range(F) if valid[i] and c[i] >= (c[i - 1] if i > 0 else -1) and c[i] >= (c[i + 1] if i + 1 < F else -1)]
    peaks.sort(key=lambda i: -c[i])
    out = []
    for i in peaks:
        s = float(np.clip(tn[i] - se[i, 0], 0, 1)); e = float(np.clip(tn[i] + se[i, 1], 0, 1))
        if e <= s:
            e = min(1.0, s + 1e-3)
        ok = True
        for o in out:
            inter = max(0, min(e, o[2]) - max(s, o[1])); uni = max(e, o[2]) - min(s, o[1])
            if inter / max(uni, 1e-6) > nms_iou:
                ok = False
                break
        if ok and c[i] >= min_score:
            out.append((float(c[i]), s, e, [tuple(map(float, pt[i, v])) for v in range(2)]))
        if len(out) >= k:
            break
    return out
