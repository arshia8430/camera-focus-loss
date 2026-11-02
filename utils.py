"""General helpers: camera containers, geometry, statistics, image embeddings and the CLI wrapper.

Nothing in this file depends on a particular loss or scene.
"""
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pytorch3d.transforms import matrix_to_quaternion
from scipy import stats

from config import IMG, dev, set_results_dir


# ======================================================================================
# Camera containers: a camera batch is a dict {"K": (B,3,3), "R": (B,3,3) cam->world, "t": (B,3) camera centre}
# ======================================================================================
def cat_cams(lst):
    return {k: torch.cat([c[k] for c in lst]) for k in lst[0]}


def idx_cams(c, i):
    return {k: v[i] for k, v in c.items()}


def n_cams(c):
    return c["R"].shape[0]


def to_tensor(a):
    return torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)


def subset_D(D, idx):
    """Select cameras `idx` from a dataset dict produced by optim.load_split."""
    t = torch.as_tensor(idx, device=dev)
    return dict(cams={k: v[t] for k, v in D["cams"].items()}, gt={k: v[t] for k, v in D["gt"].items()},
                ctx=dict(pts=D["ctx"]["pts"][t]), thr_t=D["thr_t"][t], sid=D["sid"][idx], s=D["s"][idx],
                rot0=D["rot0"][idx])


# ======================================================================================
# Geometry
# ======================================================================================
def make_K(f):
    B = len(f)
    K = torch.zeros(B, 3, 3, device=dev)
    K[:, 0, 0] = f
    K[:, 1, 1] = f
    K[:, 0, 2] = IMG / 2
    K[:, 1, 2] = IMG / 2
    K[:, 2, 2] = 1
    return K


def look_at_R(eye, target, roll):
    """Camera-to-world rotation (OpenCV convention: x right, y down, z forward) looking from eye to target."""
    B = eye.shape[0]
    z = F.normalize(target - eye, dim=1)
    up = torch.tensor([0., 1., 0.], device=dev).expand(B, 3)
    alt = torch.tensor([0., 0., 1.], device=dev).expand(B, 3)
    up = torch.where(((z * up).sum(1).abs() > .99)[:, None], alt, up)
    down = -up
    down = F.normalize(down - (down * z).sum(1, keepdim=True) * z, dim=1)
    x = torch.cross(down, z, dim=1)
    R = torch.stack([x, down, z], dim=2)
    c, s = torch.cos(roll), torch.sin(roll)
    Rz = torch.zeros(B, 3, 3, device=dev)
    Rz[:, 0, 0] = c
    Rz[:, 0, 1] = -s
    Rz[:, 1, 0] = s
    Rz[:, 1, 1] = c
    Rz[:, 2, 2] = 1
    return R @ Rz


def so3_angle(Rm):
    cos = (Rm.diagonal(dim1=1, dim2=2).sum(-1) - 1) / 2
    vee = torch.stack([Rm[:, 2, 1] - Rm[:, 1, 2], Rm[:, 0, 2] - Rm[:, 2, 0], Rm[:, 1, 0] - Rm[:, 0, 1]], 1) / 2
    return torch.atan2(torch.sqrt((vee ** 2).sum(1) + 1e-12), cos)


def l2(x):
    return torch.sqrt((x ** 2).sum(-1) + 1e-12)


def quat_pair(cam, tgt):
    """Quaternions of cam and tgt in the SAME hemisphere (q and -q encode the same rotation).

    The sign is chosen per sample by <q_cam, q_tgt> >= 0 (detached), so a loss built on them is continuous on SO(3).
    """
    qc, qt = matrix_to_quaternion(cam["R"]), matrix_to_quaternion(tgt["R"])
    qc, qt = torch.broadcast_tensors(qc, qt)
    sgn = torch.where((qc * qt).sum(1, keepdim=True) < 0, -1.0, 1.0).detach()
    return qc, qt * sgn


def rays_cam(K, P):
    """K^{-1} P for homogeneous pixels P (3,N) -> (B,3,N) rays in the camera frame."""
    fx, fy, cx, cy = K[:, 0, 0, None], K[:, 1, 1, None], K[:, 0, 2, None], K[:, 1, 2, None]
    return torch.stack([(P[0][None] - cx) / fx, (P[1][None] - cy) / fy,
                        torch.ones_like(P[0])[None].expand(K.shape[0], -1)], 1)


def project(cam, X):
    """Pinhole projection. X: (N,3) shared points or (B,N,3) per-camera points -> (B,N,2) pixels."""
    X = X if X.dim() == 3 else X[None]
    Xc = (X - cam["t"][:, None, :]) @ cam["R"]
    z = Xc[..., 2].clamp(min=1e-2)
    u = cam["K"][:, 0, 0, None] * Xc[..., 0] / z + cam["K"][:, 0, 2, None]
    v = cam["K"][:, 1, 1, None] * Xc[..., 1] / z + cam["K"][:, 1, 2, None]
    return torch.stack([u, v], -1).clamp(-10 * IMG, 10 * IMG)


def pose_errors(cam, tgt):
    return dict(rot_deg=torch.rad2deg(so3_angle(cam["R"].transpose(1, 2) @ tgt["R"])),
                trans=(cam["t"] - tgt["t"]).norm(dim=1),
                logf=(torch.log(cam["K"][:, 0, 0]) - torch.log(tgt["K"][:, 0, 0])).abs())


# ======================================================================================
# Statistics
# ======================================================================================
def rank_metrics(loss, sim, k=30):
    """Spearman, Kendall, NDCG@k and top-k overlap between a loss ranking and an image-similarity ranking."""
    if np.std(loss) == 0:
        return (np.nan,) * 4
    N = len(sim)
    rho = stats.spearmanr(loss, -sim)[0]
    tau = stats.kendalltau(loss, -sim)[0]
    rel = N - stats.rankdata(-sim)
    top = np.argsort(loss)[:k]
    disc = 1 / np.log2(np.arange(2, k + 2))
    ndcg = (rel[top] * disc).sum() / (np.sort(rel)[::-1][:k] * disc).sum()
    overlap = len(set(top) & set(np.argsort(-sim)[:k])) / k
    return rho, tau, ndcg, overlap


def boot_ci(x, B=10000, seed=0):
    """95% bootstrap CI of the mean."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return np.percentile(means, [2.5, 97.5])


def holm(p):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(p)
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1, run)
    return adj


def binned(df, xcol, ycol, edges, B=2000, seed=0):
    """Mean of `ycol` per bin of `xcol`, with a cluster bootstrap (resampling whole scenes) 95% CI."""
    b = pd.cut(df[xcol], edges, include_lowest=True, labels=False)
    ok = b.notna().values
    d = df[ok]
    b = b[ok].astype(int).values
    nb = len(edges) - 1
    scenes = np.sort(d.sid.unique())
    S = np.zeros((len(scenes), nb))
    C = np.zeros_like(S)
    for i, s_ in enumerate(scenes):
        m = (d.sid == s_).values
        S[i] = np.bincount(b[m], weights=d[ycol].values[m].astype(float), minlength=nb)
        C[i] = np.bincount(b[m], minlength=nb)
    mean = S.sum(0) / np.maximum(C.sum(0), 1)
    idx = np.random.default_rng(seed).integers(0, len(scenes), (B, len(scenes)))
    bm = S[idx].sum(1) / np.maximum(C[idx].sum(1), 1)
    return mean, np.percentile(bm, 2.5, 0), np.percentile(bm, 97.5, 0)


def cluster_binned(d, xcol, ycol, edges, min_n=15, B=2000, seed=0):
    """Like `binned`, but bins with fewer than `min_n` cameras are masked (NaN). Also returns the bin counts."""
    b = pd.cut(d[xcol], edges, include_lowest=True, labels=False)
    ok = b.notna().values
    d = d[ok]
    b = b[ok].astype(int).values
    nb = len(edges) - 1
    scenes = np.sort(d.sid.unique())
    sid = d.sid.values
    y = d[ycol].values.astype(float)
    S = np.zeros((len(scenes), nb))
    C = np.zeros_like(S)
    for i, s_ in enumerate(scenes):
        m = sid == s_
        S[i] = np.bincount(b[m], weights=y[m], minlength=nb)
        C[i] = np.bincount(b[m], minlength=nb)
    tot = C.sum(0)
    small = tot < min_n
    mean = np.where(small, np.nan, S.sum(0) / np.maximum(tot, 1))
    idx = np.random.default_rng(seed).integers(0, len(scenes), (B, len(scenes)))
    bs, bc = S[idx].sum(1), C[idx].sum(1)
    bm = np.where(bc > 0, bs / np.maximum(bc, 1), np.nan)
    lo, hi = np.nanpercentile(bm, 2.5, 0), np.nanpercentile(bm, 97.5, 0)
    lo[small] = hi[small] = np.nan
    return mean, lo, hi, tot


# ======================================================================================
# Image embeddings (DINOv2 / CLIP), loaded lazily on first use
# ======================================================================================
_ENCODERS = {}
IM_MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
IM_STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
CL_MEAN = torch.tensor([0.4815, 0.4578, 0.4082], device=dev).view(1, 3, 1, 1)
CL_STD = torch.tensor([0.2686, 0.2613, 0.2758], device=dev).view(1, 3, 1, 1)


def _encoders():
    if not _ENCODERS:
        from transformers import AutoModel, CLIPModel
        _ENCODERS["dino"] = AutoModel.from_pretrained("facebook/dinov2-base").to(dev).eval().half()
        _ENCODERS["clip"] = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval().half()
    return _ENCODERS


@torch.no_grad()
def embed(imgs, which, bs=100):
    """L2-normalised image embeddings. imgs: (N,3,H,W) in [0,1]; which in {"dino", "clip"}."""
    enc = _encoders()
    out = []
    for i in range(0, len(imgs), bs):
        x = imgs[i:i + bs]
        if which == "dino":
            z = enc["dino"](pixel_values=((x - IM_MEAN) / IM_STD).half()).last_hidden_state[:, 0]
        else:
            z = enc["clip"].get_image_features(pixel_values=((x - CL_MEAN) / CL_STD).half())
            if not torch.is_tensor(z):
                z = z.pooler_output
        out.append(F.normalize(z.float(), dim=-1))
    return torch.cat(out)


# ======================================================================================
# Command line wrapper shared by the experiment scripts
# ======================================================================================
def run_stages(description, run, analyze):
    """Parse --stage/--out and call run() (GPU, writes files) and/or analyze() (reads files, writes tables/figures)."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--stage", choices=["run", "analyze", "all"], default="all")
    ap.add_argument("--out", default=None, help="results directory (default: ./results or $FOCUS_RESULTS)")
    args = ap.parse_args()
    if args.out:
        set_results_dir(args.out)
    if args.stage in ("run", "all"):
        run()
    if args.stage in ("analyze", "all"):
        analyze()
