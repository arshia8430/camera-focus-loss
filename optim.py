"""Camera optimisation with a given loss, convergence metrics, learning-rate tuning tables and dataset loading.

Shared core of experiments 2-5.
"""
import time

import numpy as np
import pandas as pd
import torch
from pytorch3d.transforms import axis_angle_to_matrix

from config import (LOGF_T, ROT_T, T_SCALE, T_STEPS, TRANS_FRAC, cache_dir, dev)
from utils import make_K, n_cams, pose_errors

# Losses that are always compared in the final experiments (the three "best ours" entries are picked by tuning).
FIXED_CHOSEN = ["Dir-only", "P-MSE", "PoseNet(b=10)", "PoseNet(b=100)", "SE3-geo", "Ray-Plucker", "DenseRay",
                "Reproj-px", "SoftSqrt(zf=1,s=4)", "SoftGridSqrt(zf=1,s=4,r=48)"]


# ======================================================================================
# Data
# ======================================================================================
def load_split(scenes):
    """Load cached scenes (written by exp1) into one batch of cameras, GT cameras and scene points."""
    C, G, P, SID, S = [], [], [], [], []
    for sc_ in scenes:
        d = torch.load(cache_dir() / f"scene_{sc_:03d}.pt", weights_only=False)
        n = d["cams"]["R"].shape[0]
        C.append(d["cams"])
        G.append({k: v.expand(n, *v.shape[1:]).contiguous() for k, v in d["gt"].items()})
        P.append(d["pts"][:512][None].expand(n, -1, -1).contiguous())
        SID += [sc_] * n
        S.append(d["s"])
    cams = {k: torch.cat([c[k] for c in C]).to(dev) for k in C[0]}
    gt = {k: torch.cat([g[k] for g in G]).to(dev) for k in G[0]}
    pts = torch.cat(P).to(dev)
    thr_t = TRANS_FRAC * (gt["t"] - pts.mean(1)).norm(dim=1)
    return dict(cams=cams, gt=gt, ctx=dict(pts=pts), thr_t=thr_t, sid=np.array(SID),
                s=np.concatenate(S), rot0=pose_errors(cams, gt)["rot_deg"].cpu().numpy())


# ======================================================================================
# Optimisation
# ======================================================================================
def optimize(fn, D, lr, steps=T_STEPS, snaps=None, snap_every=50):
    """Optimise every camera of D independently (Adam + cosine LR) to match its GT camera under loss `fn`.

    Parameters per camera: translation offset dt (scaled by T_SCALE), rotation increment w (axis-angle) and log-focal lf.
    Returns the record of the three normalised error components, shape (steps+1, 3, B); a camera has converged
    when all components are < 1. If `snaps` is a list, the camera batch is appended to it every `snap_every` steps.
    """
    cam0, gt = D["cams"], D["gt"]
    B = n_cams(cam0)
    f0 = cam0["K"][:, 0, 0]
    dt = torch.zeros(B, 3, device=dev, requires_grad=True)
    w = torch.zeros(B, 3, device=dev, requires_grad=True)
    lf = torch.zeros(B, device=dev, requires_grad=True)
    opt = torch.optim.Adam([dt, w, lf], lr=lr, eps=1e-12)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.01)
    rec = []
    for it in range(steps + 1):
        c = dict(K=make_K(f0 * torch.exp(lf)), R=cam0["R"] @ axis_angle_to_matrix(w), t=cam0["t"] + T_SCALE * dt)
        if snaps is not None and it % snap_every == 0:
            snaps.append({k: v.detach().clone() for k, v in c.items()})
        with torch.no_grad():
            e = pose_errors(c, gt)
            comp = torch.stack([e["rot_deg"] / ROT_T, e["trans"] / D["thr_t"], e["logf"] / LOGF_T])
            rec.append(torch.nan_to_num(comp, nan=1e6, posinf=1e6, neginf=1e6))
        if it == steps:
            break
        loss = fn(c, gt, D["ctx"]).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    return torch.stack(rec)


def metrics_from_record(Rc):
    """Convergence metrics from an error record (T+1, 3, B).

    ttc = first step after which the camera stays converged (T if it never does);
    speed = 1 - ttc/T, speed_hit = same with the first hit instead, speed90 = 90% error reduction.
    """
    comp = Rc.max(1).values
    succ = comp < 1
    T = comp.shape[0] - 1
    full = torch.full((comp.shape[1],), float(T), device=dev)
    hit = torch.where(succ.any(0), succ.float().argmax(0).float(), full)
    bad = ~succ
    last_bad = (T - bad.flip(0).float().argmax(0)).float()
    stable = torch.where(bad.any(0), last_bad + 1, torch.zeros_like(last_bad))
    stable = torch.where(succ[-1], stable, full)
    red = comp <= 0.1 * comp[0]
    t90 = torch.where(red.any(0), red.float().argmax(0).float(), full)
    return dict(succ_final=succ[-1].float(), speed=1 - stable / T, speed_hit=1 - hit / T,
                speed90=1 - t90 / T, ttc=stable, comp_final=comp[-1], comp0=comp[0])


def time_per_step(fn, D, lr, reps=3, steps=30):
    """Median ms/step over `reps` repetitions after a warm-up (a single run is noisy and includes CUDA warm-up)."""
    optimize(fn, D, lr, steps=10)
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.time()
        optimize(fn, D, lr, steps=steps)
        torch.cuda.synchronize()
        ts.append((time.time() - t0) / steps * 1e3)
    return float(np.median(ts))


def run_loss(fn, D, lr, timeit=False):
    """Optimise all cameras of D with `fn`. Returns (per-camera metrics DataFrame, success-vs-step curve, ms/step)."""
    Rc = optimize(fn, D, lr)
    m = metrics_from_record(Rc)
    df = pd.DataFrame({k: v.cpu().numpy() for k, v in m.items()})
    df["sid"], df["rot0"], df["s"] = D["sid"], D["rot0"], D["s"]
    ts = torch.arange(T_STEPS + 1, device=dev)[:, None]
    curve = ((m["ttc"][None] <= ts) & m["succ_final"].bool()[None]).float().mean(1).cpu().numpy()
    ms = time_per_step(fn, D, lr) if timeit else float("nan")
    return df, curve, ms


# ======================================================================================
# Learning-rate tuning results (written by exp2, read by exp2-exp5)
# ======================================================================================
def tuning_path():
    return cache_dir() / "exp2_tuning.csv"


def best_lr_table(tune):
    """One row per loss: the learning rate with the highest tuning score."""
    return tune.loc[tune.groupby("loss").score.idxmax()].set_index("loss")


def load_best_lr():
    if not tuning_path().exists():
        raise FileNotFoundError(f"{tuning_path()} not found; run exp2_convergence.py --stage run first")
    return best_lr_table(pd.read_csv(tuning_path()))


def chosen_losses(best):
    """Losses reported in the final comparisons: best Hard / Soft / SoftGrid (by tuning score) + fixed ones."""
    pick = lambda g: best[best.group == g].score.idxmax()
    return [pick("ours-hard"), pick("ours-soft"), pick("ours-softgrid")] + FIXED_CHOSEN
