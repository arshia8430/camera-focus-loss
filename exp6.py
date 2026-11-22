"""Experiment 7: when can geometric / scene knowledge beat a (normalised) MSE?

Six small stress tests. In every test the initial camera is the GT camera plus a *structured* error whose size is
tied to the local scene depth Z (distance of the nearest visible content, estimated from the cached scene points):

  1_local        small mixed error                       control: every loss is ~quadratic here, expect a tie
  2_strafe_turn  sideways move + turn that keeps looking at the same point (image almost aligned,
                 parameters far)                         translation/rotation coupling
  3_dolly_zoom   move along the axis + focal change that keeps the object size (same story, axial)
  4_focal        large focal-length (FOV) error + small pose error      intrinsics in the loss
  5_big_rotation 90 deg rotation about a random axis                     rotation geometry / basin of attraction
  6_mixed_far    large translation + rotation + focal error together     "hard random" case

Compared losses (all from losses.py except the two scene-aware ones; each has its own learning rate, tuned on the
tuning scenes with the same scenarios):
  P-MSE, SE3-geo                  baselines of the paper
  best Hard / Soft / SoftGrid     our losses with fixed z_f (picked by the exp2 tuning table)
  Hard-Z, Soft-Z(s=1)             our Hard / Soft with z_f = local depth Z of each camera (scene-aware; built from
                                  losses.loss_hard / loss_soft, no new formula)

Two convergence notions are measured with the exp2 machinery (speed = 1 - time_to_converge / T, 'succ' = converged at end):
  param : rotation < ROT_T, translation < thr_t, |log f| < LOGF_T            (what exp2 uses)
  px    : mean pixel error of a fronto-parallel plane at depth Z (5x5 grid) < PX_T pixels   (image alignment, no renderer)

run     : builds the cameras, tunes the learning rates, optimises all cameras with every loss, writes per-camera CSV,
          and renders GIFs (GT | P-MSE | SE3-geo | best of ours) for GIF_SCENES scenes per scenario.
analyze : tables with scene-cluster bootstrap CIs, paired differences against P-MSE and SE3-geo, bar figure.
Needs   : exp1 (scene cache) and exp2 (tuning table, only to pick the best Hard / Soft / SoftGrid names).
Outputs : results/exp7/*.csv, results/exp7/exp7_summary.png, results/exp7/gifs/*.gif, results/exp7/gifs.zip
Runtime : a few minutes for the optimisation (no rendering), plus the GIF part (scene assembly ~30 s per GIF scene).
Set EXP7_GIFS=0 to skip the GIFs.
"""
import json
import os
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from pytorch3d.transforms import axis_angle_to_matrix

from config import (CFG, IMG, LOGF_T, LOSS_VERSION, LR_GRID, ROT_T, T_SCALE, T_STEPS, TEST_SCENES, TUNE_SCENES,
                    cache_dir, dev, exp_dir, seed_all)
from exp5_gifs import camera_params
from losses import LOSSES, focus_points, loss_hard, loss_soft
from optim import chosen_losses, load_best_lr, load_split, metrics_from_record
from scene import make_scene, render_batch
from utils import boot_ci, cat_cams, idx_cams, make_K, n_cams, pose_errors, project, run_stages, to_tensor

EXP7_VERSION = "v1"
N_TUNE, N_TEST = 20, 40            # cameras per scene and scenario (tuning scenes / test scenes)
PX_T = 1.0                         # pixel threshold of the image-alignment convergence
GIF_EVERY, GIF_MS, GIF_SCENES = 5, 70, 2
MAKE_GIFS = os.environ.get("EXP7_GIFS", "1") != "0"
BASE = ["P-MSE", "SE3-geo"]

# 5x5 pixel grid used for the plane-alignment error
_u = torch.linspace(0, IMG - 1, 5, device=dev)
_uu, _vv = torch.meshgrid(_u, _u, indexing="xy")
P_PX = torch.stack([_uu.flatten(), _vv.flatten(), torch.ones(25, device=dev)])


# ======================================================================================
# Scene-aware variants of our losses (local to this experiment; losses.py and LOSS_VERSION are untouched)
# ======================================================================================
EXTRA = {
    "Hard-Z": lambda c, t, x: loss_hard(c, t, x["zf"][:, None, None]),
    "Soft-Z(s=1)": lambda c, t, x: loss_soft(c, t, x["zf"][:, None, None], 1.0),
}
SCENE_AWARE = ["Hard-Z", "Soft-Z(s=1)"]


# ======================================================================================
# Scenarios: structured perturbations of the GT camera, expressed in the GT camera frame (x right, y down, z forward)
# ======================================================================================
def _unit(rng, B):
    v = rng.normal(size=(B, 3))
    return to_tensor(v / np.linalg.norm(v, axis=1, keepdims=True))


def _sign(rng, B):
    return to_tensor(rng.choice([-1.0, 1.0], size=B))


def _apply(gt, d_cam, w_cam, logf):
    """Camera = GT moved by d_cam (GT camera frame), rotated by axis-angle w_cam, focal scaled by exp(logf)."""
    return dict(K=make_K(gt["K"][:, 0, 0] * torch.exp(logf)),
                R=gt["R"] @ axis_angle_to_matrix(w_cam),
                t=gt["t"] + (gt["R"] @ d_cam[:, :, None])[:, :, 0])


def p_local(gt, Z, rng):
    B = len(Z)
    return _apply(gt, _unit(rng, B) * (0.05 * Z)[:, None], _unit(rng, B) * np.deg2rad(3), _sign(rng, B) * 0.03)


def p_strafe_turn(gt, Z, rng):
    """Move sideways by 0.4 Z and turn by atan(0.4) so that the camera still looks at the GT fixation point (depth Z)."""
    B = len(Z)
    phi = to_tensor(rng.uniform(0, 2 * np.pi, B))
    dl = 0.4 * Z
    d = torch.stack([dl * torch.cos(phi), dl * torch.sin(phi), torch.zeros_like(dl)], 1)
    axis = torch.stack([torch.sin(phi), -torch.cos(phi), torch.zeros_like(phi)], 1)
    return _apply(gt, d, axis * torch.atan2(dl, Z)[:, None], torch.zeros(B, device=dev))


def p_dolly_zoom(gt, Z, rng):
    """Move along the optical axis (back 0.5 Z or forward 0.4 Z) and rescale the focal so the plane at depth Z keeps its size."""
    B = len(Z)
    m = torch.where(_sign(rng, B) > 0, -0.5 * Z, 0.4 * Z)           # forward displacement
    d = torch.stack([torch.zeros_like(m), torch.zeros_like(m), m], 1)
    return _apply(gt, d, torch.zeros(B, 3, device=dev), torch.log(1 - m / Z))


def p_focal(gt, Z, rng):
    B = len(Z)
    return _apply(gt, _unit(rng, B) * (0.03 * Z)[:, None], _unit(rng, B) * np.deg2rad(1), _sign(rng, B) * 0.5)


def p_big_rotation(gt, Z, rng):
    B = len(Z)
    return _apply(gt, _unit(rng, B) * (0.03 * Z)[:, None], _unit(rng, B) * np.deg2rad(90), _sign(rng, B) * 0.03)


def p_mixed_far(gt, Z, rng):
    B = len(Z)
    return _apply(gt, _unit(rng, B) * (0.6 * Z)[:, None], _unit(rng, B) * np.deg2rad(45), _sign(rng, B) * 0.5)


PERTURB = {"1_local": p_local, "2_strafe_turn": p_strafe_turn, "3_dolly_zoom": p_dolly_zoom,
           "4_focal": p_focal, "5_big_rotation": p_big_rotation, "6_mixed_far": p_mixed_far}
SCEN = list(PERTURB)
SCEN_TXT = {"1_local": "small error (control)", "2_strafe_turn": "sideways move + turn, same fixation point",
            "3_dolly_zoom": "axial move + focal change, same object size", "4_focal": "FOV error (x1.65 focal)",
            "5_big_rotation": "90 deg rotation", "6_mixed_far": "large translation + rotation + FOV"}


# ======================================================================================
# Data
# ======================================================================================
def _depth(g, pts):
    """Approximate distance of the nearest visible content: 25th percentile of the depth of the cached scene points that
    fall inside the GT image (occlusion is ignored; falls back to all points in front, then to 0.5)."""
    pc = (pts - g["t"][0]) @ g["R"][0]
    z = pc[:, 2]
    zc = z.clamp_min(1e-6)
    u = g["K"][0, 0, 0] * pc[:, 0] / zc + IMG / 2
    v = g["K"][0, 1, 1] * pc[:, 1] / zc + IMG / 2
    sel = (z > 0.02) & (u >= 0) & (u < IMG) & (v >= 0) & (v < IMG)
    zz = z[sel] if int(sel.sum()) >= 8 else z[z > 0.02]
    if len(zz) < 8:
        return 0.5
    return float(torch.quantile(zz, 0.25).clamp(0.1, 1.5))


def plane_px_err(c, plane, px_gt):
    """Mean pixel error of the GT fronto-parallel plane grid (B,25,3) seen by camera c versus by the GT camera."""
    return (project(c, plane) - px_gt).norm(dim=-1).mean(1)


def make_D(scenes, n_per, seed):
    """All scenarios x `scenes` x `n_per` cameras in one batch (same keys as optim.load_split plus scen / plane / px_gt)."""
    D0 = load_split(scenes)
    sid_u, first = np.unique(D0["sid"], return_index=True)
    S = len(first)
    f_ = torch.as_tensor(first, device=dev)
    gt1 = {k: v[f_] for k, v in D0["gt"].items()}
    pts1, thr1 = D0["ctx"]["pts"][f_], D0["thr_t"][f_]
    Z1 = torch.tensor([_depth(idx_cams(gt1, [i]), pts1[i]) for i in range(S)], dtype=torch.float32, device=dev)
    rng = np.random.default_rng(seed)
    rep = torch.as_tensor(np.repeat(np.arange(S), n_per), device=dev)
    cams, gts, Zs, scen = [], [], [], []
    for si, name in enumerate(SCEN):
        gt, Z = {k: v[rep] for k, v in gt1.items()}, Z1[rep]
        cams.append(PERTURB[name](gt, Z, rng))
        gts.append(gt)
        Zs.append(Z)
        scen.append(np.full(S * n_per, si))
    nS = len(SCEN)
    cams, gt, Z = cat_cams(cams), cat_cams(gts), torch.cat(Zs)
    D = dict(cams=cams, gt=gt, ctx=dict(pts=pts1[rep].repeat(nS, 1, 1), zf=Z), thr_t=thr1[rep].repeat(nS),
             sid=np.tile(np.repeat(sid_u, n_per), nS), scen=np.concatenate(scen))
    plane = focus_points(gt, P_PX, Z[:, None, None])[0].transpose(1, 2).contiguous()
    D["plane"], D["px_gt"] = plane, project(gt, plane)
    return D


def take(D, idx):
    """Sub-batch of a dataset dict (tensors, numpy arrays and nested dicts)."""
    it = torch.as_tensor(np.asarray(idx), device=dev)

    def f(x):
        if isinstance(x, dict):
            return {k: f(v) for k, v in x.items()}
        return x[it] if torch.is_tensor(x) else x[np.asarray(idx)]
    return {k: f(v) for k, v in D.items()}


def init_table(D):
    e = pose_errors(D["cams"], D["gt"])
    Z = D["ctx"]["zf"]
    return pd.DataFrame(dict(scen=[SCEN[i] for i in D["scen"]], sid=D["sid"], Z=Z.cpu().numpy(),
                             rot0=e["rot_deg"].cpu().numpy(), trans_over_Z=(e["trans"] / Z).cpu().numpy(),
                             logf0=e["logf"].cpu().numpy(),
                             px0=plane_px_err(D["cams"], D["plane"], D["px_gt"]).cpu().numpy()))


# ======================================================================================
# Optimisation (same parametrisation / schedule / criteria as optim.optimize, plus the plane pixel error)
# ======================================================================================
def optimize7(fn, D, lr, steps=None, snaps=None, snap_every=GIF_EVERY):
    steps = T_STEPS if steps is None else steps
    cam0, gt = D["cams"], D["gt"]
    B = n_cams(cam0)
    f0 = cam0["K"][:, 0, 0]
    dt = torch.zeros(B, 3, device=dev, requires_grad=True)
    w = torch.zeros(B, 3, device=dev, requires_grad=True)
    lf = torch.zeros(B, device=dev, requires_grad=True)
    opt = torch.optim.Adam([dt, w, lf], lr=lr, eps=1e-12)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.01)
    rec, px = [], []
    for it in range(steps + 1):
        c = dict(K=make_K(f0 * torch.exp(lf)), R=cam0["R"] @ axis_angle_to_matrix(w), t=cam0["t"] + T_SCALE * dt)
        if snaps is not None and it % snap_every == 0:
            snaps.append({k: v.detach().clone() for k, v in c.items()})
        with torch.no_grad():
            e = pose_errors(c, gt)
            comp = torch.stack([e["rot_deg"] / ROT_T, e["trans"] / D["thr_t"], e["logf"] / LOGF_T])
            rec.append(torch.nan_to_num(comp, nan=1e6, posinf=1e6, neginf=1e6))
            px.append(torch.nan_to_num(plane_px_err(c, D["plane"], D["px_gt"]), nan=1e6, posinf=1e6, neginf=1e6))
        if it == steps:
            break
        loss = fn(c, gt, D["ctx"]).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    return torch.stack(rec), torch.stack(px)


def run_loss7(fn, D, lr):
    Rc, px = optimize7(fn, D, lr)
    m, mp = metrics_from_record(Rc), metrics_from_record(px[:, None, :] / PX_T)
    cols = dict(succ=m["succ_final"], speed=m["speed"], comp_final=m["comp_final"],
                px_succ=mp["succ_final"], px_speed=mp["speed"], px_final=px[-1])
    df = pd.DataFrame({k: v.cpu().numpy() for k, v in cols.items()})
    df["scen"] = [SCEN[i] for i in D["scen"]]
    df["sid"] = D["sid"]
    return df


def all_losses():
    """name -> fn(cam, tgt, ctx) for every compared loss, plus the names of our fixed-z_f losses."""
    best = load_best_lr()
    ours = chosen_losses(best)[:3]
    fns = {n: LOSSES[n][0] for n in BASE + ours}
    fns.update(EXTRA)
    order = BASE + ours + SCENE_AWARE
    return {n: fns[n] for n in order}, ours


def tune_lrs(fns, Dtune):
    """Grid search over LR_GRID on the tuning scenes (all six scenarios together); score as in exp2."""
    path = cache_dir() / "exp7_tuning.csv"
    meta_path = path.with_suffix(".csv.meta.json")
    meta = json.loads(json.dumps(dict(v=EXP7_VERSION, lv=LOSS_VERSION, lrs=LR_GRID, steps=T_STEPS, n=N_TUNE,
                                      W=CFG["LR_ERROR_WEIGHT"], scen=SCEN, seed=CFG["seed"])))
    tune = pd.DataFrame(columns=["loss", "lr", "speed", "succ", "score"])
    if path.exists() and meta_path.exists() and json.load(open(meta_path)) == meta:
        tune = pd.read_csv(path)
    rows = []
    for k, name in enumerate([n for n in fns if n not in set(tune.loss)]):
        for lr in LR_GRID:
            df = run_loss7(fns[name], Dtune, lr)
            score = df.speed.mean() - CFG["LR_ERROR_WEIGHT"] * np.median(np.log10(df.comp_final.clip(1e-3, 1e6)))
            rows.append(dict(loss=name, lr=lr, speed=df.speed.mean(), succ=df.succ.mean(), score=score))
        print(f"tuned {name:30s}", end="\r")
    if rows:
        tune = pd.concat([tune, pd.DataFrame(rows)], ignore_index=True)
    tune.to_csv(path, index=False)
    json.dump(meta, open(meta_path, "w"))
    best = tune.loc[tune.groupby("loss").score.idxmax()].set_index("loss")
    edge = [n for n in fns if best.loc[n, "lr"] in (LR_GRID[0], LR_GRID[-1])]
    if edge:
        print("\nbest lr on the edge of the grid:", edge)
    return {n: float(best.loc[n, "lr"]) for n in fns}


# ======================================================================================
# GIFs
# ======================================================================================
def _u8(x):
    return (x.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()


def make_gifs(D, R, fns, lrs, ours_set, out):
    gif_dir = out / "gifs"
    gif_dir.mkdir(exist_ok=True)
    sp = R.groupby(["scen", "loss"]).speed.mean().unstack()
    try:
        font = ImageFont.load_default(size=13)
    except TypeError:
        font = ImageFont.load_default()
    PW, HDR, FTR = IMG, 40, 70
    sids = list(dict.fromkeys(D["sid"].tolist()))[:GIF_SCENES]
    scenes, rows = {}, []
    for si, name in enumerate(SCEN):
        cols = ["P-MSE", "SE3-geo", sp.loc[name, ours_set].astype(float).idxmax()]
        for g in sids:
            idx = np.where((D["scen"] == si) & (D["sid"] == g))[0]
            if len(idx) == 0:
                continue
            Dj = take(D, [int(idx[0])])
            if g not in scenes:
                scenes[g] = make_scene(int(g))
            sc = scenes[g]
            snaps = {}
            for L in cols:
                sn = []
                optimize7(fns[L], Dj, lrs[L], snaps=sn, snap_every=GIF_EVERY)
                snaps[L] = sn
            nF = len(snaps[cols[0]])
            gt = {k: v.expand(nF, *v.shape[1:]) for k, v in Dj["gt"].items()}
            img_gt, _ = render_batch(sc["mesh"], idx_cams(Dj["gt"], [0]))
            thr = float(Dj["thr_t"][0])
            per = {}
            for L in cols:
                cf = cat_cams(snaps[L])
                fin = (torch.isfinite(cf["K"]).flatten(1).all(1) & torch.isfinite(cf["R"]).flatten(1).all(1)
                       & torch.isfinite(cf["t"]).all(1))
                c0 = idx_cams(cf, [0])
                cf = {k: torch.where(fin.view(-1, *[1] * (v.dim() - 1)), v, c0[k].expand_as(v)) for k, v in cf.items()}
                im, _ = render_batch(sc["mesh"], cf)
                e = pose_errors(cf, gt)
                rot, tr, lf = [e[k].cpu().numpy() for k in ("rot_deg", "trans", "logf")]
                pxe = plane_px_err(cf, Dj["plane"].expand(nF, -1, -1), Dj["px_gt"].expand(nF, -1, -1)).cpu().numpy()
                per[L] = dict(img=_u8(im), rot=rot, tr=tr, px=pxe, params=[camera_params(cf, k) for k in range(nF)],
                              ok=(rot < ROT_T) & (tr < thr) & (lf < LOGF_T))
            gt_arr, gt_params = _u8(img_gt)[0], camera_params(Dj["gt"], 0)
            frames = []
            for k in range(nF):
                cv = Image.new("RGB", (PW * (1 + len(cols)), HDR + PW + FTR), "white")
                d = ImageDraw.Draw(cv)
                d.text((6, 4), f"{name} | scene {g} | step {k * GIF_EVERY:3d}/{T_STEPS}", fill="black", font=font)
                d.text((6, 21), SCEN_TXT[name], fill=(80, 80, 80), font=font)
                cv.paste(Image.fromarray(gt_arr), (0, HDR))
                d.text((4, HDR + PW + 3), "GT camera", fill="black", font=font)
                for n, line in enumerate(gt_params):
                    d.text((4, HDR + PW + 20 + n * 15), line, fill="black", font=font)
                for p, L in enumerate(cols, start=1):
                    q = per[L]
                    cv.paste(Image.fromarray(q["img"][k]), (p * PW, HDR))
                    col = (0, 170, 0) if q["ok"][k] else ((230, 140, 0) if q["px"][k] < PX_T else (200, 0, 0))
                    d.rectangle([p * PW, HDR, p * PW + PW - 1, HDR + PW - 1], outline=col, width=3)
                    d.text((p * PW + 4, HDR + PW + 3), L[:30], fill="black", font=font)
                    d.text((p * PW + 4, HDR + PW + 20), f"plane err {q['px'][k]:.1f}px", fill="black", font=font)
                    d.text((p * PW + 4, HDR + PW + 35), f"rot {q['rot'][k]:.1f}deg  t {q['tr'][k]:.3f}", fill="black", font=font)
                    d.text((p * PW + 4, HDR + PW + 50), "green=params ok, orange=image ok only", fill=(110, 110, 110), font=font)
                frames.append(cv)
            frames += [frames[-1]] * 12
            fn_ = gif_dir / f"{name}_scene{g}.gif"
            frames[0].save(fn_, save_all=True, append_images=frames[1:], duration=GIF_MS, loop=0)
            r = dict(scenario=name, scene=int(g), px0=round(float(per[cols[0]]["px"][0]), 2),
                     rot0=round(float(per[cols[0]]["rot"][0]), 1))
            for L in cols:
                r[f"{L}|px_end"] = round(float(per[L]["px"][-1]), 2)
                r[f"{L}|rot_end"] = round(float(per[L]["rot"][-1]), 2)
                r[f"{L}|params_ok"] = bool(per[L]["ok"][-1])
            rows.append(r)
            print(f"gif {fn_.name} done", end="\r")
    pd.DataFrame(rows).to_csv(out / "gif_summary.csv", index=False)
    print("\nzip:", shutil.make_archive(str(out / "gifs"), "zip", gif_dir))


# ======================================================================================
# run
# ======================================================================================
def run():
    seed_all(CFG["seed"])
    out = exp_dir("exp7")
    fns, ours = all_losses()
    lrs = tune_lrs(fns, make_D(TUNE_SCENES, N_TUNE, CFG["seed"] + 1))
    print("\nlearning rates:", lrs)
    D = make_D(TEST_SCENES, N_TEST, CFG["seed"] + 2)
    init_table(D).to_csv(out / "exp7_init.csv", index=False)
    res, t0 = [], time.time()
    for k, (name, fn) in enumerate(fns.items()):
        df = run_loss7(fn, D, lrs[name])
        df["loss"] = name
        res.append(df)
        print(f"[{k + 1}/{len(fns)}] {name:30s} succ={df.succ.mean():.3f} speed={df.speed.mean():.3f} "
              f"px_speed={df.px_speed.mean():.3f}  {time.time() - t0:.0f}s")
    R = pd.concat(res, ignore_index=True)
    R.to_csv(out / "exp7_cameras.csv.gz", index=False)
    json.dump(dict(order=list(fns), ours=ours, scene_aware=SCENE_AWARE, lr=lrs, n_test=N_TEST),
              open(out / "meta.json", "w"), indent=1)
    if MAKE_GIFS:
        make_gifs(D, R, fns, lrs, ours + SCENE_AWARE, out)


# ======================================================================================
# analyze
# ======================================================================================
def _color(name, meta):
    if name in meta["scene_aware"]:
        return "#e07b00" if name == "Hard-Z" else "#f2b25c"
    if name in meta["ours"]:
        return ["#1f5fa8", "#4f8fd6", "#9cc3ee"][meta["ours"].index(name)]
    return {"P-MSE": "#555555", "SE3-geo": "#9a9a9a"}.get(name, "#777777")


def analyze():
    pd.set_option("display.width", 250, "display.max_columns", 40)
    out = exp_dir("exp7")
    R = pd.read_csv(out / "exp7_cameras.csv.gz")
    I = pd.read_csv(out / "exp7_init.csv")
    meta = json.load(open(out / "meta.json"))
    order = meta["order"]

    print("\n=== initial error per scenario (medians over cameras) ===")
    print(I.groupby("scen")[["Z", "rot0", "trans_over_Z", "logf0", "px0"]].median().round(3).to_string())

    def per_scene(metric, sc):
        return R[R.scen == sc].groupby(["sid", "loss"])[metric].mean().unstack()[order]

    tabs, pair = [], []
    for metric in ("speed", "px_speed", "succ", "px_succ"):
        for sc in SCEN:
            P = per_scene(metric, sc)
            for L in order:
                lo, hi = boot_ci(P[L].values)
                tabs.append(dict(metric=metric, scen=sc, loss=L, mean=P[L].mean(), ci_lo=lo, ci_hi=hi))
            if metric in ("speed", "px_speed"):
                for ref in ("P-MSE", "SE3-geo"):
                    for L in order:
                        if L == ref:
                            continue
                        d = (P[L] - P[ref]).values
                        lo, hi = boot_ci(d)
                        pair.append(dict(metric=metric, scen=sc, ours=L, ref=ref, diff=d.mean(), ci_lo=lo, ci_hi=hi,
                                         win_rate=(d > 0).mean(), sig=("+" if lo > 0 else "-" if hi < 0 else "")))
    T, PR = pd.DataFrame(tabs), pd.DataFrame(pair)
    T.to_csv(out / "exp7_table.csv", index=False)
    PR.to_csv(out / "exp7_paired.csv", index=False)

    for metric in ("speed", "px_speed", "succ"):
        print(f"\n=== {metric}: mean over scenes (higher = better) ===")
        print(T[T.metric == metric].pivot(index="scen", columns="loss", values="mean")[order].round(3).to_string())
    for ref in ("P-MSE", "SE3-geo"):
        for metric in ("speed", "px_speed"):
            q = PR[(PR.ref == ref) & (PR.metric == metric)]
            print(f"\n=== paired difference vs {ref} ({metric}); '+'/'-' = 95% CI excludes 0 ===")
            M = q.pivot(index="scen", columns="ours", values="diff")
            S = q.pivot(index="scen", columns="ours", values="sig")
            print(M.round(3).astype(str).add(S.fillna("")).reindex(columns=[c for c in order if c != ref]).to_string())

    fig, ax = plt.subplots(2, 1, figsize=(17, 8.5), sharex=True)
    wd = 0.8 / len(order)
    for a, metric, ttl in zip(ax, ("speed", "px_speed"), ("parameter convergence speed", "image (plane) convergence speed")):
        for j, L in enumerate(order):
            q = T[(T.metric == metric) & (T.loss == L)].set_index("scen").loc[SCEN]
            x = np.arange(len(SCEN)) + (j - (len(order) - 1) / 2) * wd
            a.bar(x, q["mean"], wd, yerr=[q["mean"] - q.ci_lo, q.ci_hi - q["mean"]], color=_color(L, meta),
                  label=L, error_kw=dict(lw=0.8))
        a.set_ylabel(ttl + "  (1 - t_conv/T)")
        a.grid(axis="y", alpha=.3)
    ax[0].legend(ncol=len(order), fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax[1].set_xticks(range(len(SCEN)))
    ax[1].set_xticklabels([f"{s}\n{SCEN_TXT[s]}" for s in SCEN], fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "exp7_summary.png", dpi=150)
    plt.close()
    print("\nwrote", out / "exp7_summary.png")


if __name__ == "__main__":
    run_stages(__doc__, run, analyze)