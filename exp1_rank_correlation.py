"""Experiment 1: does a camera loss rank random cameras like image similarity to the GT image does?

run     : renders all scenes, embeds the images with DINOv2 / CLIP, evaluates every loss on every camera and
          caches the result per scene (results/cache/scene_XXX.csv/.pt). The cache is also the input of
          experiments 2-5.
analyze : rank metrics (Spearman, Kendall, NDCG@k, top-k overlap) per scene, bootstrap CIs, Friedman test,
          paired comparison of our losses against the baselines (Holm-corrected), summary figure.
Outputs : results/exp1/*.csv and results/exp1/exp1_summary.png
"""
import glob
import json
import os
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

from config import (CFG, LOSS_VERSION, SIGMAS, TEST_SCENES, TUNE_SCENES, ZF_SOFT, cache_dir, exp_dir, seed_all)
from losses import LOSS_GROUP, LOSSES
from scene import build_scene_data
from utils import boot_ci, embed, holm, n_cams, pose_errors, rank_metrics, run_stages

K_TOP = 30
BINS = {"near": (0, 1 / 3), "mid": (1 / 3, 2 / 3), "far": (2 / 3, 1.0001)}
LOSS_COLS = list(LOSSES) + ["Photo-MSE"]
GROUP = dict(LOSS_GROUP, **{"Photo-MSE": "baseline"})


# ======================================================================================
# run: build the per-scene cache
# ======================================================================================
def _scene_meta():
    meta = dict(loss_version=LOSS_VERSION, cfg={k: CFG[k] for k in ("seed", "img", "n_cams", "min_cov")},
                losses=list(LOSSES))
    return json.loads(json.dumps(meta))


def _prepare_cache():
    """Delete scene files that were produced with other settings / loss versions."""
    cache = cache_dir()
    meta_path = cache / "scene_meta.json"
    files = glob.glob(str(cache / "scene_*.csv")) + glob.glob(str(cache / "scene_*.pt"))
    if files and not (meta_path.exists() and json.load(open(meta_path)) == _scene_meta()):
        print("scene cache is stale (settings or loss version changed) -> regenerating")
        for f in files:
            os.remove(f)
    json.dump(_scene_meta(), open(meta_path, "w"))


def run_scene(seed):
    cache = cache_dir()
    csv, pt = cache / f"scene_{seed:03d}.csv", cache / f"scene_{seed:03d}.pt"
    if csv.exists() and pt.exists():
        return
    d = build_scene_data(seed)
    cams, gt, sc = d["cams"], d["gt"], d["sc"]
    row = dict(scene=seed, cam=np.arange(n_cams(cams)), s=d["s"], cov=d["cov"])
    for w in ("dino", "clip"):
        eg, ec = embed(d["img_gt"], w), embed(d["imgs"], w)
        row[f"sim_{w}"] = (ec @ eg.T)[:, 0].cpu().numpy()
    row["Photo-MSE"] = ((d["imgs"] - d["img_gt"]) ** 2).mean((1, 2, 3)).cpu().numpy()
    with torch.no_grad():
        for name, (fn, _) in LOSSES.items():
            row[name] = fn(cams, gt, sc).cpu().numpy()
    row.update({k: v.cpu().numpy() for k, v in pose_errors(cams, gt).items()})
    torch.save(dict(cams={k: v.cpu() for k, v in cams.items()}, gt={k: v.cpu() for k, v in gt.items()},
                    s=d["s"], pts=sc["pts"].cpu(), cov=d["cov"]), pt)
    pd.DataFrame(row).to_csv(csv, index=False)
    del d
    torch.cuda.empty_cache()


def run():
    seed_all(CFG["seed"])
    _prepare_cache()
    t0 = time.time()
    for seed in range(CFG["n_scenes"]):
        run_scene(seed)
        print(f"scene {seed:2d} done | {time.time() - t0:.0f}s", end="\r")
    print("\nfinished")


# ======================================================================================
# analyze
# ======================================================================================
def compute_rank_table(df):
    rows = []
    for sc_id, g in df.groupby("scene"):
        for enc in ("dino", "clip"):
            sim = g[f"sim_{enc}"].values
            for L in LOSS_COLS:
                r = dict(scene=sc_id, enc=enc, loss=L)
                r.update(dict(zip(["rho", "tau", "ndcg", "overlap"], rank_metrics(g[L].values, sim, K_TOP))))
                for b, (lo, hi) in BINS.items():
                    m = (g.s.values >= lo) & (g.s.values < hi)
                    r[f"rho_{b}"] = rank_metrics(g[L].values[m], sim[m], K_TOP)[0]
                rows.append(r)
    return pd.DataFrame(rows)


def pivot(M, enc="dino", metric="rho", scenes=TEST_SCENES):
    return M[(M.enc == enc) & M.scene.isin(scenes)].pivot(index="scene", columns="loss", values=metric)[LOSS_COLS]


def summary(M, enc="dino", metric="rho", scenes=TEST_SCENES):
    P = pivot(M, enc, metric, scenes)
    cis = {c: boot_ci(P[c]) for c in P}
    out = pd.DataFrame({"group": pd.Series(GROUP), "mean": P.mean(),
                        "ci_lo": [cis[c][0] for c in P], "ci_hi": [cis[c][1] for c in P],
                        "avg_rank": P.rank(axis=1, ascending=False).mean()})
    return out.sort_values("mean", ascending=False)


def compare_to_baselines(M, A, enc="dino", metric="rho", scenes=TEST_SCENES):
    P = pivot(M, enc, metric, scenes)
    res = []
    for b in [c for c in LOSS_COLS if GROUP[c] == "baseline"]:
        d = (P[A] - P[b]).dropna()
        p = stats.wilcoxon(d).pvalue if (d != 0).any() else 1.0
        lo, hi = boot_ci(d)
        res.append(dict(ours=A, baseline=b, mean_diff=d.mean(), ci_lo=lo, ci_hi=hi, win_rate=(d > 0).mean(), p=p))
    r = pd.DataFrame(res)
    r["p_holm"] = holm(r.p.values)
    return r.sort_values("mean_diff", ascending=False)


def analyze():
    out = exp_dir("exp1")
    pd.set_option("display.width", 200, "display.max_rows", 100)
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(cache_dir() / "scene_*.csv")))], ignore_index=True)
    print(df.shape, "| scenes:", df.scene.nunique())
    M = compute_rank_table(df)
    M.to_csv(out / "rank_metrics_per_scene.csv", index=False)

    # Hyper-parameters of our families are selected on the tuning scenes only.
    Pt = pivot(M, "dino", "rho", TUNE_SCENES).mean()
    best = {g: Pt[[c for c in LOSS_COLS if GROUP[c] == g]].idxmax()
            for g in ("ours-hard", "ours-soft", "ours-softgrid")}
    best_hard, best_soft, best_sgrid = best["ours-hard"], best["ours-soft"], best["ours-softgrid"]
    print("best Hard (tune):", best_hard, "| best Soft (tune):", best_soft, "| best SoftGrid (tune):", best_sgrid)

    S = summary(M, "dino", "rho")
    S.to_csv(out / "summary_dino_rho.csv")
    print(S.round(3).to_string())
    P = pivot(M, "dino", "rho").dropna(axis=1)
    chi, p = stats.friedmanchisquare(*[P[c] for c in P])
    print(f"\nFriedman: chi2={chi:.1f}, p={p:.2e}, n_scenes={len(P)}, k={P.shape[1]}")

    comps = []
    for A in (best_hard, best_soft, best_sgrid, "Hard(zf=1)", "Soft(zf=1,s=1)"):
        r = compare_to_baselines(M, A)
        print(f"\n=== {A} vs baselines (Spearman, test scenes) ===")
        print(r.round(4).to_string(index=False))
        comps.append(r)
    pd.concat(comps).to_csv(out / "paired_vs_baselines_rho.csv", index=False)

    sel = [best_hard, best_soft, best_sgrid, "Dir-only", "P-MSE", "PoseNet(b=10)", "PoseNet(b=100)", "SE3-geo",
           "Ray-Plucker", "DenseRay", "Reproj-px", "Photo-MSE", "SoftSqrt(zf=1,s=4)", "SoftGridSqrt(zf=1,s=4,r=48)"]
    tab = pd.concat({f"{m}_{e}": pivot(M, e, m).mean()[sel] for e in ("dino", "clip")
                     for m in ("rho", "tau", "ndcg", "overlap")}, axis=1)
    tab.to_csv(out / "selected_losses_all_metrics.csv")
    print("\n", tab.round(3).to_string())
    a, b = summary(M, "dino")["mean"], summary(M, "clip")["mean"]
    print("\nagreement of method ordering DINO vs CLIP (Spearman):", round(stats.spearmanr(a, b.reindex(a.index))[0], 3))

    plot_summary(M, sel, out / "exp1_summary.png")


def plot_summary(M, sel, path):
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    H = [c for c in LOSS_COLS if GROUP[c] == "ours-hard"]
    Ph = pivot(M, "dino", "rho")[H]
    zfs = [float(re.search(r"zf=([\d.]+)", c).group(1)) for c in H]
    ci = np.array([boot_ci(Ph[c]) for c in H])
    ax[0].errorbar(zfs, Ph.mean(), yerr=[Ph.mean() - ci[:, 0], ci[:, 1] - Ph.mean()], marker="o", capsize=3)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("zf")
    ax[0].set_ylabel("mean Spearman")
    ax[0].set_title("Hard: sensitivity to zf")

    Sm = summary(M, "dino")["mean"]
    grid = np.array([[Sm[f"Soft(zf={zf},s={sg})"] for sg in SIGMAS] for zf in ZF_SOFT])
    im = ax[1].imshow(grid, cmap="viridis")
    plt.colorbar(im, ax=ax[1])
    ax[1].set_xticks(range(len(SIGMAS)))
    ax[1].set_xticklabels(SIGMAS)
    ax[1].set_yticks(range(len(ZF_SOFT)))
    ax[1].set_yticklabels(ZF_SOFT)
    ax[1].set_xlabel("sigma")
    ax[1].set_ylabel("zf")
    ax[1].set_title("Soft: mean Spearman")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax[1].text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)

    w = 0.8 / len(sel)
    for k, L in enumerate(sel):
        vals = [pivot(M, "dino", f"rho_{b}")[L].mean() for b in BINS]
        ax[2].bar(np.arange(3) + k * w, vals, w, label=L)
    ax[2].set_xticks(np.arange(3) + .4)
    ax[2].set_xticklabels(["near (s<1/3)", "mid", "far (s>2/3)"])
    ax[2].set_ylabel("mean Spearman")
    ax[2].legend(fontsize=7)
    ax[2].set_title("By perturbation level")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run_stages(__doc__, run, analyze)
