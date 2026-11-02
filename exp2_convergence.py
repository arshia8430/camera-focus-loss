"""Experiment 2: how fast and how reliably do the losses drive a perturbed camera back to the GT camera?

run     : (1) tunes the Adam learning rate of every loss on the tuning scenes (cached in results/cache with a
              meta file, re-tuned automatically if any setting or LOSS_VERSION changes),
          (2) optimises all test cameras with the tuned learning rates and stores per-camera convergence metrics,
              success-vs-step curves and the measured ms/step of every loss.
analyze : success / speed tables with scene-cluster bootstrap CIs, Friedman test, paired comparison against the
          baselines, results binned by perturbation level s and by initial image similarity, common-set speed
          analysis in steps and in ms, paired speed-up table.
Needs   : the scene cache from exp1_rank_correlation.py (run stage).
Outputs : results/exp2/*  (+ results/cache/exp2_tuning.csv)
"""
import json
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pytorch3d.transforms import matrix_to_quaternion
from scipy import stats

from config import (CFG, LOSS_VERSION, LR_GRID, LOGF_T, ROT_T, SIGMAS, T_SCALE, T_STEPS, TEST_SCENES, TRANS_FRAC,
                    TUNE_SCENES, ZF_SOFT, cache_dir, exp_dir, seed_all)
from losses import LOSS_GROUP, LOSSES
from optim import (best_lr_table, chosen_losses, load_best_lr, load_split, run_loss, tuning_path)
from utils import binned, boot_ci, cluster_binned, holm, run_stages

NUM_EDGES = 50            # bin edges for the "vs s" and "vs similarity" plots
MIN_N = 15                # minimum cameras per bin to be shown
COMMON_MIN_SUCC = 0.5     # a loss takes part in the common-set comparison if it converges for >= 50% of cameras


# ======================================================================================
# run
# ======================================================================================
def _tuning_meta():
    return json.loads(json.dumps(dict(LR_GRID=LR_GRID, T_SCALE=T_SCALE, T_STEPS=T_STEPS, W=CFG["LR_ERROR_WEIGHT"],
                                      ROT_T=ROT_T, LOGF_T=LOGF_T, TRANS_FRAC=TRANS_FRAC, loss_version=LOSS_VERSION)))


def tune_learning_rates(Dtune):
    """Grid search over LR_GRID for every loss. score = mean(speed) - W * median(log10(final error))."""
    path = tuning_path()
    meta_path = path.with_suffix(".csv.meta.json")
    meta = _tuning_meta()
    tune = pd.DataFrame(columns=["loss", "group", "lr", "speed", "succ", "score"])
    if path.exists():
        if meta_path.exists() and json.load(open(meta_path)) == meta:
            tune = pd.read_csv(path)
        else:
            print("tuning settings / loss version changed (or no metadata) -> re-tuning everything")
    todo = [n for n in LOSSES if n not in set(tune.loss)]
    print("to tune:", len(todo), "losses x", len(LR_GRID), "lrs")
    rows, t0 = [], time.time()
    for k, name in enumerate(todo):
        fn, grp = LOSSES[name]
        for lr in LR_GRID:
            df_, _, _ = run_loss(fn, Dtune, lr)
            score = df_.speed.mean() - CFG["LR_ERROR_WEIGHT"] * np.median(np.log10(df_.comp_final.clip(1e-3, 1e6)))
            rows.append(dict(loss=name, group=grp, lr=lr, speed=df_.speed.mean(), succ=df_.succ_final.mean(),
                             score=score))
        print(f"[{k + 1}/{len(todo)}] {name:30s} {time.time() - t0:.0f}s", end="\r")
    if rows:
        tune = pd.concat([tune, pd.DataFrame(rows)], ignore_index=True)
    tune.to_csv(path, index=False)
    json.dump(meta, open(meta_path, "w"))
    return tune


def run():
    seed_all(CFG["seed"])
    out = exp_dir("exp2")
    tune = tune_learning_rates(load_split(TUNE_SCENES))
    best = best_lr_table(tune)
    pd.set_option("display.width", 220, "display.max_rows", 100)
    print(best[["group", "lr", "speed", "succ"]].round(3).to_string())
    edge = list(best.index[best.lr.isin([LR_GRID[0], LR_GRID[-1]])])
    print("\nlosses whose best lr lies on the edge of the grid (grid may be too narrow):", edge)

    Dtest = load_split(TEST_SCENES)
    res, curves, times = [], {}, {}
    for k, (name, (fn, grp)) in enumerate(LOSSES.items()):
        df_, cv, ms = run_loss(fn, Dtest, float(best.loc[name, "lr"]), timeit=True)
        df_["loss"] = name
        df_["cam"] = df_.groupby("sid").cumcount()          # camera index inside its scene (same order as exp1 cache)
        res.append(df_)
        curves[name] = cv
        times[name] = ms
        print(f"[{k + 1}/{len(LOSSES)}] {name:30s} succ={df_.succ_final.mean():.3f} speed={df_.speed.mean():.3f}",
              end="\r")
    pd.concat(res, ignore_index=True).to_csv(out / "exp2_cameras.csv.gz", index=False)
    np.savez(out / "curves.npz", **curves)
    json.dump(times, open(out / "times.json", "w"), indent=1)
    print("\ndone")


# ======================================================================================
# analyze
# ======================================================================================
def load_results():
    out = exp_dir("exp2")
    R2 = pd.read_csv(out / "exp2_cameras.csv.gz")
    times = json.load(open(out / "times.json"))
    curves = dict(np.load(out / "curves.npz"))
    simtab = pd.concat([pd.read_csv(cache_dir() / f"scene_{s_:03d}.csv")[["scene", "cam", "sim_dino", "sim_clip"]]
                        .rename(columns={"scene": "sid"}) for s_ in TEST_SCENES])
    R2 = R2.merge(simtab, on=["sid", "cam"], how="left")
    assert R2.sim_dino.notna().all(), "similarity merge is incomplete (stale scene cache?)"
    R2["inv_steps"] = R2.succ_final / R2.ttc.clip(lower=1)
    return R2, times, curves


def overall_report(R2, best, times, curves, out):
    """Success / speed tables, tests and the six-panel overview figure."""
    G = LOSS_GROUP
    names = list(LOSSES)
    best_hard = best[best.group == "ours-hard"].score.idxmax()
    best_soft = best[best.group == "ours-soft"].score.idxmax()
    best_sg = best[best.group == "ours-softgrid"].score.idxmax()
    print("best Hard (tune):", best_hard, "| best Soft (tune):", best_soft, "| best SoftGrid (tune):", best_sg)

    scene_tab = R2.groupby(["loss", "sid"])[["succ_final", "speed", "speed_hit", "speed90"]].mean().reset_index()

    def piv(metric):
        return scene_tab.pivot(index="sid", columns="loss", values=metric)[names]

    def summary(metric):
        P = piv(metric)
        ci = {c: boot_ci(P[c]) for c in P.columns}
        return pd.DataFrame({"group": pd.Series(G).reindex(P.columns), "lr": best.lr.reindex(P.columns),
                             "mean": P.mean(), "ci_lo": pd.Series({c: ci[c][0] for c in P.columns}),
                             "ci_hi": pd.Series({c: ci[c][1] for c in P.columns}),
                             "avg_rank": P.rank(axis=1, ascending=False).mean(),
                             "ms/step": pd.Series(times).reindex(P.columns)}).sort_values("mean", ascending=False)

    def compare(A, metric):
        P = piv(metric)
        res = []
        for b in [c for c in names if G[c] == "baseline"]:
            d = (P[A] - P[b]).dropna()
            p = stats.wilcoxon(d).pvalue if (d != 0).any() else 1.0
            lo, hi = boot_ci(d)
            res.append(dict(ours=A, baseline=b, mean_diff=d.mean(), ci_lo=lo, ci_hi=hi, win_rate=(d > 0).mean(), p=p))
        r = pd.DataFrame(res)
        r["p_holm"] = holm(r.p.values)
        return r.sort_values("mean_diff", ascending=False)

    pd.set_option("display.width", 220, "display.max_rows", 100)
    for m in ("speed", "speed_hit", "succ_final"):
        S = summary(m)
        S.to_csv(out / f"summary_{m}.csv")
        print(f"\n######## {m} (test scenes) ########")
        print(S.round(3).to_string())
        P = piv(m)
        chi, p = stats.friedmanchisquare(*[P[c] for c in P])
        print(f"Friedman chi2={chi:.1f} p={p:.2e} n={len(P)}")
    comps = []
    for A in (best_hard, best_soft, best_sg, "Hard(zf=1)"):
        for m in ("speed", "succ_final"):
            r = compare(A, m)
            r["metric"] = m
            print(f"\n=== {A} vs baselines | {m} ===")
            print(r.round(4).to_string(index=False))
            comps.append(r)
    pd.concat(comps).to_csv(out / "paired_vs_baselines.csv", index=False)

    sel = chosen_losses(best)
    print("\n", R2[R2.loss.isin(sel)].groupby("loss")[["speed_hit", "speed", "succ_final"]].mean()
          .reindex(sel).round(3).to_string())

    R2 = R2.copy()
    R2["bin"] = pd.cut(R2.rot0, [0, 30, 90, 180], labels=["<30°", "30-90°", ">90°"])
    print("\n cameras count in bins:", R2[R2.loss == best_hard].bin.value_counts().to_dict())
    bt = (R2[R2.loss.isin(sel)].groupby(["loss", "bin", "sid"])[["succ_final", "speed"]].mean()
          .groupby(["loss", "bin"]).mean().unstack("bin"))
    bt.to_csv(out / "by_initial_rotation_error.csv")
    print(bt.reindex(sel).round(3).to_string())

    # ---- six-panel figure ----
    fig, ax = plt.subplots(2, 3, figsize=(20, 10))
    ax = ax.ravel()
    for L in sel:
        ax[0].plot(curves[L], label=L)
    ax[0].set_xlabel("step")
    ax[0].set_ylabel("fraction stably converged")
    ax[0].legend(fontsize=7)
    ax[0].set_title("Cumulative stable convergence (test)")
    Hn = [c for c in names if G[c] == "ours-hard"]
    zfs = [float(re.search(r"zf=([\d.]+)", c).group(1)) for c in Hn]
    Ph = piv("speed")[Hn]
    ci = np.array([boot_ci(Ph[c]) for c in Hn])
    ax[1].errorbar(zfs, Ph.mean(), yerr=[Ph.mean() - ci[:, 0], ci[:, 1] - Ph.mean()], marker="o", capsize=3)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("zf")
    ax[1].set_ylabel("speed (stable)")
    ax[1].set_title("Hard: zf sensitivity")
    Sm = summary("speed")["mean"]
    grid = np.array([[Sm[f"Soft(zf={zf},s={sg})"] for sg in SIGMAS] for zf in ZF_SOFT])
    im = ax[2].imshow(grid, cmap="viridis")
    plt.colorbar(im, ax=ax[2])
    ax[2].set_xticks(range(len(SIGMAS)))
    ax[2].set_xticklabels(SIGMAS)
    ax[2].set_yticks(range(len(ZF_SOFT)))
    ax[2].set_yticklabels(ZF_SOFT)
    ax[2].set_xlabel("sigma")
    ax[2].set_ylabel("zf")
    ax[2].set_title("Soft: speed (stable)")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax[2].text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", color="w", fontsize=7)
    w = 0.8 / len(sel)
    for k, L in enumerate(sel):
        ax[3].bar(np.arange(3) + k * w, bt.loc[L, "speed"].values, w, label=L)
        ax[4].bar(np.arange(3) + k * w, bt.loc[L, "succ_final"].values, w)
    for a, t in ((ax[3], "speed (stable) by initial rotation error"),
                 (ax[4], "final success rate by initial rotation error")):
        a.set_xticks(np.arange(3) + .4)
        a.set_xticklabels(["<30°", "30-90°", ">90°"])
        a.set_title(t)
    ax[5].barh(sel, [times[s] for s in sel])
    ax[5].set_xlabel(f"ms / step (batch of {len(R2[R2.loss == sel[0]])} cams)")
    ax[5].set_title("cost")
    plt.tight_layout()
    plt.savefig(out / "exp2_summary.png", dpi=150)
    plt.close(fig)

    # ---- diagnostic: is a quaternion loss failing scene-wise (GT quaternion close to the w = 0 cut)? ----
    qw = {s_: float(matrix_to_quaternion(torch.load(cache_dir() / f"scene_{s_:03d}.pt", weights_only=False)
                                         ["gt"]["R"][:1])[0, 0]) for s_ in TEST_SCENES}
    tab = (R2[R2.loss.isin(["P-MSE", "PoseNet(b=10)", "PoseNet(b=100)", "SE3-geo"])]
           .groupby(["sid", "loss"]).succ_final.mean().unstack("loss").round(2))
    tab["gt_w"] = pd.Series(qw).round(3)
    tab.sort_values("gt_w").to_csv(out / "per_scene_success_vs_gt_quat_w.csv")
    print("\nper-scene success (sorted by GT quaternion w):")
    print(tab.sort_values("gt_w").to_string())


def plot_vs(R2, chosen, xcol, edges, xlabel, path):
    """Mean 1/steps-to-convergence and success rate as a function of `xcol`."""
    sub = R2[R2.loss.isin(chosen)]
    xs = binned(sub[sub.loss == chosen[0]], xcol, xcol, edges)[0]
    fig, ax = plt.subplots(1, 2, figsize=(18, 6))
    T_inv, T_sr = {}, {}
    for L in chosen:
        d = sub[sub.loss == L]
        m, lo, hi = binned(d, xcol, "inv_steps", edges)
        sr, slo, shi = binned(d, xcol, "succ_final", edges)
        ax[0].plot(xs, m, marker="o", label=L)
        ax[0].fill_between(xs, lo, hi, alpha=.10)
        ax[1].plot(xs, sr, marker="o", label=L)
        ax[1].fill_between(xs, slo, shi, alpha=.10)
        T_inv[L], T_sr[L] = m, sr
    ax[0].set_ylabel("mean of 1 / steps-to-stable-convergence  (failed = 0)")
    ax[1].set_ylabel("fraction of cameras that converge (and stay)")
    for a in ax:
        a.set_xlabel(xlabel)
        a.grid(alpha=.3)
    ax[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    idxs = [f"{x:.3f}" for x in xs]
    for title, T in (("mean 1/steps", T_inv), ("success rate", T_sr)):
        T = pd.DataFrame(T, index=idxs)
        T.to_csv(str(path).replace(".png", f"_{title.replace(' ', '_').replace('/', '_over_')}.csv"))
        print(f"\n--- {title} per bin of {xcol} ---")
        print(T.round(4).to_string())
        print(pd.DataFrame({"winner": T.idxmax(axis=1), "best": T.max(axis=1).round(4),
                            "runner_up": T.apply(lambda r: r.nlargest(2).index[1], axis=1)}).to_string())


def speed_analysis(R2, CH, times, unit, out):
    """Convergence time in `unit` ("steps" or "ms") on cameras that converge, on the common set, and paired speed-ups."""
    R2 = R2.copy()
    ulab = "steps" if unit == "steps" else "ms (batch compute)"
    scale = {L: (times[L] if unit == "ms" else 1.0) for L in R2.loss.unique()}
    R2["t_conv"] = np.where(R2.succ_final == 1, R2.ttc.clip(lower=1) * R2.loss.map(scale), np.nan)
    R2["inv_t"] = 1.0 / R2["t_conv"]

    succ_rate = R2[R2.loss.isin(CH)].groupby("loss").succ_final.mean()
    COMP = [L for L in CH if succ_rate[L] >= COMMON_MIN_SUCC]
    W = R2[R2.loss.isin(COMP)].pivot(index=["sid", "cam"], columns="loss", values="inv_t")[COMP].dropna()
    R2["in_common"] = pd.MultiIndex.from_frame(R2[["sid", "cam"]]).isin(W.index)
    n_all = R2.drop_duplicates(["sid", "cam"]).shape[0]
    print(f"\n######## unit = {unit} ########")
    print(f"competitive losses (success>={COMMON_MIN_SUCC}): {COMP}")
    print(f"common set: {len(W)} cameras of {n_all} ({100 * len(W) / n_all:.1f}%)")
    W = W.join(R2.drop_duplicates(["sid", "cam"]).set_index(["sid", "cam"])[["s", "sim_dino"]])

    rows = []
    for L in CH:
        d = R2[R2.loss == L]
        c = d[d.succ_final == 1]
        cc = d[d.in_common]
        rows.append(dict(loss=L, success=d.succ_final.mean(),
                         cond_median_t=c.t_conv.median(), cond_typical_t=(1 / c.inv_t.mean() if len(c) else np.nan),
                         common_median_t=cc.t_conv.median() if L in COMP else np.nan,
                         common_typical_t=(1 / cc.inv_t.mean() if (L in COMP and len(cc)) else np.nan)))
    overall = pd.DataFrame(rows).set_index("loss")
    overall.to_csv(out / f"overall_time_{unit}.csv")
    print(f"\n=== overall (time in {ulab}; typical = 1/mean(1/t)) ===")
    print(overall.round(3).to_string())

    col = {L: plt.cm.tab20(i) for i, L in enumerate(CH)}

    def speed_plots(xcol, edges, xlabel, fname):
        nb = len(edges) - 1
        uniq = R2.drop_duplicates(["sid", "cam"])
        bb = pd.cut(uniq[xcol], edges, include_lowest=True, labels=False)
        xs = uniq.groupby(bb)[xcol].mean().reindex(range(nb)).values
        blabel = [f"[{edges[k]:.2f},{edges[k + 1]:.2f}]" for k in range(nb)]

        def med_bin(d, c):
            b = pd.cut(d[xcol], edges, include_lowest=True, labels=False)
            g = d.groupby(b)[c]
            med = g.median().reindex(range(nb)).values
            cnt = g.size().reindex(range(nb)).fillna(0).values
            return np.where(cnt >= MIN_N, med, np.nan)

        fig, ax = plt.subplots(2, 3, figsize=(22, 12))
        ax = ax.ravel()
        cond_m, succ_m, com_m, ncom = {}, {}, {}, None
        for L in CH:
            d = R2[R2.loss == L]
            c = d[d.succ_final == 1]
            m, lo, hi, _ = cluster_binned(c, xcol, "inv_t", edges, MIN_N)
            cond_m[L] = m
            ax[0].plot(xs, m, marker="o", color=col[L], label=L)
            ax[0].fill_between(xs, lo, hi, color=col[L], alpha=.10)
            sm, slo, shi, _ = cluster_binned(d, xcol, "succ_final", edges, 1)
            succ_m[L] = sm
            ax[1].plot(xs, sm, marker="o", color=col[L], label=L)
            ax[1].fill_between(xs, slo, shi, color=col[L], alpha=.10)
            ax[3].plot(xs, med_bin(c, "t_conv"), marker="o", color=col[L], label=L)
            if L in COMP:
                cc = d[d.in_common]
                m2, lo2, hi2, ncom = cluster_binned(cc, xcol, "inv_t", edges, MIN_N)
                com_m[L] = m2
                ax[2].plot(xs, m2, marker="o", color=col[L], label=L)
                ax[2].fill_between(xs, lo2, hi2, color=col[L], alpha=.10)
                ax[4].plot(xs, med_bin(cc, "t_conv"), marker="o", color=col[L], label=L)
        ax[0].set_title("(a) conditional: mean 1/t over cameras THAT converged (own set)")
        ax[1].set_title("(b) fraction of cameras that converge (and stay)")
        ax[2].set_title("(c) common set: mean 1/t on cameras converged by ALL competitive losses")
        ax[3].set_title("(d) conditional: median convergence time")
        ax[4].set_title("(e) common set: median convergence time")
        ax[5].bar(range(nb), ncom if ncom is not None else np.zeros(nb), color="gray")
        ax[5].set_xticks(range(nb))
        ax[5].set_xticklabels(blabel, rotation=60, fontsize=7)
        ax[5].set_title("(f) # cameras in common set per bin")
        for i in (0, 2):
            ax[i].set_ylabel(f"mean of 1/t   [1/{ulab}]")
        ax[1].set_ylabel("success rate")
        for i in (3, 4):
            ax[i].set_ylabel(f"median t  [{ulab}]")
            ax[i].set_yscale("log")
        for i in range(5):
            ax[i].set_xlabel(xlabel)
            ax[i].grid(alpha=.3)
        ax[0].legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out / fname, dpi=150)
        plt.close(fig)

        Tc = pd.DataFrame(com_m, index=blabel)
        Tc.to_csv(out / fname.replace(".png", "_common_mean_inv_t.csv"))
        print(f"\n######## common set | mean 1/t per bin of {xcol} ########")
        print(Tc.round(5).to_string())
        print(f"\n--- typical time (= 1/mean(1/t), {ulab}) on the common set ---")
        print((1 / Tc).round(1).to_string())
        print(f"\n--- conditional typical time (own converged set, {ulab}) ---")
        print((1 / pd.DataFrame(cond_m, index=blabel)).round(1).to_string())
        print("\n--- success rate ---")
        print(pd.DataFrame(succ_m, index=blabel).round(3).to_string())
        Wb = W.copy()
        Wb["bin"] = pd.cut(Wb[xcol], edges, include_lowest=True, labels=False)
        wins = []
        for k in range(nb):
            r = Tc.iloc[k].dropna().sort_values(ascending=False)
            if len(r) < 2:
                continue
            a, b = r.index[:2]
            Wk = Wb[Wb.bin == k].assign(diff=lambda x: x[a] - x[b]).reset_index()
            md, lo, hi, n = cluster_binned(Wk, xcol, "diff", [-1e9, 1e9], MIN_N)
            wins.append(dict(bin=blabel[k], n_common=int(n[0]), winner=a, runner_up=b,
                             t_winner=1 / r[a], t_runner_up=1 / r[b], diff_1_over_t=md[0], ci_lo=lo[0], ci_hi=hi[0],
                             significant="*" if lo[0] > 0 else ""))
        wins = pd.DataFrame(wins)
        wins.to_csv(out / fname.replace(".png", "_winners.csv"), index=False)
        print("\n--- winner per bin (common set; paired cluster-bootstrap CI of winner - runner-up in 1/t) ---")
        print(wins.round(5).to_string(index=False))

    speed_plots("s", np.linspace(0, 1, NUM_EDGES), "s  (0 = near GT camera, 1 = fully random)",
                f"exp2_speed_vs_s_{unit}.png")
    u = R2.drop_duplicates(["sid", "cam"])
    edges_sim = np.unique(np.quantile(u.sim_dino, np.linspace(0, 1, NUM_EDGES)))
    speed_plots("sim_dino", edges_sim, "initial DINOv2 similarity (camera image vs GT image)",
                f"exp2_speed_vs_sim_{unit}.png")

    # ---- paired comparison of OUR losses vs baselines (log-ratio of 1/t per scene; cluster bootstrap; Holm) ----
    STRATA = {"near (s<1/3)": (0, 1 / 3), "mid": (1 / 3, 2 / 3), "far (s>=2/3)": (2 / 3, 1.0001), "all": (0, 1.0001)}
    OURS = [L for L in CH if L.startswith(("Hard", "Soft"))]
    BASE = [L for L in ["P-MSE", "SE3-geo", "PoseNet(b=10)", "PoseNet(b=100)"] if L in W.columns]
    rows_ = []
    for A in OURS:
        if A not in W.columns:
            continue
        for B in BASE:
            for nm, (lo_s, hi_s) in STRATA.items():
                m_ = (W.s >= lo_s) & (W.s < hi_s)
                lr_ = np.log(W.loc[m_, A] / W.loc[m_, B])          # > 0  <=>  ours is faster
                g_ = lr_.groupby(level="sid")
                per = g_.mean()[g_.size() >= 3]
                if len(per) < 5:
                    continue
                p_ = stats.wilcoxon(per).pvalue if (per != 0).any() else 1.0
                lo_, hi_ = boot_ci(per.values)
                rows_.append(dict(unit=unit, ours=A, base=B, stratum=nm, n_scenes=len(per),
                                  speedup_pct=100 * (np.exp(per.mean()) - 1), ci_lo=100 * (np.exp(lo_) - 1),
                                  ci_hi=100 * (np.exp(hi_) - 1), p=p_))
    CMP = pd.DataFrame(rows_)
    if len(CMP):
        CMP["p_holm"] = holm(CMP.p.values)
    CMP.to_csv(out / f"paired_speedup_vs_baselines_{unit}.csv", index=False)
    pd.set_option("display.width", 220, "display.max_rows", 400)
    print(f"\n######## paired speed-up of ours vs baselines [{unit}] (positive = ours faster; common set) ########")
    print(CMP.round(3).to_string(index=False))


def analyze():
    out = exp_dir("exp2")
    R2, times, curves = load_results()
    best = load_best_lr()
    overall_report(R2, best, times, curves, out)

    chosen = chosen_losses(best)
    plot_vs(R2, chosen, "s", np.linspace(0, 1, NUM_EDGES), "s  (0 = near GT camera, 1 = fully random)",
            out / "exp2_vs_s.png")
    uniq = R2.drop_duplicates(["sid", "cam"])
    edges_sim = np.unique(np.quantile(uniq.sim_dino, np.linspace(0, 1, NUM_EDGES)))
    plot_vs(R2, chosen, "sim_dino", edges_sim, "initial DINOv2 cosine similarity (camera image vs GT image)",
            out / "exp2_vs_sim.png")

    # If PoseNet(b=100) behaves exactly like PoseNet(b=10) (e.g. both never converge) it is dropped from the speed study.
    CH = list(chosen)
    a_ = R2[R2.loss == "PoseNet(b=10)"].ttc.values
    b_ = R2[R2.loss == "PoseNet(b=100)"].ttc.values
    if len(a_) == len(b_) and np.array_equal(a_, b_):
        CH.remove("PoseNet(b=100)")
    print("losses in the speed study:", CH)
    for unit in ("steps", "ms"):
        speed_analysis(R2, CH, times, unit, out)


if __name__ == "__main__":
    run_stages(__doc__, run, analyze)
