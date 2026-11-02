"""Experiment 3: how does the image similarity to the GT image evolve during the camera optimisation?

run     : optimises a random subset of the test cameras with each chosen loss, stores the camera every
          SNAP_EVERY steps, renders the snapshots and computes the DINOv2 similarity to the GT image.
analyze : per-scene means with bootstrap CIs over scenes, stratified by perturbation level s; figures and tables.
Needs   : exp1 (scene cache) and exp2 (tuned learning rates).
Outputs : results/exp3/simtraj.npz, exp3_simtraj_per_loss.png, exp3_simtraj_compare.png, exp3_simtraj_table_*.csv
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from config import CFG, T_STEPS, TEST_SCENES, exp_dir, seed_all
from losses import LOSSES
from optim import chosen_losses, load_best_lr, load_split, optimize
from scene import make_scene, render_batch
from utils import boot_ci, embed, idx_cams, run_stages, subset_D

SNAP_EVERY = 50
N_SNAP = T_STEPS // SNAP_EVERY + 1          # steps 0, 50, ..., 500
CAMS_PER_SCENE = 80
SUBSET_SEED = 123


def run():
    seed_all(CFG["seed"])
    out = exp_dir("exp3")
    best = load_best_lr()
    chosen = chosen_losses(best)
    Dtest = load_split(TEST_SCENES)

    rng = np.random.default_rng(SUBSET_SEED)
    sub_idx = []
    for s_ in TEST_SCENES:
        ids = np.where(Dtest["sid"] == s_)[0]
        sub_idx.append(ids[rng.permutation(len(ids))[:CAMS_PER_SCENE]])
    sub_idx = np.sort(np.concatenate(sub_idx))
    Dsub = subset_D(Dtest, sub_idx)
    sid_sub, s_sub = Dsub["sid"], Dsub["s"]
    print("subset cams:", len(sub_idx))

    # 1) optimise and keep the cameras every SNAP_EVERY steps
    snaps = {}
    for L in chosen:
        sn = []
        optimize(LOSSES[L][0], Dsub, float(best.loc[L, "lr"]), steps=T_STEPS, snaps=sn, snap_every=SNAP_EVERY)
        snaps[L] = [{k: v.cpu() for k, v in s.items()} for s in sn]

    # 2) render snapshots scene by scene (each scene is built once) and embed
    SIM = {L: np.full((N_SNAP, len(sub_idx)), np.nan, dtype=np.float32) for L in chosen}
    t0 = time.time()
    for si, s_ in enumerate(TEST_SCENES):
        sc = make_scene(s_)
        cols = np.where(sid_sub == s_)[0]
        ct = torch.as_tensor(cols)
        gt1 = idx_cams(Dsub["gt"], [int(cols[0])])                  # the GT camera is identical for the whole scene
        img_gt, _ = render_batch(sc["mesh"], gt1)
        e_gt = embed(img_gt, "dino")[0]
        for L in chosen:
            c0 = {k: v[ct].to(Dsub["gt"]["R"].device) for k, v in snaps[L][0].items()}
            for k, sn in enumerate(snaps[L]):
                c = {kk: v[ct].to(c0["R"].device) for kk, v in sn.items()}
                fin = (torch.isfinite(c["K"]).all(1).all(1) & torch.isfinite(c["R"]).all(1).all(1)
                       & torch.isfinite(c["t"]).all(1))
                # a diverged (NaN/inf) camera is replaced by its initial camera for rendering, similarity set to NaN
                c = {kk: torch.where(fin.view(-1, *[1] * (c[kk].dim() - 1)), c[kk], c0[kk]) for kk in c}
                im, _ = render_batch(sc["mesh"], c)
                sm = (embed(im, "dino") @ e_gt).cpu().numpy()
                sm[~fin.cpu().numpy()] = np.nan
                SIM[L][k, cols] = sm
        del sc
        torch.cuda.empty_cache()
        print(f"scene {si + 1}/{len(TEST_SCENES)} | {time.time() - t0:.0f}s", end="\r")
    np.savez_compressed(out / "simtraj.npz", sid=sid_sub, s=s_sub, steps=np.arange(N_SNAP) * SNAP_EVERY,
                        chosen=np.array(chosen), **{f"sim::{L}": v for L, v in SIM.items()})
    print("\nrender done")


def analyze():
    out = exp_dir("exp3")
    z = np.load(out / "simtraj.npz")
    sid_sub, s_sub, steps = z["sid"], z["s"], z["steps"]
    chosen = [str(c) for c in z["chosen"]]
    SIM = {L: z[f"sim::{L}"] for L in chosen}

    strata = {"all": np.ones(len(s_sub), bool), "near (s<1/3)": s_sub < 1 / 3,
              "mid (1/3<=s<2/3)": (s_sub >= 1 / 3) & (s_sub < 2 / 3), "far (s>=2/3)": s_sub >= 2 / 3}
    scenes = [s_ for s_ in TEST_SCENES]

    def curve(L, mask):
        P = np.array([np.nanmean(SIM[L][:, mask & (sid_sub == s_)], axis=1)
                      for s_ in scenes if (mask & (sid_sub == s_)).any()])
        lo = np.array([boot_ci(P[:, k], B=2000)[0] for k in range(P.shape[1])])
        hi = np.array([boot_ci(P[:, k], B=2000)[1] for k in range(P.shape[1])])
        return np.nanmean(P, 0), lo, hi

    CURVES = {(L, n): curve(L, m) for L in chosen for n, m in strata.items()}
    ccol = {"all": "k", "near (s<1/3)": "tab:green", "mid (1/3<=s<2/3)": "tab:orange", "far (s>=2/3)": "tab:red"}

    # Figure A: one panel per loss
    nr = int(np.ceil(len(chosen) / 4))
    fig, axs = plt.subplots(nr, 4, figsize=(22, 4.4 * nr))
    axs = axs.ravel()
    for ax_, L in zip(axs, chosen):
        for n in strata:
            m, lo, hi = CURVES[(L, n)]
            ax_.plot(steps, m, marker="o", color=ccol[n], label=n, lw=2.6 if n == "all" else 1.3)
            ax_.fill_between(steps, lo, hi, color=ccol[n], alpha=.12)
        ax_.set_title(L)
        ax_.set_xlabel("optimization step")
        ax_.set_ylabel("DINOv2 similarity to GT image")
        ax_.set_ylim(top=1.0)
        ax_.grid(alpha=.3)
    axs[0].legend(fontsize=8)
    for ax_ in axs[len(chosen):]:
        ax_.axis("off")
    plt.tight_layout()
    plt.savefig(out / "exp3_simtraj_per_loss.png", dpi=150)
    plt.close(fig)

    # Figure B: all losses in one panel per stratum
    fig, axs = plt.subplots(2, 2, figsize=(18, 11))
    axs = axs.ravel()
    for ax_, n in zip(axs, strata):
        for L in chosen:
            m, lo, hi = CURVES[(L, n)]
            ax_.plot(steps, m, marker="o", label=L)
            ax_.fill_between(steps, lo, hi, alpha=.08)
        ax_.set_title(n)
        ax_.set_xlabel("optimization step")
        ax_.set_ylabel("DINOv2 similarity to GT image")
        ax_.set_ylim(top=1.0)
        ax_.grid(alpha=.3)
    axs[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "exp3_simtraj_compare.png", dpi=150)
    plt.close(fig)

    # Tables: mean similarity every SNAP_EVERY steps, and the mean over steps 50..500 (normalised AUC) per stratum
    rows = []
    for L in chosen:
        for n in strata:
            m = CURVES[(L, n)][0]
            rows.append(dict(loss=L, stratum=n, **{f"step{s}": v for s, v in zip(steps, m)}, auc=m[1:].mean()))
    Sdf = pd.DataFrame(rows)
    pd.set_option("display.width", 220, "display.max_rows", 200)
    per_step = Sdf[Sdf.stratum == "all"].set_index("loss").drop(columns="stratum")
    auc = Sdf.pivot(index="loss", columns="stratum", values="auc")[list(strata)]
    per_step.to_csv(out / "exp3_simtraj_table_per_step.csv")
    auc.to_csv(out / "exp3_simtraj_table_auc.csv")
    print("\n=== all cameras: mean similarity per step ===")
    print(per_step.round(3).to_string())
    print("\n=== auc (mean over steps 50..500) per stratum ===")
    print(auc.round(3).to_string())


if __name__ == "__main__":
    run_stages(__doc__, run, analyze)
