"""Experiment 4: sensitivity of the loss ranking to the convergence thresholds.

The convergence criterion of experiment 2 (rotation < ROT_T deg, translation < TRANS_FRAC of the GT distance,
|log focal| < LOGF_T) is replaced by a strict and a loose variant. The error records of one optimisation run are
re-thresholded, so no extra optimisation is needed per threshold setting.

run     : optimises all test cameras with every chosen loss, evaluates the convergence metrics under each
          threshold setting and stores success / steps-to-convergence per camera.
analyze : success rate, typical convergence time (1 / mean(1/t) on the common set) and the speed advantage over
          P-MSE for each setting.
Needs   : exp1 (scene cache) and exp2 (tuned learning rates).
Outputs : results/exp4/metrics.npz, results/exp4/sensitivity_<setting>.csv
"""
import numpy as np
import pandas as pd
import torch

from config import CFG, LOGF_T, ROT_T, TEST_SCENES, TRANS_FRAC, exp_dir, seed_all
from losses import LOSSES
from optim import chosen_losses, load_best_lr, load_split, metrics_from_record, optimize
from utils import run_stages

# (rotation [deg], translation fraction, |log focal|)
SENS = {"strict": (0.5, 0.002, 0.005), "default": (ROT_T, TRANS_FRAC, LOGF_T), "loose": (5.0, 0.02, 0.05)}
COMMON_MIN_SUCC = 0.5


def run():
    seed_all(CFG["seed"])
    out = exp_dir("exp4")
    best = load_best_lr()
    chosen = chosen_losses(best)
    Dtest = load_split(TEST_SCENES)

    store = {"chosen": np.array(chosen)}
    for L in chosen:
        rec = optimize(LOSSES[L][0], Dtest, float(best.loc[L, "lr"]))     # (T+1, 3, B), normalised by default thresholds
        for tag, (rt, tf, lt) in SENS.items():
            scale = torch.tensor([ROT_T / rt, TRANS_FRAC / tf, LOGF_T / lt], device=rec.device).view(1, 3, 1)
            m = metrics_from_record(rec * scale)
            store[f"succ::{tag}::{L}"] = m["succ_final"].cpu().numpy()
            store[f"ttc::{tag}::{L}"] = m["ttc"].cpu().numpy().clip(1)
        del rec
        torch.cuda.empty_cache()
        print("done:", L)
    np.savez_compressed(out / "metrics.npz", **store)


def analyze():
    out = exp_dir("exp4")
    z = np.load(out / "metrics.npz")
    chosen = [str(c) for c in z["chosen"]]
    pd.set_option("display.width", 200)
    for tag, (rt, tf, lt) in SENS.items():
        succ = {L: z[f"succ::{tag}::{L}"] for L in chosen}
        ttc = {L: z[f"ttc::{tag}::{L}"] for L in chosen}
        comp = [L for L in chosen if succ[L].mean() >= COMMON_MIN_SUCC]
        common = np.all([succ[L] > 0 for L in comp], axis=0)
        o = pd.DataFrame({"success": {L: succ[L].mean() for L in chosen},
                          "typical_steps": {L: (1 / np.mean(1 / ttc[L][common]) if (L in comp and common.any())
                                                else np.nan) for L in chosen}})
        o["faster_than_P-MSE_%"] = 100 * (o.loc["P-MSE", "typical_steps"] / o.typical_steps - 1)
        print(f"\n=== threshold sensitivity [{tag}]: rot<{rt} deg, trans<{tf * 100:.1f}%, logf<{lt} "
              f"| common n={common.sum()} ===")
        print(o.round(3).to_string())
        o.to_csv(out / f"sensitivity_{tag}.csv")


if __name__ == "__main__":
    run_stages(__doc__, run, analyze)
