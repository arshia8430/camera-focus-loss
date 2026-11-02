"""Experiment 5: qualitative convergence GIFs (best SoftGrid vs P-MSE vs SE3-geo) for near / mid / far cameras.

Each GIF shows the GT image and the rendered image of every loss during the optimisation, with the similarity,
rotation / translation error and the camera parameters. The border is green when the camera is converged.

run     : picks GIF_N cameras per perturbation stratum (all from different scenes), optimises, renders, writes the
          GIFs, a summary table and a zip archive.
analyze : prints the saved summary table.
Needs   : exp1 (scene cache) and exp2 (tuned learning rates).
Outputs : results/exp5/gifs/*.gif, results/exp5/gif_summary.csv, results/exp5/gifs.zip
"""
import shutil

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from config import CFG, IMG, LOGF_T, ROT_T, T_STEPS, TEST_SCENES, exp_dir, seed_all
from losses import LOSSES
from optim import load_best_lr, load_split, optimize
from scene import make_scene, render_batch
from utils import cat_cams, embed, idx_cams, pose_errors, run_stages, subset_D

GIF_EVERY, GIF_N, GIF_MS, GIF_SEED = 5, 10, 70, 7
BASELINES_IN_GIF = ["P-MSE", "SE3-geo"]          # SE3-geo is the strongest baseline and is included on purpose
PW, HDR, FTR = IMG, 26, 105


def camera_params(cam, idx):
    """Text lines describing camera `idx`: centre in world coordinates, XYZ Euler angles (deg) and field of view."""
    R, t, K = cam["R"][idx], cam["t"][idx], cam["K"][idx]
    rot = Rotation.from_matrix(R.detach().cpu().numpy()).as_euler("xyz", degrees=True)
    fov = 2 * np.degrees(np.arctan(IMG / (2 * K[0, 0].item())))
    return (f"Pos: ({t[0].item():.2f}, {t[1].item():.2f}, {t[2].item():.2f})",
            f"Rot: ({rot[0]:.1f}, {rot[1]:.1f}, {rot[2]:.1f})",
            f"FOV: {fov:.1f} deg")


def _u8(x):
    return (x.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()


def _stable_step(ok):
    """First step after which the camera stays converged (NaN if it ends unconverged)."""
    if not ok[-1]:
        return np.nan
    bad = np.where(~ok)[0]
    return (bad[-1] + 1 if len(bad) else 0) * GIF_EVERY


def run():
    seed_all(CFG["seed"])
    out = exp_dir("exp5")
    gif_dir = out / "gifs"
    gif_dir.mkdir(exist_ok=True)
    best = load_best_lr()
    gif_ours = best[best.group == "ours-softgrid"].score.idxmax()
    gifl = [gif_ours] + BASELINES_IN_GIF
    print("GIF losses:", gifl, "| lr:", [float(best.loc[L, "lr"]) for L in gifl])

    Dtest = load_split(TEST_SCENES)
    rng = np.random.default_rng(GIF_SEED)
    s_all, sid_all = Dtest["s"], Dtest["sid"]
    strata = {"near": s_all < 1 / 3, "mid": (s_all >= 1 / 3) & (s_all < 2 / 3), "far": s_all >= 2 / 3}
    pick = {}
    for nm, m in strata.items():
        ch, used = [], set()
        for i in rng.permutation(np.where(m)[0]):
            if sid_all[i] not in used:
                ch.append(int(i))
                used.add(sid_all[i])
            if len(ch) == GIF_N:
                break
        pick[nm] = ch
    strat_list = [nm for nm in strata for _ in pick[nm]]
    sel_idx = np.array([i for nm in strata for i in pick[nm]])
    Dg = subset_D(Dtest, sel_idx)
    print("cameras:", len(sel_idx), "| scenes:", sorted(set(Dg["sid"].tolist())))

    snaps = {}
    for L in gifl:
        sn = []
        optimize(LOSSES[L][0], Dg, float(best.loc[L, "lr"]), steps=T_STEPS, snaps=sn, snap_every=GIF_EVERY)
        snaps[L] = sn
    nF = len(snaps[gifl[0]])
    print("frames per gif:", nF)

    try:
        font = ImageFont.load_default(size=13)
    except TypeError:
        font = ImageFont.load_default()

    scenes, counts, rows = {}, {nm: 0 for nm in strata}, []
    for j in range(len(sel_idx)):
        sid_j, sj, strat = int(Dg["sid"][j]), float(Dg["s"][j]), strat_list[j]
        if sid_j not in scenes:
            scenes[sid_j] = make_scene(sid_j)
        sc = scenes[sid_j]
        gt1 = idx_cams(Dg["gt"], [j])
        thr = float(Dg["thr_t"][j])
        img_gt, _ = render_batch(sc["mesh"], gt1)
        e_gt = embed(img_gt, "dino")[0]
        gt_arr = _u8(img_gt)[0]
        gt_params = camera_params(gt1, 0)

        per = {}
        for L in gifl:
            cf = cat_cams([idx_cams(s_, [j]) for s_ in snaps[L]])
            fin = (torch.isfinite(cf["K"]).flatten(1).all(1) & torch.isfinite(cf["R"]).flatten(1).all(1)
                   & torch.isfinite(cf["t"]).all(1))
            c0 = idx_cams(cf, [0])
            cf = {k: torch.where(fin.view(-1, *[1] * (v.dim() - 1)), v, c0[k].expand_as(v)) for k, v in cf.items()}
            params = [camera_params(cf, k) for k in range(nF)]
            im, _ = render_batch(sc["mesh"], cf)
            e = pose_errors(cf, gt1)
            rot, tr, lf = [e[k].cpu().numpy() for k in ("rot_deg", "trans", "logf")]
            per[L] = dict(img=_u8(im), sim=(embed(im, "dino") @ e_gt).cpu().numpy(), rot=rot, tr=tr,
                          ok=(rot < ROT_T) & (tr < thr) & (lf < LOGF_T), params=params)

        frames = []
        for k in range(nF):
            cv = Image.new("RGB", (PW * (1 + len(gifl)), HDR + PW + FTR), "white")
            d = ImageDraw.Draw(cv)
            d.text((6, 5), f"scene {sid_j} | s={sj:.2f} | step {k * GIF_EVERY:3d}/{T_STEPS}", fill="black", font=font)
            cv.paste(Image.fromarray(gt_arr), (0, HDR))
            d.text((4, HDR + PW + 3), "GT camera", fill="black", font=font)
            for n, line in enumerate(gt_params):
                d.text((4, HDR + PW + 20 + n * 15), line, fill="black", font=font)
            for p, L in enumerate(gifl, start=1):
                q = per[L]
                cv.paste(Image.fromarray(q["img"][k]), (p * PW, HDR))
                d.rectangle([p * PW, HDR, p * PW + PW - 1, HDR + PW - 1],
                            outline=(0, 170, 0) if q["ok"][k] else (200, 0, 0), width=3)
                d.text((p * PW + 4, HDR + PW + 3), L[:30], fill="black", font=font)
                d.text((p * PW + 4, HDR + PW + 20),
                       f"sim {q['sim'][k]:.3f}  rot {q['rot'][k]:.1f}deg  t {q['tr'][k]:.2f}", fill="black", font=font)
                for n, line in enumerate(q["params"][k]):
                    d.text((p * PW + 4, HDR + PW + 37 + n * 15), line, fill="black", font=font)
            frames.append(cv)
        frames += [frames[-1]] * 12                                        # hold the last frame
        fn_ = gif_dir / f"{strat}_{counts[strat]:02d}_scene{sid_j}_s{sj:.2f}.gif"
        frames[0].save(fn_, save_all=True, append_images=frames[1:], duration=GIF_MS, loop=0)
        counts[strat] += 1

        r = dict(stratum=strat, scene=sid_j, s=round(sj, 2), sim0=round(float(per[gifl[0]]["sim"][0]), 3))
        for L in gifl:
            r[f"{L}|sim_end"] = round(float(per[L]["sim"][-1]), 3)
            r[f"{L}|rot_end"] = round(float(per[L]["rot"][-1]), 2)
            r[f"{L}|ok"] = bool(per[L]["ok"][-1])
            r[f"{L}|stable_step"] = _stable_step(per[L]["ok"])
        rows.append(r)
        print(f"gif {j + 1}/{len(sel_idx)} done", end="\r")

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "gif_summary.csv", index=False)
    zp = shutil.make_archive(str(out / "gifs"), "zip", gif_dir)
    print("\nzip:", zp)


def analyze():
    pd.set_option("display.width", 250, "display.max_columns", 50)
    print(pd.read_csv(exp_dir("exp5") / "gif_summary.csv").to_string(index=False))


if __name__ == "__main__":
    run_stages(__doc__, run, analyze)
