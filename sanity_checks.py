"""Sanity checks to run before the experiments (needs a GPU).

1. every loss is ~0 at the ground-truth camera,
2. the quaternion losses are continuous (sign-invariant) on SO(3),
3. the closed form of Soft agrees with its numerical integral,
4. our pinhole projection agrees with PyTorch3D,
5. (optional, --plot) a grid of sample cameras and a summary of the initial pose errors.

Usage:  python sanity_checks.py [--plot] [--out results]
"""
import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pytorch3d.transforms import axis_angle_to_matrix

from config import CFG, exp_dir, seed_all, set_results_dir
from losses import LOSSES, loss_pmse, loss_soft, loss_soft_numeric
from scene import build_scene_data, to_p3d
from utils import n_cams, pose_errors, project


def check_self_loss(gt, sc):
    worst = {name: float(fn(gt, gt, sc).abs().max()) for name, (fn, _) in LOSSES.items()}
    for name, v in worst.items():
        if v > 1e-3:
            print("!! non-zero self-loss:", name, v)
    assert max(worst.values()) < 1e-3, "some loss is not ~0 at the ground truth"
    print(f"[ok] max self-loss over all {len(worst)} losses: {max(worst.values()):.2e}")


def check_quaternion_continuity(gt):
    """Rotating by 120 deg vs 121 deg about random axes must change P-MSE only slightly."""
    dev = gt["R"].device
    Rg = gt["R"].expand(64, -1, -1).contiguous()
    ax = F.normalize(torch.randn(64, 3, device=dev), dim=1)
    t, K = gt["t"].expand(64, -1), gt["K"].expand(64, -1, -1)
    cm = dict(R=Rg @ axis_angle_to_matrix(ax * math.radians(120)), t=t, K=K)
    cp = dict(R=Rg @ axis_angle_to_matrix(ax * math.radians(121)), t=t, K=K)
    d = (loss_pmse(cm, gt) - loss_pmse(cp, gt)).abs().max()
    assert d < 0.05, f"P-MSE is discontinuous ({float(d):.3f})"
    print(f"[ok] P-MSE continuity: max |dL| for a 1 deg step at 120 deg = {float(d):.4f}")


def check_closed_form(cams, gt):
    for zf, sg in [(1, 1.0), (2, 4.0), (0.5, 0.25)]:
        a = loss_soft(cams, gt, zf, sg)
        b = loss_soft_numeric(cams, gt, zf, sg)
        err = float(((a - b).abs() / b.abs().clamp(min=1e-9)).max())
        print(f"[info] zf={zf} sigma={sg}: max relative error closed-vs-numeric = {err:.2e}")


def check_projection(cams, sc):
    P3 = to_p3d(cams)
    pts = sc["pts"][None].expand(n_cams(cams), -1, -1)
    uv_p3d = P3.transform_points_screen(pts)[..., :2]
    Xc = (sc["pts"][None] - cams["t"][:, None]) @ cams["R"]
    ok = Xc[..., 2] > 0.5
    uv_me = project(cams, sc["pts"])
    print("[info] max |uv_pytorch3d - uv_ours| (px, points in front):", float((uv_p3d - uv_me)[ok].abs().max()))


def plot_overview(data, out):
    cams = data["cams"]
    order = np.argsort(data["s"])[np.linspace(0, n_cams(cams) - 1, 11).astype(int)]
    fig, ax = plt.subplots(2, 6, figsize=(18, 6))
    ax[0, 0].imshow(data["img_gt"][0].permute(1, 2, 0).cpu())
    ax[0, 0].set_title("GT")
    for k, i in enumerate(order):
        a = ax.flat[k + 1]
        a.imshow(data["imgs"][i].permute(1, 2, 0).cpu())
        a.set_title(f"s={data['s'][i]:.2f}")
    for a in ax.flat:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(out / "sanity_cameras.png", dpi=120)
    plt.close(fig)
    e = pose_errors(cams, data["gt"])
    print(pd.DataFrame({k: v.cpu().numpy() for k, v in e.items()}).describe().round(2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out:
        set_results_dir(args.out)
    seed_all(CFG["seed"])
    data = build_scene_data(seed=0)
    cams, gt, sc = data["cams"], data["gt"], data["sc"]
    print("n cams:", n_cams(cams), "| coverage min/mean:", data["cov"].min().round(3), data["cov"].mean().round(3))
    check_self_loss(gt, sc)
    check_quaternion_continuity(gt)
    check_closed_form(cams, gt)
    check_projection(cams, sc)
    if args.plot:
        plot_overview(data, exp_dir("sanity"))


if __name__ == "__main__":
    main()
