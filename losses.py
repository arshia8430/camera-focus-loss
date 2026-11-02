"""Loss definitions: the proposed focus-point losses (Hard / Soft / SoftGrid) and all baselines.

Every loss has the signature  fn(cam, tgt, ctx=None) -> (B,)  where `cam` and `tgt` are camera dicts
(K, R, t) and `ctx` is a dict with scene points (`ctx["pts"]`) used only by the reprojection loss.
`LOSSES` maps a loss name to (fn, group); the group label ("ours-hard", "ours-soft", "ours-softgrid",
"ours-sqrt", "ablation", "baseline") drives all downstream analyses.
"""
import torch

from config import IMG, SG_RHO, SG_SIG, SG_ZF, SIGMAS, ZF_HARD, ZF_SOFT, dev
from utils import l2, project, quat_pair, rays_cam, so3_angle

# Pixel sets (homogeneous coordinates, shape (3,N))
P_CORN = torch.tensor([[0, IMG - 1, IMG - 1, 0], [0, 0, IMG - 1, IMG - 1], [1, 1, 1, 1]],
                      dtype=torch.float32, device=dev)


def _grid_pix(G):
    u = torch.linspace(0, IMG - 1, G, device=dev)
    uu, vv = torch.meshgrid(u, u, indexing="xy")
    return torch.stack([uu.flatten(), vv.flatten(), torch.ones(G * G, device=dev)])


P_GRID = _grid_pix(6)

_c = (torch.arange(16, device=dev, dtype=torch.float32) + 0.5) * IMG / 16 - 0.5
_uu, _vv = torch.meshgrid(_c, _c, indexing="xy")
P_DENSE = torch.stack([_uu.flatten(), _vv.flatten(), torch.ones(256, device=dev)])


# ======================================================================================
# Proposed losses: focus points
# ======================================================================================
def focus_points(cam, P=P_CORN, zf=1.0):
    """World-space points at depth `zf` along the rays of pixels P, and the unit ray directions."""
    q = rays_cam(cam["K"], P)
    Rq = cam["R"] @ q
    return cam["t"][:, :, None] + zf * Rq, Rq / q.norm(dim=1, keepdim=True)


def loss_hard(cam, tgt, zf=1.0):
    A, _ = focus_points(cam, zf=zf)
    B, _ = focus_points(tgt, zf=zf)
    return ((A - B) ** 2).mean((1, 2))


def loss_dir(cam, tgt):
    _, d = focus_points(cam)
    _, ds = focus_points(tgt)
    return ((d - ds) ** 2).mean((1, 2))


def loss_soft(cam, tgt, zf=1.0, sigma=1.0):
    return loss_hard(cam, tgt, zf) + sigma ** 2 / 2 * loss_dir(cam, tgt)


def loss_soft_sqrt(cam, tgt, zf=1.0, sigma=1.0):
    """Ablation: non-squared (RMS-type) Soft; the gradient magnitude does not vanish near the optimum."""
    return torch.sqrt(loss_soft(cam, tgt, zf, sigma) + 1e-12)


def loss_soft_numeric(cam, tgt, zf, sigma, n=4001):
    """Numerical (trapezoid) evaluation of the Gaussian-weighted ray loss; used to verify the closed form of Soft."""
    s = torch.linspace(-8 * sigma, 8 * sigma, n, device=dev)
    A, d = focus_points(cam, zf=zf)
    B, ds = focus_points(tgt, zf=zf)
    X = A[..., None] + s * d[..., None]
    Xs = B[..., None] + s * ds[..., None]
    w = torch.exp(-s ** 2 / sigma ** 2)
    num = torch.trapz(w * ((X - Xs) ** 2).sum(1), s, dim=-1)
    return (num / torch.trapz(w, s)).sum(1) / 12


def make_softgrid(rho, zf=1.0, sigma=4.0, n=5, trunc=2.0, inward=False, root=False):
    """Soft loss averaged over a Gaussian pixel grid around each image corner. rho = Gaussian width in pixels."""
    off = torch.linspace(-trunc * rho, trunc * rho, n, device=dev)
    ox, oy = torch.meshgrid(off, off, indexing="xy")
    ox, oy = ox.flatten(), oy.flatten()
    w = torch.exp(-(ox ** 2 + oy ** 2) / (2 * rho ** 2))
    w = w / w.sum()                                              # Gaussian weights, sum = 1 per corner
    pix, wts = [], []
    for i in range(4):
        cu, cv = float(P_CORN[0, i]), float(P_CORN[1, i])
        if inward:                                               # fold the offsets towards the image centre
            sx = 1.0 if cu < IMG / 2 else -1.0
            sy = 1.0 if cv < IMG / 2 else -1.0
            dx, dy = sx * ox.abs(), sy * oy.abs()
        else:
            dx, dy = ox, oy
        pix.append(torch.stack([cu + dx, cv + dy, torch.ones_like(dx)]))
        wts.append(w / 4)
    Pg, om = torch.cat(pix, 1), torch.cat(wts)                   # (3, 4G), (4G,) with sum(om) = 1

    def fn(cam, tgt, ctx=None):
        A, da = focus_points(cam, Pg, zf)
        B, db = focus_points(tgt, Pg, zf)
        hard = (((A - B) ** 2).mean(1) * om).sum(1)              # mean over xyz, weighted over points
        dirr = (((da - db) ** 2).mean(1) * om).sum(1)
        val = hard + sigma ** 2 / 2 * dirr
        return torch.sqrt(val + 1e-12) if root else val

    return fn


# ======================================================================================
# Baselines
# ======================================================================================
def loss_pmse(cam, tgt, ctx=None):
    """Pose MSE on [t, q, fx/IMG, fy/IMG]; quaternions are sign-aligned so the loss is continuous on SO(3)."""
    qc, qt = quat_pair(cam, tgt)
    B = qc.shape[0]

    def v(c, q):
        t = c["t"].expand(B, -1)
        fx = (c["K"][:, 0, 0, None] / IMG).expand(B, -1)
        fy = (c["K"][:, 1, 1, None] / IMG).expand(B, -1)
        return torch.cat([t, q, fx, fy], 1)

    return ((v(cam, qc) - v(tgt, qt)) ** 2).mean(1)


def loss_posenet(cam, tgt, ctx=None, beta=100.):
    qc, qt = quat_pair(cam, tgt)
    lf = (torch.log(cam["K"][:, 0, 0]) - torch.log(tgt["K"][:, 0, 0])).abs()
    return l2(cam["t"] - tgt["t"]) + beta * l2(qc - qt) + lf


def loss_se3(cam, tgt, ctx=None):
    lf = (torch.log(cam["K"][:, 0, 0]) - torch.log(tgt["K"][:, 0, 0])).abs()
    return l2(cam["t"] - tgt["t"]) + so3_angle(cam["R"].transpose(1, 2) @ tgt["R"]) + lf


def _plucker(c, P):
    _, d = focus_points(c, P)
    m = torch.cross(c["t"][:, :, None].expand_as(d), d, dim=1)
    return d, m


def _ray_loss(cam, tgt, P):
    d, m = _plucker(cam, P)
    ds, ms = _plucker(tgt, P)
    return ((d - ds) ** 2).mean((1, 2)) + ((m - ms) ** 2).mean((1, 2))


def loss_ray(cam, tgt, ctx=None):
    return _ray_loss(cam, tgt, P_GRID)


def loss_denseray(cam, tgt, ctx=None):
    return _ray_loss(cam, tgt, P_DENSE)


def loss_reproj(cam, tgt, ctx):
    """Reprojection error (normalised by IMG^2) of the first 512 scene points."""
    X = ctx["pts"]
    X = X[:, :512] if X.dim() == 3 else X[:512]
    return ((project(cam, X) - project(tgt, X)) ** 2).mean((1, 2)) / IMG ** 2


# ======================================================================================
# Registry
# ======================================================================================
def build_losses():
    """name -> (fn, group). Grids come from config.py."""
    L = {}
    for zf in ZF_HARD:
        L[f"Hard(zf={zf})"] = (lambda c, t, x=None, zf=zf: loss_hard(c, t, zf), "ours-hard")
    for zf in ZF_SOFT:
        for sg in SIGMAS:
            L[f"Soft(zf={zf},s={sg})"] = (lambda c, t, x=None, zf=zf, sg=sg: loss_soft(c, t, zf, sg), "ours-soft")
    for zf in SG_ZF:
        for sg in SG_SIG:
            for rho in SG_RHO:
                L[f"SoftGrid(zf={zf},s={sg},r={rho})"] = (make_softgrid(rho, zf, sg), "ours-softgrid")
    L["Dir-only"] = (lambda c, t, x=None: loss_dir(c, t), "ablation")
    L["P-MSE"] = (lambda c, t, x=None: loss_pmse(c, t, x), "baseline")
    L["PoseNet(b=10)"] = (lambda c, t, x=None: loss_posenet(c, t, x, 10.), "baseline")
    L["PoseNet(b=100)"] = (lambda c, t, x=None: loss_posenet(c, t, x, 100.), "baseline")
    L["SE3-geo"] = (lambda c, t, x=None: loss_se3(c, t, x), "baseline")
    L["Ray-Plucker"] = (lambda c, t, x=None: loss_ray(c, t, x), "baseline")
    L["Reproj-px"] = (lambda c, t, x=None: loss_reproj(c, t, x), "baseline")
    L["DenseRay"] = (lambda c, t, x=None: loss_denseray(c, t, x), "baseline")
    # Ablation: non-squared versions of our losses (tests the "squared loss + Adam -> slow tail" hypothesis)
    L["SoftSqrt(zf=1,s=4)"] = (lambda c, t, x=None: loss_soft_sqrt(c, t, 1, 4.0), "ours-sqrt")
    L["SoftGridSqrt(zf=1,s=4,r=48)"] = (make_softgrid(48, 1, 4.0, root=True), "ours-sqrt")
    return L


LOSSES = build_losses()
LOSS_GROUP = {n: g for n, (_, g) in LOSSES.items()}
