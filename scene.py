"""Synthetic scenes, camera sampling and the PyTorch3D renderer.

This is the only module that depends on rendering. A scene is a random collection of coloured primitives;
for each scene one ground-truth (GT) camera and `n_cams` perturbed cameras are sampled, where the perturbation
strength s in [0, 1] controls how far a camera is from the GT camera (0 = near, 1 = fully random).
"""
import colorsys
import math

import numpy as np
import torch
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.renderer import (BlendParams, DirectionalLights, HardPhongShader, Materials, MeshRasterizer,
                                MeshRenderer, RasterizationSettings, TexturesVertex)
from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection, ico_sphere
from scipy.spatial.transform import Rotation as Rot

from config import CFG, IMG, dev
from utils import cat_cams, idx_cams, look_at_R, make_K, n_cams, to_tensor


# ======================================================================================
# Primitives
# ======================================================================================
def _orient_outward(v, f, centers=None):
    tri = v[f]
    n = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)
    c = tri.mean(1) if centers is None else tri.mean(1) - centers
    flip = (n * c).sum(1) < 0
    f = f.clone()
    f[flip] = f[flip][:, [0, 2, 1]]
    return f


def make_box():
    v = torch.tensor([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])
    f = torch.tensor([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                      [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return v, _orient_outward(v, f)


def make_sphere():
    m = ico_sphere(2)
    return m.verts_packed().cpu() * 0.5, m.faces_packed().cpu()


def make_cyl(n=24):
    th = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
    ring = torch.stack([.5 * torch.cos(th), torch.zeros(n), .5 * torch.sin(th)], 1)
    bot, top = ring.clone(), ring.clone()
    bot[:, 1] = -.5
    top[:, 1] = .5
    v = torch.cat([bot, top, torch.tensor([[0, -.5, 0], [0, .5, 0]])])
    f = []
    for i in range(n):
        j = (i + 1) % n
        f += [[i, j, n + j], [i, n + j, n + i], [2 * n, j, i], [2 * n + 1, n + i, n + j]]
    f = torch.tensor(f)
    return v, _orient_outward(v, f)


def make_cone(n=24):
    th = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
    ring = torch.stack([.5 * torch.cos(th), -.5 * torch.ones(n), .5 * torch.sin(th)], 1)
    v = torch.cat([ring, torch.tensor([[0, .5, 0], [0, -.5, 0]])])
    f = []
    for i in range(n):
        j = (i + 1) % n
        f += [[i, j, n], [n + 1, j, i]]
    f = torch.tensor(f)
    return v, _orient_outward(v, f)


def make_octa():
    v = torch.tensor([[.6, 0, 0], [-.6, 0, 0], [0, .6, 0], [0, -.6, 0], [0, 0, .6], [0, 0, -.6]])
    f = torch.tensor([[a, b, c] for a in (0, 1) for b in (2, 3) for c in (4, 5)])
    return v, _orient_outward(v, f)


def make_torus(R=.35, r=.15, ns=16, nr=24):
    u = torch.linspace(0, 2 * math.pi, nr + 1)[:-1]
    w = torch.linspace(0, 2 * math.pi, ns + 1)[:-1]
    U, W_ = torch.meshgrid(u, w, indexing="ij")
    v = torch.stack([(R + r * torch.cos(W_)) * torch.cos(U), r * torch.sin(W_),
                     (R + r * torch.cos(W_)) * torch.sin(U)], -1).reshape(-1, 3)
    id_ = lambda i, j: (i % nr) * ns + (j % ns)
    f = []
    for i in range(nr):
        for j in range(ns):
            a, b, c, d = id_(i, j), id_(i + 1, j), id_(i + 1, j + 1), id_(i, j + 1)
            f += [[a, b, c], [a, c, d]]
    f = torch.tensor(f)
    tc = v[f].mean(1)
    ring = torch.stack([tc[:, 0], torch.zeros_like(tc[:, 0]), tc[:, 2]], 1)
    ring = R * ring / ring.norm(dim=1, keepdim=True)
    return v, _orient_outward(v, f, ring)


SHAPES = dict(box=make_box, sphere=make_sphere, cyl=make_cyl, cone=make_cone, octa=make_octa, torus=make_torus)


def make_scene(seed):
    """Random scene of 10-19 coloured primitives. Returns mesh, centre, bounding radius and 2048 surface points."""
    g = np.random.default_rng(1000 + seed)
    kinds = list(SHAPES)
    V, Fc, C, off = [], [], [], 0
    n_obj = int(g.integers(10, 20))
    h0 = g.uniform()
    for k in range(n_obj):
        v, f = SHAPES[kinds[int(g.integers(len(kinds)))]]()
        scale = torch.tensor(g.uniform(.7, 1.4, 3) * g.uniform(.6, 1.1), dtype=torch.float32)
        Rm = torch.tensor(Rot.random(random_state=int(g.integers(1 << 31))).as_matrix(), dtype=torch.float32)
        pos = torch.tensor(g.uniform(-1.3, 1.3, 3), dtype=torch.float32)
        v = (v * scale) @ Rm.T + pos
        base = np.array(colorsys.hsv_to_rgb((h0 + .618 * k) % 1, g.uniform(.55, 1), g.uniform(.65, 1)))
        col = np.clip(base + g.normal(0, .12, (len(v), 3)), 0, 1)
        V.append(v)
        Fc.append(f + off)
        C.append(torch.tensor(col, dtype=torch.float32))
        off += len(v)
    V, Fc, C = torch.cat(V), torch.cat(Fc), torch.cat(C)
    mesh = Meshes([V], [Fc], TexturesVertex([C])).to(dev)
    center = (V.max(0).values + V.min(0).values) / 2
    rad = float((V - center).norm(dim=1).max())
    torch.manual_seed(seed)
    pts = sample_points_from_meshes(mesh, 2048)[0]
    return dict(mesh=mesh, center=center.to(dev), rad=rad, pts=pts)


# ======================================================================================
# Renderer
# ======================================================================================
_raster = RasterizationSettings(image_size=IMG, blur_radius=0.0, faces_per_pixel=1)
_lights = DirectionalLights(device=dev, direction=[[-0.4, -1.0, -0.3]],
                            ambient_color=[[.55] * 3], diffuse_color=[[.5] * 3], specular_color=[[0.] * 3])
_mat = Materials(device=dev, specular_color=[[0., 0., 0.]], shininess=1.0)
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(raster_settings=_raster),
    shader=HardPhongShader(device=dev, lights=_lights, materials=_mat,
                           blend_params=BlendParams(background_color=(.5, .5, .5))))


def to_p3d(cam):
    """Convert our camera dict (K, R cam->world, t = camera centre) to PyTorch3D cameras."""
    Rwc = cam["R"].transpose(1, 2)
    tvec = -(Rwc @ cam["t"][:, :, None])[:, :, 0]
    size = torch.tensor([[IMG, IMG]], dtype=torch.float32, device=dev).expand(n_cams(cam), 2)
    return cameras_from_opencv_projection(Rwc, tvec, cam["K"], size)


@torch.no_grad()
def render_batch(mesh, cam, bs=CFG["render_bs"]):
    """Returns RGB images (B,3,H,W) in [0,1] and binary object masks (B,H,W)."""
    out = []
    for i in range(0, n_cams(cam), bs):
        c = idx_cams(cam, slice(i, i + bs))
        out.append(renderer(mesh.extend(n_cams(c)), cameras=to_p3d(c)))
    o = torch.cat(out)
    return o[..., :3].permute(0, 3, 1, 2).clamp(0, 1), (o[..., 3] > .5).float()


# ======================================================================================
# Camera sampling
# ======================================================================================
def sample_gt(g, sc):
    """One GT camera on a sphere around the scene, looking near the scene centre."""
    az = g.uniform(0, 2 * math.pi)
    el = math.asin(g.uniform(-.85, .85))
    r = g.uniform(2.6, 3.2) * sc["rad"]
    d = np.array([math.cos(el) * math.cos(az), math.sin(el), math.cos(el) * math.sin(az)])
    eye = sc["center"].cpu().numpy() + r * d
    tgt = sc["center"].cpu().numpy() + g.normal(0, .2, 3)
    fov = math.radians(g.uniform(40, 60))
    f = .5 * IMG / math.tan(fov / 2)
    R = look_at_R(to_tensor(eye)[None], to_tensor(tgt)[None], to_tensor([g.normal(0, math.radians(10))]))
    return dict(K=make_K(to_tensor([f])), R=R, t=to_tensor(eye)[None])


def sample_random_cams(g, n, gt, sc):
    """`n` cameras whose distance from the GT camera grows with s ~ U(0,1) (viewpoint, distance, target, roll, focal)."""
    c0 = sc["center"].cpu().numpy()
    rad = sc["rad"]
    d0 = gt["t"][0].cpu().numpy() - c0
    r0 = np.linalg.norm(d0)
    u0 = d0 / r0
    f0 = float(gt["K"][0, 0, 0])
    s = g.uniform(0, 1, n)
    eyes, tgts, rolls, fs = [], [], [], []
    for i in range(n):
        ax = g.normal(size=3)
        ax /= np.linalg.norm(ax)
        u = Rot.from_rotvec(ax * s[i] * math.pi).apply(u0)
        r = max(r0 * math.exp(g.uniform(-1, 1) * s[i] * math.log(1.6)), 1.6 * rad)
        eyes.append(c0 + r * u)
        tgts.append(c0 + g.normal(0, 1, 3) * .35 * rad * s[i])
        rolls.append(g.uniform(-math.pi, math.pi) * s[i])
        fs.append(f0 * math.exp(g.uniform(-1, 1) * s[i] * math.log(2)))
    eye, tgt = to_tensor(eyes), to_tensor(tgts)
    return dict(K=make_K(to_tensor(fs)), R=look_at_R(eye, tgt, to_tensor(rolls)), t=eye), s


def build_scene_data(seed, n_cams_=CFG["n_cams"]):
    """Scene + GT camera/image + `n_cams_` perturbed cameras/images (all with object coverage >= min_cov)."""
    sc = make_scene(seed)
    g = np.random.default_rng(5000 + seed)
    for _ in range(200):
        gt = sample_gt(g, sc)
        img_gt, m_gt = render_batch(sc["mesh"], gt)
        if m_gt.mean() >= CFG["min_cov"]:
            break
    else:
        raise RuntimeError(f"scene {seed}: no valid GT camera found (last coverage {float(m_gt.mean()):.3f})")

    cams, S, imgs, covs = [], [], [], []
    have = 0
    while have < n_cams_:
        m = int(1.5 * (n_cams_ - have)) + 10
        c, s = sample_random_cams(g, m, gt, sc)
        im, mk = render_batch(sc["mesh"], c)
        cov = mk.mean((1, 2))
        keep = (cov >= CFG["min_cov"]).nonzero()[:, 0][: n_cams_ - have]
        cams.append(idx_cams(c, keep))
        S.append(s[keep.cpu().numpy()])
        imgs.append(im[keep])
        covs.append(cov[keep])
        have += len(keep)
    return dict(seed=seed, sc=sc, gt=gt, cams=cat_cams(cams), s=np.concatenate(S),
                img_gt=img_gt, imgs=torch.cat(imgs), cov=torch.cat(covs).cpu().numpy())
