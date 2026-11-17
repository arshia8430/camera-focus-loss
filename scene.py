"""BlendedMVS textured meshes + camera sampling + the PyTorch3D renderer.

Same module as the synthetic version, except that `make_scene` loads a real textured mesh from the BlendedMVS
"textured meshes" supplementary package instead of generating random primitives. Everything else (renderer, camera
sampling, `build_scene_data`, `to_p3d`, `render_batch`) is unchanged, so arbitrary new cameras can be rendered.

Mesh files: every `*.obj` found below the mesh root is one scene (`seed % n_files` selects it).
Mesh root: env var BLENDEDMVS_MESH_ROOT, else CFG["bmvs_mesh_root"], else "data/BlendedMVS_textured_meshes".

The mesh is recentred at the origin and rescaled so that its (robust) bounding radius is NORM_RAD (default 2.0).
This matters: PyTorch3D's default near plane is 1.0, so a scene with a tiny/huge native scale would be clipped.
"""
import functools
import glob
import math
import os

import numpy as np
import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.renderer import (BlendParams, DirectionalLights, HardPhongShader, Materials, MeshRasterizer,
                                MeshRenderer, RasterizationSettings)
from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection
from scipy.spatial.transform import Rotation as Rot

from config import CFG, IMG, dev
from utils import cat_cams, idx_cams, look_at_R, make_K, n_cams, to_tensor


def _cfg(key, default):
    try:
        return CFG[key]
    except KeyError:
        return default


MESH_ROOT = os.environ.get("BLENDEDMVS_MESH_ROOT") or _cfg("bmvs_mesh_root", "data/BlendedMVS_textured_meshes")
NORM_RAD = float(_cfg("bmvs_rad", 2.0))  # bounding radius after normalisation
ATLAS_SIZE = int(_cfg("bmvs_atlas_size", 4))  # texels per face side (higher = sharper texture, more memory)
MIN_VIS = float(_cfg("min_vis", 0.5))  # min fraction of scene points that must be in front of the camera and in the image


# ======================================================================================
# Scene (BlendedMVS textured mesh)
# ======================================================================================
_FILES = None


def _mesh_files():
    global _FILES
    if _FILES is None:
        files = sorted(glob.glob(os.path.join(MESH_ROOT, "**", "*.obj"), recursive=True))
        if not files:
            raise FileNotFoundError(f"no .obj meshes found under '{MESH_ROOT}' "
                                    f"(set BLENDEDMVS_MESH_ROOT or CFG['bmvs_mesh_root'])")
        _FILES = files
    return _FILES


@functools.lru_cache(maxsize=2)  # loading a big textured obj is slow; build_scene_data / callers reuse it
def make_scene(seed):
    """BlendedMVS mesh number `seed % n_meshes`. Returns mesh, centre, bounding radius and 2048 surface points."""
    files = _mesh_files()
    path = files[seed % len(files)]
    # create_texture_atlas=True is needed for meshes with several texture images / materials
    mesh = load_objs_as_meshes([path], device=dev, load_textures=True, create_texture_atlas=True,
                               texture_atlas_size=ATLAS_SIZE, texture_wrap="repeat")

    torch.manual_seed(seed)
    P = sample_points_from_meshes(mesh, 20000)[0]
    lo, hi = torch.quantile(P, .02, dim=0), torch.quantile(P, .98, dim=0)  # robust to floating outliers
    centre = (lo + hi) / 2
    rad = float(torch.quantile((P - centre).norm(dim=1), .99))
    if not math.isfinite(rad) or rad <= 0:
        raise RuntimeError(f"mesh '{path}' is degenerate (radius {rad})")
    k = NORM_RAD / rad

    V = (mesh.verts_packed() - centre) * k
    mesh = Meshes([V], [mesh.faces_packed()], mesh.textures)
    pts = ((P - centre) * k)[:2048]
    return dict(mesh=mesh, center=torch.zeros(3, dtype=torch.float32, device=dev), rad=NORM_RAD, pts=pts)


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


@torch.no_grad()
def visible_frac(cam, pts):
    """Fraction of scene points (N,3) that lie in front of each camera and inside the image. Returns (B,)."""
    pc = (pts[None] - cam["t"][:, None]) @ cam["R"]  # camera coords = R^T (p - t), shape (B,N,3)
    uv = pc @ cam["K"].transpose(1, 2)
    z = pc[..., 2].clamp_min(1e-6)
    u, v = uv[..., 0] / z, uv[..., 1] / z
    ok = (pc[..., 2] > 0) & (u >= 0) & (u < IMG) & (v >= 0) & (v < IMG)
    return ok.float().mean(1)


def build_scene_data(seed, n_cams_=CFG["n_cams"]):
    """Scene + GT camera/image + `n_cams_` perturbed cameras/images.
    Every camera must have object coverage >= min_cov and >= MIN_VIS of the scene points inside its image."""
    sc = make_scene(seed)
    g = np.random.default_rng(5000 + seed)
    for _ in range(200):
        gt = sample_gt(g, sc)
        img_gt, m_gt = render_batch(sc["mesh"], gt)
        cov_gt, vis_gt = float(m_gt.mean()), float(visible_frac(gt, sc["pts"])[0])
        if cov_gt >= CFG["min_cov"] and vis_gt >= MIN_VIS:
            break
    else:
        raise RuntimeError(f"scene {seed}: no valid GT camera found "
                           f"(last coverage {cov_gt:.3f}, visible points {vis_gt:.2f})")

    cams, S, imgs, covs = [], [], [], []
    have, tries = 0, 0
    while have < n_cams_:
        tries += 1
        if tries > 200:
            raise RuntimeError(f"scene {seed}: only {have}/{n_cams_} valid cameras after {tries - 1} attempts "
                               f"(try lowering min_cov / min_vis)")
        m = int(1.5 * (n_cams_ - have)) + 10
        c, s = sample_random_cams(g, m, gt, sc)
        vis_ok = (visible_frac(c, sc["pts"]) >= MIN_VIS).nonzero()[:, 0]  # cheap pre-filter, no rendering needed
        if len(vis_ok) == 0:
            continue
        c, s = idx_cams(c, vis_ok), s[vis_ok.cpu().numpy()]
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