"""HSSD-hab (Hugging Face `hssd/hssd-hab`) full-house scenes + camera sampling + the PyTorch3D renderer.

Same module interface as the BlendedMVS version: `make_scene`, `build_scene_data`, `render_batch`, `to_p3d`
(names, arguments and return structure unchanged), so the rest of the project keeps working.

What a scene is here
    scenes/<id>.scene_instance.json  -> stage (the house shell) + ~50-300 object instances (pose + scale)
    stages/<id>.glb, objects/**/*.glb -> render meshes with textures
`make_scene(seed)` downloads (only what that scene references), assembles stage + ALL objects into one textured
mesh (one packed texture atlas), recentres it at the origin and rescales it to bounding radius NORM_RAD.

Download policy (nothing extra is ever fetched)
    * only the NUM_SCENES scenes selected below are used (spread evenly over the sorted list of scenes),
    * per scene: scene json, stage glb+config, and the glb+config of exactly the objects that scene references,
    * everything lands in HSSD_ROOT and is reused on later runs (no network access for files already there).

Cameras
    CAM_MODE = "interior" (default): eye-height cameras inside the house, away from walls/furniture.
    CAM_MODE = "orbit": the original sphere-around-the-scene sampler (sees the house from outside).

Extra pip dependencies: huggingface_hub, trimesh, pillow (scipy/numpy/torch/pytorch3d as before).
"""
import functools
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import trimesh
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.renderer import (BlendParams, DirectionalLights, HardPhongShader, Materials, MeshRasterizer,
                                MeshRenderer, RasterizationSettings, TexturesUV)
from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

from config import CFG, IMG, dev
from utils import cat_cams, idx_cams, look_at_R, make_K, n_cams, to_tensor


def _cfg(key, default):
    try:
        return CFG[key]
    except KeyError:
        return default


# ======================================================================================
# Settings
# ======================================================================================
NUM_SCENES = 40  # <<< how many HSSD scenes to download / use. Change this number.

HSSD_REPO = "hssd/hssd-hab"
HSSD_ROOT = os.environ.get("HSSD_ROOT") or _cfg("hssd_root", "data/hssd-hab")  # local download folder
HSSD_SCENE_DIR = _cfg("hssd_scene_dir", "scenes")  # "scenes" (full, cluttered) or "scenes-uncluttered"
CAM_MODE = _cfg("hssd_cam_mode", "interior")  # "interior" | "orbit"

NORM_RAD = float(_cfg("bmvs_rad", 2.0))  # bounding radius after normalisation (kept name/default from before)
TEX_SIZE = int(_cfg("hssd_tex_size", 4096))  # side of the packed texture atlas (8192 = sharper, ~4x memory)
TILE_MAX = int(_cfg("hssd_tile_max", 2048))  # per-texture side cap before packing
FLIP_V = bool(_cfg("hssd_flip_v", False))  # set True if the rendered textures come out upside-down
MIN_VIS = float(_cfg("min_vis", 0.5))  # orbit mode: min fraction of scene points in front of camera and in image
ZNEAR = 0.01  # PyTorch3D default is 1.0, which clips everything closer than 0.5 units (we are inside a house)

# interior-camera parameters, in metres (converted to scene units with sc["unit"])
EYE_H = (1.35, 1.70)  # eye height above the floor
CLEARANCE = 0.45  # min distance from eye to any surface
MAX_SHIFT = 5.0  # perturbed cameras move up to this far (horizontally) at s = 1
MIN_NEAR, MIN_MED = 0.35, 1.0  # reject views with > 3% of pixels closer than MIN_NEAR or median depth < MIN_MED


# ======================================================================================
# Hugging Face access (selective download)
# ======================================================================================
def _dl(rel):
    """Download one repo file into HSSD_ROOT (skipped if it is already there). Returns the local path."""
    p = os.path.join(HSSD_ROOT, rel)
    if os.path.exists(p) and os.path.getsize(p) > 0:
        return p
    return hf_hub_download(HSSD_REPO, rel, repo_type="dataset", local_dir=HSSD_ROOT)


def _json_cache(name):
    return os.path.join(HSSD_ROOT, name)


def _paths_exist(api, paths):
    found = set()
    for i in range(0, len(paths), 200):
        for info in api.get_paths_info(HSSD_REPO, paths[i:i + 200], repo_type="dataset"):
            found.add(info.path)
    return found


_IDS = None


def _scene_ids():
    """The NUM_SCENES scene ids that are used (deterministic, evenly spread over all available scenes)."""
    global _IDS
    if _IDS is not None:
        return _IDS
    if NUM_SCENES < 1:
        raise ValueError("NUM_SCENES must be >= 1")
    os.makedirs(HSSD_ROOT, exist_ok=True)
    f = _json_cache(f"_selected_{HSSD_SCENE_DIR}.json")
    if os.path.exists(f):
        d = json.load(open(f))
        if d.get("n") == NUM_SCENES:
            _IDS = d["ids"]
            return _IDS
    api = HfApi()
    suf = ".scene_instance.json"
    ids = sorted(os.path.basename(e.path)[:-len(suf)]
                 for e in api.list_repo_tree(HSSD_REPO, path_in_repo=HSSD_SCENE_DIR, repo_type="dataset")
                 if e.path.endswith(suf))
    ok = _paths_exist(api, [f"stages/{i}.glb" for i in ids])  # keep scenes whose stage is in this repo
    ids = [i for i in ids if f"stages/{i}.glb" in ok]
    if not ids:
        raise FileNotFoundError(f"no usable scenes found in {HSSD_REPO}/{HSSD_SCENE_DIR}")
    pick = np.unique(np.linspace(0, len(ids) - 1, min(NUM_SCENES, len(ids))).round().astype(int))
    _IDS = [ids[i] for i in pick]
    json.dump(dict(n=NUM_SCENES, ids=_IDS), open(f, "w"))
    return _IDS


def _full_index():
    """Fallback: names -> paths for every object (metadata only, cached)."""
    f = _json_cache("_objects_index.json")
    if os.path.exists(f):
        return json.load(open(f))
    glb, cfg = {}, {}
    for e in HfApi().list_repo_tree(HSSD_REPO, path_in_repo="objects", recursive=True, repo_type="dataset"):
        p = e.path
        b = os.path.basename(p)
        if p.endswith(".collider.glb"):
            continue
        if p.endswith(".object_config.json"):
            cfg[b[:-len(".object_config.json")]] = p
        elif p.endswith(".glb"):
            glb[b[:-4]] = p
    idx = dict(glb=glb, cfg=cfg)
    json.dump(idx, open(f, "w"))
    return idx


def _resolve(names):
    """template_name -> (glb repo path, config repo path or None)."""
    f = _json_cache("_objects_resolved.json")
    cache = json.load(open(f)) if os.path.exists(f) else {}
    todo = [n for n in names if n not in cache]
    if todo:
        api = HfApi()
        dirs = lambda n: (f"objects/{n[0]}", f"objects/decomposed/{n.split('_part_')[0]}", "objects/openings")
        cand = [f"{d}/{n}{e}" for n in todo for d in dirs(n) for e in (".glb", ".object_config.json")]
        found = _paths_exist(api, cand)
        for n in todo:
            for d in dirs(n):
                if f"{d}/{n}.glb" in found:
                    c = f"{d}/{n}.object_config.json"
                    cache[n] = [f"{d}/{n}.glb", c if c in found else None]
                    break
        rest = [n for n in todo if n not in cache]
        if rest:  # unusual naming: look them up in the full (metadata-only) index
            idx = _full_index()
            for n in rest:
                if n in idx["glb"] or n in idx["cfg"]:
                    cache[n] = [idx["glb"].get(n), idx["cfg"].get(n)]
        json.dump(cache, open(f, "w"))
    return {n: cache[n] for n in names if n in cache}


def _fetch_scene(sid):
    """Download (if needed) and read everything one scene needs. Returns the parsed description."""
    scene = json.load(open(_dl(f"{HSSD_SCENE_DIR}/{sid}.scene_instance.json")))
    tn = scene["stage_instance"]["template_name"]
    if "/" not in tn:
        tn = f"stages/{tn}"
    stage_cfg = json.load(open(_dl(f"{tn}.stage_config.json")))
    stage_glb = _dl(f"{os.path.dirname(tn)}/{stage_cfg.get('render_asset', os.path.basename(tn) + '.glb')}")

    names = sorted({o["template_name"] for o in scene.get("object_instances", [])})
    res = _resolve(names)

    def get(n):
        try:
            if n not in res:
                return n, None
            glb_rel, cfg_rel = res[n]
            cfg = {}
            if cfg_rel:
                cfg = json.load(open(_dl(cfg_rel)))
                if cfg.get("render_asset"):
                    glb_rel = os.path.normpath(os.path.join(os.path.dirname(cfg_rel), cfg["render_asset"])).replace(os.sep, "/")
            return n, (_dl(glb_rel), cfg)
        except Exception as e:  # one broken object must not kill the scene
            print(f"[hssd] scene {sid}: skipping object '{n}' ({type(e).__name__}: {e})")
            return n, None

    print(f"[hssd] scene {sid}: {len(scene.get('object_instances', []))} instances, {len(names)} unique objects")
    with ThreadPoolExecutor(8) as ex:
        objs = {n: v for n, v in ex.map(get, names) if v is not None}
    if len(objs) < len(names):
        print(f"[hssd] scene {sid}: {len(names) - len(objs)} object types unavailable, skipped")
    return dict(scene=scene, stage_glb=stage_glb, stage_cfg=stage_cfg, objs=objs)


def download_scenes():
    """Optional: prefetch all NUM_SCENES scenes now (otherwise each scene is fetched on first use)."""
    for sid in _scene_ids():
        _fetch_scene(sid)


# ======================================================================================
# GLB -> parts, texture packing, scene assembly
# ======================================================================================
def _orient(cfg):
    """Rotation (asset -> Habitat canonical frame, up=+Y, front=-Z) from the config's up/front vectors."""
    up = np.asarray(cfg.get("up", [0, 1, 0]), float)
    fr = np.asarray(cfg.get("front", [0, 0, -1]), float)
    up, fr = up / np.linalg.norm(up), fr / np.linalg.norm(fr)
    right = np.cross(fr, up)
    return np.stack([right, up, -fr], axis=1).T


def _mat_info(vis):
    """(PIL image or None, rgb factor (3,)) of a trimesh visual's base colour."""
    mat = getattr(vis, "material", None)
    img, fac = None, np.ones(3)
    if mat is not None:
        img = getattr(mat, "baseColorTexture", None)
        if img is None:
            img = getattr(mat, "image", None)
        f = getattr(mat, "baseColorFactor", None)
        if f is None:
            f = getattr(mat, "diffuse", None)
        if f is not None:
            f = np.asarray(f, float).ravel()[:3]
            fac = f / 255. if f.max() > 1.0 else f
    return img, fac


def _read_glb(path):
    """List of dicts (V, F, uv|None, img|None, color) with node transforms applied."""
    try:
        sc = trimesh.load(path, force="scene", process=False)
    except Exception as e:
        raise RuntimeError(f"cannot read '{path}' with trimesh ({e}). If the GLB uses Draco/meshopt/KTX2 "
                           f"compression it has to be decoded first.") from e
    out = []
    for g in sc.dump(concatenate=False):
        if getattr(g, "faces", None) is None or len(g.faces) == 0:
            continue
        V, F = np.asarray(g.vertices, np.float32), np.asarray(g.faces, np.int64)
        vis, uv, img = g.visual, None, None
        color = np.array([.6, .6, .6])
        if getattr(vis, "kind", None) == "texture":
            img, fac = _mat_info(vis)
            color = fac
            if img is not None and getattr(vis, "uv", None) is not None:
                uv = np.asarray(vis.uv, np.float32).copy()
                img = img.convert("RGB")
                if max(img.size) > TILE_MAX:
                    s = TILE_MAX / max(img.size)
                    img = img.resize((max(2, int(img.width * s)), max(2, int(img.height * s))), Image.LANCZOS)
                if np.abs(fac - 1).max() > 1e-3:
                    img = Image.fromarray((np.asarray(img, np.float32) * fac).clip(0, 255).astype(np.uint8))
            else:
                img = None
        elif getattr(vis, "kind", None) in ("vertex", "face"):
            color = np.asarray(vis.vertex_colors, float)[:, :3].mean(0) / 255.
        if uv is not None:
            if FLIP_V:
                uv[:, 1] = 1 - uv[:, 1]
            if uv.min() < -1e-3 or uv.max() > 1 + 1e-3:  # repeating texture: un-weld and shift each face into [0,1]
                uvf = uv[F]
                shift = np.floor(uvf.mean(1, keepdims=True))
                V = V[F].reshape(-1, 3)
                uv = (uvf - shift).reshape(-1, 2).astype(np.float32)
                F = np.arange(len(V), dtype=np.int64).reshape(-1, 3)
            uv = np.clip(uv, 0, 1)
        out.append(dict(V=V, F=F, uv=uv, img=img, color=color))
    if not out:
        raise RuntimeError(f"'{path}' contains no triangle meshes")
    return out


def _pack(tiles, prio, size, pad=4):
    """Shelf-pack PIL tiles into one (size,size,3) uint8 atlas. Returns atlas and inner rects (x, y, w, h)."""
    dims = [(t.width, t.height) for t in tiles]
    area = sum(((w * p) + 2 * pad) * ((h * p) + 2 * pad) for (w, h), p in zip(dims, prio))
    s = min(1.0, math.sqrt(0.75 * size * size / area))
    for _ in range(60):
        sz = [(max(2, min(w, int(w * s * p))), max(2, min(h, int(h * s * p)))) for (w, h), p in zip(dims, prio)]
        order = sorted(range(len(tiles)), key=lambda i: -sz[i][1])
        pos, x, y, rowh, ok = [None] * len(tiles), 0, 0, 0, True
        for i in order:
            w, h = sz[i][0] + 2 * pad, sz[i][1] + 2 * pad
            if x + w > size:
                x, y, rowh = 0, y + rowh, 0
            if w > size or y + h > size:
                ok = False
                break
            pos[i] = (x, y)
            x += w
            rowh = max(rowh, h)
        if ok:
            break
        s *= 0.92
    else:
        raise RuntimeError("texture packing failed (raise hssd_tex_size)")
    atlas = np.full((size, size, 3), 127, np.uint8)
    rects = [None] * len(tiles)
    for i, t in enumerate(tiles):
        w, h = sz[i]
        if (w, h) != t.size:
            t = t.resize((w, h), Image.LANCZOS)
        arr = np.pad(np.asarray(t.convert("RGB")), ((pad, pad), (pad, pad), (0, 0)), mode="edge")
        x, y = pos[i]
        atlas[y:y + h + 2 * pad, x:x + w + 2 * pad] = arr
        rects[i] = (x + pad, y + pad, w, h)
    return atlas, rects


def _assemble(data):
    """Stage + all object instances -> one Meshes with a single packed texture atlas (un-normalised)."""
    scene = data["scene"]
    assets, tiles, prio, color_tid = {}, [], [], {}

    def load(key, glb, cfg, p):
        if key in assets:
            return
        Ro = _orient(cfg)
        parts = []
        for gm in _read_glb(glb):
            V = (gm["V"] @ Ro.T).astype(np.float32)
            uv = gm["uv"]
            if gm["img"] is not None and uv is not None:
                tid = len(tiles)
                tiles.append(gm["img"])
                prio.append(p)
            else:
                c = tuple(int(round(float(x) * 255)) for x in np.clip(gm["color"], 0, 1))
                if c not in color_tid:
                    color_tid[c] = len(tiles)
                    tiles.append(Image.new("RGB", (8, 8), c))
                    prio.append(1.0)
                tid = color_tid[c]
                uv = np.full((len(V), 2), .5, np.float32)
            parts.append(dict(V=torch.from_numpy(V).to(dev), F=torch.from_numpy(gm["F"]).to(dev), uv=uv, tid=tid))
        assets[key] = parts

    inst = []  # (asset key, R (3,3), t (3,), scale (3,))
    load("__stage__", data["stage_glb"], data["stage_cfg"], 2.0)  # stage textures get more atlas area
    si = scene["stage_instance"]
    inst.append(("__stage__", Rot.from_quat(_wxyz_to_xyzw(si.get("rotation", [1, 0, 0, 0]))).as_matrix(),
                 np.asarray(si.get("translation", [0, 0, 0]), float), np.ones(3)))
    skipped = 0
    for o in scene.get("object_instances", []):
        n = o["template_name"]
        if n not in data["objs"]:
            skipped += 1
            continue
        glb, cfg = data["objs"][n]
        load(n, glb, cfg, 1.0)
        tscale = np.broadcast_to(np.asarray(cfg.get("scale", 1.0), float), (3,))
        sc = tscale * np.asarray(o.get("non_uniform_scale", [1, 1, 1]), float)
        inst.append((n, Rot.from_quat(_wxyz_to_xyzw(o.get("rotation", [1, 0, 0, 0]))).as_matrix(),
                     np.asarray(o.get("translation", [0, 0, 0]), float), sc))

    atlas, rects = _pack(tiles, prio, TEX_SIZE)
    for parts in assets.values():  # remap per-vertex uv into the packed atlas
        for p in parts:
            x, y, w, h = rects[p["tid"]]
            u, v = p["uv"][:, 0], p["uv"][:, 1]
            U = (x + u * w) / TEX_SIZE
            Vv = 1.0 - (y + (1.0 - v) * h) / TEX_SIZE
            p["uvp"] = torch.from_numpy(np.stack([U, Vv], 1).astype(np.float32)).to(dev)

    Vs, Fs, UVs, off = [], [], [], 0
    for key, R, t, s in inst:
        Rt = torch.from_numpy(R.astype(np.float32)).to(dev)
        tt = torch.from_numpy(t.astype(np.float32)).to(dev)
        st = torch.from_numpy(s.astype(np.float32)).to(dev)
        flip = float(np.prod(s)) < 0  # mirrored instance -> flip winding so normals stay outward
        for p in assets[key]:
            Vs.append((p["V"] * st) @ Rt.T + tt)
            F = p["F"] + off
            Fs.append(F[:, [0, 2, 1]] if flip else F)
            UVs.append(p["uvp"])
            off += len(p["V"])
    V, F, UV = torch.cat(Vs), torch.cat(Fs), torch.cat(UVs)
    print(f"[hssd] assembled {len(inst)} instances ({skipped} skipped), {len(V):,} verts, {len(F):,} faces, "
          f"{len(tiles)} textures -> {TEX_SIZE}x{TEX_SIZE} atlas")
    maps = torch.from_numpy(atlas).to(dev).float().div_(255.)[None]
    mesh = Meshes([V], [F], textures=TexturesUV(maps=maps, faces_uvs=[F], verts_uvs=[UV]))
    del assets, Vs, Fs, UVs  # per-part tensors are no longer needed
    torch.cuda.empty_cache()
    return mesh


def _wxyz_to_xyzw(q):  # Habitat json stores quaternions as [w, x, y, z]
    q = list(q)
    return [q[1], q[2], q[3], q[0]]


# ======================================================================================
# Scene
# ======================================================================================
def _interior_setup(Pn, Nn, k, rng):
    """Valid eye positions inside the house: on the floor, at eye height, >= CLEARANCE from every surface."""
    y = Pn[:, 1]
    horiz = Nn[:, 1].abs() > 0.9
    if int(horiz.sum()) < 100:
        raise RuntimeError("no horizontal surfaces found, cannot place interior cameras")
    floor_y = float(torch.quantile(y[horiz], .02))
    on_floor = horiz & (y > floor_y - .08 * k) & (y < floor_y + .08 * k)
    fp = Pn[on_floor].cpu().numpy()
    tree = cKDTree(Pn[y > floor_y + .15 * k].cpu().numpy())
    for clear in (CLEARANCE, .30, .20):
        eyes = fp.copy()
        eyes[:, 1] = floor_y + rng.uniform(*EYE_H, len(fp)) * k
        ok = tree.query(eyes)[0] >= clear * k
        if ok.sum() >= 50:
            break
    else:
        raise RuntimeError("could not find free space for interior cameras")
    cands = eyes[ok]
    if len(cands) > 5000:
        cands = cands[rng.choice(len(cands), 5000, replace=False)]
    return cands.astype(np.float32)


@functools.lru_cache(maxsize=2)  # assembling a whole house is slow; build_scene_data / callers reuse it
def make_scene(seed):
    """HSSD scene number `seed % NUM_SCENES`. Returns mesh, centre, bounding radius and 2048 surface points."""
    ids = _scene_ids()
    sid = ids[seed % len(ids)]
    mesh = _assemble(_fetch_scene(sid))

    torch.manual_seed(seed)
    P, Nn = sample_points_from_meshes(mesh, 300000, return_normals=True)
    P, Nn = P[0], Nn[0]
    lo, hi = torch.quantile(P, .02, dim=0), torch.quantile(P, .98, dim=0)  # robust to floating outliers
    centre = (lo + hi) / 2
    rad = float(torch.quantile((P - centre).norm(dim=1), .99))
    if not math.isfinite(rad) or rad <= 0:
        raise RuntimeError(f"scene '{sid}' is degenerate (radius {rad})")
    k = NORM_RAD / rad

    V = (mesh.verts_packed() - centre) * k
    mesh = Meshes([V], [mesh.faces_packed()], mesh.textures)
    Pn = (P - centre) * k
    cands = _interior_setup(Pn, Nn, k, np.random.default_rng(seed)) if CAM_MODE == "interior" else None
    print(f"[hssd] scene {sid} ready: radius {rad:.1f} m -> {NORM_RAD}, "
          f"{0 if cands is None else len(cands)} interior camera positions")
    return dict(mesh=mesh, center=torch.zeros(3, dtype=torch.float32, device=dev), rad=NORM_RAD, pts=Pn[:2048],
                unit=k, cands=cands)  # unit = scene units per metre; cands = valid eye positions (interior mode)


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
    cams = cameras_from_opencv_projection(Rwc, tvec, cam["K"], size)
    cams.znear = torch.full((n_cams(cam),), ZNEAR, dtype=torch.float32, device=dev)  # see ZNEAR note above
    return cams


def _safe_bs(mesh, bs, textured=True, budget=3e9, overhead=4):
    """Cap the batch size so `mesh.extend(bs)` fits in `budget` bytes (a whole-house mesh is ~100-300 MB per copy)."""
    per = mesh.verts_packed().numel() * 4 + mesh.faces_packed().numel() * 8
    if textured and mesh.textures is not None:
        per += mesh.textures.maps_padded()[0].numel() * 4
    return int(max(1, min(bs, budget // (overhead * per))))


@torch.no_grad()
def render_batch(mesh, cam, bs=CFG["render_bs"]):
    """Returns RGB images (B,3,H,W) in [0,1] and binary object masks (B,H,W)."""
    bs = _safe_bs(mesh, bs)
    out = []
    for i in range(0, n_cams(cam), bs):
        c = idx_cams(cam, slice(i, i + bs))
        out.append(renderer(mesh.extend(n_cams(c)), cameras=to_p3d(c)))
    o = torch.cat(out)
    return o[..., :3].permute(0, 3, 1, 2).clamp(0, 1), (o[..., 3] > .5).float()


# ======================================================================================
# Camera sampling
# ======================================================================================
def _dir(yaw, pitch):
    return np.array([math.cos(pitch) * math.cos(yaw), math.sin(pitch), math.cos(pitch) * math.sin(yaw)])


def sample_gt(g, sc):
    """One GT camera. interior: eye-height camera inside the house. orbit: sphere around the scene centre."""
    if CAM_MODE == "orbit":
        az = g.uniform(0, 2 * math.pi)
        el = math.asin(g.uniform(-.85, .85))
        r = g.uniform(2.6, 3.2) * sc["rad"]
        d = np.array([math.cos(el) * math.cos(az), math.sin(el), math.cos(el) * math.sin(az)])
        eye = sc["center"].cpu().numpy() + r * d
        tgt = sc["center"].cpu().numpy() + g.normal(0, .2, 3)
        fov = math.radians(g.uniform(40, 60))
        roll = g.normal(0, math.radians(10))
    else:
        eye = sc["cands"][g.integers(len(sc["cands"]))]
        yaw = g.uniform(0, 2 * math.pi)
        pitch = float(np.clip(g.normal(0, math.radians(8)), -.4, .4))
        tgt = eye + _dir(yaw, pitch)
        fov = math.radians(g.uniform(55, 80))
        roll = g.normal(0, math.radians(2))
    f = .5 * IMG / math.tan(fov / 2)
    R = look_at_R(to_tensor(eye)[None], to_tensor(tgt)[None], to_tensor([roll]))
    return dict(K=make_K(to_tensor([f])), R=R, t=to_tensor(eye)[None])


def sample_random_cams(g, n, gt, sc):
    """`n` cameras whose distance from the GT camera grows with s ~ U(0,1) (viewpoint, distance, target, roll, focal)."""
    f0 = float(gt["K"][0, 0, 0])
    s = g.uniform(0, 1, n)
    if CAM_MODE == "interior":
        c = sc["cands"]
        unit = sc["unit"]
        eye0 = gt["t"][0].cpu().numpy()
        fwd = gt["R"][0, :, 2].cpu().numpy()  # camera z axis (forward) in world coordinates
        yaw0, pitch0 = math.atan2(fwd[2], fwd[0]), math.asin(float(np.clip(fwd[1], -1, 1)))
        dxz = np.linalg.norm(c[:, [0, 2]] - eye0[[0, 2]], axis=1)
        eyes, tgts, rolls, fs = [], [], [], []
        for i in range(n):
            pool = np.nonzero(dxz <= max(s[i] * MAX_SHIFT, .3) * unit)[0]
            j = int(g.choice(pool)) if len(pool) else int(dxz.argmin())
            eye = c[j]
            yaw = yaw0 + g.uniform(-math.pi, math.pi) * s[i]
            pitch = float(np.clip(pitch0 + g.normal(0, math.radians(15)) * s[i], -.7, .7))
            eyes.append(eye)
            tgts.append(eye + _dir(yaw, pitch))
            rolls.append(g.uniform(-math.pi, math.pi) * s[i])
            fs.append(f0 * math.exp(g.uniform(-1, 1) * s[i] * math.log(2)))
    else:
        c0 = sc["center"].cpu().numpy()
        rad = sc["rad"]
        d0 = gt["t"][0].cpu().numpy() - c0
        r0 = np.linalg.norm(d0)
        u0 = d0 / r0
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


@torch.no_grad()
def _depth_ok(mesh, cam, unit, bs=CFG["render_bs"]):
    """Interior views only: reject cameras staring at a wall from close range / looking out of the house. Returns (B,)."""
    geom = Meshes([mesh.verts_packed()], [mesh.faces_packed()])  # geometry only: cheap to extend
    bs = _safe_bs(geom, bs, textured=False)
    res = []
    for i in range(0, n_cams(cam), bs):
        c = idx_cams(cam, slice(i, i + bs))
        z = renderer.rasterizer(geom.extend(n_cams(c)), cameras=to_p3d(c)).zbuf[..., 0]  # (B,H,W), -1 = no hit
        hit = z > 0
        zi = torch.where(hit, z, torch.full_like(z, 1e9))
        near = (zi < MIN_NEAR * unit).float().mean((1, 2))
        med = zi.flatten(1).median(1).values
        res.append((hit.float().mean((1, 2)) >= .9) & (near <= .03) & (med >= MIN_MED * unit))
    return torch.cat(res)


def _pre_ok(cam, sc):
    """Cheap validity test that needs no shading. orbit: MIN_VIS of the scene points visible. interior: depth test."""
    if CAM_MODE == "orbit":
        return visible_frac(cam, sc["pts"]) >= MIN_VIS
    return _depth_ok(sc["mesh"], cam, sc["unit"])


def build_scene_data(seed, n_cams_=CFG["n_cams"]):
    """Scene + GT camera/image + `n_cams_` perturbed cameras/images.
    Every camera must have object coverage >= min_cov and pass the view test (_pre_ok)."""
    sc = make_scene(seed)
    g = np.random.default_rng(5000 + seed)
    cov_gt = float("nan")
    for _ in range(200):
        gt = sample_gt(g, sc)
        if not bool(_pre_ok(gt, sc)[0]):
            continue
        img_gt, m_gt = render_batch(sc["mesh"], gt)
        cov_gt = float(m_gt.mean())
        if cov_gt >= CFG["min_cov"]:
            break
    else:
        raise RuntimeError(f"scene {seed}: no valid GT camera found (last coverage {cov_gt:.3f}, mode {CAM_MODE})")

    cams, S, imgs, covs = [], [], [], []
    have, tries = 0, 0
    while have < n_cams_:
        tries += 1
        if tries > 200:
            raise RuntimeError(f"scene {seed}: only {have}/{n_cams_} valid cameras after {tries - 1} attempts "
                               f"(try lowering min_cov / min_vis)")
        m = int(1.5 * (n_cams_ - have)) + 10
        c, s = sample_random_cams(g, m, gt, sc)
        vis_ok = _pre_ok(c, sc).nonzero()[:, 0]  # cheap pre-filter, no shading needed
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
