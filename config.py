"""Global configuration shared by every module: constants, grids, thresholds and output paths.

Dependency order of the project (one-directional, no circular imports):

    config -> utils -> scene -> losses -> optim -> exp*.py
"""
import ctypes
import glob
import os
import random
import site
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")


def _ensure_cuda_runtime():
    """Make `pytorch3d._C` importable on machines (e.g. Colab) where libcudart.so.12 is not on the loader path."""
    try:
        import pytorch3d._C  # noqa: F401
        return
    except ModuleNotFoundError:
        raise
    except (ImportError, OSError):
        pass

    def find():
        pats = []
        for sp in site.getsitepackages() + [site.getusersitepackages()]:
            pats += [f"{sp}/nvidia/cuda_runtime/lib/libcudart.so.12*", f"{sp}/nvidia/cu12/lib/libcudart.so.12*"]
        pats += ["/usr/local/cuda*/lib64/libcudart.so.12*", "/usr/lib/x86_64-linux-gnu/libcudart.so.12*"]
        return [p for pat in pats for p in glob.glob(pat)]

    libs = find()
    if not libs:
        subprocess.run([sys.executable, "-m", "pip", "-q", "install", "nvidia-cuda-runtime-cu12"], check=False)
        libs = find()
    if not libs:
        raise RuntimeError("libcudart.so.12 was not found; install a CUDA 12 runtime.")
    ctypes.CDLL(libs[0], mode=ctypes.RTLD_GLOBAL)


_ensure_cuda_runtime()

dev = torch.device("cuda")

# --------------------------------------------------------------------------------------
# Data / rendering
# --------------------------------------------------------------------------------------
# LR_ERROR_WEIGHT is an important hyper-parameter of the learning-rate tuning (experiment 2).
# The tuning score of a (loss, lr) pair is
#       score = mean(speed) - LR_ERROR_WEIGHT * median(log10(final error)),
# i.e. it controls how much the final error matters relative to the convergence speed.
CFG = dict(seed=0, img=224, n_scenes=40, n_cams=200, min_cov=0.1, render_bs=50, LR_ERROR_WEIGHT=0.1)
IMG = CFG["img"]

TUNE_SCENES = list(range(10))
TEST_SCENES = list(range(10, CFG["n_scenes"]))

# --------------------------------------------------------------------------------------
# Loss grids
# --------------------------------------------------------------------------------------
ZF_HARD = [0.1, 0.5, 1, 2, 5, 10]
ZF_SOFT = [0.1, 0.5, 1, 2, 5, 10]
SIGMAS = [0.25, 1, 2, 4, 8, 16, 32, 64]
SG_RHO, SG_SIG, SG_ZF = [4, 12, 24, 48], [2, 4, 8], [1]      # SoftGrid: Gaussian width (px), sigma, zf

# Bump this whenever ANY loss formula changes: it invalidates every cache that depends on the losses.
LOSS_VERSION = "v2-signinv-sqrt"

# --------------------------------------------------------------------------------------
# Optimisation / convergence criteria (experiment 2 and its derivatives)
# --------------------------------------------------------------------------------------
T_STEPS, T_SCALE = 500, 3.0
LR_GRID = [0.005, 0.008, 0.01, 0.03, 0.05, 0.08, 0.1, 0.3, 1.0]
ROT_T, LOGF_T, TRANS_FRAC = 2.0, 0.02, 0.01     # rotation [deg], |log focal|, translation (fraction of GT distance)

# --------------------------------------------------------------------------------------
# Output paths (can be changed with --out or the FOCUS_RESULTS environment variable)
# --------------------------------------------------------------------------------------
_RESULTS = Path(os.environ.get("FOCUS_RESULTS", "results"))


def set_results_dir(path):
    global _RESULTS
    _RESULTS = Path(path)


def results_dir():
    return _RESULTS


def cache_dir():
    """Heavy intermediate files (scene caches, tuning table); not meant to be committed."""
    p = _RESULTS / "cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def exp_dir(name):
    """Output folder of one experiment, e.g. exp_dir("exp2") -> results/exp2."""
    p = _RESULTS / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
