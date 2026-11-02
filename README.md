# Camera focus-point losses

Code for the experiments of the paper: focus-point based losses (Hard / Soft / SoftGrid) for camera pose
optimisation, compared against pose-regression, SE(3), ray and reprojection baselines on synthetic scenes.

## Layout

```
config.py                       all constants, loss grids, thresholds, output paths
utils.py                        camera helpers, geometry, statistics, DINOv2/CLIP embeddings, CLI wrapper
scene.py                        procedural scenes, camera sampling, PyTorch3D renderer
losses.py                       proposed losses + baselines + registry (LOSSES, LOSS_GROUP)
optim.py                        optimize(), convergence metrics, dataset loading, tuning tables

sanity_checks.py                loss/geometry unit checks (run first)
exp1_rank_correlation.py        losses vs. image-similarity ranking (also builds the scene cache)
exp2_convergence.py             LR tuning + convergence speed / success on the test scenes
exp3_similarity_trajectory.py   image similarity along the optimisation
exp4_threshold_sensitivity.py   sensitivity to the convergence thresholds
exp5_gifs.py                    qualitative convergence GIFs

scripts/                        install_colab.sh, run_all.sh
results/                        outputs (cache/ is git-ignored, paper/ holds the final tables and figures)
```

Dependency order (no circular imports): `config -> utils -> scene -> losses -> optim -> exp*.py`.

## Setup

A CUDA 12 GPU is required. `scripts/install_colab.sh` contains the recipe used for the paper
(PyTorch3D wheel for CUDA 12.8); other packages are listed in `requirements.txt`.

## Reproducing the results

```bash
bash scripts/run_all.sh results          # everything, in the right order
```

or step by step (the order matters, later experiments read the files of the earlier ones):

```bash
python sanity_checks.py                  # must pass before anything else
python exp1_rank_correlation.py          # scene cache  -> results/cache, results/exp1
python exp2_convergence.py               # LR tuning + test run -> results/cache/exp2_tuning.csv, results/exp2
python exp3_similarity_trajectory.py     # results/exp3
python exp4_threshold_sensitivity.py     # results/exp4
python exp5_gifs.py                      # results/exp5
```

Every experiment script has two stages, selectable with `--stage`:

| stage     | needs GPU | what it does                                                  |
|-----------|-----------|---------------------------------------------------------------|
| `run`     | yes       | produces CSV / npz files                                      |
| `analyze` | no        | reads those files and writes tables and figures               |
| `all`     | yes       | both (default)                                                |

So the tables and figures can be regenerated without re-running the experiments, e.g.
`python exp2_convergence.py --stage analyze`. Use `--out DIR` (or the `FOCUS_RESULTS` environment variable) to
choose the results directory.

## Which file produces what

| experiment | main outputs |
|------------|--------------|
| exp1 | `results/exp1/summary_dino_rho.csv`, `paired_vs_baselines_rho.csv`, `selected_losses_all_metrics.csv`, `exp1_summary.png` |
| exp2 | `results/exp2/summary_{speed,speed_hit,succ_final}.csv`, `paired_vs_baselines.csv`, `exp2_summary.png`, `exp2_vs_s.png`, `exp2_vs_sim.png`, `exp2_speed_vs_{s,sim}_{steps,ms}.png`, `paired_speedup_vs_baselines_{steps,ms}.csv` |
| exp3 | `results/exp3/exp3_simtraj_per_loss.png`, `exp3_simtraj_compare.png`, `exp3_simtraj_table_*.csv` |
| exp4 | `results/exp4/sensitivity_{strict,default,loose}.csv` |
| exp5 | `results/exp5/gifs/*.gif`, `gif_summary.csv` |

## Reproducibility notes

* All seeds and settings live in `config.py`; `CFG["seed"]` is applied by every `run` stage.
* The scene cache and the learning-rate tuning table are stored together with a metadata file. If a setting or
  `LOSS_VERSION` (in `config.py`) changes, the cache is rebuilt / the learning rates are re-tuned automatically.
  **Increase `LOSS_VERSION` whenever a loss formula changes.**
* Hyper-parameters of our loss families and all learning rates are selected on the tuning scenes
  (`TUNE_SCENES`); all reported numbers are on the disjoint test scenes (`TEST_SCENES`).
* Heavy artefacts (`results/cache/`, GIFs) are not committed; publish them as a release or on Zenodo.
