#!/usr/bin/env bash
# Environment used for the paper (Colab, CUDA 12.8). Run once, then restart the runtime.
set -e
pip install -q fvcore iopath
pip install -q --index https://wheels.astral.sh/simple/cu128/ pytorch3d
pip uninstall -y torchaudio
pip install -q transformers scipy pandas matplotlib pillow
echo "Done. Restart the runtime, then run scripts/run_all.sh"
