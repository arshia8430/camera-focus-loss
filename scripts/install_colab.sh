#!/usr/bin/env bash
# Environment + dataset used for the paper (Colab, CUDA 12.8). Run once from the repo root, then restart the runtime.
#   KEEP_ARCHIVES=1  keep the downloaded zip parts (default: delete them after extraction)
#   BLENDEDMVS_MESH_ROOT=<dir>  where to put the meshes (default data/BlendedMVS_textured_meshes, must match the code)
set -e

# ---- Python environment ----------------------------------------------------------------
pip install -q fvcore iopath
pip install -q --index https://wheels.astral.sh/simple/cu128/ pytorch3d
pip uninstall -y torchaudio
pip install -q transformers scipy pandas matplotlib pillow

# ---- Dataset: BlendedMVS textured meshes (~9.4 GB download, ~30 GB free disk needed at peak) ----
MESH_ROOT="${BLENDEDMVS_MESH_ROOT:-data/BlendedMVS_textured_meshes}"
BASE_URL="https://github.com/YoYo000/BlendedMVS/releases/download/v1.0.4"
NAME="dataset_textured_meshes"
PARTS=4  # .z01 ... .z04 plus the final .zip

count_obj() { find "$MESH_ROOT" -name '*.obj' 2>/dev/null | wc -l; }

if [ "$(count_obj)" -gt 0 ]; then
    echo "Meshes already present in $MESH_ROOT ($(count_obj) .obj files), skipping download."
else
    if ! command -v zip >/dev/null || ! command -v unzip >/dev/null; then
        apt-get update -qq && apt-get install -y -qq zip unzip
    fi
    mkdir -p "$MESH_ROOT"
    OUT="$(realpath "$MESH_ROOT")"
    TMP="$(mktemp -d ./bmvs_dl.XXXXXX)"
    (
        cd "$TMP"
        for i in $(seq 1 "$PARTS"); do
            wget -c -q --show-progress "$BASE_URL/$NAME.z$(printf %02d "$i")"
        done
        wget -c -q --show-progress "$BASE_URL/$NAME.zip"
        # the release is a split zip: join the parts, then extract
        zip -s 0 "$NAME.zip" --out "${NAME}_joined.zip"
        if [ "${KEEP_ARCHIVES:-0}" != "1" ]; then rm -f "$NAME".z[0-9][0-9] "$NAME.zip"; fi
        unzip -q -o "${NAME}_joined.zip" -d "$OUT"
    )
    if [ "${KEEP_ARCHIVES:-0}" != "1" ]; then rm -rf "$TMP"; else echo "Archives kept in $TMP"; fi

    N=$(count_obj)
    if [ "$N" -eq 0 ]; then
        echo "ERROR: no .obj files found under $MESH_ROOT after extraction; check the folder structure:" >&2
        find "$MESH_ROOT" -maxdepth 3 | head -30 >&2
        exit 1
    fi
    echo "Extracted $N .obj meshes into $MESH_ROOT"
fi

echo "Done. Restart the runtime, then run scripts/run_all.sh"