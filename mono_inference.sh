#!/usr/bin/env bash
set -euo pipefail

# ---- Edit these to change a run -----------------------------------------
SCENES_ROOT="/iopsstor/scratch/cscs/yknzi/anysplat/MultiCamVideo-Dataset/MultiCamVideo-Dataset/train/f35_aperture2.4"
OUTPUT_ROOT="outputs/mono"
HEIGHT=335
WIDTH=559
BATCH_SIZE=40
RESIZE_MODE="center_crop"
NON_STATIC_CAMERAS=0      # 1 = pass --non-static-cameras,  0 = omit
LOW_VRAM=-1                # 1 = pass --low_vram,            0 = omit
MODEL_PATH="/iopsstor/scratch/cscs/yknzi/anysplat/neoverse-models/"
RECONSTRUCTOR_PATH="/iopsstor/scratch/cscs/yknzi/anysplat/neoverse-models/NeoVerse/reconstructor.ckpt"
MAX_SCENES=15
# -------------------------------------------------------------------------

if [[ ! -d "$SCENES_ROOT" ]]; then
    echo "SCENES_ROOT not found: $SCENES_ROOT" >&2
    exit 1
fi

common_args=(
    --evaluate
    --height "$HEIGHT"
    --width "$WIDTH"
    --batch_size "$BATCH_SIZE"
    --resize_mode "$RESIZE_MODE"
    --model_path "$MODEL_PATH"
    --reconstructor_path "$RECONSTRUCTOR_PATH"
)
(( NON_STATIC_CAMERAS )) && common_args+=(--non-static-cameras)
(( LOW_VRAM ))           && common_args+=(--low_vram)

shopt -s nullglob
for scene_dir in "$SCENES_ROOT"/*/; do
    scene_name=$(basename "$scene_dir")
    cam_dirs=("$scene_dir"camera_{0001..0004}/)
    if (( ${#cam_dirs[@]} == 0 )); then
        echo "[mono] $scene_name: no camera subdirs, skipping"
        continue
    fi
    for cam_dir in "${cam_dirs[@]}"; do
        cam_name=$(basename "$cam_dir")
        out_dir="$OUTPUT_ROOT/$scene_name/$cam_name"
        echo "[mono] $scene_name / $cam_name -> $out_dir"
        python inference.py \
            "${common_args[@]}" \
            --input_path "$cam_dir" \
            --evaluate_output_path "$out_dir"
    done
done
