#!/usr/bin/env bash
set -euo pipefail

# ---- Edit these to change a run -----------------------------------------
SCENES_ROOT="data/scenes"
OUTPUT_ROOT="outputs/multi"
HEIGHT=336
WIDTH=560
BATCH_SIZE=41
RESIZE_MODE="center_crop"
LOW_VRAM=0                # 1 = pass --low_vram, 0 = omit
MODEL_PATH="models"
RECONSTRUCTOR_PATH="models/NeoVerse/reconstructor.ckpt"
# -------------------------------------------------------------------------

if [[ ! -d "$SCENES_ROOT" ]]; then
    echo "SCENES_ROOT not found: $SCENES_ROOT" >&2
    exit 1
fi

common_args=(
    --height "$HEIGHT"
    --width "$WIDTH"
    --batch_size "$BATCH_SIZE"
    --resize_mode "$RESIZE_MODE"
    --model_path "$MODEL_PATH"
    --reconstructor_path "$RECONSTRUCTOR_PATH"
)
(( LOW_VRAM )) && common_args+=(--low_vram)

shopt -s nullglob
for scene_dir in "$SCENES_ROOT"/*/; do
    scene_name=$(basename "$scene_dir")
    if [[ ! -f "$scene_dir/models.json" ]]; then
        echo "[multi] $scene_name: no models.json, skipping"
        continue
    fi
    cam_dirs=("$scene_dir"*/)
    if (( ${#cam_dirs[@]} == 0 )); then
        echo "[multi] $scene_name: no camera subdirs, skipping"
        continue
    fi
    cameras=()
    for cam_dir in "${cam_dirs[@]}"; do
        cameras+=("$(basename "$cam_dir")")
    done
    out_dir="$OUTPUT_ROOT/$scene_name"
    echo "[multi] $scene_name (cameras: ${cameras[*]}) -> $out_dir"
    python inference_multiview.py \
        "${common_args[@]}" \
        --input_path "$scene_dir" \
        --cameras "${cameras[@]}" \
        --output_path "$out_dir"
done
