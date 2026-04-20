#!/usr/bin/env bash
set -euo pipefail

# ---- Edit these to change a run -----------------------------------------
SCENES_ROOT="/iopsstor/scratch/cscs/yknzi/anysplat/MultiCamVideo-Dataset/MultiCamVideo-Dataset/train/f35_aperture2.4"
OUTPUT_ROOT="outputs/multi"
HEIGHT=336
WIDTH=560
BATCH_SIZE=41
RESIZE_MODE="center_crop"
LOW_VRAM=0                # 1 = pass --low_vram, 0 = omit
MODEL_PATH="/iopsstor/scratch/cscs/yknzi/anysplat/neoverse-models/"
RECONSTRUCTOR_PATH="/iopsstor/scratch/cscs/yknzi/anysplat/neoverse-models/NeoVerse/reconstructor.ckpt"
MAX_SCENES=15
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
num_scenes=0
for scene_dir in "$SCENES_ROOT"/*/; do
    scene_name=$(basename "$scene_dir")
    if [[ ! -f "$scene_dir/models.json" && ! -f "$scene_dir/cameras/camera_extrinsics.json" ]]; then
        echo "[multi] $scene_name: no models.json or cameras/camera_extrinsics.json, skipping"
        continue
    fi
    out_dir="$OUTPUT_ROOT/$scene_name"
    echo "[multi] $scene_name -> $out_dir"
    python inference_multiview.py \
        "${common_args[@]}" \
        --input_path "$scene_dir" \
        --output_path "$out_dir"
    num_scenes=$((num_scenes + 1))
    if [ $num_scenes -ge $MAX_SCENES ]; then
        break;
    fi
done
