#!/usr/bin/env bash
set -euo pipefail

# ---- Edit these to change a run -----------------------------------------
SCENES_ROOT="/iopsstor/scratch/cscs/yknzi/anysplat/kubric-eval"
OUTPUT_ROOT="outputs/multi"
HEIGHT=336
WIDTH=560
BATCH_SIZE=41
RESIZE_MODE="center_crop"
LOW_VRAM=0                # 1 = pass --low_vram, 0 = omit
MODEL_PATH="/iopsstor/scratch/cscs/yknzi/anysplat/neoverse-models/"
RECONSTRUCTOR_PATH="/iopsstor/scratch/cscs/yknzi/anysplat/neoverse-models/NeoVerse/reconstructor.ckpt"
CAMERAS=("camera_0001" "camera_0002" "camera_0003" "camera_0004")
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
    --cameras "$CAMERAS"
)
(( LOW_VRAM )) && common_args+=(--low_vram)


shopt -s nullglob
num_scenes=0
for scene_dir in "$SCENES_ROOT"/*/; do
    scene_name=$(basename "$scene_dir")
    has_kubric=0
    for json_path in "$scene_dir"*.json; do
        base="${json_path%.json}"
        if [[ -d "$base" ]]; then
            has_kubric=1
            break
        fi
    done
    if [[ ! -f "$scene_dir/models.json" \
       && ! -f "$scene_dir/cameras/camera_extrinsics.json" \
       && $has_kubric -eq 0 ]]; then
        echo "[multi] $scene_name: no models.json, cameras/camera_extrinsics.json, or Kubric <name>.json+<name>/ pair, skipping"
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
