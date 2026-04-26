#!/usr/bin/env bash
set -euo pipefail

# ---- Edit these to change a run -----------------------------------------
SCENES_TXT="/iopsstor/scratch/cscs/yknzi/anysplat/MultiCamVideo-Dataset/multicamvideo_test.txt"
SCENES_ROOT="/iopsstor/scratch/cscs/yknzi/anysplat/MultiCamVideo-Dataset/MultiCamVideo-Dataset/train/f35_aperture2.4"
OUTPUT_PATH="outputs/multi_new"
HEIGHT=336
WIDTH=560
RESIZE_MODE="center_crop"
LOW_VRAM=0                          # 1 = pass --low_vram, 0 = omit
MODEL_PATH="/iopsstor/scratch/cscs/yknzi/anysplat/neoverse-models/"
RECONSTRUCTOR_PATH="/iopsstor/scratch/cscs/yknzi/anysplat/neoverse-models/NeoVerse/reconstructor.ckpt"
MAX_WINDOWS=                        # empty = evaluate every window per scene
NO_SKIP_FRAMES_IN_MOTION_BRANCH=0   # 1 = motion stride 1 (ablation), 0 = stride 4 (default)
SAVE_COMPARISON_IMAGES=0            # 1 = pass --save_comparison_images
MAX_SAVED_IMAGES=0                  # 0 = unlimited (only used with SAVE_COMPARISON_IMAGES=1)
# -------------------------------------------------------------------------

if [[ ! -f "$SCENES_TXT" ]]; then
    echo "SCENES_TXT not found: $SCENES_TXT" >&2
    exit 1
fi
if [[ ! -d "$SCENES_ROOT" ]]; then
    echo "SCENES_ROOT not found: $SCENES_ROOT" >&2
    exit 1
fi

args=(
    --scenes_txt "$SCENES_TXT"
    --scenes_root "$SCENES_ROOT"
    --output_path "$OUTPUT_PATH"
    --height "$HEIGHT"
    --width "$WIDTH"
    --resize_mode "$RESIZE_MODE"
    --model_path "$MODEL_PATH"
    --reconstructor_path "$RECONSTRUCTOR_PATH"
)
(( LOW_VRAM )) && args+=(--low_vram)
(( NO_SKIP_FRAMES_IN_MOTION_BRANCH )) && args+=(--no_skip_frames_in_motion_branch)
(( SAVE_COMPARISON_IMAGES )) && args+=(--save_comparison_images --max_saved_images "$MAX_SAVED_IMAGES")
[[ -n "${MAX_WINDOWS}" ]] && args+=(--max_windows "$MAX_WINDOWS")

echo "[multi_new] scenes from $SCENES_TXT -> $OUTPUT_PATH"
python inference_multi_new.py "${args[@]}"
