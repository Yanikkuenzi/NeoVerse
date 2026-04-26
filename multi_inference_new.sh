#!/usr/bin/env bash
set -euo pipefail

# Render-only multi-camera NVS run.
# Output PNGs land at $OUTPUT_PATH/<take_name>/<cam>/<original_filename>.
# A separate script computes PSNR/SSIM/LPIPS by diffing against the GT tree.

# ---- Edit these to change a run -----------------------------------------
SCENES_TXT="/iopsstor/scratch/cscs/yknzi/anysplat/MultiCamVideo-Dataset/multicamvideo_test.txt"
SCENES_ROOT="/iopsstor/scratch/cscs/yknzi/anysplat/MultiCamVideo-Dataset/MultiCamVideo-Dataset/train/f35_aperture2.4"
OUTPUT_PATH="outputs/multi_new"
CAMERAS=("camera_0000" "camera_0001" "camera_0002" "camera_0003")  # exactly 4; empty = auto-discover per scene
HEIGHT=336
WIDTH=560
RESIZE_MODE="center_crop"
LOW_VRAM=0                          # 1 = pass --low_vram, 0 = omit
MODEL_PATH="/iopsstor/scratch/cscs/yknzi/anysplat/neoverse-models/"
RECONSTRUCTOR_PATH="/iopsstor/scratch/cscs/yknzi/anysplat/neoverse-models/NeoVerse/reconstructor.ckpt"
MAX_WINDOWS=                        # empty = render every window per scene
NO_SKIP_FRAMES_IN_MOTION_BRANCH=0   # 1 = motion stride 1 (ablation), 0 = stride 4 (default)
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
(( ${#CAMERAS[@]} )) && args+=(--cameras "${CAMERAS[@]}")
(( LOW_VRAM )) && args+=(--low_vram)
(( NO_SKIP_FRAMES_IN_MOTION_BRANCH )) && args+=(--no_skip_frames_in_motion_branch)
[[ -n "${MAX_WINDOWS}" ]] && args+=(--max_windows "$MAX_WINDOWS")

echo "[multi_new] scenes from $SCENES_TXT -> $OUTPUT_PATH"
python inference_multi_new.py "${args[@]}"
