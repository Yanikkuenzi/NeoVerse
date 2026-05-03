#!/usr/bin/env bash
set -euo pipefail

# Render-only multi-camera NVS run.
# Output PNGs land at $OUTPUT_PATH/<take_name>/<cam>/<original_filename>.
# A separate script computes PSNR/SSIM/LPIPS by diffing against the GT tree.

# ---- Edit these to change a run -----------------------------------------
for num_cams in {2..12}
do
    SCENES_TXT="/iopsstor/scratch/cscs/yknzi/NeoVerse/n3dv-scenes.txt"
    SCENES_ROOT="/iopsstor/scratch/cscs/yknzi/anysplat/N3DV"
    OUTPUT_PATH="outputs/multi_new"
    ALL_CAMERAS=(camera_{0001..0012})  # exactly 4; empty = auto-discover per scene
    HEIGHT=336
    WIDTH=560
    RESIZE_MODE="center_crop"
    LOW_VRAM=0                          # 1 = pass --low_vram, 0 = omit
    MODEL_PATH="/iopsstor/scratch/cscs/yknzi/anysplat/neoverse-models/"
    RECONSTRUCTOR_PATH="/iopsstor/scratch/cscs/yknzi/anysplat/neoverse-models/NeoVerse/reconstructor.ckpt"
    MAX_WINDOWS=                        # empty = render every window per scene
    NO_SKIP_FRAMES_IN_MOTION_BRANCH=0   # 1 = motion stride 1 (ablation), 0 = stride 4 (default)
    # -------------------------------------------------------------------------

    CAMERAS=("${ALL_CAMERAS[@]:0:$num_cams}")
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
        --output_path "${OUTPUT_PATH}/eval_${num_cams}"
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

    echo "[multi_new] scenes from $SCENES_TXT -> $OUTPUT_PATH with $num_cams cameras"
    python inference_multi_new.py "${args[@]}"
done