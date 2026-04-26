"""Extract frames from the N3DV dataset.

Input layout:
    <dataset_root>/<scene>/cam<XX>.mp4        (XX in 00..20)

Output layout (written in-place under each scene):
    <dataset_root>/<scene>/camera_<XXXX>/<NNNNN>.png

Frames are decoded with ffmpeg (libx264 -> PNG). Each scene's cameras are
extracted in parallel via a process pool.

Example:
    python extract_n3dv_frames.py --dataset_root /data/n3dv
    python extract_n3dv_frames.py --dataset_root /data/n3dv --scenes coffee_martini cook_spinach
    python extract_n3dv_frames.py --dataset_root /data/n3dv --workers 8 --overwrite
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

CAM_RE = re.compile(r"^cam(\d{2})\.mp4$", re.IGNORECASE)


def discover_scenes(dataset_root: Path, scenes_filter: Optional[List[str]]) -> List[Path]:
    if scenes_filter:
        scenes = [dataset_root / s for s in scenes_filter]
        missing = [str(s) for s in scenes if not s.is_dir()]
        if missing:
            raise FileNotFoundError(f"Scene folder(s) not found: {missing}")
        return scenes
    return sorted(p for p in dataset_root.iterdir() if p.is_dir())


def discover_cam_videos(scene_dir: Path) -> List[Tuple[int, Path]]:
    cams: List[Tuple[int, Path]] = []
    for entry in sorted(scene_dir.iterdir()):
        if not entry.is_file():
            continue
        m = CAM_RE.match(entry.name)
        if m:
            cams.append((int(m.group(1)), entry))
    return cams


def extract_one(
    video_path_str: str,
    out_dir_str: str,
    frame_pattern: str,
    overwrite: bool,
    qscale: int,
) -> Tuple[str, int, Optional[str]]:
    """Extract all frames of one video into ``out_dir``. Runs in a worker process."""
    video_path = Path(video_path_str)
    out_dir = Path(out_dir_str)

    if out_dir.exists():
        if overwrite:
            shutil.rmtree(out_dir)
        elif any(out_dir.iterdir()):
            return (str(out_dir), 0, "exists (skipped; use --overwrite to redo)")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_template = str(out_dir / frame_pattern)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-y",
        "-i", str(video_path),
        "-start_number", "0",
        "-vsync", "0",
        "-qscale:v", str(qscale),
        out_template,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        return (str(out_dir), 0, e.stderr.decode("utf-8", errors="replace").strip() or "ffmpeg failed")

    n = sum(1 for p in out_dir.iterdir() if p.suffix.lower() == ".png")
    return (str(out_dir), n, None)


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract N3DV mp4 frames to camera_XXXX/ PNG folders.")
    ap.add_argument("--dataset_root", type=Path, required=True,
                    help="Root containing <scene>/cam<XX>.mp4 files.")
    ap.add_argument("--scenes", nargs="*", default=None,
                    help="Optional subset of scene folder names. Default: all subdirs.")
    ap.add_argument("--workers", type=int, default=4,
                    help="Number of videos to extract in parallel (default 4).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Wipe existing camera_XXXX/ output dirs before extracting.")
    ap.add_argument("--frame_pattern", default="%05d.png",
                    help="ffmpeg output pattern relative to camera_XXXX/ (default '%%05d.png').")
    ap.add_argument("--qscale", type=int, default=2,
                    help="ffmpeg PNG qscale:v (lower=better; default 2).")
    args = ap.parse_args()

    if not args.dataset_root.is_dir():
        print(f"[ERROR] dataset_root not found: {args.dataset_root}", file=sys.stderr)
        return 2

    scenes = discover_scenes(args.dataset_root, args.scenes)
    if not scenes:
        print(f"[ERROR] No scenes found under {args.dataset_root}", file=sys.stderr)
        return 2

    jobs: List[Tuple[str, str]] = []  # (video_path, out_dir)
    for scene_dir in scenes:
        cams = discover_cam_videos(scene_dir)
        if not cams:
            print(f"[WARN] No cam<XX>.mp4 under {scene_dir}; skipping.")
            continue
        print(f"[{scene_dir.name}] {len(cams)} camera videos")
        for cam_idx, video_path in cams:
            out_dir = scene_dir / f"camera_{cam_idx:04d}"
            jobs.append((str(video_path), str(out_dir)))

    if not jobs:
        print("[ERROR] No videos to extract.", file=sys.stderr)
        return 2

    print(f"Extracting {len(jobs)} videos with {args.workers} worker(s)...")
    failures = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(
                extract_one,
                vid, out, args.frame_pattern, args.overwrite, args.qscale,
            ): (vid, out)
            for (vid, out) in jobs
        }
        for fut in as_completed(futs):
            vid, out = futs[fut]
            try:
                out_dir, n, err = fut.result()
            except Exception as e:
                failures += 1
                print(f"  [FAIL] {Path(vid).name} -> {Path(out).name}: {e}")
                continue
            if err:
                if err.startswith("exists"):
                    print(f"  [SKIP] {Path(out).name}: {err}")
                else:
                    failures += 1
                    print(f"  [FAIL] {Path(vid).name} -> {Path(out).name}: {err}")
            else:
                print(f"  [OK]   {Path(vid).name} -> {Path(out).name}  ({n} frames)")

    print(f"\nDone. {len(jobs) - failures}/{len(jobs)} videos extracted; "
          f"{failures} failure(s).")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
