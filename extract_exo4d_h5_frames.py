"""Extract per-camera PNG frames from Exo4D-style HDF5 scene files.

Input layout:
    <h5_root>/<scene>/<scene>.h5

H5 schema (see dataset_exo4d.py):
    hf["frames"]      uint8, shape [T, num_cameras, H, W, 3]
    hf["camera_ids"]  num_cameras strings (file order)
    hf.attrs["num_frames"], hf.attrs["width"], hf.attrs["height"]

Output layout:
    <out_root>/<scene>/camera_<XXXX>/<NNNNN>.png

XXXX is the 1-based camera index in the h5 file's storage order
(camera_0001 == frames[t, 0]). NNNNN is the zero-padded frame index.

Example:
    python extract_exo4d_h5_frames.py \\
        --h5_root /data/exo4d \\
        --scenes_file scenes.txt \\
        --out_root /data/exo4d_png \\
        --workers 8
"""

from __future__ import annotations

import argparse
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import h5py


def read_scenes_file(scenes_file: Path) -> List[str]:
    scenes: List[str] = []
    with scenes_file.open("r") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            scenes.append(s)
    return scenes


def scene_h5_path(h5_root: Path, scene: str) -> Path:
    return h5_root / scene / f"{scene}.h5"


def discover_jobs(
    h5_root: Path, scenes: List[str], out_root: Path,
) -> Tuple[List[Tuple[str, int, str, str]], List[str]]:
    """Return (jobs, missing_scenes).

    Each job is (h5_path, cam_idx, cam_label, out_dir).
    """
    jobs: List[Tuple[str, int, str, str]] = []
    missing: List[str] = []
    for scene in scenes:
        h5_path = scene_h5_path(h5_root, scene)
        if not h5_path.is_file():
            missing.append(scene)
            continue
        with h5py.File(h5_path, "r") as hf:
            num_cameras = int(hf["frames"].shape[1])
        print(f"[{scene}] {num_cameras} cameras")
        for cam_idx in range(num_cameras):
            cam_label = f"camera_{cam_idx + 1:04d}"
            out_dir = out_root / scene / cam_label
            jobs.append((str(h5_path), cam_idx, cam_label, str(out_dir)))
    return jobs, missing


def extract_one(
    h5_path_str: str,
    cam_idx: int,
    cam_label: str,
    out_dir_str: str,
    frame_pattern: str,
    overwrite: bool,
    png_compression: int,
) -> Tuple[str, int, Optional[str]]:
    """Extract every frame of one camera into ``out_dir``. Runs in a worker."""
    out_dir = Path(out_dir_str)

    if out_dir.exists():
        if overwrite:
            shutil.rmtree(out_dir)
        elif any(out_dir.iterdir()):
            return (str(out_dir), 0, "exists (skipped; use --overwrite to redo)")
    out_dir.mkdir(parents=True, exist_ok=True)

    write_params = [cv2.IMWRITE_PNG_COMPRESSION, int(png_compression)]
    n = 0
    with h5py.File(h5_path_str, "r") as hf:
        frames_ds = hf["frames"]
        num_frames = int(frames_ds.shape[0])
        for t in range(num_frames):
            rgb = frames_ds[t, cam_idx]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            out_path = str(out_dir / (frame_pattern % t))
            if not cv2.imwrite(out_path, bgr, write_params):
                return (str(out_dir), n, f"cv2.imwrite failed at frame {t}: {out_path}")
            n += 1

    if n == 0:
        return (str(out_dir), 0, "decoded 0 frames (h5 frames dataset is empty?)")
    return (str(out_dir), n, None)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Extract Exo4D h5 frames to <scene>/camera_XXXX/NNNNN.png folders.",
    )
    ap.add_argument("--h5_root", type=Path, required=True,
                    help="Root containing <scene>/<scene>.h5 files.")
    ap.add_argument("--scenes_file", type=Path, required=True,
                    help="Text file with one scene name per line (# comments allowed).")
    ap.add_argument("--out_root", type=Path, required=True,
                    help="Output root; PNGs land at <out_root>/<scene>/camera_XXXX/NNNNN.png.")
    ap.add_argument("--workers", type=int, default=4,
                    help="Number of (scene, camera) jobs to run in parallel (default 4).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Wipe existing camera_XXXX/ output dirs before extracting.")
    ap.add_argument("--frame_pattern", default="%05d.png",
                    help="Per-frame filename %% pattern (default '%%05d.png').")
    ap.add_argument("--png_compression", type=int, default=3,
                    help="cv2 PNG compression 0..9 (higher=smaller/slower; default 3).")
    args = ap.parse_args()

    if not args.h5_root.is_dir():
        print(f"[ERROR] h5_root not found: {args.h5_root}", file=sys.stderr)
        return 2
    if not args.scenes_file.is_file():
        print(f"[ERROR] scenes_file not found: {args.scenes_file}", file=sys.stderr)
        return 2

    scenes = read_scenes_file(args.scenes_file)
    if not scenes:
        print(f"[ERROR] No scenes listed in {args.scenes_file}", file=sys.stderr)
        return 2

    jobs, missing = discover_jobs(args.h5_root, scenes, args.out_root)
    for scene in missing:
        print(f"[WARN] Missing h5 for scene '{scene}' (expected {scene_h5_path(args.h5_root, scene)}); skipping.")

    if not jobs:
        print("[ERROR] No (scene, camera) jobs to run.", file=sys.stderr)
        return 2

    print(f"Extracting {len(jobs)} (scene, camera) jobs with {args.workers} worker(s)...")
    failures = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(
                extract_one,
                h5_path, cam_idx, cam_label, out_dir,
                args.frame_pattern, args.overwrite, args.png_compression,
            ): (h5_path, cam_label, out_dir)
            for (h5_path, cam_idx, cam_label, out_dir) in jobs
        }
        for fut in as_completed(futs):
            h5_path, cam_label, out_dir = futs[fut]
            scene_dir = Path(out_dir).parent.name
            try:
                _, n, err = fut.result()
            except Exception as e:
                failures += 1
                print(f"  [FAIL] {scene_dir}/{cam_label}: {e}")
                continue
            if err:
                if err.startswith("exists"):
                    print(f"  [SKIP] {scene_dir}/{cam_label}: {err}")
                else:
                    failures += 1
                    print(f"  [FAIL] {scene_dir}/{cam_label}: {err}")
            else:
                print(f"  [OK]   {scene_dir}/{cam_label}  ({n} frames)")

    print(f"\nDone. {len(jobs) - failures}/{len(jobs)} jobs extracted; "
          f"{failures} failure(s); {len(missing)} scene(s) missing.")
    return 0 if (failures == 0 and not missing) else 1


if __name__ == "__main__":
    raise SystemExit(main())
