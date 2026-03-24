import json
import os
from pathlib import Path

import torch
from torch import Tensor

from diffsynth.auxiliary_models.worldmirror.models.models.rasterization import Gaussians
from diffsynth.auxiliary_models.worldmirror.models.utils.rotation import quat_to_rotmat, rotmat_to_quat


def transform_gaussians_to_world(gaussians: Gaussians, gt_c2w: Tensor) -> Gaussians:
    """Transform Gaussians from camera-local frame to world frame via GT c2w.

    Args:
        gaussians: Gaussians with parameters in the reconstructor's local
            coordinate frame (frame 0 at origin).
        gt_c2w: [4, 4] ground-truth camera-to-world matrix for this view.

    Returns:
        New Gaussians with means and rotations transformed to world frame.
    """
    R_gt = gt_c2w[:3, :3]  # [3, 3]
    t_gt = gt_c2w[:3, 3]   # [3]

    # Transform means: world_pt = R_gt @ local_pt + t_gt
    means_world = gaussians.means @ R_gt.T + t_gt.unsqueeze(0)

    # Transform rotations (wxyz convention in Gaussians)
    # Convert wxyz -> xyzw for rotation utilities
    quats_wxyz = gaussians.rotations  # [N, 4]
    quats_xyzw = quats_wxyz[:, [1, 2, 3, 0]]

    # Convert to rotation matrices, compose with GT rotation, convert back
    R_local = quat_to_rotmat(quats_xyzw)    # [N, 3, 3]
    R_world = R_gt.unsqueeze(0) @ R_local    # [N, 3, 3]
    quats_world_xyzw = rotmat_to_quat(R_world)  # [N, 4] xyzw

    # Convert xyzw -> wxyz
    quats_world_wxyz = quats_world_xyzw[:, [3, 0, 1, 2]]

    return Gaussians(
        means=means_world,
        harmonics=gaussians.harmonics,
        opacities=gaussians.opacities,
        scales=gaussians.scales,
        rotations=quats_world_wxyz,
        confidences=getattr(gaussians, 'confidences', None),
        timestamp=gaussians.timestamp,
        life_span=gaussians.life_span,
    )


def load_multiview_config(path: str) -> dict:
    """Load and validate a multi-view JSON config file.

    Expected format:
    {
      "views": [
        {
          "image_dir": "path/to/frames/",
          "c2w": [[4x4 matrix]]
        }, ...
      ],
      "render_viewpoint": {
        "c2w": [[4x4 matrix]],
        "intrinsics": [[3x3 matrix]]
      }
    }
    """
    with open(path, 'r') as f:
        config = json.load(f)

    if "views" not in config or not config["views"]:
        raise ValueError("Config must contain a non-empty 'views' list.")
    if "render_viewpoint" not in config:
        raise ValueError("Config must contain 'render_viewpoint'.")

    for i, view in enumerate(config["views"]):
        if "image_dir" not in view:
            raise ValueError(f"View {i} must contain 'image_dir'.")
        if "c2w" not in view:
            raise ValueError(f"View {i} must contain 'c2w'.")
        c2w = view["c2w"]
        if len(c2w) != 4 or any(len(row) != 4 for row in c2w):
            raise ValueError(f"View {i} 'c2w' must be a 4x4 matrix.")

    rv = config["render_viewpoint"]
    if "c2w" not in rv:
        raise ValueError("render_viewpoint must contain 'c2w'.")
    if "intrinsics" not in rv:
        raise ValueError("render_viewpoint must contain 'intrinsics'.")
    if len(rv["intrinsics"]) != 3 or any(len(row) != 3 for row in rv["intrinsics"]):
        raise ValueError("render_viewpoint 'intrinsics' must be a 3x3 matrix.")

    return config


def load_frames_from_dir(image_dir: str) -> list[str]:
    """Load image file paths from a directory in sorted order.

    Returns:
        Sorted list of absolute image file paths.
    """
    supported_ext = {'.png', '.jpg', '.jpeg'}
    dir_path = Path(image_dir)
    if not dir_path.is_dir():
        raise ValueError(f"Image directory does not exist: {image_dir}")

    files = sorted(
        p for p in dir_path.iterdir()
        if p.suffix.lower() in supported_ext
    )
    if not files:
        raise ValueError(f"No image files found in {image_dir}")

    return [str(f) for f in files]
