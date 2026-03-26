import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from scipy.spatial.transform import Rotation

from diffsynth.auxiliary_models.worldmirror.models.models.rasterization import Gaussians
from diffsynth.auxiliary_models.worldmirror.models.utils.rotation import quat_to_rotmat, rotmat_to_quat


def get_camera_extrinsics(base_path: Path, camera_name: str) -> tuple[Tensor, Tensor]:
    """Read models.json and return (c2w, K) for the named camera.

    Args:
        base_path: Directory containing models.json and camera subdirs.
        camera_name: Camera name matching the 'name' field in models.json.

    Returns:
        c2w: [4, 4] camera-to-world matrix (float32).
        K:   [3, 3] intrinsic matrix in pixel units (float32).
    """
    with open(base_path / "models.json") as f:
        views = json.load(f)

    view = None
    for v in views:
        if v["name"] == camera_name:
            view = v
            break
    assert view is not None, f"Camera {camera_name} not found in models.json"

    # Build w2c (same convention as AnySplat)
    R = torch.from_numpy(
        Rotation.from_rotvec(view["orientation"]).as_matrix()
    ).to(torch.float32)
    t = torch.tensor(view["position"], dtype=torch.float32)[:, None]

    w2c = torch.eye(4, dtype=torch.float32)
    w2c[:3, :3] = R
    w2c[:3, 3:] = -R @ t

    # Invert to get c2w
    R_inv = R.T
    t_inv = -R_inv @ (-R @ t)  # = t
    c2w = torch.eye(4, dtype=torch.float32)
    c2w[:3, :3] = R_inv
    c2w[:3, 3:] = t_inv

    # Intrinsic matrix (pixel units)
    K = torch.tensor([
        [view["focal_length"], 0.0, view["principal_point"][0]],
        [0.0, view["focal_length"], view["principal_point"][1]],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)

    return c2w, K


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
