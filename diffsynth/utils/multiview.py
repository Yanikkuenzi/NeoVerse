import json
import re
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from scipy.spatial.transform import Rotation

from diffsynth.auxiliary_models.worldmirror.models.models.rasterization import Gaussians
from diffsynth.auxiliary_models.worldmirror.models.utils.rotation import quat_to_rotmat, rotmat_to_quat


def _parse_ue5_matrix(matrix_str: str) -> np.ndarray:
    """Parse a MultiCamVideo-style matrix string into a (4, 4) numpy array.

    Ported from ReCamMaster's vis_cam.py. The string looks like
    ``"[r00 r01 r02 0] [r10 r11 r12 0] [r20 r21 r22 0] [tx ty tz 1]"``.
    3-element rows are padded with a trailing zero.
    """
    rows = matrix_str.strip().split('] [')
    matrix = []
    for row in rows:
        row = row.replace('[', '').replace(']', '')
        vals = list(map(float, row.split()))
        if len(vals) == 3:
            vals.append(0.0)
        matrix.append(vals)
    return np.array(matrix, dtype=np.float64)


def _folder_to_json_cam(folder_name: str) -> str:
    """Map a camera folder name like ``camera_0001`` to its JSON key ``cam01``."""
    m = re.search(r"(\d+)$", folder_name)
    if m is None:
        raise ValueError(
            f"Cannot derive camera index from folder name '{folder_name}'"
        )
    return f"cam{int(m.group(1)):02d}"


def load_scene_c2w(
    base_path: Path,
    camera_folders: list[str],
    num_frames: int,
) -> dict[str, Tensor]:
    """Load per-frame c2w matrices for every camera in the scene.

    Dispatches between the legacy ``models.json`` format (one static c2w per
    camera) and the MultiCamVideo ``cameras/camera_extrinsics.json`` format
    (per-frame w2c that is transformed to c2w following ReCamMaster's
    vis_cam.py).

    Returns a dict mapping each input folder name to a ``[num_frames, 4, 4]``
    float32 tensor of c2w matrices.
    """
    base_path = Path(base_path)
    models_path = base_path / "models.json"
    extrinsics_path = base_path / "cameras" / "camera_extrinsics.json"

    if models_path.exists():
        out: dict[str, Tensor] = {}
        for cam in camera_folders:
            c2w, _ = get_camera_extrinsics(base_path, cam)
            out[cam] = c2w.unsqueeze(0).expand(num_frames, -1, -1).contiguous()
        return out

    if extrinsics_path.exists():
        with open(extrinsics_path) as f:
            data = json.load(f)

        frame_keys = sorted(data.keys(), key=lambda k: int(re.search(r"\d+", k).group()))
        if len(frame_keys) != num_frames:
            raise ValueError(
                f"{extrinsics_path} has {len(frame_keys)} frames but scene has "
                f"{num_frames} image frames; counts must match."
            )

        out = {}
        for cam in camera_folders:
            json_key = _folder_to_json_cam(cam)
            mats = np.empty((num_frames, 4, 4), dtype=np.float32)
            for i, fk in enumerate(frame_keys):
                if json_key not in data[fk]:
                    raise KeyError(
                        f"Camera key '{json_key}' (folder '{cam}') not found in "
                        f"{extrinsics_path} at '{fk}'"
                    )
                m = _parse_ue5_matrix(data[fk][json_key])
                m = m.T
                m = m[:, [1, 2, 0, 3]]
                m[:3, 1] *= -1.0
                mats[i] = np.linalg.inv(m).astype(np.float32)
            out[cam] = torch.from_numpy(mats)
        return out

    raise FileNotFoundError(
        f"Neither {models_path} nor {extrinsics_path} exists; cannot load "
        f"camera poses for scene at {base_path}."
    )


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

    # Rotate velocity vectors to world frame
    fwd_vel = gaussians.forward_vel @ R_gt.T if gaussians.forward_vel is not None else None
    bwd_vel = gaussians.backward_vel @ R_gt.T if gaussians.backward_vel is not None else None

    return Gaussians(
        means=means_world,
        harmonics=gaussians.harmonics,
        opacities=gaussians.opacities,
        scales=gaussians.scales,
        rotations=quats_world_wxyz,
        confidences=getattr(gaussians, 'confidences', None),
        timestamp=gaussians.timestamp,
        life_span=gaussians.life_span,
        life_span_gamma=getattr(gaussians, 'life_span_gamma', 0.0),
        forward_timestamp=gaussians.forward_timestamp,
        forward_vel=fwd_vel,
        forward_scales=getattr(gaussians, 'forward_scales', None),
        forward_rotations=getattr(gaussians, 'forward_rotations', None),
        backward_timestamp=gaussians.backward_timestamp,
        backward_vel=bwd_vel,
        backward_scales=getattr(gaussians, 'backward_scales', None),
        backward_rotations=getattr(gaussians, 'backward_rotations', None),
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
