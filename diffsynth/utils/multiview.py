import json
import re
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from scipy.spatial.transform import Rotation

from diffsynth.auxiliary_models.worldmirror.models.models.rasterization import Gaussians
from diffsynth.auxiliary_models.worldmirror.models.utils.rotation import quat_to_rotmat, rotmat_to_quat
from diffsynth.auxiliary_models.depth_anything_3.utils.pose_align import align_poses_umeyama
from diffsynth.utils.auxiliary import homo_matrix_inverse


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


def discover_cameras(base_path: Path) -> list[str]:
    """Discover camera folder names for a scene from its metadata.

    - ``models.json`` -> camera names from the ``name`` field.
    - ``cameras/camera_extrinsics.json`` -> folder names whose
      ``_folder_to_json_cam(...)`` matches a key in the JSON.
    """
    base_path = Path(base_path)
    models_path = base_path / "models.json"
    extrinsics_path = base_path / "cameras" / "camera_extrinsics.json"

    if models_path.exists():
        with open(models_path) as f:
            views = json.load(f)
        return sorted(v["name"] for v in views)

    if extrinsics_path.exists():
        with open(extrinsics_path) as f:
            data = json.load(f)
        first_frame = next(iter(data.values()))
        json_keys = set(first_frame.keys())
        cams = []
        for entry in sorted(base_path.iterdir()):
            if not entry.is_dir() or entry.name == "cameras":
                continue
            try:
                key = _folder_to_json_cam(entry.name)
            except ValueError:
                continue
            if key in json_keys:
                cams.append(entry.name)
        return cams

    raise FileNotFoundError(
        f"Neither {models_path} nor {extrinsics_path} exists; cannot "
        f"discover cameras for scene at {base_path}."
    )


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


def estimate_sim3_local_to_gt(
    pred_c2w: Tensor, gt_c2w: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Estimate a Sim(3) that maps the reconstructor's local frame to GT world.

    Runs Umeyama alignment on W paired predicted / GT camera poses (camera
    centers + orientations). The returned transform is defined by
    ``p_world = s * (R @ p_local) + t`` and is the same Sim(3) applied to
    Gaussian centers, scales and rotations in ``transform_gaussians_to_world``.

    Args:
        pred_c2w: [W, 4, 4] predicted camera-to-world from the reconstructor.
        gt_c2w:   [W, 4, 4] GT camera-to-world for the same context frames.

    Returns:
        Tuple ``(R, t, s)`` on ``pred_c2w``'s device/dtype, with shapes
        ``[3, 3]``, ``[3]``, ``[]``.
    """
    assert pred_c2w.shape == gt_c2w.shape and pred_c2w.shape[-2:] == (4, 4), (
        f"pred_c2w {tuple(pred_c2w.shape)} and gt_c2w {tuple(gt_c2w.shape)} "
        "must share shape [W, 4, 4]"
    )
    assert pred_c2w.shape[0] >= 3, "Umeyama alignment needs at least 3 poses"

    pred_w2c_np = homo_matrix_inverse(pred_c2w).detach().cpu().double().numpy()
    gt_w2c_np = homo_matrix_inverse(gt_c2w).detach().cpu().double().numpy()

    R_np, t_np, s_np = align_poses_umeyama(gt_w2c_np, pred_w2c_np)

    device, dtype = pred_c2w.device, pred_c2w.dtype
    R = torch.from_numpy(np.ascontiguousarray(R_np)).to(device=device, dtype=dtype)
    t = torch.from_numpy(np.asarray(t_np, dtype=np.float64).reshape(3)).to(
        device=device, dtype=dtype
    )
    s = torch.tensor(float(s_np), device=device, dtype=dtype)
    return R, t, s


def transform_gaussians_to_world(
    gaussians: Gaussians, R: Tensor, t: Tensor, s: Tensor
) -> Gaussians:
    """Apply a Sim(3) ``(R, t, s)`` to Gaussians living in the reconstructor's
    scale-ambiguous local frame, producing Gaussians in the target world frame.

    Mapping: ``p_world = s * (R @ p_local) + t``. Rotation composes with each
    Gaussian's orientation; ``scales`` is multiplied by ``s``; velocities are
    rotated and rescaled. Photometric / opacity / timestamp fields pass through.

    Args:
        gaussians: Gaussians in the reconstructor's local frame.
        R: [3, 3] rotation.
        t: [3] translation.
        s: scalar scale (0-d tensor).
    """
    means_world = (gaussians.means @ R.T) * s + t.unsqueeze(0)

    scales_world = gaussians.scales * s

    # wxyz -> xyzw, compose with R, back to wxyz
    quats_wxyz = gaussians.rotations
    quats_xyzw = quats_wxyz[:, [1, 2, 3, 0]]
    R_local = quat_to_rotmat(quats_xyzw)
    R_world = R.unsqueeze(0) @ R_local
    quats_world_xyzw = rotmat_to_quat(R_world)
    quats_world_wxyz = quats_world_xyzw[:, [3, 0, 1, 2]]

    def _vel(v):
        return (v @ R.T) * s if v is not None else None

    return Gaussians(
        means=means_world,
        harmonics=gaussians.harmonics,
        opacities=gaussians.opacities,
        scales=scales_world,
        rotations=quats_world_wxyz,
        confidences=getattr(gaussians, 'confidences', None),
        timestamp=gaussians.timestamp,
        life_span=gaussians.life_span,
        life_span_gamma=getattr(gaussians, 'life_span_gamma', 0.0),
        forward_timestamp=gaussians.forward_timestamp,
        forward_vel=_vel(gaussians.forward_vel),
        forward_scales=getattr(gaussians, 'forward_scales', None),
        forward_rotations=getattr(gaussians, 'forward_rotations', None),
        backward_timestamp=gaussians.backward_timestamp,
        backward_vel=_vel(gaussians.backward_vel),
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
