from pathlib import Path
import os
import sys
import struct

# HuggingFace's Xet transfer backend (hf-xet) buffers large downloads in RAM
# and can stall on a memory-constrained / headless box, wedging the first-time
# MASt3R checkpoint fetch. Force the classic streaming HTTP downloader unless
# the caller has deliberately chosen otherwise.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import cv2
import torch
import trimesh
import gc
import json

from colmap_utils import find_best_sparse_model
from image_geometry import original_to_prediction, prediction_to_original

from pipeline_config import (
    MAST3R_ROOT,
    get_sfm_dir,
    get_frames_dir,
    get_fusion_dir,
    ensure_job_directories,
)


# ============================================================
# PATHS
# ============================================================


SPARSE_DIR = get_sfm_dir() / "sparse"
FRAME_DIR = get_frames_dir()

OUTPUT_DIR = get_fusion_dir()

sys.path.insert(0, str(MAST3R_ROOT))


# ============================================================
# MAST3R
# ============================================================

from mast3r.model import AsymmetricMASt3R
from mast3r.cloud_opt.sparse_ga import symmetric_inference

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy


# ============================================================
# CONFIG
# ============================================================

def _select_device():
    """Pick the best available torch backend.

    The pipeline was originally written for Apple Silicon (mps). On a
    headless Linux/CPU box we fall back to CUDA when present, otherwise CPU.
    Override with SKYFORM_FUSION_DEVICE if needed.
    """
    forced = os.environ.get("SKYFORM_FUSION_DEVICE", "").strip().lower()
    if forced:
        return forced

    if torch.cuda.is_available():
        return "cuda"

    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def _empty_device_cache():
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()


DEVICE = _select_device()

# CPU inference is compute-bound here; the pipeline is normally launched with
# OMP_NUM_THREADS=1, which would make ViT-Large inference painfully slow. Give
# the ATen intra-op pool a sane default (overridable via SKYFORM_FUSION_THREADS).
if DEVICE == "cpu":
    import multiprocessing as _mp

    _default_threads = max(1, min(8, (_mp.cpu_count() or 2) - 2))
    _threads = int(os.environ.get("SKYFORM_FUSION_THREADS", str(_default_threads)))
    torch.set_num_threads(max(1, _threads))

MODEL_NAME = (
    "naver/"
    "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
)

IMAGE_SIZE = 512


def load_mast3r_model(model_name, device):
    """Load the MASt3R checkpoint without holding two full copies of the weights.

    HuggingFace's ``PyTorchModelHubMixin.from_pretrained`` first allocates the
    module with random init (~2 GB for ViT-Large) and then loads a second full
    copy of the state dict before assigning - a transient ~4 GB peak that thrashes
    a memory-constrained box. Instead we build the module on the ``meta`` device
    (no parameter storage) and assign the checkpoint tensors straight in.

    Falls back to the stock loader for local checkpoint files or non-CPU devices.
    """
    from mast3r.model import AsymmetricMASt3R

    if os.path.isfile(model_name) or device != "cpu":
        return AsymmetricMASt3R.from_pretrained(model_name).to(device).eval()

    import json as _json
    import gc as _gc
    import safetensors.torch as _st
    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(model_name, "config.json")
    weights_path = hf_hub_download(model_name, "model.safetensors")

    with open(config_path) as fh:
        config = _json.load(fh)

    with torch.device("meta"):
        model = AsymmetricMASt3R(**config)

    state_dict = _st.load_file(weights_path)
    missing, unexpected = model.load_state_dict(
        state_dict, strict=False, assign=True
    )
    del state_dict
    _gc.collect()

    # The only tolerated "missing" keys are the DPT ``scratch.layer_rn.N``
    # aliases, which share module objects with ``scratch.layerN_rn`` and are
    # already populated. Anything else - or any tensor still on meta - is fatal.
    real_missing = [k for k in missing if ".scratch.layer_rn." not in k]
    stranded = [n for n, p in list(model.named_parameters()) + list(model.named_buffers())
                if p.is_meta]
    if real_missing or unexpected or stranded:
        raise RuntimeError(
            "MASt3R checkpoint load mismatch: "
            f"missing={real_missing} unexpected={unexpected} still_meta={stranded}"
        )

    return model.to(device).eval()

def get_registered_frame_names(images):
    frame_names = []

    for name in sorted(images.keys()):
        frame_path = FRAME_DIR / name

        if frame_path.exists():
            frame_names.append(name)

    if len(frame_names) < 2:
        raise RuntimeError(
            "Fewer than 2 registered frame images were found."
        )

    return frame_names

# MASt3R confidence filter
CONF_THRESHOLD = 1.5

# Keep every Nth pixel in x/y.
# 2 = roughly 1/4 of all pixels.
PIXEL_STEP = 2

# Minimum number of COLMAP sparse anchors required
# before trusting a frame.
MIN_ANCHORS = 20


# ============================================================
# COLMAP CAMERA MODELS
# ============================================================

CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


# ============================================================
# BINARY HELPERS
# ============================================================

def read_bytes(fid, n, fmt):
    data = fid.read(n)

    if len(data) != n:
        raise EOFError(
            "Unexpected end of COLMAP binary file."
        )

    return struct.unpack(
        "<" + fmt,
        data
    )


# ============================================================
# QUATERNION -> ROTATION
# ============================================================

def qvec_to_rotmat(qvec):

    qw, qx, qy, qz = qvec

    return np.array([
        [
            1 - 2 * qy * qy - 2 * qz * qz,
            2 * qx * qy - 2 * qz * qw,
            2 * qx * qz + 2 * qy * qw,
        ],
        [
            2 * qx * qy + 2 * qz * qw,
            1 - 2 * qx * qx - 2 * qz * qz,
            2 * qy * qz - 2 * qx * qw,
        ],
        [
            2 * qx * qz - 2 * qy * qw,
            2 * qy * qz + 2 * qx * qw,
            1 - 2 * qx * qx - 2 * qy * qy,
        ],
    ], dtype=np.float64)


# ============================================================
# READ CAMERAS.BIN
# ============================================================

def read_cameras_binary(path):

    cameras = {}

    with open(path, "rb") as fid:

        num_cameras = read_bytes(
            fid,
            8,
            "Q"
        )[0]

        for _ in range(num_cameras):

            camera_id, model_id, width, height = read_bytes(
                fid,
                24,
                "iiQQ"
            )

            if model_id not in CAMERA_MODELS:
                raise RuntimeError(
                    f"Unsupported COLMAP camera model ID {model_id}"
                )

            model_name, num_params = CAMERA_MODELS[
                model_id
            ]

            params = read_bytes(
                fid,
                8 * num_params,
                "d" * num_params
            )

            cameras[camera_id] = {
                "model": model_name,
                "width": int(width),
                "height": int(height),
                "params": np.array(
                    params,
                    dtype=np.float64
                ),
            }

    return cameras


# ============================================================
# READ IMAGES.BIN + 2D OBSERVATIONS
# ============================================================

def read_images_binary(path):

    images = {}

    with open(path, "rb") as fid:

        num_images = read_bytes(
            fid,
            8,
            "Q"
        )[0]

        for _ in range(num_images):

            values = read_bytes(
                fid,
                64,
                "idddddddi"
            )

            image_id = values[0]

            qvec = np.array(
                values[1:5],
                dtype=np.float64
            )

            tvec = np.array(
                values[5:8],
                dtype=np.float64
            )

            camera_id = values[8]

            name_bytes = []

            while True:

                b = fid.read(1)

                if not b:
                    raise EOFError("Truncated image name")

                if b == b"\x00":
                    break

                name_bytes.append(b)

            name = (
                b"".join(name_bytes)
                .decode("utf-8")
            )

            num_points2d = read_bytes(
                fid,
                8,
                "Q"
            )[0]

            observations = []

            for _ in range(num_points2d):

                x, y, point3d_id = read_bytes(
                    fid,
                    24,
                    "ddq"
                )

                if point3d_id >= 0:

                    observations.append(
                        (
                            float(x),
                            float(y),
                            int(point3d_id)
                        )
                    )

            R = qvec_to_rotmat(
                qvec
            )

            camera_center = (
                -R.T @ tvec
            )

            cam_to_world = np.eye(
                4,
                dtype=np.float64
            )

            cam_to_world[:3, :3] = R.T
            cam_to_world[:3, 3] = camera_center

            images[name] = {
                "image_id": image_id,
                "camera_id": camera_id,
                "qvec": qvec,
                "tvec": tvec,
                "R": R,
                "camera_center": camera_center,
                "cam_to_world": cam_to_world,
                "observations": observations,
            }

    return images


# ============================================================
# READ POINTS3D.BIN
# ============================================================

def read_points3d_binary(path):

    points = {}

    with open(path, "rb") as fid:

        num_points = read_bytes(
            fid,
            8,
            "Q"
        )[0]

        for _ in range(num_points):

            values = read_bytes(
                fid,
                43,
                "QdddBBBd"
            )

            point_id = values[0]

            xyz = np.array(
                values[1:4],
                dtype=np.float64
            )

            track_length = read_bytes(
                fid,
                8,
                "Q"
            )[0]

            # Each track entry:
            # image_id (int32)
            # point2D_idx (int32)
            fid.seek(
                track_length * 8,
                1
            )

            points[point_id] = xyz

    return points


# ============================================================
# ROBUST DEPTH SCALE
# ============================================================

def estimate_depth_scale(
    mast3r_depth,
    image_data,
    points3d,
    camera,
):

    H, W = mast3r_depth.shape

    original_w = camera["width"]
    original_h = camera["height"]

    ratios = []

    R = image_data["R"]
    t = image_data["tvec"]

    for x, y, point_id in image_data["observations"]:

        if point_id not in points3d:
            continue

        # Original image -> MASt3R image coordinates
        mx, my = original_to_prediction(x, y, camera, W, H)
        mx, my = int(round(float(mx))), int(round(float(my)))

        if (
            mx < 0
            or mx >= W
            or my < 0
            or my >= H
        ):
            continue

        pred_z = float(
            mast3r_depth[my, mx]
        )

        if (
            not np.isfinite(pred_z)
            or pred_z <= 1e-6
        ):
            continue

        X_world = points3d[
            point_id
        ]

        X_cam = (
            R @ X_world + t
        )

        colmap_z = float(
            X_cam[2]
        )

        if (
            not np.isfinite(colmap_z)
            or colmap_z <= 1e-6
        ):
            continue

        ratio = (
            colmap_z / pred_z
        )

        if (
            np.isfinite(ratio)
            and ratio > 0
        ):
            ratios.append(
                ratio
            )

    ratios = np.array(
        ratios,
        dtype=np.float64
    )

    if len(ratios) < MIN_ANCHORS:

        return None, len(ratios)

    # --------------------------------------------------------
    # ROBUST OUTLIER REJECTION
    # --------------------------------------------------------

    median = np.median(
        ratios
    )

    mad = np.median(
        np.abs(
            ratios - median
        )
    )

    if mad > 1e-12:

        valid = (
            np.abs(
                ratios - median
            )
            < 3.5 * 1.4826 * mad
        )

        filtered = ratios[
            valid
        ]

    else:

        filtered = ratios

    if len(filtered) < MIN_ANCHORS:

        return None, len(filtered)

    scale = float(
        np.median(
            filtered
        )
    )

    return scale, len(filtered)


# ============================================================
# UNDISTORT PIXEL RAYS
# ============================================================

def make_camera_rays(
    width,
    height,
    camera,
):

    if camera["model"] != "SIMPLE_RADIAL":

        raise RuntimeError(
            "This test currently expects SIMPLE_RADIAL."
        )

    f, cx, cy, k1 = camera["params"]

    original_w = camera["width"]
    original_h = camera["height"]

    xs = np.arange(
        0,
        width,
        PIXEL_STEP
    )

    ys = np.arange(
        0,
        height,
        PIXEL_STEP
    )

    xx, yy = np.meshgrid(
        xs,
        ys
    )

    # Convert MASt3R resized coordinates
    # back to original image coordinates.

    u, v = prediction_to_original(xx, yy, camera, width, height)

    pixels = np.stack(
        [u, v],
        axis=-1
    ).astype(
        np.float64
    )

    pixels_cv = pixels.reshape(
        -1,
        1,
        2
    )

    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1],
    ], dtype=np.float64)

    dist = np.array(
        [k1, 0, 0, 0, 0],
        dtype=np.float64
    )

    normalized = cv2.undistortPoints(
        pixels_cv,
        K,
        dist
    )

    normalized = normalized.reshape(
        -1,
        2
    )

    rays = np.column_stack([
        normalized[:, 0],
        normalized[:, 1],
        np.ones(
            len(normalized),
            dtype=np.float64
        ),
    ])

    return (
        rays,
        xx.reshape(-1),
        yy.reshape(-1)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n==============================================")
    print(" SKYFORM - COLMAP + MASt3R FIXED-POSE FUSION")
    print("==============================================")

    print(f"\nMASt3R inference device: {DEVICE}")

    if DEVICE == "cpu":
        print(
            "  (running on CPU - expect slower per-pair inference; "
            "peak RAM is dominated by the ViT-Large model + one image pair)"
        )

    print("\nReading COLMAP reconstruction...")

    model_dir = find_best_sparse_model(
        SPARSE_DIR
    )

    cameras = read_cameras_binary(
        model_dir / "cameras.bin"
    )

    images = read_images_binary(
        model_dir / "images.bin"
    )

    points3d = read_points3d_binary(
        model_dir / "points3D.bin"
    )

    print(
        "COLMAP cameras:",
        len(cameras)
    )

    print(
        "COLMAP registered images:",
        len(images)
    )

    print(
        "COLMAP sparse points:",
        len(points3d)
    )

    # ========================================================
    # LOAD REGISTERED FRAMES
    # ========================================================

    frame_names = get_registered_frame_names(
        images
    )

    print(
        "Registered frames available for MASt3R:",
        len(frame_names)
    )

    paths = [
        str(
            FRAME_DIR / name
        )
        for name in frame_names
    ]

    for p in paths:

        if not Path(p).exists():

            raise FileNotFoundError(
                p
            )

        if Path(p).name not in images:

            raise RuntimeError(
                f"{Path(p).name} is not registered in COLMAP."
            )

    print("\nLoading MASt3R model...")

    model = load_mast3r_model(MODEL_NAME, DEVICE)

    print(
        f"MASt3R model loaded (device={DEVICE}, "
        f"torch_threads={torch.get_num_threads()})."
    )

    print("\nLoading selected images...")

    mast3r_images = load_images(
        paths,
        size=IMAGE_SIZE,
        square_ok=True,
        verbose=True
    )

    # ========================================================
    # STORAGE FOR PER-FRAME PREDICTIONS
    # ========================================================

    depth_predictions = {
        i: []
        for i in range(
            len(frame_names)
        )
    }

    confidence_predictions = {
        i: []
        for i in range(
            len(frame_names)
        )
    }

    # ========================================================
    # ADJACENT PAIR INFERENCE ONLY
    # ========================================================

    print(
        "\nRunning local MASt3R pair inference..."
    )

    for i in range(len(mast3r_images) - 1):

        j = i + 1

        print(
            f"\nPair "
            f"{frame_names[i]} "
            f"<-> "
            f"{frame_names[j]}"
        )

        # ----------------------------------------------------
        # Run one pair only.
        # ----------------------------------------------------

        with torch.inference_mode():

            res11, res21, res22, res12 = symmetric_inference(
                model,
                mast3r_images[i],
                mast3r_images[j],
                device=DEVICE,
            )

        # ----------------------------------------------------
        # Immediately move ONLY the data we need to CPU/NumPy.
        # ----------------------------------------------------

        pts_i = np.asarray(
            to_numpy(res11["pts3d"][0])
        )

        conf_i = np.asarray(
            to_numpy(res11["conf"][0])
        )

        pts_j = np.asarray(
            to_numpy(res22["pts3d"][0])
        )

        conf_j = np.asarray(
            to_numpy(res22["conf"][0])
        )

        # Make independent CPU copies.
        #
        # This prevents NumPy arrays from accidentally retaining
        # references to tensors/storage owned by the inference result.
        depth_i = np.array(
            pts_i[..., 2],
            dtype=np.float32,
            copy=True,
        )

        confidence_i = np.array(
            conf_i,
            dtype=np.float32,
            copy=True,
        )

        depth_j = np.array(
            pts_j[..., 2],
            dtype=np.float32,
            copy=True,
        )

        confidence_j = np.array(
            conf_j,
            dtype=np.float32,
            copy=True,
        )

        depth_predictions[i].append(depth_i)
        confidence_predictions[i].append(confidence_i)

        depth_predictions[j].append(depth_j)
        confidence_predictions[j].append(confidence_j)

        # ----------------------------------------------------
        # CRITICAL:
        # Release all pair-specific GPU/MPS tensors before
        # processing the next pair.
        # ----------------------------------------------------

        del res11
        del res21
        del res22
        del res12

        del pts_i
        del conf_i
        del pts_j
        del conf_j

        del depth_i
        del confidence_i
        del depth_j
        del confidence_j

        gc.collect()

        _empty_device_cache()

        print(
            f"  Pair {i + 1}/"
            f"{len(mast3r_images) - 1} complete."
        )

    # ========================================================
    # FREE MODEL MEMORY
    # ========================================================

    del model
    gc.collect()

    _empty_device_cache()

    # ========================================================
    # FUSE INTO COLMAP WORLD
    # ========================================================

    print(
        "\nFusing frames using COLMAP poses..."
    )

    global_points = []
    global_colors = []
    frame_support = []

    for i, name in enumerate(
        frame_names
    ):

        print(
            f"\nProcessing {name}"
        )

        if not depth_predictions[i]:

            print(
                "  SKIPPED: no MASt3R depth."
            )

            continue

        # ----------------------------------------------------
        # WEIGHTED DEPTH AVERAGE
        # ----------------------------------------------------

        depths = np.stack(
            depth_predictions[i],
            axis=0
        )

        confs = np.stack(
            confidence_predictions[i],
            axis=0
        )

        weights = np.maximum(
            confs - 1.0,
            0.0
        )

        weight_sum = np.sum(
            weights,
            axis=0
        )

        safe_weight_sum = np.maximum(
            weight_sum,
            1e-8
        )

        depth = np.sum(
            depths * weights,
            axis=0
        ) / safe_weight_sum

        confidence = np.max(
            confs,
            axis=0
        )

        H, W = depth.shape

        # ----------------------------------------------------
        # COLMAP CAMERA
        # ----------------------------------------------------

        image_data = images[
            name
        ]

        camera = cameras[
            image_data["camera_id"]
        ]

        # ----------------------------------------------------
        # DEPTH -> COLMAP SCALE
        # ----------------------------------------------------

        scale, anchors = estimate_depth_scale(
            depth,
            image_data,
            points3d,
            camera,
        )

        if scale is None:

            print(
                f"  SKIPPED: only "
                f"{anchors} usable sparse anchors."
            )

            continue

        print(
            f"  Sparse anchors: {anchors}"
        )

        print(
            f"  Depth scale: {scale:.6f}"
        )

        scaled_depth = (
            depth * scale
        )

        # ----------------------------------------------------
        # CAMERA RAYS
        # ----------------------------------------------------

        rays, xs, ys = make_camera_rays(
            W,
            H,
            camera,
        )

        z = scaled_depth[
            ys,
            xs
        ].reshape(-1)

        conf = confidence[
            ys,
            xs
        ].reshape(-1)

        # Because ray is [x/z, y/z, 1],
        # multiplying by camera-space Z gives XYZ.

        points_camera = (
            rays
            * z[:, None]
        )

        # ----------------------------------------------------
        # VALID MASK
        # ----------------------------------------------------

        valid = (
            np.isfinite(
                points_camera
            ).all(axis=1)
            & np.isfinite(z)
            & (z > 0)
            & np.isfinite(conf)
            & (conf > CONF_THRESHOLD)
        )

        frame_support.append({"image": name, "sampled_pixels": int(len(valid)), "supported_pixels": int(valid.sum()), "supported_fraction": float(valid.mean()), "sparse_anchors": int(anchors)})

        points_camera = (
            points_camera[
                valid
            ]
        )

        # ----------------------------------------------------
        # CAMERA -> COLMAP WORLD
        # ----------------------------------------------------

        R_world_to_cam = (
            image_data["R"]
        )

        t_world_to_cam = (
            image_data["tvec"]
        )

        # X_cam = R * X_world + t
        #
        # therefore:
        #
        # X_world = R^T * (X_cam - t)

        points_world = (
            R_world_to_cam.T
            @ (
                points_camera
                - t_world_to_cam
            ).T
        ).T

        # ----------------------------------------------------
        # COLORS
        # ----------------------------------------------------

        image_bgr = cv2.imread(
            str(
                FRAME_DIR / name
            )
        )

        if image_bgr is None:

            raise RuntimeError(
                f"Could not read {name}"
            )

        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB
        )

        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        source_x, source_y = prediction_to_original(grid_x, grid_y, camera, W, H)
        image_rgb = cv2.remap(image_rgb, source_x.astype(np.float32),
                              source_y.astype(np.float32), cv2.INTER_LINEAR)

        colors = image_rgb[
            ys,
            xs
        ].reshape(
            -1,
            3
        )

        colors = colors[
            valid
        ]

        print(
            f"  Kept points: "
            f"{len(points_world):,}"
        )

        global_points.append(
            points_world
        )

        global_colors.append(
            colors
        )

    # ========================================================
    # CONCATENATE
    # ========================================================

    if not global_points:

        raise RuntimeError(
            "No frames survived fusion."
        )

    points = np.concatenate(
        global_points,
        axis=0
    )

    colors = np.concatenate(
        global_colors,
        axis=0
    )

    # ========================================================
    # SIMPLE EXTREME OUTLIER FILTER
    # ========================================================

    center = np.median(
        points,
        axis=0
    )

    dist = np.linalg.norm(
        points - center,
        axis=1
    )

    cutoff = np.percentile(
        dist,
        99.5
    )

    keep = (
        np.isfinite(
            points
        ).all(axis=1)
        & (dist <= cutoff)
    )

    points = points[
        keep
    ]

    colors = colors[
        keep
    ]

    print(
        "\nFinal fused points:",
        f"{len(points):,}"
    )

    # ========================================================
    # EXPORT
    # ========================================================

    cloud = trimesh.points.PointCloud(
        vertices=points,
        colors=colors,
    )

    output_path = (
        OUTPUT_DIR
        / "dense_colmap_fixed_poses.ply"
    )

    (OUTPUT_DIR / "coverage.json").write_text(json.dumps({
        "registered_frames": len(frame_names), "fused_frames": len(frame_support),
        "frames": frame_support,
        "entire_visible_scene_validated": False,
        "note": "Confidence-supported pixel fractions are diagnostics, not ground-truth coverage. Unregistered, occluded and unsupported regions remain unknown."
    }, indent=2))

    cloud.export(
        str(
            output_path
        )
    )

    print("\n==============================================")
    print(" FIXED-POSE FUSION COMPLETE")
    print("==============================================")

    print(
        "Output:",
        output_path
    )

    print(
        "Points:",
        f"{len(points):,}"
    )

    print("==============================================\n")


if __name__ == "__main__":
    main()
