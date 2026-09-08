import json
import struct

import numpy as np

from colmap_utils import find_best_sparse_model
from pipeline_config import (
    get_sfm_dir,
    get_colmap_poses_dir,
    ensure_job_directories,
)


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


def read_next_bytes(fid, num_bytes, format_char_sequence):
    data = fid.read(num_bytes)

    if len(data) != num_bytes:
        raise EOFError("Unexpected end of COLMAP binary file.")

    return struct.unpack("<" + format_char_sequence, data)


def qvec_to_rotmat(qvec):
    qw, qx, qy, qz = qvec

    return np.array(
        [
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
        ],
        dtype=np.float64,
    )


def read_cameras_binary(path):
    cameras = {}

    with open(path, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]

        for _ in range(num_cameras):
            camera_id, model_id, width, height = read_next_bytes(
                fid, 24, "iiQQ"
            )

            if model_id not in CAMERA_MODELS:
                raise RuntimeError(
                    f"Unsupported COLMAP camera model ID: {model_id}"
                )

            model_name, num_params = CAMERA_MODELS[model_id]

            params = read_next_bytes(
                fid,
                8 * num_params,
                "d" * num_params,
            )

            cameras[camera_id] = {
                "camera_id": int(camera_id),
                "model": model_name,
                "width": int(width),
                "height": int(height),
                "params": [float(x) for x in params],
            }

    return cameras


def read_images_binary(path):
    images = {}

    with open(path, "rb") as fid:
        num_images = read_next_bytes(fid, 8, "Q")[0]

        for _ in range(num_images):
            binary_data = read_next_bytes(
                fid,
                64,
                "idddddddi",
            )

            image_id = binary_data[0]
            qvec = np.array(binary_data[1:5], dtype=np.float64)
            tvec = np.array(binary_data[5:8], dtype=np.float64)
            camera_id = binary_data[8]

            name_bytes = []

            while True:
                current = fid.read(1)

                if current == b"\x00":
                    break

                if current == b"":
                    raise EOFError(
                        "Unexpected EOF while reading COLMAP image name."
                    )

                name_bytes.append(current)

            image_name = b"".join(name_bytes).decode("utf-8")

            num_points2d = read_next_bytes(fid, 8, "Q")[0]

            # Each observation:
            # double x + double y + int64 point3D_id = 24 bytes
            fid.seek(num_points2d * 24, 1)

            # COLMAP:
            # X_cam = R * X_world + t
            R_world_to_cam = qvec_to_rotmat(qvec)

            # Camera centre in COLMAP world coordinates.
            camera_center = -R_world_to_cam.T @ tvec

            # Camera-to-world rotation.
            R_cam_to_world = R_world_to_cam.T

            cam_to_world = np.eye(4, dtype=np.float64)
            cam_to_world[:3, :3] = R_cam_to_world
            cam_to_world[:3, 3] = camera_center

            images[image_id] = {
                "image_id": int(image_id),
                "name": image_name,
                "camera_id": int(camera_id),
                "qvec": qvec.tolist(),
                "tvec": tvec.tolist(),
                "camera_center": camera_center.tolist(),
                "position": camera_center.tolist(),
                "rotation_matrix": R_cam_to_world.tolist(),
                "cam_to_world": cam_to_world.tolist(),
            }

    return images


def make_intrinsics(camera):
    model = camera["model"]
    p = camera["params"]

    if model == "SIMPLE_PINHOLE":
        f, cx, cy = p
        fx = fy = f
        distortion = []

    elif model == "PINHOLE":
        fx, fy, cx, cy = p
        distortion = []

    elif model == "SIMPLE_RADIAL":
        f, cx, cy, k = p
        fx = fy = f
        distortion = [k]

    elif model == "RADIAL":
        f, cx, cy, k1, k2 = p
        fx = fy = f
        distortion = [k1, k2]

    elif model == "OPENCV":
        fx, fy, cx, cy = p[:4]
        distortion = p[4:]

    else:
        return None

    return {
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "distortion": [float(x) for x in distortion],
    }


def main():
    ensure_job_directories()

    sparse_dir = get_sfm_dir() / "sparse"
    output_dir = get_colmap_poses_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 30)
    print(" SKYFORM - READ COLMAP POSES")
    print("=" * 30)

    # IMPORTANT:
    # Resolve the actual COLMAP model here, after the job paths exist.
    model_dir = find_best_sparse_model(sparse_dir)

    if model_dir is None:
        raise RuntimeError(
            f"No valid COLMAP sparse model found inside: {sparse_dir}"
        )

    cameras_bin = model_dir / "cameras.bin"
    images_bin = model_dir / "images.bin"

    if not cameras_bin.exists():
        raise FileNotFoundError(
            f"Missing COLMAP cameras.bin: {cameras_bin}"
        )

    if not images_bin.exists():
        raise FileNotFoundError(
            f"Missing COLMAP images.bin: {images_bin}"
        )

    print(f"COLMAP model: {model_dir}")

    cameras = read_cameras_binary(cameras_bin)
    images = read_images_binary(images_bin)

    if not cameras:
        raise RuntimeError("COLMAP model contains no cameras.")

    if not images:
        raise RuntimeError("COLMAP model contains no registered images.")

    # Sort by filename so frame order is deterministic.
    ordered_images = sorted(
        images.values(),
        key=lambda item: item["name"],
    )

    camera_records = []

    for camera_id in sorted(cameras):
        camera = cameras[camera_id].copy()
        camera["intrinsics"] = make_intrinsics(camera)
        camera_records.append(camera)

    output = {
        "model_directory": str(model_dir),
        "registered_images": len(ordered_images),
        "camera_count": len(camera_records),
        "cameras": camera_records,
        "images": ordered_images,
        # Compatibility alias for the existing viewer service.
        "poses": ordered_images,
    }

    output_path = output_dir / "colmap_poses.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Registered images: {len(ordered_images)}")
    print(f"Cameras: {len(camera_records)}")

    for camera in camera_records:
        print(
            f"Camera #{camera['camera_id']}: "
            f"{camera['model']} "
            f"{camera['params']}"
        )

    print(f"Saved: {output_path}")

    print("=" * 30)
    print(" COLMAP POSE EXTRACTION COMPLETE")
    print("=" * 30)


if __name__ == "__main__":
    main()
