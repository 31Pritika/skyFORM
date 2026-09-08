from pathlib import Path
import json
import numpy as np


def quaternion_to_rotation(
    qw,
    qx,
    qy,
    qz,
):
    return np.array([
        [
            1 - 2 * (qy*qy + qz*qz),
            2 * (qx*qy - qz*qw),
            2 * (qx*qz + qy*qw),
        ],
        [
            2 * (qx*qy + qz*qw),
            1 - 2 * (qx*qx + qz*qz),
            2 * (qy*qz - qx*qw),
        ],
        [
            2 * (qx*qz - qy*qw),
            2 * (qy*qz + qx*qw),
            1 - 2 * (qx*qx + qy*qy),
        ],
    ])


def get_active_video_id(base_dir):
    state_path = (
        Path(base_dir)
        / "outputs"
        / "pipeline_state.json"
    )

    if not state_path.exists():
        return None

    try:
        with open(
            state_path,
            "r",
            encoding="utf-8",
        ) as file:
            state = json.load(file)

        return state.get("video_id")

    except Exception:
        return None


def normalize_pose_data(data):
    """
    Accept Stage-4 JSON without forcing one exact JSON shape.
    """
    if isinstance(data, list):
        entries = data

    elif isinstance(data, dict):
        entries = (
            data.get("poses")
            or data.get("images")
            or data.get("camera_poses")
            or []
        )

    else:
        return []

    poses = []

    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue

        image_name = (
            entry.get("image")
            or entry.get("name")
            or entry.get("image_name")
            or f"frame_{index:04d}.jpg"
        )

        image_id = (
            entry.get("id")
            or entry.get("image_id")
            or index
        )

        # Stage 4 may already store camera center.
        position = (
            entry.get("position")
            or entry.get("camera_center")
            or entry.get("center")
        )

        rotation = (
            entry.get("rotation_matrix")
            or entry.get("R")
        )

        # Or it may store COLMAP qvec/tvec.
        if position is None:
            qvec = entry.get("qvec")
            tvec = entry.get("tvec")

            if (
                qvec is not None
                and tvec is not None
                and len(qvec) == 4
                and len(tvec) == 3
            ):
                R = quaternion_to_rotation(
                    float(qvec[0]),
                    float(qvec[1]),
                    float(qvec[2]),
                    float(qvec[3]),
                )

                t = np.asarray(
                    tvec,
                    dtype=float,
                )

                center = -R.T @ t

                position = center.tolist()

                if rotation is None:
                    rotation = R.T.tolist()

        if position is None:
            continue

        poses.append({
            "id": int(image_id),
            "image": str(image_name),
            "position": [
                float(position[0]),
                float(position[1]),
                float(position[2]),
            ],
            "rotation_matrix":
                rotation
                if rotation is not None
                else [],
        })

    return poses


def get_camera_poses(base_dir):
    base_dir = Path(base_dir)

    video_id = get_active_video_id(
        base_dir
    )

    candidates = []

    if video_id:
        candidates.append(
            base_dir
            / "outputs"
            / "jobs"
            / video_id
            / "colmap_poses"
            / "colmap_poses.json"
        )

    if not video_id:
        candidates.append(
        base_dir
        / "outputs"
        / "reconstruction"
        / "camera_poses.json"
    )

    for path in candidates:
        if not path.exists():
            continue

        try:
            with open(
                path,
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)

            poses = normalize_pose_data(
                data
            )

            if poses:
                from urllib.parse import quote
                for pose in poses:
                    if video_id:
                        pose["image_url"] = f"/outputs/jobs/{quote(video_id)}/frames/{quote(pose['image'])}"
                return poses

        except Exception as error:
            print(
                "Camera pose read error:",
                error,
            )

    return []
