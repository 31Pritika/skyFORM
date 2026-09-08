from pathlib import Path
import numpy as np


def quaternion_to_rotation(qw, qx, qy, qz):
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


def get_camera_poses(base_dir):
    base_dir = Path(base_dir)

    images_file = (
        base_dir
        / "outputs"
        / "colmap_new"
        / "dense"
        / "sparse_txt"
        / "images.txt"
    )

    if not images_file.exists():
        return []

    poses = []

    with open(images_file, "r") as file:
        lines = file.readlines()

    # COLMAP images.txt uses two lines per image:
    # image metadata
    # POINTS2D observations
    valid_lines = [
        line.strip()
        for line in lines
        if line.strip()
        and not line.startswith("#")
    ]

    for i in range(0, len(valid_lines), 2):
        parts = valid_lines[i].split()

        if len(parts) < 10:
            continue

        image_id = int(parts[0])

        qw = float(parts[1])
        qx = float(parts[2])
        qy = float(parts[3])
        qz = float(parts[4])

        tx = float(parts[5])
        ty = float(parts[6])
        tz = float(parts[7])

        image_name = parts[9]

        R = quaternion_to_rotation(
            qw, qx, qy, qz
        )

        t = np.array([
            tx,
            ty,
            tz
        ])

        # COLMAP world-space camera center
        center = -R.T @ t

        poses.append({
            "id": image_id,
            "image": image_name,

            "position": [
                float(center[0]),
                float(center[1]),
                float(center[2]),
            ],

            "rotation_matrix":
                R.tolist(),
        })

    return poses