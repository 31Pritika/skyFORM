from pathlib import Path

import cv2
import numpy as np


def depth_to_point_cloud(
    image_path,
    depth_path,
    output_path,
    sample_step=4
):
    """
    Converts one RGB image + relative depth map
    into a colored relative 3D point cloud.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Could not load image: {image_path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    depth = np.load(depth_path)

    height, width = depth.shape

    if image.shape[:2] != (height, width):
        image = cv2.resize(
            image,
            (width, height)
        )

    # Approximate intrinsics.
    focal = 0.9 * max(
        width,
        height
    )

    cx = width / 2.0
    cy = height / 2.0

    points = []
    colors = []

    for v in range(
        0,
        height,
        sample_step
    ):
        for u in range(
            0,
            width,
            sample_step
        ):
            z = float(depth[v, u])

            if (
                not np.isfinite(z)
                or z <= 0
            ):
                continue

            x = (
                (u - cx) * z / focal
            )

            y = (
                (v - cy) * z / focal
            )

            points.append([
                x,
                -y,
                -z
            ])

            colors.append(
                image[v, u]
            )

    points = np.asarray(
        points,
        dtype=np.float32
    )

    colors = np.asarray(
        colors,
        dtype=np.uint8
    )

    save_colored_ply(
        points,
        colors,
        output_path
    )

    return {
        "points": len(points),
        "output": str(output_path)
    }


def save_colored_ply(
    points,
    colors,
    output_path
):
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w"
    ) as file:

        file.write("ply\n")
        file.write(
            "format ascii 1.0\n"
        )

        file.write(
            f"element vertex {len(points)}\n"
        )

        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")

        file.write(
            "property uchar red\n"
        )
        file.write(
            "property uchar green\n"
        )
        file.write(
            "property uchar blue\n"
        )

        file.write("end_header\n")

        for point, color in zip(
            points,
            colors
        ):
            x, y, z = point
            r, g, b = color

            file.write(
                f"{x} {y} {z} "
                f"{int(r)} "
                f"{int(g)} "
                f"{int(b)}\n"
            )