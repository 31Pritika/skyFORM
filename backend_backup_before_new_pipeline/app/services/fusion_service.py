from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def quaternion_to_rotation(qw, qx, qy, qz):
    q = np.array(
        [qw, qx, qy, qz],
        dtype=np.float64
    )

    q /= np.linalg.norm(q)

    qw, qx, qy, qz = q

    return np.array([
        [
            1 - 2 * (qy * qy + qz * qz),
            2 * (qx * qy - qz * qw),
            2 * (qx * qz + qy * qw)
        ],
        [
            2 * (qx * qy + qz * qw),
            1 - 2 * (qx * qx + qz * qz),
            2 * (qy * qz - qx * qw)
        ],
        [
            2 * (qx * qz - qy * qw),
            2 * (qy * qz + qx * qw),
            1 - 2 * (qx * qx + qy * qy)
        ]
    ], dtype=np.float64)


def read_cameras(path):
    cameras = {}

    with open(path, "r") as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])

            params = list(
                map(float, parts[4:])
            )

            if model == "PINHOLE":
                fx, fy, cx, cy = params[:4]

            elif model == "SIMPLE_PINHOLE":
                f, cx, cy = params[:3]
                fx = fy = f

            elif model == "SIMPLE_RADIAL":
                f, cx, cy = params[:3]
                fx = fy = f

            else:
                raise ValueError(
                    f"Unsupported camera model: {model}"
                )

            cameras[camera_id] = {
                "width": width,
                "height": height,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy
            }

    return cameras


def read_points3d(path):
    points = {}

    with open(path, "r") as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            point_id = int(parts[0])

            xyz = np.array(
                list(map(float, parts[1:4])),
                dtype=np.float64
            )

            points[point_id] = xyz

    return points


def read_images(path):
    images = {}

    with open(path, "r") as file:
        lines = [
            line.strip()
            for line in file
            if line.strip()
            and not line.startswith("#")
        ]

    i = 0

    while i < len(lines):
        header = lines[i].split()

        image_id = int(header[0])

        qw, qx, qy, qz = map(
            float,
            header[1:5]
        )

        tx, ty, tz = map(
            float,
            header[5:8]
        )

        camera_id = int(header[8])
        image_name = header[9]

        points_line = (
            lines[i + 1].split()
            if i + 1 < len(lines)
            else []
        )

        observations = []

        for j in range(
            0,
            len(points_line),
            3
        ):
            if j + 2 >= len(points_line):
                break

            x = float(points_line[j])
            y = float(points_line[j + 1])
            point3d_id = int(
                points_line[j + 2]
            )

            if point3d_id != -1:
                observations.append({
                    "x": x,
                    "y": y,
                    "point3d_id": point3d_id
                })

        images[image_name] = {
            "image_id": image_id,
            "camera_id": camera_id,
            "rotation": quaternion_to_rotation(
                qw,
                qx,
                qy,
                qz
            ),
            "translation": np.array(
                [tx, ty, tz],
                dtype=np.float64
            ),
            "observations": observations
        }

        i += 2

    return images


def fit_depth_alignment(
    predicted_depth,
    image_info,
    points3d
):
    """
    Fits relative Depth Anything output
    to COLMAP camera-space depth.

    Tests both direct depth and inverse depth,
    then chooses the better fit.
    """

    R = image_info["rotation"]
    t = image_info["translation"]

    predicted_values = []
    colmap_depths = []

    height, width = predicted_depth.shape

    for obs in image_info["observations"]:
        point_id = obs["point3d_id"]

        if point_id not in points3d:
            continue

        u = int(round(obs["x"]))
        v = int(round(obs["y"]))

        if (
            u < 0
            or u >= width
            or v < 0
            or v >= height
        ):
            continue

        world_point = points3d[
            point_id
        ]

        camera_point = (
            R @ world_point + t
        )

        z = camera_point[2]

        if z <= 0:
            continue

        depth_value = float(
            predicted_depth[v, u]
        )

        if (
            not np.isfinite(depth_value)
            or depth_value <= 0
        ):
            continue

        predicted_values.append(
            depth_value
        )

        colmap_depths.append(
            z
        )

    if len(predicted_values) < 10:
        return None

    d = np.asarray(
        predicted_values,
        dtype=np.float64
    )

    z = np.asarray(
        colmap_depths,
        dtype=np.float64
    )

    # Remove extreme observations.
    z_low, z_high = np.percentile(
        z,
        [5, 95]
    )

    mask = (
        (z >= z_low)
        & (z <= z_high)
    )

    d = d[mask]
    z = z[mask]

    if len(d) < 10:
        return None

    # Model 1:
    # z = a * predicted_depth + b
    A_direct = np.column_stack([
        d,
        np.ones_like(d)
    ])

    direct_coeff, _, _, _ = (
        np.linalg.lstsq(
            A_direct,
            z,
            rcond=None
        )
    )

    direct_prediction = (
        A_direct @ direct_coeff
    )

    direct_error = np.median(
        np.abs(
            direct_prediction - z
        )
    )

    # Model 2:
    # z = a / predicted_depth + b
    inverse_d = 1.0 / (
        d + 1e-8
    )

    A_inverse = np.column_stack([
        inverse_d,
        np.ones_like(inverse_d)
    ])

    inverse_coeff, _, _, _ = (
        np.linalg.lstsq(
            A_inverse,
            z,
            rcond=None
        )
    )

    inverse_prediction = (
        A_inverse @ inverse_coeff
    )

    inverse_error = np.median(
        np.abs(
            inverse_prediction - z
        )
    )

    if inverse_error < direct_error:
        return {
            "mode": "inverse",
            "a": inverse_coeff[0],
            "b": inverse_coeff[1],
            "error": inverse_error,
            "samples": len(d)
        }

    return {
        "mode": "direct",
        "a": direct_coeff[0],
        "b": direct_coeff[1],
        "error": direct_error,
        "samples": len(d)
    }


def align_depth(depth, alignment):
    if alignment["mode"] == "inverse":
        result = (
            alignment["a"]
            / (depth + 1e-8)
            + alignment["b"]
        )

    else:
        result = (
            alignment["a"]
            * depth
            + alignment["b"]
        )

    return result


def save_colored_ply(
    points,
    colors,
    output_path
):
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(
            f"element vertex {len(points)}\n"
        )
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
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



def clean_fused_point_cloud(
    points,
    colors,
    voxel_size=0.06,
    statistical_neighbors=20,
    statistical_std_ratio=2.0,
    radius=0.18,
    radius_neighbors=3
):
    """
    Cleans the fused dense cloud before meshing.

    The thresholds are intentionally conservative:
    - voxel downsampling removes redundant nearby samples
    - statistical filtering removes isolated global outliers
    - radius filtering removes small floating fragments

    Returns cleaned points, cleaned colors, and cleanup statistics.
    """

    points = np.asarray(
        points,
        dtype=np.float64
    )

    colors = np.asarray(
        colors,
        dtype=np.float64
    )

    if len(points) == 0:
        raise RuntimeError(
            "Cannot clean an empty point cloud."
        )

    cloud = o3d.geometry.PointCloud()

    cloud.points = o3d.utility.Vector3dVector(
        points
    )

    cloud.colors = o3d.utility.Vector3dVector(
        np.clip(
            colors / 255.0,
            0.0,
            1.0
        )
    )

    raw_count = len(cloud.points)

    # 1. Reduce duplicate / extremely dense nearby samples.
    if voxel_size > 0:
        cloud = cloud.voxel_down_sample(
            voxel_size=voxel_size
        )

    after_voxel = len(cloud.points)

    # 2. Remove points that are statistically far from their neighbours.
    if (
        len(cloud.points)
        > statistical_neighbors
    ):
        cloud, _ = (
            cloud.remove_statistical_outlier(
                nb_neighbors=statistical_neighbors,
                std_ratio=statistical_std_ratio
            )
        )

    after_statistical = len(
        cloud.points
    )

    # 3. Remove tiny isolated floating fragments.
    if (
        radius > 0
        and len(cloud.points)
        > radius_neighbors
    ):
        radius_cloud, _ = (
            cloud.remove_radius_outlier(
                nb_points=radius_neighbors,
                radius=radius
            )
        )

        # Guard against an overly aggressive radius threshold.
        # If it destroys most of the reconstruction, keep the
        # statistically-cleaned cloud instead.
        if (
            len(radius_cloud.points)
            >= max(
                1000,
                int(
                    0.35
                    * after_statistical
                )
            )
        ):
            cloud = radius_cloud

    after_radius = len(
        cloud.points
    )

    cleaned_points = np.asarray(
        cloud.points,
        dtype=np.float32
    )

    cleaned_colors = np.asarray(
        cloud.colors
    )

    cleaned_colors = np.clip(
        np.rint(
            cleaned_colors * 255.0
        ),
        0,
        255
    ).astype(np.uint8)

    print(
        "Dense cloud cleanup:",
        f"raw={raw_count},",
        f"voxel={after_voxel},",
        f"statistical={after_statistical},",
        f"final={after_radius}"
    )

    return (
        cleaned_points,
        cleaned_colors,
        {
            "raw_points": raw_count,
            "after_voxel": after_voxel,
            "after_statistical": (
                after_statistical
            ),
            "cleaned_points": after_radius
        }
    )


def fuse_depth_maps(
    image_dir,
    depth_dir,
    model_dir,
    output_path,
    sample_step=12
):
    image_dir = Path(image_dir)
    depth_dir = Path(depth_dir)
    model_dir = Path(model_dir)

    cameras = read_cameras(
        model_dir / "cameras.txt"
    )

    images = read_images(
        model_dir / "images.txt"
    )

    points3d = read_points3d(
        model_dir / "points3D.txt"
    )

    world_points = []
    world_colors = []

    successful_views = 0

    for image_name, info in images.items():
        image_path = (
            image_dir / image_name
        )

        depth_path = (
            depth_dir
            / f"{Path(image_name).stem}_depth.npy"
        )

        if (
            not image_path.exists()
            or not depth_path.exists()
        ):
            continue

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        depth = np.load(
            depth_path
        )

        alignment = fit_depth_alignment(
            depth,
            info,
            points3d
        )

        if alignment is None:
            print(
                "Skipping:",
                image_name,
                "- insufficient COLMAP anchors"
            )
            continue

        metric_depth = align_depth(
            depth,
            alignment
        )

        camera = cameras[
            info["camera_id"]
        ]

        fx = camera["fx"]
        fy = camera["fy"]
        cx = camera["cx"]
        cy = camera["cy"]

        R = info["rotation"]
        t = info["translation"]

        height, width = metric_depth.shape

        print(
            image_name,
            "| mode:",
            alignment["mode"],
            "| anchors:",
            alignment["samples"],
            "| median error:",
            round(
                float(
                    alignment["error"]
                ),
                4
            )
        )

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
                z = float(
                    metric_depth[v, u]
                )

                if (
                    not np.isfinite(z)
                    or z <= 0
                ):
                    continue

                x = (
                    (u - cx)
                    * z
                    / fx
                )

                y = (
                    (v - cy)
                    * z
                    / fy
                )

                camera_point = np.array([
                    x,
                    y,
                    z
                ])

                # COLMAP:
                # X_camera = R X_world + t
                #
                # therefore:
                # X_world = R.T (X_camera - t)
                world_point = (
                    R.T
                    @ (
                        camera_point - t
                    )
                )

                if not np.all(
                    np.isfinite(
                        world_point
                    )
                ):
                    continue

                world_points.append(
                    world_point
                )

                if (
                    v < image.shape[0]
                    and u < image.shape[1]
                ):
                    world_colors.append(
                        image[v, u]
                    )

        successful_views += 1

    if not world_points:
        raise RuntimeError(
            "No dense points could be fused."
        )

    world_points = np.asarray(
        world_points,
        dtype=np.float32
    )

    world_colors = np.asarray(
        world_colors,
        dtype=np.uint8
    )

    (
        cleaned_points,
        cleaned_colors,
        cleanup_stats
    ) = clean_fused_point_cloud(
        world_points,
        world_colors
    )

    save_colored_ply(
        cleaned_points,
        cleaned_colors,
        output_path
    )

    return {
        "views": successful_views,
        "points": len(cleaned_points),
        "raw_points": len(world_points),
        "cleanup": cleanup_stats,
        "output": str(output_path)
    }