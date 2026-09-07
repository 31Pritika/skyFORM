from pathlib import Path
import json
import math
import re

import cv2
import numpy as np
import open3d as o3d


EARTH_RADIUS_METERS = 6378137.0


# ============================================================
# GPS → LOCAL ENU
# ============================================================

def gps_to_local_enu(
    latitude,
    longitude,
    altitude,
    reference_latitude,
    reference_longitude,
    reference_altitude,
):
    lat = math.radians(latitude)
    lon = math.radians(longitude)

    lat0 = math.radians(
        reference_latitude
    )

    lon0 = math.radians(
        reference_longitude
    )

    east = (
        (lon - lon0)
        * math.cos(lat0)
        * EARTH_RADIUS_METERS
    )

    north = (
        (lat - lat0)
        * EARTH_RADIUS_METERS
    )

    up = (
        altitude
        - reference_altitude
    )

    return np.array(
        [east, north, up],
        dtype=np.float64,
    )


# ============================================================
# CAMERA POSES
# ============================================================

def quaternion_to_rotation(
    qw,
    qx,
    qy,
    qz,
):
    q = np.array(
        [qw, qx, qy, qz],
        dtype=np.float64,
    )

    q /= np.linalg.norm(q)

    qw, qx, qy, qz = q

    return np.array([
        [
            1 - 2 * (
                qy * qy
                + qz * qz
            ),
            2 * (
                qx * qy
                - qz * qw
            ),
            2 * (
                qx * qz
                + qy * qw
            ),
        ],
        [
            2 * (
                qx * qy
                + qz * qw
            ),
            1 - 2 * (
                qx * qx
                + qz * qz
            ),
            2 * (
                qy * qz
                - qx * qw
            ),
        ],
        [
            2 * (
                qx * qz
                - qy * qw
            ),
            2 * (
                qy * qz
                + qx * qw
            ),
            1 - 2 * (
                qx * qx
                + qy * qy
            ),
        ],
    ])


def read_colmap_camera_centers(
    images_txt_path,
):
    images_txt_path = Path(
        images_txt_path
    )

    if not images_txt_path.exists():
        raise FileNotFoundError(
            "COLMAP images.txt not found."
        )

    with open(
        images_txt_path,
        "r",
        encoding="utf-8",
    ) as file:

        lines = [
            line.strip()
            for line in file
            if line.strip()
            and not line.startswith("#")
        ]

    cameras = []

    index = 0

    while index < len(lines):
        header = lines[
            index
        ].split()

        if len(header) < 10:
            index += 2
            continue

        image_id = int(
            header[0]
        )

        qw, qx, qy, qz = map(
            float,
            header[1:5],
        )

        tx, ty, tz = map(
            float,
            header[5:8],
        )

        image_name = (
            header[9]
        )

        rotation = (
            quaternion_to_rotation(
                qw,
                qx,
                qy,
                qz,
            )
        )

        translation = np.array(
            [tx, ty, tz],
            dtype=np.float64,
        )

        camera_center = (
            -rotation.T
            @ translation
        )

        cameras.append({
            "image_id":
                image_id,

            "image_name":
                image_name,

            "position":
                camera_center,
        })

        index += 2

    return cameras


# ============================================================
# KEYFRAME TIMESTAMP
# ============================================================

def extract_frame_number(
    image_name,
):
    match = re.search(
        r"keyframe_(\d+)",
        image_name,
    )

    if not match:
        return None

    return int(
        match.group(1)
    )


def get_video_fps(
    video_path,
):
    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            "Unable to open source video."
        )

    fps = float(
        capture.get(
            cv2.CAP_PROP_FPS
        )
    )

    capture.release()

    if fps <= 0:
        raise RuntimeError(
            "Invalid video FPS."
        )

    return fps


# ============================================================
# GPS MATCHING
# ============================================================

def match_cameras_to_gps(
    cameras,
    gps_points,
    fps,
    max_time_difference=1.5,
):
    if not gps_points:
        raise RuntimeError(
            "GPS telemetry is empty."
        )

    matches = []

    for camera in cameras:
        frame_number = (
            extract_frame_number(
                camera[
                    "image_name"
                ]
            )
        )

        if frame_number is None:
            continue

        timestamp = (
            frame_number
            / fps
        )

        nearest = min(
            gps_points,
            key=lambda point:
                abs(
                    float(
                        point[
                            "timestamp"
                        ]
                    )
                    - timestamp
                ),
        )

        time_difference = abs(
            float(
                nearest[
                    "timestamp"
                ]
            )
            - timestamp
        )

        if (
            time_difference
            > max_time_difference
        ):
            continue

        matches.append({
            "image_name":
                camera[
                    "image_name"
                ],

            "frame_number":
                frame_number,

            "timestamp":
                timestamp,

            "time_difference":
                time_difference,

            "sfm_position":
                camera[
                    "position"
                ],

            "gps":
                nearest,
        })

    return matches


# ============================================================
# UMEYAMA SIMILARITY TRANSFORM
# ============================================================

def estimate_similarity_transform(
    source_points,
    target_points,
):
    source_points = np.asarray(
        source_points,
        dtype=np.float64,
    )

    target_points = np.asarray(
        target_points,
        dtype=np.float64,
    )

    if (
        len(source_points) < 3
        or len(target_points) < 3
    ):
        raise RuntimeError(
            "At least 3 matched camera/GPS "
            "positions are required."
        )

    source_mean = (
        source_points.mean(
            axis=0
        )
    )

    target_mean = (
        target_points.mean(
            axis=0
        )
    )

    source_centered = (
        source_points
        - source_mean
    )

    target_centered = (
        target_points
        - target_mean
    )

    covariance = (
        target_centered.T
        @ source_centered
        / len(source_points)
    )

    U, singular_values, Vt = (
        np.linalg.svd(
            covariance
        )
    )

    correction = np.eye(3)

    if (
        np.linalg.det(U)
        * np.linalg.det(Vt)
        < 0
    ):
        correction[
            2,
            2
        ] = -1

    rotation = (
        U
        @ correction
        @ Vt
    )

    source_variance = (
        np.mean(
            np.sum(
                source_centered ** 2,
                axis=1,
            )
        )
    )

    if source_variance <= 1e-12:
        raise RuntimeError(
            "Degenerate camera trajectory."
        )

    scale = (
        np.sum(
            singular_values
            * np.diag(
                correction
            )
        )
        / source_variance
    )

    translation = (
        target_mean
        - scale
        * (
            rotation
            @ source_mean
        )
    )

    transformed = (
        scale
        * (
            source_points
            @ rotation.T
        )
        + translation
    )

    residuals = (
        transformed
        - target_points
    )

    distances = np.linalg.norm(
        residuals,
        axis=1,
    )

    rmse = float(
        np.sqrt(
            np.mean(
                distances ** 2
            )
        )
    )

    median_error = float(
        np.median(
            distances
        )
    )

    return {
        "scale":
            float(scale),

        "rotation":
            rotation,

        "translation":
            translation,

        "rmse_m":
            rmse,

        "median_error_m":
            median_error,

        "transformed_source":
            transformed,
    }


# ============================================================
# APPLY TRANSFORM
# ============================================================

def transform_points(
    points,
    scale,
    rotation,
    translation,
):
    points = np.asarray(
        points,
        dtype=np.float64,
    )

    return (
        scale
        * (
            points
            @ rotation.T
        )
        + translation
    )


def transform_point_cloud(
    input_path,
    output_path,
    scale,
    rotation,
    translation,
):
    cloud = o3d.io.read_point_cloud(
        str(input_path)
    )

    points = np.asarray(
        cloud.points
    )

    if len(points) == 0:
        raise RuntimeError(
            "Point cloud is empty."
        )

    transformed = (
        transform_points(
            points,
            scale,
            rotation,
            translation,
        )
    )

    cloud.points = (
        o3d.utility
        .Vector3dVector(
            transformed
        )
    )

    success = (
        o3d.io.write_point_cloud(
            str(output_path),
            cloud,
        )
    )

    if not success:
        raise RuntimeError(
            "Failed to write "
            "georeferenced point cloud."
        )


def transform_mesh(
    input_path,
    output_path,
    scale,
    rotation,
    translation,
):
    mesh = o3d.io.read_triangle_mesh(
        str(input_path)
    )

    vertices = np.asarray(
        mesh.vertices
    )

    if len(vertices) == 0:
        raise RuntimeError(
            "Mesh is empty."
        )

    transformed = (
        transform_points(
            vertices,
            scale,
            rotation,
            translation,
        )
    )

    mesh.vertices = (
        o3d.utility
        .Vector3dVector(
            transformed
        )
    )

    mesh.compute_vertex_normals()

    success = (
        o3d.io.write_triangle_mesh(
            str(output_path),
            mesh,
            write_vertex_normals=True,
            write_vertex_colors=True,
        )
    )

    if not success:
        raise RuntimeError(
            "Failed to write "
            "georeferenced mesh."
        )


# ============================================================
# COMPLETE GEOREFERENCING
# ============================================================

def georeference_reconstruction(
    base_dir,
    video_path,
    gps_data,
):
    base_dir = Path(
        base_dir
    )

    video_path = Path(
        video_path
    )

    reconstruction_dir = (
        base_dir
        / "outputs"
        / "reconstruction"
    )

    images_txt = (
        base_dir
        / "outputs"
        / "colmap_new"
        / "dense"
        / "sparse_txt"
        / "images.txt"
    )

    source_cloud = (
        reconstruction_dir
        / "dense_fused.ply"
    )

    source_mesh = (
        reconstruction_dir
        / "mesh.ply"
    )

    output_cloud = (
        reconstruction_dir
        / "dense_fused_georef.ply"
    )

    output_mesh = (
        reconstruction_dir
        / "mesh_georef.ply"
    )

    video_id = video_path.stem

    telemetry_dir = (
        base_dir
        / "outputs"
        / "telemetry"
        / video_id
    )

    telemetry_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    transform_path = (
        telemetry_dir
        / "geospatial_transform.json"
    )

    cameras = (
        read_colmap_camera_centers(
            images_txt
        )
    )

    fps = get_video_fps(
        video_path
    )

    gps_points = (
        gps_data[
            "points"
        ]
    )

    if len(gps_points) < 3:
        raise RuntimeError(
            "At least 3 GPS telemetry "
            "samples are required."
        )

    reference = (
        gps_points[0]
    )

    reference_latitude = float(
        reference[
            "latitude"
        ]
    )

    reference_longitude = float(
        reference[
            "longitude"
        ]
    )

    reference_altitude = float(
        reference.get(
            "altitude",
            0.0,
        )
    )

    matches = (
        match_cameras_to_gps(
            cameras,
            gps_points,
            fps,
        )
    )

    if len(matches) < 3:
        raise RuntimeError(
            "Not enough camera/GPS "
            "timestamp matches."
        )

    sfm_points = []

    enu_points = []

    for match in matches:
        gps = match["gps"]

        enu = gps_to_local_enu(
            float(
                gps["latitude"]
            ),
            float(
                gps["longitude"]
            ),
            float(
                gps.get(
                    "altitude",
                    reference_altitude,
                )
            ),
            reference_latitude,
            reference_longitude,
            reference_altitude,
        )

        sfm_points.append(
            match[
                "sfm_position"
            ]
        )

        enu_points.append(
            enu
        )

    transform = (
        estimate_similarity_transform(
            sfm_points,
            enu_points,
        )
    )

    scale = (
        transform[
            "scale"
        ]
    )

    rotation = (
        transform[
            "rotation"
        ]
    )

    translation = (
        transform[
            "translation"
        ]
    )

    transform_point_cloud(
        source_cloud,
        output_cloud,
        scale,
        rotation,
        translation,
    )

    transform_mesh(
        source_mesh,
        output_mesh,
        scale,
        rotation,
        translation,
    )

    result = {
        "coordinate_system":
            "local_enu_meters",

        "reference": {
            "latitude":
                reference_latitude,

            "longitude":
                reference_longitude,

            "altitude":
                reference_altitude,
        },

        "matched_cameras":
            len(matches),

        "scale":
            scale,

        "alignment_rmse_m":
            transform[
                "rmse_m"
            ],

        "alignment_median_error_m":
            transform[
                "median_error_m"
            ],

        "rotation":
            rotation.tolist(),

        "translation":
            translation.tolist(),

        "outputs": {
            "point_cloud":
                str(
                    output_cloud
                ),

            "mesh":
                str(
                    output_mesh
                ),
        },
    }

    with open(
        transform_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
        )

    return result