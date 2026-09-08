from pathlib import Path
import json
import struct


# ---------------------------------------------------------
# Basic PLY helpers
# ---------------------------------------------------------

def read_ply_vertex_count(path):
    path = Path(path)

    if not path.exists():
        return 0

    try:
        with open(
            path,
            "rb",
        ) as file:
            while True:
                line = file.readline()

                if not line:
                    break

                text = line.decode(
                    "ascii",
                    errors="ignore",
                ).strip()

                if text.startswith(
                    "element vertex"
                ):
                    return int(
                        text.split()[-1]
                    )

                if text == "end_header":
                    break

    except Exception as error:
        print(
            "PLY status error:",
            error,
        )

    return 0


# ---------------------------------------------------------
# COLMAP binary helpers
# ---------------------------------------------------------

def read_exact(file, size):
    data = file.read(size)

    if len(data) != size:
        raise EOFError(
            "Unexpected end of COLMAP file."
        )

    return data


def read_null_terminated_string(file):
    chars = []

    while True:
        char = read_exact(
            file,
            1,
        )

        if char == b"\x00":
            break

        chars.append(char)

    return b"".join(
        chars
    ).decode(
        "utf-8",
        errors="ignore",
    )


def read_colmap_images_bin(path):
    """
    Returns number of registered images in
    COLMAP images.bin.
    """
    path = Path(path)

    if not path.exists():
        return 0

    with open(
        path,
        "rb",
    ) as file:
        num_images = struct.unpack(
            "<Q",
            read_exact(file, 8),
        )[0]

        for _ in range(
            num_images
        ):
            # image_id
            read_exact(
                file,
                4,
            )

            # qvec = 4 doubles
            read_exact(
                file,
                8 * 4,
            )

            # tvec = 3 doubles
            read_exact(
                file,
                8 * 3,
            )

            # camera_id
            read_exact(
                file,
                4,
            )

            # image filename
            read_null_terminated_string(
                file
            )

            # number of 2D observations
            num_points2d = struct.unpack(
                "<Q",
                read_exact(file, 8),
            )[0]

            # each observation:
            # x double
            # y double
            # point3D_id int64
            file.seek(
                num_points2d * 24,
                1,
            )

    return int(
        num_images
    )


def read_colmap_points3d_bin(path):
    """
    Reads COLMAP points3D.bin.

    Returns:
    - point count
    - observation count
    - mean track length
    - weighted mean reprojection error
    """
    path = Path(path)

    if not path.exists():
        return {
            "points": 0,
            "observations": 0,
            "mean_track_length": None,
            "reprojection_error": None,
        }

    point_count = 0
    total_observations = 0

    weighted_error_sum = 0.0
    total_error_weight = 0

    with open(
        path,
        "rb",
    ) as file:
        num_points = struct.unpack(
            "<Q",
            read_exact(file, 8),
        )[0]

        point_count = int(
            num_points
        )

        for _ in range(
            num_points
        ):
            # point3D_id uint64
            read_exact(
                file,
                8,
            )

            # XYZ = 3 doubles
            read_exact(
                file,
                24,
            )

            # RGB = 3 uint8
            read_exact(
                file,
                3,
            )

            # reprojection error double
            error = struct.unpack(
                "<d",
                read_exact(file, 8),
            )[0]

            # track length uint64
            track_length = struct.unpack(
                "<Q",
                read_exact(file, 8),
            )[0]

            total_observations += (
                track_length
            )

            weighted_error_sum += (
                error
                * track_length
            )

            total_error_weight += (
                track_length
            )

            # Each track entry:
            # image_id int32
            # point2D_idx int32
            file.seek(
                track_length * 8,
                1,
            )

    mean_track_length = (
        total_observations
        / point_count
        if point_count > 0
        else None
    )

    reprojection_error = (
        weighted_error_sum
        / total_error_weight
        if total_error_weight > 0
        else None
    )

    return {
        "points":
            point_count,

        "observations":
            int(
                total_observations
            ),

        "mean_track_length":
            (
                round(
                    mean_track_length,
                    6,
                )
                if mean_track_length
                is not None
                else None
            ),

        "reprojection_error":
            (
                round(
                    reprojection_error,
                    6,
                )
                if reprojection_error
                is not None
                else None
            ),
    }


# ---------------------------------------------------------
# Sparse model discovery
# ---------------------------------------------------------

def is_valid_sparse_model(path):
    path = Path(path)

    return (
        path.is_dir()
        and (
            path
            / "images.bin"
        ).exists()
        and (
            path
            / "points3D.bin"
        ).exists()
    )


def find_sparse_models(
    base_dir,
):
    """
    Search both the old verified COLMAP folder
    and the newer pipeline folder.
    """
    base_dir = Path(base_dir)

    output_dir = (
        base_dir / "outputs"
    )

    sparse_roots = [
        output_dir
        / "colmap_new"
        / "sparse",

        output_dir
        / "colmap"
        / "sparse",
    ]

    models = []

    for root in sparse_roots:
        if not root.exists():
            continue

        for model_dir in (
            root.iterdir()
        ):
            if is_valid_sparse_model(
                model_dir
            ):
                models.append(
                    model_dir
                )

    return models


def find_best_sparse_model(
    base_dir,
):
    """
    Choose the model with the largest number
    of registered images.
    """
    models = find_sparse_models(
        base_dir
    )

    if not models:
        return None

    best_model = None
    best_registered = -1

    for model_dir in models:
        try:
            registered = (
                read_colmap_images_bin(
                    model_dir
                    / "images.bin"
                )
            )

            if (
                registered
                > best_registered
            ):
                best_registered = (
                    registered
                )

                best_model = (
                    model_dir
                )

        except Exception as error:
            print(
                "COLMAP model read error:",
                model_dir,
                error,
            )

    return best_model


# ---------------------------------------------------------
# Active video
# ---------------------------------------------------------

def find_active_video_id(
    base_dir,
):
    base_dir = Path(base_dir)

    output_dir = (
        base_dir / "outputs"
    )

    # Try common pipeline-state locations.
    possible_state_paths = [
        output_dir
        / "pipeline_state.json",

        output_dir
        / "reconstruction"
        / "pipeline_state.json",
    ]

    for pipeline_state_path in (
        possible_state_paths
    ):
        if not pipeline_state_path.exists():
            continue

        try:
            with open(
                pipeline_state_path,
                "r",
                encoding="utf-8",
            ) as file:
                state = json.load(
                    file
                )

            video_id = state.get(
                "video_id"
            )

            if video_id:
                return str(
                    video_id
                )

        except Exception:
            pass

    # Safe fallback:
    # most recently modified uploaded video.
    upload_dir = (
        base_dir
        / "uploads"
    )

    if not upload_dir.exists():
        return None

    allowed_extensions = {
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
    }

    videos = [
        path
        for path
        in upload_dir.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in allowed_extensions
        )
    ]

    if not videos:
        return None

    newest_video = max(
        videos,
        key=lambda path:
            path.stat().st_mtime,
    )

    return (
        newest_video.stem
    )


# ---------------------------------------------------------
# Geospatial status
# ---------------------------------------------------------

def get_geospatial_state(
    base_dir,
    video_id,
):
    base_dir = Path(base_dir)

    default_state = {
        "video_id":
            video_id,

        "gps_available":
            False,

        "aligned":
            False,

        "coordinate_system":
            "relative_unscaled",

        "alignment":
            None,
    }

    if not video_id:
        return default_state

    telemetry_dir = (
        base_dir
        / "outputs"
        / "telemetry"
        / video_id
    )

    gps_path = (
        telemetry_dir
        / "gps.csv"
    )

    transform_path = (
        telemetry_dir
        / "geospatial_transform.json"
    )

    gps_available = (
        gps_path.exists()
    )

    if not transform_path.exists():
        return {
            **default_state,

            "gps_available":
                gps_available,
        }

    try:
        with open(
            transform_path,
            "r",
            encoding="utf-8",
        ) as file:
            transform = (
                json.load(file)
            )

        if (
            transform.get(
                "coordinate_system"
            )
            != "local_enu_meters"
        ):
            return {
                **default_state,

                "gps_available":
                    gps_available,
            }

        return {
            "video_id":
                video_id,

            "gps_available":
                gps_available,

            "aligned":
                True,

            "coordinate_system":
                "local_enu_meters",

            "alignment":
                transform,
        }

    except Exception as error:
        print(
            "Geospatial status error:",
            error,
        )

        return {
            **default_state,

            "gps_available":
                gps_available,
        }


# ---------------------------------------------------------
# Main reconstruction status
# ---------------------------------------------------------

def get_reconstruction_status(
    base_dir,
):
    base_dir = Path(base_dir)

    output_dir = (
        base_dir
        / "outputs"
    )

    reconstruction_dir = (
        output_dir
        / "reconstruction"
    )

    keyframe_dir = (
        output_dir
        / "keyframes"
        / "new_test"
    )

    depth_dir = (
        output_dir
        / "depth_maps"
    )

    fused_cloud = (
        reconstruction_dir
        / "dense_fused.ply"
    )

    mesh_file = (
        reconstruction_dir
        / "mesh.ply"
    )

    georef_cloud = (
        reconstruction_dir
        / "dense_fused_georef.ply"
    )

    georef_mesh = (
        reconstruction_dir
        / "mesh_georef.ply"
    )

    # -----------------------------
    # Keyframes
    # -----------------------------

    keyframes = 0

    if keyframe_dir.exists():
        keyframes = len(
            list(
                keyframe_dir.glob(
                    "*.jpg"
                )
            )
        )

    # -----------------------------
    # Depth maps
    # -----------------------------

    depth_maps = 0

    if depth_dir.exists():
        depth_maps = len(
            list(
                depth_dir.glob(
                    "*_depth.npy"
                )
            )
        )

    # -----------------------------
    # Dense cloud
    # -----------------------------

    fused_points = (
        read_ply_vertex_count(
            fused_cloud
        )
    )

    # -----------------------------
    # Sparse reconstruction
    # -----------------------------

    sparse_model = (
        find_best_sparse_model(
            base_dir
        )
    )

    registered_images = 0
    sparse_points = 0
    observations = 0
    mean_track_length = None
    reprojection_error = None

    if sparse_model:
        try:
            registered_images = (
                read_colmap_images_bin(
                    sparse_model
                    / "images.bin"
                )
            )

            point_metrics = (
                read_colmap_points3d_bin(
                    sparse_model
                    / "points3D.bin"
                )
            )

            sparse_points = (
                point_metrics[
                    "points"
                ]
            )

            observations = (
                point_metrics[
                    "observations"
                ]
            )

            mean_track_length = (
                point_metrics[
                    "mean_track_length"
                ]
            )

            reprojection_error = (
                point_metrics[
                    "reprojection_error"
                ]
            )

        except Exception as error:
            print(
                "COLMAP binary status error:",
                error,
            )

    # -----------------------------
    # Artifacts
    # -----------------------------

    dense_ready = (
        fused_cloud.exists()
    )

    mesh_ready = (
        mesh_file.exists()
    )

    # -----------------------------
    # Current video / GPS
    # -----------------------------

    active_video_id = (
        find_active_video_id(
            base_dir
        )
    )

    geospatial = (
        get_geospatial_state(
            base_dir,
            active_video_id,
        )
    )

    geospatial_aligned = (
        geospatial[
            "aligned"
        ]
    )

    # -----------------------------
    # Pipeline state
    # -----------------------------

    pipeline = {
        "video_ingestion": (
            active_video_id
            is not None
        ),

        "frame_intelligence": (
            keyframes > 0
        ),

        "spatial_computation": (
            registered_images > 0
        ),

        "reconstruction": (
            dense_ready
            and mesh_ready
        ),

        "geospatial_alignment":
            geospatial_aligned,
    }

    # -----------------------------
    # Response
    # -----------------------------

    return {
        "status": (
            "ready"
            if (
                dense_ready
                and mesh_ready
            )
            else "processing"
        ),

        "video_id":
            active_video_id,

        "coordinate_system":
            geospatial[
                "coordinate_system"
            ],

        "gps_available":
            geospatial[
                "gps_available"
            ],

        "aligned":
            geospatial[
                "aligned"
            ],

        "metrics": {
            "keyframes":
                keyframes,

            "registered_cameras":
                registered_images,

            "sparse_points":
                sparse_points,

            "observations":
                observations,

            "mean_track_length":
                mean_track_length,

            "fused_points":
                fused_points,

            "depth_maps":
                depth_maps,

            "reprojection_error_px":
                reprojection_error,
        },

        "geospatial": {
            "gps_available":
                geospatial[
                    "gps_available"
                ],

            "aligned":
                geospatial[
                    "aligned"
                ],

            "coordinate_system":
                geospatial[
                    "coordinate_system"
                ],

            "alignment":
                geospatial[
                    "alignment"
                ],
        },

        "artifacts": {
            "sparse_model":
                sparse_model is not None,

            "sparse_model_path":
                (
                    str(
                        sparse_model
                    )
                    if sparse_model
                    else None
                ),

            "dense_point_cloud":
                dense_ready,

            "mesh":
                mesh_ready,

            "georeferenced_point_cloud":
                (
                    geospatial_aligned
                    and georef_cloud.exists()
                ),

            "georeferenced_mesh":
                (
                    geospatial_aligned
                    and georef_mesh.exists()
                ),

            "dense_point_cloud_url":
                (
                    "/outputs/"
                    "reconstruction/"
                    "dense_fused.ply"
                    if dense_ready
                    else None
                ),

            "mesh_url":
                (
                    "/outputs/"
                    "reconstruction/"
                    "mesh.ply"
                    if mesh_ready
                    else None
                ),

            "georeferenced_point_cloud_url":
                (
                    "/outputs/"
                    "reconstruction/"
                    "dense_fused_georef.ply"
                    if (
                        geospatial_aligned
                        and georef_cloud.exists()
                    )
                    else None
                ),

            "georeferenced_mesh_url":
                (
                    "/outputs/"
                    "reconstruction/"
                    "mesh_georef.ply"
                    if (
                        geospatial_aligned
                        and georef_mesh.exists()
                    )
                    else None
                ),
        },

        "pipeline":
            pipeline,
    }