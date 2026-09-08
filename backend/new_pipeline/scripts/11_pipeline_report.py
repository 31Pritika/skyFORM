import json
import struct

import trimesh

from pipeline_config import (
    get_job_id,
    get_input_video,
    get_frames_dir,
    get_sfm_dir,
    get_colmap_poses_dir,
    get_fusion_dir,
    get_cleanup_dir,
    get_mesh_dir,
    get_georeferencing_dir,
    get_final_dir,
    ensure_job_directories,
)

from colmap_utils import find_best_sparse_model


def read_registered_count(images_bin):
    with open(images_bin, "rb") as f:
        return struct.unpack("<Q", f.read(8))[0]


def read_sparse_point_count(points_bin):
    with open(points_bin, "rb") as f:
        return struct.unpack("<Q", f.read(8))[0]


def count_frames():
    return len(
        list(get_frames_dir().glob("frame_*.jpg"))
    )


def count_points(path):
    if not path.exists():
        return 0

    try:
        cloud = trimesh.load(
            str(path),
            process=False,
        )
        return len(cloud.vertices)
    except Exception:
        return 0


def load_json(path):
    if not path.exists():
        return {}

    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def main():
    ensure_job_directories()

    final_dir = get_final_dir()
    final_dir.mkdir(parents=True, exist_ok=True)

    report_path = final_dir / "pipeline_report.json"

    frames = count_frames()

    registered_frames = None
    sparse_points = None
    selected_model = None

    try:
        model = find_best_sparse_model(
            get_sfm_dir() / "sparse"
        )

        selected_model = model.name

        registered_frames = read_registered_count(
            model / "images.bin"
        )

        sparse_points = read_sparse_point_count(
            model / "points3D.bin"
        )

    except Exception:
        pass

    clean_cloud = (
        get_cleanup_dir()
        / "dense_clean.ply"
    )

    mesh_path = (
        get_mesh_dir()
        / "skyform_mesh.ply"
    )

    telemetry_info = load_json(
        get_georeferencing_dir()
        / "telemetry_validation.json"
    )

    timing_info = load_json(
        final_dir / "pipeline_runtime.json"
    )

    reconstruction_info = load_json(
        final_dir / "reconstruction_info.json"
    )

    validation = reconstruction_info.get(
        "validation",
        {},
    )

    final_point_count = count_points(
        clean_cloud
    )

    report = {
        "project": "SkyFORM",
        "job_id": get_job_id(),

        "input": {
            "video_file": get_input_video().name,
            "frames_extracted": frames,
            "telemetry_samples":
                telemetry_info.get("samples", 0),
        },

        "structure_from_motion": {
            "selected_model": selected_model,
            "registered_frames": registered_frames,
            "registration_rate_percent": (
                registered_frames / frames * 100.0
                if registered_frames is not None
                and frames > 0
                else None
            ),
            "sparse_points": sparse_points,
            "camera_model": "SIMPLE_RADIAL",
        },

        "dense_reconstruction": {
            "method": (
                "MASt3R local geometry with "
                "COLMAP-fixed camera poses"
            ),
            "final_point_count": final_point_count,
            "pointcloud_available":
                clean_cloud.exists(),
            "mesh_available":
                mesh_path.exists(),
            "mesh_status": (
                "experimental"
                if mesh_path.exists()
                else "not generated"
            ),
        },

        "georeferencing": {
            "telemetry_available":
                telemetry_info.get(
                    "samples",
                    0,
                ) > 0,

            "georeferenced":
                validation.get(
                    "georeferenced",
                    False,
                ),

            "metric_scale_validated":
                validation.get(
                    "metric_scale_validated",
                    False,
                ),

            "spatial_accuracy_validated":
                validation.get(
                    "spatial_accuracy_validated",
                    False,
                ),
        },

        "runtime": timing_info,

        "outputs": {
            "pointcloud_ply": str(
                final_dir
                / "skyform_pointcloud.ply"
            ),

            "pointcloud_xyz": str(
                final_dir
                / "skyform_pointcloud.xyz"
            ),

            "results_zip": str(
                final_dir
                / "skyform_results.zip"
            ),
        },

        "claims": {
            "validated": [
                "COLMAP camera reconstruction",
                "Dense point-cloud generation",
                "PLY point-cloud export",
            ],

            "not_yet_validated": [
                "Spatial accuracy <= 1 metre",
                "Metric scale accuracy",
                "GPS georeferencing without supplied telemetry",
                "10-minute video under 15-minute benchmark",
            ],
        },
    }

    report["dense_reconstruction"]["mesh_details"] = load_json(get_mesh_dir() / "mesh_info.json")
    report["georeferencing"]["telemetry_validation"] = load_json(get_georeferencing_dir() / "telemetry_validation.json")
    report["dense_reconstruction"]["coverage"] = load_json(get_fusion_dir() / "coverage.json")
    alignment = load_json(final_dir / "georeferencing.json")
    report["georeferencing"]["georeferenced"] = bool(alignment)
    report["georeferencing"]["alignment"] = alignment
    report["requirements"] = {
        "formats_available": [path.name for path in sorted(final_dir.iterdir())
                              if path.suffix in {".ply", ".obj", ".las", ".glb", ".gltf", ".fbx", ".tif"}],
        "formats_unavailable": {},
        "geotiff_coordinate_note": "Local unscaled raster until GPS/control-point alignment is supplied; not a geographic elevation model.",
        "entire_visible_scene_coverage": "Not validated; water, occlusions and low-texture areas can remain incomplete",
        "spatial_accuracy_le_1m": "Not validated; requires independent metric ground truth",
        "ten_minute_video_under_fifteen_minutes": "Not benchmarked on a ten-minute input",
    }

    required_formats = {"OBJ": ".obj", "PLY": ".ply", "LAS": ".las", "GeoTIFF": ".tif", "GLB": ".glb", "glTF": ".gltf", "FBX": ".fbx"}
    report["requirements"]["formats_unavailable"] = {
        name: "No artifact produced" for name, extension in required_formats.items()
        if not any(filename.endswith(extension) for filename in report["requirements"]["formats_available"])
    }

    with open(report_path, "w") as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    print("\n========================================")
    print(" SKYFORM - PIPELINE REPORT")
    print("========================================")
    print("Job:", get_job_id())
    print("Frames:", frames)
    print("Registered:", registered_frames)
    print("Sparse points:", sparse_points)
    print("Dense points:", final_point_count)
    print(
        "Telemetry samples:",
        telemetry_info.get("samples", 0),
    )
    print("Report:", report_path)
    print("========================================\n")


if __name__ == "__main__":
    main()
