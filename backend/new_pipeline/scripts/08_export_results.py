from pathlib import Path
import json
import shutil
import zipfile

import numpy as np
import trimesh
import laspy
from format_exports import export_surface_raster, export_fbx

from pipeline_config import (
    get_cleanup_dir,
    get_mesh_dir,
    get_final_dir,
    ensure_job_directories,
)


# ============================================================
# PATHS
# ============================================================

CLEAN_PLY = (
    get_cleanup_dir()
    / "dense_clean.ply"
)

MESH_PLY = (
    get_mesh_dir()
    / "skyform_mesh.ply"
)

MESH_OBJ = (
    get_mesh_dir()
    / "skyform_mesh.obj"
)

FINAL_DIR = get_final_dir()

FINAL_PLY = FINAL_DIR / "skyform_pointcloud.ply"
FINAL_XYZ = FINAL_DIR / "skyform_pointcloud.xyz"

FINAL_MESH_PLY = FINAL_DIR / "skyform_mesh_experimental.ply"
FINAL_MESH_OBJ = FINAL_DIR / "skyform_mesh_experimental.obj"

INFO_JSON = FINAL_DIR / "reconstruction_info.json"
ZIP_PATH = FINAL_DIR / "skyform_results.zip"


def export_xyz(points, colors, path):
    colors = colors.astype(np.uint8)

    data = np.column_stack([
        points,
        colors
    ])

    np.savetxt(
        path,
        data,
        fmt=[
            "%.6f",
            "%.6f",
            "%.6f",
            "%d",
            "%d",
            "%d",
        ]
    )


def main():
    ensure_job_directories()

    print("\n========================================")
    print(" SKYFORM - FINAL EXPORT")
    print("========================================")

    if not CLEAN_PLY.exists():
        raise FileNotFoundError(
            f"Clean point cloud missing:\n{CLEAN_PLY}"
        )

    FINAL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nCopying primary point cloud...")

    shutil.copy2(
        CLEAN_PLY,
        FINAL_PLY
    )

    cloud = trimesh.load(
        str(CLEAN_PLY),
        process=False
    )

    points = np.asarray(
        cloud.vertices,
        dtype=np.float64
    )

    if len(points) == 0:
        raise RuntimeError(
            "Clean point cloud contains zero points."
        )

    finite = np.isfinite(points).all(axis=1)
    points = points[finite]

    if len(points) == 0:
        raise RuntimeError(
            "Clean point cloud contains no finite points."
        )

    if hasattr(cloud.visual, "vertex_colors"):
        colors = np.asarray(
            cloud.visual.vertex_colors
        )

        colors = colors[finite]

        if (
            colors.ndim == 2
            and colors.shape[1] >= 3
        ):
            colors = colors[:, :3]
        else:
            colors = np.full(
                (len(points), 3),
                200,
                dtype=np.uint8
            )
    else:
        colors = np.full(
            (len(points), 3),
            200,
            dtype=np.uint8
        )

    print("Exporting XYZ...")

    export_xyz(
        points,
        colors,
        FINAL_XYZ
    )

    raster_info = export_surface_raster(points, FINAL_DIR / "skyform_surface_local.tif")

    # LAS coordinates retain the SfM frame; no metric CRS is invented.
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = points.min(axis=0)
    header.scales = np.maximum(np.ptp(points, axis=0) / 2_000_000_000, 1e-6)
    las = laspy.LasData(header)
    las.x, las.y, las.z = points.T
    las.red, las.green, las.blue = (colors.astype(np.uint16) * 257).T
    las.write(FINAL_DIR / "skyform_pointcloud.las")

    # --------------------------------------------------------
    # OPTIONAL / EXPERIMENTAL MESH
    # --------------------------------------------------------

    mesh_available = False
    fbx_error = None

    if MESH_PLY.exists():
        shutil.copy2(
            MESH_PLY,
            FINAL_MESH_PLY
        )
        surface = trimesh.load(str(MESH_PLY), process=False)
        surface.export(str(FINAL_DIR / "skyform_mesh_experimental.glb"))
        # FBX conversion depends on the external Assimp CLI. Treat it as a
        # best-effort extra format: a missing/failing Assimp must not sink the
        # whole export, which also produces PLY, OBJ, GLB and glTF.
        try:
            export_fbx(FINAL_DIR / "skyform_mesh_experimental.glb", FINAL_DIR / "skyform_mesh_experimental.fbx")
        except Exception as error:
            fbx_error = str(error)
            print(f"WARNING: FBX export skipped ({fbx_error})")
        for filename, content in trimesh.exchange.gltf.export_gltf(surface).items():
            (FINAL_DIR / filename).write_bytes(content)
        mesh_available = True

    if MESH_OBJ.exists():
        shutil.copy2(
            MESH_OBJ,
            FINAL_MESH_OBJ
        )
        mesh_available = True

    # --------------------------------------------------------
    # POINT CLOUD BOUNDS
    # --------------------------------------------------------

    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    dimensions = max_bound - min_bound

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------
    #
    # Do not hard-code metrics from an earlier reconstruction.
    # Stage 11 reads the actual COLMAP model and creates the
    # detailed pipeline report.

    info = {
        "project": "SkyFORM",

        "pipeline": [
            "Drone video input",
            "Frame extraction",
            "COLMAP structure-from-motion",
            "MASt3R local dense geometry",
            "COLMAP fixed-pose depth fusion",
            "Point-cloud outlier cleanup",
            "Final export"
        ],

        "dense_reconstruction": {
            "method": (
                "MASt3R local depth with "
                "COLMAP-fixed camera trajectory"
            ),
            "global_pose_source": "COLMAP",
            "primary_output": "point cloud"
        },

        "output": {
            "point_count": int(len(points)),
            "bounding_box_min": min_bound.tolist(),
            "bounding_box_max": max_bound.tolist(),
            "scene_dimensions_colmap_units": dimensions.tolist(),
            "formats": [
                "PLY",
                "XYZ", "LAS", "GeoTIFF (relative unscaled)"
            ],
            "experimental_mesh_available": mesh_available
        },

        "surface_raster": raster_info,

        "validation": {
            "georeferenced": False,
            "metric_scale_validated": False,
            "spatial_accuracy_validated": False,
            "note": (
                "Current reconstruction is expressed "
                "in COLMAP reconstruction coordinates. "
                "GPS/RTK/GCP alignment and independent "
                "metric accuracy validation have not "
                "yet been performed."
            )
        }
    }

    if mesh_available:
        mesh_formats = [
            "experimental mesh PLY",
            "experimental mesh OBJ", "GLB", "glTF",
        ]
        if fbx_error is None and (FINAL_DIR / "skyform_mesh_experimental.fbx").exists():
            mesh_formats.append("FBX")
        else:
            info["output"]["fbx_unavailable_reason"] = (
                fbx_error or "Assimp CLI not found"
            )
        info["output"]["formats"].extend(mesh_formats)

    print("Writing metadata...")

    with open(
        INFO_JSON,
        "w"
    ) as f:
        json.dump(
            info,
            f,
            indent=2
        )

    print("Creating ZIP package...")

    files_to_zip = [path for path in sorted(FINAL_DIR.iterdir())
                    if path.is_file() and path.suffix != ".zip"]

    with zipfile.ZipFile(
        ZIP_PATH,
        "w",
        compression=zipfile.ZIP_DEFLATED
    ) as zf:
        for path in files_to_zip:
            zf.write(
                path,
                arcname=path.name
            )

    print("\n========================================")
    print(" FINAL EXPORT COMPLETE")
    print("========================================")

    print(
        f"Point cloud: {FINAL_PLY}"
    )

    print(
        f"XYZ:         {FINAL_XYZ}"
    )

    print(
        f"Metadata:    {INFO_JSON}"
    )

    print(
        f"ZIP:         {ZIP_PATH}"
    )

    print(
        f"Points:      {len(points):,}"
    )

    print(
        f"Experimental mesh: "
        f"{'YES' if mesh_available else 'NO'}"
    )

    print("========================================\n")


if __name__ == "__main__":
    main()
