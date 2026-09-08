from pathlib import Path
import json
import shutil
import tempfile
import zipfile

import open3d as o3d


def convert_mesh_formats(
    source_mesh,
    models_dir,
):
    """
    Convert the genuine reconstructed PLY mesh
    into additional standard 3D formats.

    No geometry is fabricated. These files contain
    the same reconstructed mesh in other formats.
    """
    source_mesh = Path(source_mesh)
    models_dir = Path(models_dir)

    results = {
        "obj": False,
        "glb": False,
    }

    if not source_mesh.exists():
        return results

    try:
        mesh = o3d.io.read_triangle_mesh(
            str(source_mesh)
        )

        if (
            mesh.is_empty()
            or len(mesh.vertices) == 0
            or len(mesh.triangles) == 0
        ):
            print(
                "Export conversion skipped: "
                "mesh contains no triangles."
            )
            return results

        # Ensure normals exist for viewers.
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()

        # -----------------------------------------
        # OBJ
        # -----------------------------------------

        obj_path = (
            models_dir
            / "mesh.obj"
        )

        try:
            success = (
                o3d.io.write_triangle_mesh(
                    str(obj_path),
                    mesh,
                    write_ascii=False,
                    compressed=False,
                    write_vertex_normals=True,
                    write_vertex_colors=True,
                    write_triangle_uvs=False,
                )
            )

            results["obj"] = bool(
                success
                and obj_path.exists()
            )

        except Exception as error:
            print(
                "OBJ export error:",
                error,
            )

        # -----------------------------------------
        # GLB
        # -----------------------------------------

        glb_path = (
            models_dir
            / "mesh.glb"
        )

        try:
            success = (
                o3d.io.write_triangle_mesh(
                    str(glb_path),
                    mesh,
                    write_ascii=False,
                    compressed=False,
                    write_vertex_normals=True,
                    write_vertex_colors=True,
                    write_triangle_uvs=False,
                )
            )

            results["glb"] = bool(
                success
                and glb_path.exists()
            )

        except Exception as error:
            print(
                "GLB export error:",
                error,
            )

    except Exception as error:
        print(
            "Mesh conversion error:",
            error,
        )

    return results


def create_reconstruction_export(
    base_dir,
    reconstruction_status,
    quality_data,
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

    export_dir = (
        output_dir
        / "exports"
    )

    export_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    export_path = (
        export_dir
        / "skyform_reconstruction.zip"
    )

    # -----------------------------------------
    # DETERMINE COORDINATE STATE
    # -----------------------------------------

    coordinate_system = (
        reconstruction_status.get(
            "coordinate_system",
            "relative_unscaled",
        )
    )

    aligned = bool(
        reconstruction_status.get(
            "aligned",
            False,
        )
    )

    gps_available = bool(
        reconstruction_status.get(
            "gps_available",
            False,
        )
    )

    # If GPS alignment genuinely exists, export
    # the georeferenced geometry as the primary
    # geometry. Otherwise use relative SfM output.
    relative_cloud = (
        reconstruction_dir
        / "dense_fused.ply"
    )

    relative_mesh = (
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

    if (
        aligned
        and georef_cloud.exists()
        and georef_mesh.exists()
    ):
        primary_cloud = (
            georef_cloud
        )

        primary_mesh = (
            georef_mesh
        )

        geometry_state = (
            "georeferenced"
        )

    else:
        primary_cloud = (
            relative_cloud
        )

        primary_mesh = (
            relative_mesh
        )

        geometry_state = (
            "relative"
        )

    # -----------------------------------------
    # BUILD TEMPORARY EXPORT PACKAGE
    # -----------------------------------------

    with tempfile.TemporaryDirectory() as temp:
        temp_dir = Path(temp)

        package_dir = (
            temp_dir
            / "SkyFORM"
        )

        package_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        models_dir = (
            package_dir
            / "models"
        )

        models_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # -----------------------------------------
        # REAL PLY ARTIFACTS
        # -----------------------------------------

        cloud_exported = False
        mesh_exported = False

        if primary_cloud.exists():
            shutil.copy2(
                primary_cloud,
                models_dir
                / "dense_point_cloud.ply",
            )

            cloud_exported = True

        if primary_mesh.exists():
            shutil.copy2(
                primary_mesh,
                models_dir
                / "mesh.ply",
            )

            mesh_exported = True

        # -----------------------------------------
        # OBJ + GLB
        # -----------------------------------------

        conversions = {
            "obj": False,
            "glb": False,
        }

        if primary_mesh.exists():
            conversions = (
                convert_mesh_formats(
                    primary_mesh,
                    models_dir,
                )
            )

        # -----------------------------------------
        # RECONSTRUCTION METRICS
        # -----------------------------------------

        metrics = {
            "project":
                "SkyFORM",

            "coordinate_system":
                coordinate_system,

            "geometry_state":
                geometry_state,

            "gps_available":
                gps_available,

            "aligned":
                aligned,

            "reconstruction":
                reconstruction_status.get(
                    "metrics",
                    {},
                ),

            "quality":
                quality_data.get(
                    "summary",
                    {},
                ),

            "formats": {
                "point_cloud_ply":
                    cloud_exported,

                "mesh_ply":
                    mesh_exported,

                "mesh_obj":
                    conversions[
                        "obj"
                    ],

                "mesh_glb":
                    conversions[
                        "glb"
                    ],
            },
        }

        # Include genuine alignment information
        # when available.
        geospatial_data = (
            reconstruction_status.get(
                "geospatial",
                {},
            )
        )

        if geospatial_data:
            metrics[
                "geospatial"
            ] = geospatial_data

        metrics_path = (
            package_dir
            / "reconstruction_metrics.json"
        )

        with open(
            metrics_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metrics,
                file,
                indent=2,
            )

        # -----------------------------------------
        # README
        # -----------------------------------------

        if aligned:
            coordinate_description = """
The exported primary geometry has been transformed
into a local East-North-Up metric coordinate frame
using supplied GPS telemetry.

The GPS/SfM alignment residual describes the fit
between reconstructed camera positions and GPS
samples. It is NOT an independently measured
ground-truth reconstruction accuracy.
"""
        elif gps_available:
            coordinate_description = """
GPS telemetry is available for this project, but
geospatial alignment has not been completed.

The exported geometry therefore remains in the
relative, unscaled Structure-from-Motion coordinate
frame.
"""
        else:
            coordinate_description = """
No validated GPS alignment is associated with this
reconstruction.

The exported geometry therefore remains in the
relative, unscaled Structure-from-Motion coordinate
frame. Distances must not be interpreted as metres.
"""

        readme = f"""SkyFORM Reconstruction Export

Generated by SkyFORM
Single-Pass UAV Video to 3D Reconstruction System


CONTENTS

models/dense_point_cloud.ply
Pose-aware dense point cloud generated using
COLMAP camera geometry and AI depth fusion.

models/mesh.ply
Surface mesh generated from the fused point cloud.

models/mesh.obj
OBJ representation of the reconstructed mesh,
when conversion is supported.

models/mesh.glb
Binary glTF representation of the reconstructed
mesh, when conversion is supported.

reconstruction_metrics.json
Measured reconstruction metrics, quality
information, coordinate state and available
export formats.


COORDINATE SYSTEM
{coordinate_description}


QUALITY / CONFIDENCE

SkyFORM confidence is a reconstruction-quality
heuristic based on feature-track support and
reprojection residuals.

It must NOT be interpreted as ground-truth
geometric accuracy or as a calibrated probability.


FORMAT NOTES

PLY:
Native SkyFORM point-cloud and mesh output.

OBJ:
Converted representation of the same reconstructed
mesh geometry.

GLB:
Converted binary glTF representation of the same
reconstructed mesh geometry.

GeoTIFF is not generated because SkyFORM does not
currently generate a validated orthorectified
raster or DEM.

FBX is not generated by the current export pipeline.

LAS is not generated by the current export pipeline.
"""

        with open(
            package_dir
            / "README.txt",
            "w",
            encoding="utf-8",
        ) as file:
            file.write(
                readme
            )

        # -----------------------------------------
        # ZIP PACKAGE
        # -----------------------------------------

        if export_path.exists():
            export_path.unlink()

        with zipfile.ZipFile(
            export_path,
            "w",
            zipfile.ZIP_DEFLATED,
        ) as archive:

            for path in (
                package_dir.rglob("*")
            ):
                if path.is_file():
                    archive.write(
                        path,
                        path.relative_to(
                            temp_dir
                        ),
                    )

    return export_path