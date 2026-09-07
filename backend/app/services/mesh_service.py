from pathlib import Path

import numpy as np
import open3d as o3d


def _remove_long_triangles(
    mesh,
    max_edge_length
):
    """
    Removes triangles that bridge points that are too far apart.

    This helps suppress stretched surfaces across holes and disconnected
    geometry without changing the valid parts of the reconstruction.
    """

    vertices = np.asarray(
        mesh.vertices
    )

    triangles = np.asarray(
        mesh.triangles
    )

    if len(triangles) == 0:
        return mesh, 0

    tri_vertices = vertices[
        triangles
    ]

    edge_01 = np.linalg.norm(
        tri_vertices[:, 0]
        - tri_vertices[:, 1],
        axis=1
    )

    edge_12 = np.linalg.norm(
        tri_vertices[:, 1]
        - tri_vertices[:, 2],
        axis=1
    )

    edge_20 = np.linalg.norm(
        tri_vertices[:, 2]
        - tri_vertices[:, 0],
        axis=1
    )

    keep_mask = (
        (edge_01 <= max_edge_length)
        & (edge_12 <= max_edge_length)
        & (edge_20 <= max_edge_length)
    )

    remove_mask = ~keep_mask

    removed_count = int(
        np.count_nonzero(
            remove_mask
        )
    )

    if removed_count > 0:
        mesh.remove_triangles_by_mask(
            remove_mask
        )

        mesh.remove_unreferenced_vertices()

    return mesh, removed_count


def _remove_tiny_components(
    mesh,
    minimum_triangles=25
):
    """
    Removes only very small disconnected mesh fragments.

    Large disconnected objects are kept, because a real scene may contain
    multiple buildings, vegetation clusters, roads, or other structures.
    """

    if len(mesh.triangles) == 0:
        return mesh, 0

    (
        triangle_clusters,
        cluster_triangle_counts,
        _
    ) = mesh.cluster_connected_triangles()

    triangle_clusters = np.asarray(
        triangle_clusters
    )

    cluster_triangle_counts = np.asarray(
        cluster_triangle_counts
    )

    if len(cluster_triangle_counts) == 0:
        return mesh, 0

    dynamic_threshold = max(
        minimum_triangles,
        int(
            0.0005
            * len(mesh.triangles)
        )
    )

    tiny_clusters = np.where(
        cluster_triangle_counts
        < dynamic_threshold
    )[0]

    if len(tiny_clusters) == 0:
        return mesh, 0

    remove_mask = np.isin(
        triangle_clusters,
        tiny_clusters
    )

    removed_count = int(
        np.count_nonzero(
            remove_mask
        )
    )

    mesh.remove_triangles_by_mask(
        remove_mask
    )

    mesh.remove_unreferenced_vertices()

    return mesh, removed_count


def create_mesh_from_point_cloud(
    point_cloud_path,
    mesh_output_path
):
    point_cloud_path = Path(
        point_cloud_path
    )

    mesh_output_path = Path(
        mesh_output_path
    )

    mesh_output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "Loading cleaned point cloud..."
    )

    cloud = o3d.io.read_point_cloud(
        str(point_cloud_path)
    )

    points = np.asarray(
        cloud.points
    )

    if len(points) == 0:
        raise RuntimeError(
            "Point cloud is empty."
        )

    print(
        "Points loaded:",
        len(points)
    )

    # ---------------------------------------------------------
    # 1. Remove invalid values only.
    # The fusion stage already performs geometric outlier
    # filtering, so we avoid aggressive duplicate filtering here.
    # ---------------------------------------------------------

    valid_mask = np.all(
        np.isfinite(points),
        axis=1
    )

    points = points[
        valid_mask
    ]

    colors = np.asarray(
        cloud.colors
    )

    cloud_clean = (
        o3d.geometry.PointCloud()
    )

    cloud_clean.points = (
        o3d.utility.Vector3dVector(
            points
        )
    )

    if len(colors) == len(valid_mask):
        colors = colors[
            valid_mask
        ]

        cloud_clean.colors = (
            o3d.utility.Vector3dVector(
                colors
            )
        )

    cloud = cloud_clean

    if len(cloud.points) < 100:
        raise RuntimeError(
            "Too few valid points for meshing."
        )

    # ---------------------------------------------------------
    # 2. Very light statistical cleanup.
    # Fusion already removed most outliers; this only catches
    # anything that survived file round-tripping.
    # ---------------------------------------------------------

    print(
        "Applying light mesh-stage cleanup..."
    )

    if len(cloud.points) > 20:
        filtered_cloud, _ = (
            cloud.remove_statistical_outlier(
                nb_neighbors=20,
                std_ratio=2.5
            )
        )

        # Safety guard: never accept a filter that destroys
        # a large part of the already-clean fused cloud.
        if (
            len(filtered_cloud.points)
            >= 0.85
            * len(cloud.points)
        ):
            cloud = filtered_cloud

    print(
        "Points after light filtering:",
        len(cloud.points)
    )

    # ---------------------------------------------------------
    # 3. Conservative adaptive voxel downsampling.
    # ---------------------------------------------------------

    bbox = (
        cloud.get_axis_aligned_bounding_box()
    )

    extent = np.asarray(
        bbox.get_extent()
    )

    scene_size = float(
        np.max(extent)
    )

    if (
        not np.isfinite(scene_size)
        or scene_size <= 0
    ):
        raise RuntimeError(
            "Invalid point-cloud bounds."
        )

    voxel_size = max(
        scene_size / 450.0,
        1e-5
    )

    downsampled = (
        cloud.voxel_down_sample(
            voxel_size
        )
    )

    # Guard against over-downsampling.
    if (
        len(downsampled.points)
        >= 0.60
        * len(cloud.points)
    ):
        cloud = downsampled

    print(
        "Voxel size:",
        voxel_size
    )

    print(
        "Points after downsampling:",
        len(cloud.points)
    )

    # ---------------------------------------------------------
    # 4. Estimate stable normals.
    # ---------------------------------------------------------

    print(
        "Estimating normals..."
    )

    distances = np.asarray(
        cloud.compute_nearest_neighbor_distance()
    )

    distances = distances[
        np.isfinite(distances)
        & (distances > 0)
    ]

    if len(distances) == 0:
        raise RuntimeError(
            "Could not estimate point spacing."
        )

    # Median is more robust than mean when a few isolated
    # samples remain.
    point_spacing = float(
        np.median(distances)
    )

    print(
        "Median point spacing:",
        point_spacing
    )

    normal_radius = max(
        point_spacing * 8.0,
        voxel_size * 3.0
    )

    cloud.estimate_normals(
        search_param=(
            o3d.geometry
            .KDTreeSearchParamHybrid(
                radius=normal_radius,
                max_nn=60
            )
        )
    )

    cloud.normalize_normals()

    print(
        "Orienting normals..."
    )

    try:
        cloud.orient_normals_consistent_tangent_plane(
            20
        )

    except Exception as error:
        print(
            "Normal orientation warning:",
            error
        )

    # ---------------------------------------------------------
    # 5. Ball Pivoting.
    # Poisson is intentionally avoided because it previously
    # caused native crashes on this Apple Silicon environment.
    # ---------------------------------------------------------

    print(
        "Running Ball Pivoting..."
    )

    radii = [
        point_spacing * 1.5,
        point_spacing * 2.5,
        point_spacing * 4.0,
        point_spacing * 6.0
    ]

    print(
        "Ball radii:",
        radii
    )

    mesh = (
        o3d.geometry.TriangleMesh
        .create_from_point_cloud_ball_pivoting(
            cloud,
            o3d.utility.DoubleVector(
                radii
            )
        )
    )

    if len(mesh.triangles) == 0:
        raise RuntimeError(
            "Ball Pivoting generated no triangles."
        )

    print(
        "Raw vertices:",
        len(mesh.vertices)
    )

    print(
        "Raw triangles:",
        len(mesh.triangles)
    )

    # ---------------------------------------------------------
    # 6. Conservative mesh cleanup.
    # ---------------------------------------------------------

    print(
        "Cleaning mesh..."
    )

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()

    # Remove stretched bridges across holes/noisy regions.
    max_edge_length = (
        point_spacing * 7.0
    )

    mesh, long_removed = (
        _remove_long_triangles(
            mesh,
            max_edge_length
        )
    )

    print(
        "Long-edge triangles removed:",
        long_removed
    )

    # Remove only tiny disconnected fragments.
    mesh, fragment_removed = (
        _remove_tiny_components(
            mesh,
            minimum_triangles=25
        )
    )

    print(
        "Tiny-fragment triangles removed:",
        fragment_removed
    )

    # Keep this after fragment cleanup.
    mesh.remove_unreferenced_vertices()

    # Non-manifold cleanup can remove a small number of bad
    # edges produced by Ball Pivoting.
    mesh.remove_non_manifold_edges()

    mesh.compute_vertex_normals()

    if len(mesh.triangles) == 0:
        raise RuntimeError(
            "Mesh cleanup removed all triangles."
        )

    print(
        "Final vertices:",
        len(mesh.vertices)
    )

    print(
        "Final triangles:",
        len(mesh.triangles)
    )

    # ---------------------------------------------------------
    # 7. Save.
    # ---------------------------------------------------------

    success = (
        o3d.io.write_triangle_mesh(
            str(mesh_output_path),
            mesh,
            write_vertex_normals=True,
            write_vertex_colors=True
        )
    )

    if not success:
        raise RuntimeError(
            "Failed to save mesh."
        )

    return {
        "vertices": len(
            mesh.vertices
        ),
        "triangles": len(
            mesh.triangles
        ),
        "point_spacing": (
            point_spacing
        ),
        "voxel_size": voxel_size,
        "long_triangles_removed": (
            long_removed
        ),
        "fragment_triangles_removed": (
            fragment_removed
        ),
        "output": str(
            mesh_output_path
        )
    }
