from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from pipeline_config import (
    get_fusion_dir,
    get_cleanup_dir,
    ensure_job_directories,
)


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = (
    get_fusion_dir()
    / "dense_colmap_fixed_poses.ply"
)

OUTPUT_DIR = get_cleanup_dir()

OUTPUT_PATH = (
    OUTPUT_DIR
    / "dense_clean.ply"
)


# ============================================================
# CONFIG
# ============================================================

VOXEL_RATIO = 0.0015

# Statistical filtering
STAT_NEIGHBORS = 20
STAT_STD_RATIO = 2.0

# Radius filtering
RADIUS_MULTIPLIER = 4.0
MIN_RADIUS_NEIGHBORS = 5


# ============================================================
# VOXEL DOWNSAMPLING
# ============================================================

def voxel_downsample(points, colors, voxel_size):
    voxel_indices = np.floor(
        points / voxel_size
    ).astype(np.int64)

    _, unique_indices = np.unique(
        voxel_indices,
        axis=0,
        return_index=True
    )

    unique_indices = np.sort(unique_indices)

    return (
        points[unique_indices],
        colors[unique_indices]
    )


# ============================================================
# STATISTICAL OUTLIER REMOVAL
# ============================================================

def statistical_filter(points, colors):
    if len(points) <= STAT_NEIGHBORS:
        return points, colors

    tree = cKDTree(points)

    distances, _ = tree.query(
        points,
        k=STAT_NEIGHBORS + 1,
        workers=-1
    )

    # Ignore distance to itself.
    mean_distances = distances[:, 1:].mean(
        axis=1
    )

    global_mean = mean_distances.mean()
    global_std = mean_distances.std()

    threshold = (
        global_mean
        + STAT_STD_RATIO * global_std
    )

    keep = mean_distances <= threshold

    return (
        points[keep],
        colors[keep]
    )


# ============================================================
# RADIUS FILTER
# ============================================================

def radius_filter(
    points,
    colors,
    radius,
):
    if len(points) == 0:
        return points, colors

    tree = cKDTree(points)

    # query_ball_point with return_length avoids
    # constructing a giant Python list of neighbors.
    counts = tree.query_ball_point(
        points,
        r=radius,
        return_length=True,
        workers=-1
    )

    # Includes the point itself.
    keep = counts >= MIN_RADIUS_NEIGHBORS

    return (
        points[keep],
        colors[keep]
    )


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_job_directories()

    print("\n========================================")
    print(" SKYFORM - POINT CLOUD CLEANUP")
    print("========================================")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Point cloud not found:\n{INPUT_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print("\nLoading point cloud...")

    cloud = trimesh.load(
        str(INPUT_PATH),
        process=False
    )

    points = np.asarray(
        cloud.vertices,
        dtype=np.float64
    )

    if hasattr(cloud.visual, "vertex_colors"):
        colors = np.asarray(
            cloud.visual.vertex_colors
        )

        # RGBA -> RGB
        if colors.ndim == 2 and colors.shape[1] >= 3:
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

    original_count = len(points)

    print(
        f"Original points: {original_count:,}"
    )

    if original_count == 0:
        raise RuntimeError(
            "Input cloud contains zero points."
        )

    # --------------------------------------------------------
    # INVALID POINTS
    # --------------------------------------------------------

    finite = np.isfinite(
        points
    ).all(axis=1)

    points = points[finite]
    colors = colors[finite]

    print(
        f"After non-finite removal: {len(points):,}"
    )

    if len(points) == 0:
        raise RuntimeError(
            "No finite points remain after cleanup."
        )

    # --------------------------------------------------------
    # SCENE SIZE
    # --------------------------------------------------------

    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)

    scene_diagonal = float(
        np.linalg.norm(
            max_bound - min_bound
        )
    )

    if (
        not np.isfinite(scene_diagonal)
        or scene_diagonal <= 0
    ):
        raise RuntimeError(
            "Invalid scene dimensions."
        )

    print(
        f"Scene diagonal: {scene_diagonal:.6f}"
    )

    # --------------------------------------------------------
    # VOXEL DOWNSAMPLING
    # --------------------------------------------------------

    voxel_size = (
        scene_diagonal
        * VOXEL_RATIO
    )

    print(
        f"\nVoxel size: {voxel_size:.6f}"
    )

    points, colors = voxel_downsample(
        points,
        colors,
        voxel_size
    )

    print(
        f"After voxel downsampling: {len(points):,}"
    )

    # --------------------------------------------------------
    # STATISTICAL FILTER
    # --------------------------------------------------------

    print(
        "\nRunning statistical outlier removal..."
    )

    points, colors = statistical_filter(
        points,
        colors
    )

    print(
        f"After statistical filtering: {len(points):,}"
    )

    # --------------------------------------------------------
    # RADIUS FILTER
    # --------------------------------------------------------

    radius = (
        voxel_size
        * RADIUS_MULTIPLIER
    )

    print(
        f"\nRadius filter radius: {radius:.6f}"
    )

    print(
        "Running radius outlier removal..."
    )

    points, colors = radius_filter(
        points,
        colors,
        radius
    )

    final_count = len(points)

    print(
        f"After radius filtering: {final_count:,}"
    )

    if final_count == 0:
        raise RuntimeError(
            "Filtering removed every point."
        )

    retained = (
        final_count
        / original_count
        * 100
    )

    print(
        f"\nRetained: {retained:.2f}% "
        "of original points"
    )

    # --------------------------------------------------------
    # EXPORT
    # --------------------------------------------------------

    print(
        "\nSaving cleaned point cloud..."
    )

    cleaned_cloud = trimesh.points.PointCloud(
        vertices=points,
        colors=colors
    )

    cleaned_cloud.export(
        str(OUTPUT_PATH)
    )

    print("\n========================================")
    print(" CLEANUP COMPLETE")
    print("========================================")

    print(
        f"Before: {original_count:,}"
    )

    print(
        f"After:  {final_count:,}"
    )

    print(
        f"Kept:   {retained:.2f}%"
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )

    print("========================================\n")


if __name__ == "__main__":
    main()
