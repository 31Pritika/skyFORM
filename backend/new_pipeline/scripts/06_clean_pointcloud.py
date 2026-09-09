import json
import os
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from pipeline_config import (
    get_fusion_dir,
    get_cleanup_dir,
    get_colmap_poses_dir,
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


# ------------------------------------------------------------
# Path-distance filter (drops sky / distant background points
# that sit far from the camera trajectory).
# ------------------------------------------------------------

def _env_flag(name, default):
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name, default):
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"  WARNING: {name}={raw!r} is not a number; using {default}.")
        return default


# Toggle the whole step. Env override: SKYFORM_ENABLE_PATH_DIST_FILTER=0
ENABLE_PATH_DISTANCE_FILTER = _env_flag(
    "SKYFORM_ENABLE_PATH_DIST_FILTER", True
)

# Cutoff = PATH_DIST_MULTIPLIER * median spacing between neighbouring camera
# centres. Env override: SKYFORM_PATH_DIST_MULTIPLIER=<float> for quick tests.
PATH_DIST_MULTIPLIER = _env_float(
    "SKYFORM_PATH_DIST_MULTIPLIER", 8.0
)

# Safety valve: if the filter would keep less than this fraction of points,
# assume the poses / multiplier are untrustworthy and leave the cloud alone.
PATH_DIST_MIN_RETAIN = 0.30


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
# PATH-DISTANCE FILTER
# ============================================================

def load_camera_centers():
    """Return an (N, 3) array of COLMAP camera centres, or None.

    Reads colmap_poses.json (written by stage 04) - the same source stage 07
    uses. Centres are in the same COLMAP world frame as the fused cloud.
    """
    poses_path = (
        get_colmap_poses_dir()
        / "colmap_poses.json"
    )

    if not poses_path.exists():
        print(f"  SKIPPED: {poses_path.name} not found.")
        return None

    try:
        poses = json.loads(poses_path.read_text())
    except (ValueError, OSError) as error:
        print(f"  SKIPPED: could not read poses ({error}).")
        return None

    if isinstance(poses, list):
        entries = poses
    else:
        entries = poses.get(
            "poses",
            poses.get("images", [])
        )

    centers = []

    for entry in entries:
        if not isinstance(entry, dict):
            continue

        center = entry.get(
            "camera_center",
            entry.get("position")
        )

        if center is not None:
            centers.append(center)

    centers = np.asarray(centers, dtype=np.float64)

    if centers.ndim != 2 or centers.shape[1] != 3:
        print(
            f"  SKIPPED: unexpected camera-centre shape {centers.shape}."
        )
        return None

    centers = centers[
        np.isfinite(centers).all(axis=1)
    ]

    if len(centers) < 2:
        print("  SKIPPED: need at least 2 finite camera centres.")
        return None

    return centers


def path_distance_filter(points, colors):
    """Drop points that sit far from the camera path.

    Distance is measured to the nearest camera centre. The cutoff scales with
    the median spacing between neighbouring cameras, so it adapts to COLMAP's
    arbitrary (unscaled) units.

    Returns (points, colors, info). On any problem the cloud is returned
    unchanged with info["applied"] == False.
    """
    centers = load_camera_centers()

    if centers is None:
        return points, colors, {
            "applied": False,
            "reason": "no_camera_centers",
        }

    camera_tree = cKDTree(centers)

    # Column 0 is the camera itself; column 1 is the nearest other camera.
    neighbor_dist, _ = camera_tree.query(
        centers,
        k=2,
        workers=-1
    )

    camera_spacing = float(
        np.median(neighbor_dist[:, 1])
    )

    if not np.isfinite(camera_spacing) or camera_spacing <= 0:
        print("  SKIPPED: degenerate camera spacing.")
        return points, colors, {
            "applied": False,
            "reason": "degenerate_spacing",
        }

    cutoff = PATH_DIST_MULTIPLIER * camera_spacing

    point_dist, _ = camera_tree.query(
        points,
        k=1,
        workers=-1
    )

    keep = point_dist <= cutoff

    retain = float(keep.mean()) if len(keep) else 0.0

    print(f"  Cameras: {len(centers)}")
    print(
        f"  Camera spacing (median NN): {camera_spacing:.6f}"
    )
    print(
        f"  Cutoff ({PATH_DIST_MULTIPLIER:g}x): {cutoff:.6f}"
    )
    print(
        f"  Within cutoff: {int(keep.sum()):,} / {len(keep):,} "
        f"({retain * 100:.2f}%)"
    )

    info = {
        "applied": True,
        "cameras": int(len(centers)),
        "camera_spacing": camera_spacing,
        "multiplier": PATH_DIST_MULTIPLIER,
        "cutoff": cutoff,
        "input_points": int(len(keep)),
        "kept_points": int(keep.sum()),
        "removed_points": int((~keep).sum()),
        "retain_fraction": retain,
    }

    if retain < PATH_DIST_MIN_RETAIN:
        print(
            f"  SKIPPED: would keep only {retain * 100:.2f}% "
            f"(< {PATH_DIST_MIN_RETAIN * 100:.0f}% floor); "
            "leaving cloud unchanged."
        )
        info["applied"] = False
        info["reason"] = "below_retain_floor"
        return points, colors, info

    return points[keep], colors[keep], info


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
    # PATH-DISTANCE FILTER (sky / distant background)
    # --------------------------------------------------------
    #
    # Runs before scene-size estimation so distant points do not
    # inflate scene_diagonal (and thus the voxel / radius sizes).

    if ENABLE_PATH_DISTANCE_FILTER:
        print("\nRunning path-distance filter...")

        points, colors, path_filter_info = path_distance_filter(
            points,
            colors
        )

        print(
            f"After path-distance filter: {len(points):,}"
        )

        if len(points) == 0:
            raise RuntimeError(
                "Path-distance filter removed every point."
            )
    else:
        print(
            "\nPath-distance filter disabled "
            "(SKYFORM_ENABLE_PATH_DIST_FILTER=0)."
        )
        path_filter_info = {
            "applied": False,
            "reason": "disabled",
        }

    (OUTPUT_DIR / "path_filter.json").write_text(
        json.dumps(path_filter_info, indent=2)
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
