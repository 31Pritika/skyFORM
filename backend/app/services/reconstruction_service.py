import cv2
import numpy as np
from pathlib import Path

from app.services.feature_service import (
    load_keyframe_images,
    analyze_keyframe_pair,
    verify_matches_ransac,
)


def estimate_camera_matrix(image):
    """
    Approximate camera intrinsics when calibration data is unavailable.
    """

    height, width = image.shape[:2]

    focal_length = 0.9 * max(width, height)

    camera_matrix = np.array([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    return camera_matrix


def estimate_relative_pose(
    keypoints_a,
    keypoints_b,
    matches,
    camera_matrix
):
    """
    Estimates relative rotation and translation between two cameras.
    """

    if len(matches) < 8:
        return None

    points_a = np.float32([
        keypoints_a[m.queryIdx].pt
        for m in matches
    ])

    points_b = np.float32([
        keypoints_b[m.trainIdx].pt
        for m in matches
    ])

    essential_matrix, mask = cv2.findEssentialMat(
        points_a,
        points_b,
        camera_matrix,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )

    if essential_matrix is None:
        return None

    _, rotation, translation, pose_mask = cv2.recoverPose(
        essential_matrix,
        points_a,
        points_b,
        camera_matrix
    )

    valid = pose_mask.ravel() > 0

    return {
        "rotation": rotation,
        "translation": translation,
        "points_a": points_a[valid],
        "points_b": points_b[valid]
    }


def triangulate_points(
    points_a,
    points_b,
    camera_matrix,
    rotation,
    translation
):
    """
    Triangulates matched 2D observations into 3D points.
    """

    projection_a = camera_matrix @ np.hstack((
        np.eye(3),
        np.zeros((3, 1))
    ))

    projection_b = camera_matrix @ np.hstack((
        rotation,
        translation
    ))

    homogeneous_points = cv2.triangulatePoints(
        projection_a,
        projection_b,
        points_a.T,
        points_b.T
    )

    valid_w = (
        np.abs(homogeneous_points[3]) > 1e-8
    )

    homogeneous_points = (
        homogeneous_points[:, valid_w]
    )

    points_3d = (
        homogeneous_points[:3]
        / homogeneous_points[3]
    ).T

    finite = np.all(
        np.isfinite(points_3d),
        axis=1
    )

    points_3d = points_3d[finite]

    # Remove extreme numerical outliers.
    if len(points_3d) > 0:
        distances = np.linalg.norm(
            points_3d,
            axis=1
        )

        cutoff = np.percentile(
            distances,
            99
        )

        points_3d = points_3d[
            distances <= cutoff
        ]

    return points_3d


def save_ply(points, output_path):
    """
    Saves XYZ points as an ASCII PLY point cloud.
    """

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
        file.write("end_header\n")

        for x, y, z in points:
            file.write(
                f"{x} {y} {z}\n"
            )


def reconstruct_first_pair(
    keyframe_dir,
    output_path
):
    """
    Creates an initial sparse reconstruction
    from the first usable keyframe pair.
    """

    images = load_keyframe_images(
        keyframe_dir
    )

    for index in range(len(images) - 1):

        image_a = images[index]
        image_b = images[index + 1]

        analysis = analyze_keyframe_pair(
            image_a["image"],
            image_b["image"]
        )

        verification = verify_matches_ransac(
            analysis["keypoints_a"],
            analysis["keypoints_b"],
            analysis["matches"]
        )

        inliers = verification[
            "inlier_matches"
        ]

        if len(inliers) < 8:
            continue

        camera_matrix = (
            estimate_camera_matrix(
                image_a["image"]
            )
        )

        pose = estimate_relative_pose(
            analysis["keypoints_a"],
            analysis["keypoints_b"],
            inliers,
            camera_matrix
        )

        if pose is None:
            continue

        points_3d = triangulate_points(
            pose["points_a"],
            pose["points_b"],
            camera_matrix,
            pose["rotation"],
            pose["translation"]
        )

        if len(points_3d) == 0:
            continue

        save_ply(
            points_3d,
            output_path
        )

        return {
            "success": True,
            "image_a": image_a["filename"],
            "image_b": image_b["filename"],
            "ransac_inliers": len(inliers),
            "pose_inliers": len(
                pose["points_a"]
            ),
            "points_3d": len(points_3d),
            "rotation": pose[
                "rotation"
            ].tolist(),
            "translation": pose[
                "translation"
            ].reshape(-1).tolist(),
            "output": str(output_path),
            "coordinate_system": "relative_unscaled"
        }

    return {
        "success": False,
        "message": "No suitable image pair found."
    }