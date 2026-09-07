from app.services.reconstruction_service import (
    reconstruct_first_pair
)


result = reconstruct_first_pair(
    keyframe_dir="outputs/keyframes/test",
    output_path="outputs/reconstruction/sparse_initial.ply"
)


print(
    "\n--- SkyFORM Initial 3D Reconstruction ---"
)

print(
    "Success:",
    result["success"]
)


if result["success"]:

    print(
        "Images:",
        result["image_a"],
        "->",
        result["image_b"]
    )

    print(
        "RANSAC inliers:",
        result["ransac_inliers"]
    )

    print(
        "Pose inliers:",
        result["pose_inliers"]
    )

    print(
        "3D points:",
        result["points_3d"]
    )

    print(
        "Translation:",
        result["translation"]
    )

    print(
        "Coordinate system:",
        result["coordinate_system"]
    )

    print(
        "Saved:",
        result["output"]
    )

else:

    print(
        result["message"]
    )