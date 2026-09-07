import cv2

from app.services.feature_service import (
    load_keyframe_images,
    analyze_keyframe_pair,
    verify_matches_ransac
)


images = load_keyframe_images(
    "outputs/keyframes/test"
)


print(
    "\n--- SkyFORM RANSAC Verification ---"
)


total_raw = 0
total_inliers = 0
processed_pairs = 0


for i in range(len(images) - 1):

    a = images[i]
    b = images[i + 1]

    analysis = analyze_keyframe_pair(
        a["image"],
        b["image"]
    )

    verification = (
        verify_matches_ransac(
            analysis["keypoints_a"],
            analysis["keypoints_b"],
            analysis["matches"]
        )
    )

    raw = analysis["match_count"]

    inliers = verification[
        "inlier_count"
    ]

    total_raw += raw
    total_inliers += inliers
    processed_pairs += 1

    print(
        f"{a['filename']} -> "
        f"{b['filename']} | "
        f"Raw: {raw} | "
        f"Inliers: {inliers} | "
        f"Ratio: "
        f"{verification['inlier_ratio']}"
    )


average_raw = (
    total_raw / processed_pairs
    if processed_pairs
    else 0
)

average_inliers = (
    total_inliers / processed_pairs
    if processed_pairs
    else 0
)


print("\nPairs:", processed_pairs)

print(
    "Average raw matches:",
    round(average_raw, 2)
)

print(
    "Average RANSAC inliers:",
    round(average_inliers, 2)
)