import cv2
from pathlib import Path


def load_keyframe_images(keyframe_dir):
    """
    Loads all saved keyframe images in order.
    """
    directory = Path(keyframe_dir)

    image_paths = sorted(
        directory.glob("keyframe_*.jpg")
    )

    images = []

    for path in image_paths:
        image = cv2.imread(str(path))

        if image is None:
            continue

        images.append({
            "filename": path.name,
            "path": str(path),
            "image": image
        })

    return images


def detect_features(image, max_features=4000):
    """
    Detects ORB keypoints and descriptors.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    orb = cv2.ORB_create(
        nfeatures=max_features
    )

    keypoints, descriptors = (
        orb.detectAndCompute(
            gray,
            None
        )
    )

    return keypoints, descriptors


def match_features(
    descriptors_a,
    descriptors_b,
    ratio_threshold=0.75
):
    """
    Matches ORB descriptors using KNN
    and Lowe's ratio test.
    """

    if (
        descriptors_a is None
        or descriptors_b is None
    ):
        return []

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=False
    )

    raw_matches = matcher.knnMatch(
        descriptors_a,
        descriptors_b,
        k=2
    )

    good_matches = []

    for pair in raw_matches:
        if len(pair) < 2:
            continue

        best, second_best = pair

        if (
            best.distance
            < ratio_threshold
            * second_best.distance
        ):
            good_matches.append(best)

    return good_matches


def analyze_keyframe_pair(
    image_a,
    image_b
):
    """
    Detects and matches features between
    two keyframes.
    """

    keypoints_a, descriptors_a = (
        detect_features(image_a)
    )

    keypoints_b, descriptors_b = (
        detect_features(image_b)
    )

    matches = match_features(
        descriptors_a,
        descriptors_b
    )

    return {
        "keypoints_a": keypoints_a,
        "keypoints_b": keypoints_b,
        "descriptors_a": descriptors_a,
        "descriptors_b": descriptors_b,
        "matches": matches,
        "feature_count_a": len(
            keypoints_a
        ),
        "feature_count_b": len(
            keypoints_b
        ),
        "match_count": len(matches)
    }


def analyze_keyframe_sequence(
    keyframe_dir
):
    """
    Performs feature matching across all
    consecutive keyframe pairs.
    """

    images = load_keyframe_images(
        keyframe_dir
    )

    results = []

    for index in range(
        len(images) - 1
    ):
        current = images[index]
        next_image = images[index + 1]

        analysis = analyze_keyframe_pair(
            current["image"],
            next_image["image"]
        )

        results.append({
            "image_a":
                current["filename"],

            "image_b":
                next_image["filename"],

            "features_a":
                analysis["feature_count_a"],

            "features_b":
                analysis["feature_count_b"],

            "matches":
                analysis["match_count"]
        })

    return results


def verify_matches_ransac(
    keypoints_a,
    keypoints_b,
    matches,
    ransac_threshold=1.0
):
    """
    Uses the Fundamental Matrix + RANSAC
    to reject geometrically inconsistent matches.
    """

    if len(matches) < 8:
        return {
            "fundamental_matrix": None,
            "inlier_matches": [],
            "inlier_count": 0,
            "inlier_ratio": 0.0
        }

    import numpy as np

    points_a = np.float32([
        keypoints_a[m.queryIdx].pt
        for m in matches
    ])

    points_b = np.float32([
        keypoints_b[m.trainIdx].pt
        for m in matches
    ])

    fundamental_matrix, mask = (
        cv2.findFundamentalMat(
            points_a,
            points_b,
            cv2.FM_RANSAC,
            ransac_threshold,
            0.99
        )
    )

    if (
        fundamental_matrix is None
        or mask is None
    ):
        return {
            "fundamental_matrix": None,
            "inlier_matches": [],
            "inlier_count": 0,
            "inlier_ratio": 0.0
        }

    mask = mask.ravel().astype(bool)

    inlier_matches = [
        match
        for match, keep
        in zip(matches, mask)
        if keep
    ]

    inlier_count = len(
        inlier_matches
    )

    inlier_ratio = (
        inlier_count / len(matches)
        if matches
        else 0
    )

    return {
        "fundamental_matrix":
            fundamental_matrix,

        "inlier_matches":
            inlier_matches,

        "inlier_count":
            inlier_count,

        "inlier_ratio":
            round(inlier_ratio, 3)
    }