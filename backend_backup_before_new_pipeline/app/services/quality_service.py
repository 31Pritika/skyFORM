from pathlib import Path


def get_quality_metrics(base_dir):
    base_dir = Path(base_dir)

    points_file = (
        base_dir
        / "outputs"
        / "colmap_new"
        / "dense"
        / "sparse_txt"
        / "points3D.txt"
    )

    if not points_file.exists():
        return {
            "available": False,
            "points": [],
            "summary": {},
        }

    points = []

    with open(points_file, "r") as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) < 8:
                continue

            point_id = int(parts[0])

            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])

            r = int(parts[4])
            g = int(parts[5])
            b = int(parts[6])

            reprojection_error = float(parts[7])

            track_data = parts[8:]
            track_length = len(track_data) // 2

            # Heuristic quality score:
            # lower reprojection error = better
            # longer feature track = better
            error_score = max(
                0.0,
                1.0 - reprojection_error / 3.0
            )

            track_score = min(
                1.0,
                track_length / 8.0
            )

            confidence = (
                0.65 * error_score
                + 0.35 * track_score
            )

            confidence = max(
                0.0,
                min(1.0, confidence)
            )

            points.append({
                "id": point_id,

                "position": [
                    x,
                    y,
                    z,
                ],

                "rgb": [
                    r,
                    g,
                    b,
                ],

                "reprojection_error":
                    reprojection_error,

                "track_length":
                    track_length,

                "confidence":
                    round(confidence, 4),
            })

    if not points:
        return {
            "available": False,
            "points": [],
            "summary": {},
        }

    point_count = len(points)

    average_confidence = (
        sum(
            point["confidence"]
            for point in points
        )
        / point_count
    )

    average_error = (
        sum(
            point["reprojection_error"]
            for point in points
        )
        / point_count
    )

    average_track_length = (
        sum(
            point["track_length"]
            for point in points
        )
        / point_count
    )

    high_confidence = sum(
        1
        for point in points
        if point["confidence"] >= 0.75
    )

    medium_confidence = sum(
        1
        for point in points
        if 0.45 <= point["confidence"] < 0.75
    )

    low_confidence = sum(
        1
        for point in points
        if point["confidence"] < 0.45
    )

    return {
        "available": True,

        "summary": {
            "point_count":
                point_count,

            "average_confidence":
                round(
                    average_confidence,
                    4
                ),

            "average_reprojection_error":
                round(
                    average_error,
                    4
                ),

            "average_track_length":
                round(
                    average_track_length,
                    4
                ),

            "high_confidence_points":
                high_confidence,

            "medium_confidence_points":
                medium_confidence,

            "low_confidence_points":
                low_confidence,
        },

        "points": points,
    }