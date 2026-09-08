from pathlib import Path
import json
import math
import struct


def load_json(path):
    if not path.exists():
        return {}

    try:
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)
    except Exception:
        return {}


def get_active_video_id(base_dir):
    state = load_json(
        Path(base_dir)
        / "outputs"
        / "pipeline_state.json"
    )

    return state.get("video_id")


def load_sparse_confidence(job_dir):
    """Read COLMAP sparse tracks; scores describe support, not accuracy."""
    root = job_dir / "sfm" / "sparse"
    candidates = list(root.glob("*/points3D.bin"))
    def registered(path):
        try:
            return struct.unpack("<Q", (path.parent / "images.bin").read_bytes()[:8])[0]
        except (OSError, struct.error):
            return 0
    path = max(candidates, key=registered) if candidates else root / "points3D.bin"
    if not path.exists():
        return []
    points = []
    try:
        with path.open("rb") as stream:
            count, = struct.unpack("<Q", stream.read(8))
            stride = max(1, math.ceil(count / 30000))
            for index in range(count):
                record = struct.unpack("<QdddBBBd", stream.read(43))
                track_length, = struct.unpack("<Q", stream.read(8))
                stream.seek(track_length * 8, 1)
                position, error = list(record[1:4]), record[7]
                if index % stride or not all(math.isfinite(v) for v in position + [error]) or error < 0:
                    continue
                score = min(track_length / 6.0, 1.0) / (1.0 + error / 2.0)
                points.append({"position": position, "confidence": score,
                               "reprojection_error": error, "track_length": track_length})
    except (OSError, struct.error):
        return []
    return points


def get_quality_metrics(base_dir):
    base_dir = Path(base_dir)

    video_id = get_active_video_id(
        base_dir
    )

    if not video_id:
        return {
            "available": False,
            "points": [],
            "summary": {},
        }

    job_dir = (
        base_dir
        / "outputs"
        / "jobs"
        / video_id
    )

    report = load_json(
        job_dir
        / "final"
        / "pipeline_report.json"
    )

    sfm = report.get(
        "structure_from_motion",
        {},
    )

    dense = report.get(
        "dense_reconstruction",
        {},
    )

    registered = sfm.get(
        "registered_frames"
    )

    frame_count = (
        report.get("input", {})
        .get("frames_extracted")
    )

    registration_rate = (
        sfm.get(
            "registration_rate_percent"
        )
    )

    point_count = dense.get(
        "final_point_count",
        0,
    )

    # This is a reconstruction-support indicator,
    # NOT geometric accuracy.
    if registration_rate is None:
        heuristic_confidence = None
    else:
        heuristic_confidence = round(
            min(
                max(
                    registration_rate / 100.0,
                    0.0,
                ),
                1.0,
            ),
            4,
        )

    points = load_sparse_confidence(job_dir)
    spatial_summary = {}
    if points:
        spatial_summary = {
            "point_count": len(points),
            "average_confidence": sum(p["confidence"] for p in points) / len(points),
            "average_reprojection_error": sum(p["reprojection_error"] for p in points) / len(points),
            "high_confidence_points": sum(p["confidence"] >= .75 for p in points),
            "medium_confidence_points": sum(.45 <= p["confidence"] < .75 for p in points),
            "low_confidence_points": sum(p["confidence"] < .45 for p in points),
            "confidence_note": "Sparse SfM support: min(track length / 6, 1) / (1 + error / 2). Not dense-surface accuracy. At most 30,000 sampled points.",
        }

    return {
        "available": bool(points or report),

        # Kept for frontend/API compatibility.
        "points": points,

        "summary": {
            "point_count":
                point_count,

            "registered_cameras":
                registered,

            "input_frames":
                frame_count,

            "registration_rate_percent":
                registration_rate,

            "average_confidence":
                heuristic_confidence,

            "average_reprojection_error":
                sfm.get(
                    "mean_reprojection_error_px"
                ),

            "confidence_type":
                "heuristic",

            "confidence_note": (
                "Registration support indicator only; "
                "not ground-truth geometric accuracy."
            ),
            **spatial_summary,
        },
    }
