from pathlib import Path
import csv


def _validate_video_id(video_id):
    value = str(video_id).strip()

    if not value:
        raise ValueError(
            "video_id is required."
        )

    if (
        "/" in value
        or "\\" in value
        or value in {".", ".."}
    ):
        raise ValueError(
            "Invalid video_id."
        )

    return value


def get_video_telemetry_dir(
    base_dir,
    video_id,
):
    base_dir = Path(base_dir)
    video_id = _validate_video_id(
        video_id
    )

    directory = (
        base_dir
        / "outputs"
        / "telemetry"
        / video_id
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return directory


def get_gps_path(
    base_dir,
    video_id,
):
    return (
        get_video_telemetry_dir(
            base_dir,
            video_id,
        )
        / "gps.csv"
    )


def parse_gps_csv(csv_path):
    csv_path = Path(csv_path)

    points = []

    with open(
        csv_path,
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError(
                "GPS CSV has no header."
            )

        normalized = {
            name.strip().lower(): name
            for name in reader.fieldnames
        }

        required = [
            "latitude",
            "longitude",
        ]

        for field in required:
            if field not in normalized:
                raise ValueError(
                    "CSV must contain latitude "
                    "and longitude columns."
                )

        for index, row in enumerate(
            reader
        ):
            try:
                latitude = float(
                    row[
                        normalized[
                            "latitude"
                        ]
                    ]
                )

                longitude = float(
                    row[
                        normalized[
                            "longitude"
                        ]
                    ]
                )

                altitude = None

                if (
                    "altitude"
                    in normalized
                ):
                    raw_altitude = row[
                        normalized[
                            "altitude"
                        ]
                    ]

                    if raw_altitude:
                        altitude = float(
                            raw_altitude
                        )

                timestamp = None

                if (
                    "timestamp"
                    in normalized
                ):
                    raw_timestamp = row[
                        normalized[
                            "timestamp"
                        ]
                    ]

                    if (
                        raw_timestamp
                        is not None
                    ):
                        raw_timestamp = (
                            raw_timestamp
                            .strip()
                        )

                    if raw_timestamp:
                        timestamp = (
                            raw_timestamp
                        )

                points.append({
                    "index": index,
                    "latitude":
                        latitude,
                    "longitude":
                        longitude,
                    "altitude":
                        altitude,
                    "timestamp":
                        timestamp,
                })

            except (
                ValueError,
                TypeError,
                KeyError,
            ):
                continue

    if not points:
        raise ValueError(
            "No valid GPS rows found."
        )

    latitudes = [
        point["latitude"]
        for point in points
    ]

    longitudes = [
        point["longitude"]
        for point in points
    ]

    altitudes = [
        point["altitude"]
        for point in points
        if (
            point["altitude"]
            is not None
        )
    ]

    timestamped_count = sum(
        1
        for point in points
        if point["timestamp"]
        not in (
            None,
            "",
        )
    )

    return {
        "point_count":
            len(points),

        "timestamped_point_count":
            timestamped_count,

        "bounds": {
            "min_latitude":
                min(latitudes),

            "max_latitude":
                max(latitudes),

            "min_longitude":
                min(longitudes),

            "max_longitude":
                max(longitudes),
        },

        "altitude_range": (
            {
                "minimum":
                    min(altitudes),

                "maximum":
                    max(altitudes),
            }
            if altitudes
            else None
        ),

        "points":
            points,
    }
