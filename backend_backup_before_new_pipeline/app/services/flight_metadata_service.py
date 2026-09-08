from pathlib import Path
import json


ALLOWED_NUMERIC_FIELDS = {
    "flight_altitude_m",
    "speed_mps",
    "heading_deg",
    "yaw_deg",
    "pitch_deg",
    "roll_deg",
    "focal_length_mm",
    "sensor_width_mm",
    "sensor_height_mm",
}


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


def get_flight_metadata_path(
    base_dir,
    video_id,
):
    base_dir = Path(base_dir)
    video_id = _validate_video_id(
        video_id
    )

    output_dir = (
        base_dir
        / "outputs"
        / "telemetry"
        / video_id
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        output_dir
        / "flight_metadata.json"
    )


def validate_flight_metadata(data):
    if not isinstance(
        data,
        dict,
    ):
        raise ValueError(
            "Flight metadata must "
            "be a JSON object."
        )

    if not data:
        raise ValueError(
            "Flight metadata is empty."
        )

    cleaned = {}

    for key, value in data.items():
        if value is None:
            continue

        if (
            isinstance(value, str)
            and not value.strip()
        ):
            continue

        if (
            key
            in ALLOWED_NUMERIC_FIELDS
        ):
            try:
                cleaned[key] = float(
                    value
                )
            except Exception:
                raise ValueError(
                    f"{key} must be numeric."
                )

        else:
            cleaned[key] = value

    if not cleaned:
        raise ValueError(
            "Flight metadata is empty."
        )

    return cleaned


def save_flight_metadata(
    base_dir,
    video_id,
    metadata,
):
    path = (
        get_flight_metadata_path(
            base_dir,
            video_id,
        )
    )

    cleaned = (
        validate_flight_metadata(
            metadata
        )
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            cleaned,
            file,
            indent=2,
        )

    return {
        "available": True,
        "video_id":
            str(video_id),
        "metadata":
            cleaned,
    }


def load_flight_metadata(
    base_dir,
    video_id,
):
    path = (
        get_flight_metadata_path(
            base_dir,
            video_id,
        )
    )

    if not path.exists():
        return {
            "available": False,
            "video_id":
                str(video_id),
            "metadata": {},
        }

    try:
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        return {
            "available": True,
            "video_id":
                str(video_id),
            "metadata":
                data,
        }

    except Exception as error:
        return {
            "available": False,
            "video_id":
                str(video_id),
            "metadata": {},
            "error":
                str(error),
        }
