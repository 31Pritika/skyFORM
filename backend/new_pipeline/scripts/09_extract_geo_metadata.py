import json
import subprocess

from pipeline_config import (
    get_input_video,
    get_georeferencing_dir,
    ensure_job_directories,
)


GPS_KEYWORDS = [
    "gps",
    "latitude",
    "longitude",
    "location",
    "altitude",
    "speed",
    "heading",
    "yaw",
    "pitch",
    "roll",
    "imu",
    "telemetry",
]


def search_metadata(obj, path="root"):
    matches = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            key_lower = str(key).lower()

            if any(word in key_lower for word in GPS_KEYWORDS):
                matches.append((f"{path}.{key}", value))

            matches.extend(
                search_metadata(value, f"{path}.{key}")
            )

    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            matches.extend(
                search_metadata(value, f"{path}[{i}]")
            )

    return matches


def main():
    ensure_job_directories()

    video = get_input_video()
    output_dir = get_georeferencing_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_json = output_dir / "video_metadata.json"

    print("\n========================================")
    print(" SKYFORM - GEO METADATA INSPECTION")
    print("========================================")
    print("Video:", video)

    command = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    metadata = json.loads(result.stdout)

    with open(output_json, "w") as f:
        json.dump(metadata, f, indent=2)

    matches = search_metadata(metadata)

    print("\nPotential GPS / telemetry fields:")
    print("----------------------------------------")

    if matches:
        for key, value in matches:
            print(f"{key}: {value}")
    else:
        print("NONE FOUND")
        print(
            "\nNo usable embedded GPS/flight telemetry "
            "was detected. Georeferencing is not claimed."
        )

    print("\nMetadata saved:", output_json)
    print("========================================\n")


if __name__ == "__main__":
    main()
