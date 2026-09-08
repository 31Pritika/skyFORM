from pathlib import Path
import json
import subprocess

from pipeline_config import (
    get_input_video,
    get_job_dir,
    ensure_job_directories,
)


OUTPUT_DIR = get_job_dir() / "video_probe"


def main():
    ensure_job_directories()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    video_path = get_input_video()

    print("\n==============================")
    print(" SKYFORM - VIDEO PROBE")
    print("==============================")
    print("Video:", video_path)

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(video_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    metadata = json.loads(result.stdout)

    output_path = OUTPUT_DIR / "video_probe.json"

    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    video_stream = next(
        (
            stream
            for stream in metadata.get("streams", [])
            if stream.get("codec_type") == "video"
        ),
        None,
    )

    if video_stream is None:
        raise RuntimeError(
            "No video stream found in input file."
        )

    width = video_stream.get("width")
    height = video_stream.get("height")

    duration = (
        metadata
        .get("format", {})
        .get("duration")
    )

    print("Resolution:", f"{width}x{height}")
    print("Duration:", duration, "seconds")
    print("Probe saved:", output_path)

    print("\nVIDEO PROBE COMPLETE\n")


if __name__ == "__main__":
    main()
