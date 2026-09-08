from pathlib import Path
import cv2
import shutil
import json
import os
import math

from pipeline_config import (
    get_input_video,
    get_frames_dir,
    ensure_job_directories,
)


TARGET_FPS = 2.0
JPEG_QUALITY = 95


def main():
    ensure_job_directories()

    video_path = get_input_video()
    output_dir = get_frames_dir()

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n==============================")
    print(" SKYFORM - FRAME EXTRACTION")
    print("==============================")
    print("Video:", video_path)
    print("Target FPS:", TARGET_FPS)

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    source_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if source_fps <= 0:
        cap.release()
        raise RuntimeError(
            "Could not determine source video FPS."
        )

    duration = (
        total_frames / source_fps
        if total_frames > 0
        else 0
    )

    max_frames = int(os.environ.get("SKYFORM_MAX_FRAMES", "300"))
    if max_frames < 2:
        raise ValueError("SKYFORM_MAX_FRAMES must be at least 2")
    frame_interval = max(
        math.ceil(total_frames / max_frames),
        1,
        int(round(source_fps / TARGET_FPS))
    )

    print("Source FPS:", round(source_fps, 3))
    print("Source frames:", total_frames)
    print("Duration:", round(duration, 2), "seconds")
    print("Frame interval:", frame_interval)

    source_index = 0
    saved_index = 0
    manifest = []

    while True:
        success, frame = cap.read()

        if not success:
            break

        if source_index % frame_interval == 0:

            output_path = (
                output_dir
                / f"frame_{saved_index:04d}.jpg"
            )

            ok = cv2.imwrite(
                str(output_path),
                frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    JPEG_QUALITY,
                ],
            )

            if not ok:
                cap.release()
                raise RuntimeError(
                    f"Failed to save frame: {output_path}"
                )

            manifest.append({"image": output_path.name, "source_frame": source_index, "timestamp": source_index / source_fps})
            saved_index += 1

        source_index += 1

    cap.release()

    if saved_index < 2:
        raise RuntimeError(
            "Fewer than 2 frames were extracted."
        )

    (output_dir / "frames.json").write_text(json.dumps({"source_fps": source_fps, "duration_seconds": duration, "sampling_interval_frames": frame_interval, "frames": manifest}, indent=2))

    print("\nExtracted frames:", saved_index)
    print("Output:", output_dir)

    print("\nFRAME EXTRACTION COMPLETE\n")


if __name__ == "__main__":
    main()
