import cv2
import os
from pathlib import Path


def analyze_video(video_path: str):
    """
    Reads a video file and extracts basic technical metadata.
    """

    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        raise ValueError("Unable to open video file.")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration = frame_count / fps if fps > 0 else 0

    codec_value = int(capture.get(cv2.CAP_PROP_FOURCC))

    codec = "".join(
        chr((codec_value >> 8 * i) & 0xFF)
        for i in range(4)
    )

    capture.release()

    file_size_bytes = os.path.getsize(video_path)
    file_size_mb = file_size_bytes / (1024 * 1024)

    return {
        "fps": round(fps, 2),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
        "duration_seconds": round(duration, 2),
        "codec": codec,
        "file_size_mb": round(file_size_mb, 2)
    }


def extract_preview_frames(
    video_path: str,
    output_dir: str,
    video_id: str,
    frame_count_to_extract: int = 5
):
    """
    Extracts evenly spaced preview frames from the video.
    """

    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        raise ValueError("Unable to open video file.")

    total_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    output_path = Path(output_dir) / video_id
    output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    extracted_frames = []

    if total_frames <= 0:
        capture.release()
        return extracted_frames

    frame_positions = []

    for i in range(frame_count_to_extract):

        ratio = i / max(
            frame_count_to_extract - 1,
            1
        )

        frame_number = int(
            ratio * (total_frames - 1)
        )

        frame_positions.append(frame_number)

    for index, frame_number in enumerate(
        frame_positions
    ):

        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number
        )

        success, frame = capture.read()

        if not success:
            continue

        filename = f"preview_{index + 1}.jpg"

        file_path = (
            output_path / filename
        )

        cv2.imwrite(
            str(file_path),
            frame
        )

        extracted_frames.append({
            "filename": filename,
            "frame_number": frame_number
        })

    capture.release()

    return extracted_frames