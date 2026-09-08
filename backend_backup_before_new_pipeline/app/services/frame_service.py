import cv2
import numpy as np
from pathlib import Path


def calculate_sharpness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return round(float(laplacian.var()), 2)


def calculate_brightness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return round(float(np.mean(gray)), 2)


def calculate_exposure_quality(frame):
    brightness = calculate_brightness(frame)

    if brightness < 45:
        status = "too_dark"
    elif brightness > 210:
        status = "too_bright"
    else:
        status = "good"

    return {
        "brightness": brightness,
        "status": status
    }


def calculate_frame_difference(frame_a, frame_b):
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

    gray_a = cv2.resize(gray_a, (320, 180))
    gray_b = cv2.resize(gray_b, (320, 180))

    difference = cv2.absdiff(gray_a, gray_b)

    return round(float(np.mean(difference)), 2)


def is_redundant_frame(
    previous_frame,
    current_frame,
    threshold=18.0
):
    difference_score = calculate_frame_difference(
        previous_frame,
        current_frame
    )

    return {
        "difference_score": difference_score,
        "redundant": difference_score < threshold
    }


def select_keyframes(
    video_path,
    sample_interval=5,
    sharpness_threshold=80.0,
    difference_threshold=18.0
):
    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        raise ValueError("Unable to open video file.")

    total_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = capture.get(cv2.CAP_PROP_FPS)

    selected_keyframes = []
    previous_selected_frame = None

    frame_number = 0
    analyzed_frames = 0

    while True:
        success, frame = capture.read()

        if not success:
            break

        if frame_number % sample_interval != 0:
            frame_number += 1
            continue

        analyzed_frames += 1

        sharpness = calculate_sharpness(frame)

        if sharpness < sharpness_threshold:
            frame_number += 1
            continue

        exposure = calculate_exposure_quality(frame)

        if exposure["status"] != "good":
            frame_number += 1
            continue

        difference_score = None

        if previous_selected_frame is not None:
            comparison = is_redundant_frame(
                previous_selected_frame,
                frame,
                threshold=difference_threshold
            )

            difference_score = comparison[
                "difference_score"
            ]

            if comparison["redundant"]:
                frame_number += 1
                continue

        timestamp = (
            frame_number / fps
            if fps > 0
            else 0
        )

        selected_keyframes.append({
            "frame_number": frame_number,
            "timestamp_seconds": round(timestamp, 2),
            "sharpness": sharpness,
            "brightness": exposure["brightness"],
            "difference_score": difference_score
        })

        previous_selected_frame = frame.copy()

        frame_number += 1

    capture.release()

    return {
        "total_frames": total_frames,
        "analyzed_frames": analyzed_frames,
        "selected_count": len(selected_keyframes),
        "keyframes": selected_keyframes
    }


def save_selected_keyframes(
    video_path,
    keyframes,
    output_dir
):
    output_path = Path(output_dir)

    output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        raise ValueError("Unable to open video file.")

    saved_frames = []

    for keyframe in keyframes:
        frame_number = keyframe["frame_number"]

        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number
        )

        success, frame = capture.read()

        if not success:
            continue

        filename = (
            f"keyframe_{frame_number:06d}.jpg"
        )

        file_path = output_path / filename

        success = cv2.imwrite(
            str(file_path),
            frame
        )

        if not success:
            continue

        saved_frames.append({
            **keyframe,
            "filename": filename,
            "path": str(file_path)
        })

    capture.release()

    return saved_frames