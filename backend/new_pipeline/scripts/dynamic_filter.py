"""Dynamic-object masking for SkyFORM frames.

Ported from ``app/services/dynamic_object_service.py`` so the standalone
new_pipeline scripts can run it without importing the FastAPI ``app`` package
(these scripts run with ``cwd=new_pipeline/scripts`` and no ``app`` on the path).

Detects movable COCO classes (vehicles, people, animals) with YOLOv8n and
paints each detection box solid black, in place, over the extracted frames.
Removing those regions before COLMAP keeps moving objects from corrupting
multi-view feature matches and camera poses, and - because stage 05 fusion
reads the same frame directory - keeps them out of the MASt3R dense cloud too.

Guarded by ``SKYFORM_FILTER_DYNAMIC=1`` at the call site (``03_sfm.py``).
"""

from pathlib import Path
import json

import cv2
from ultralytics import YOLO


# yolov8n.pt ships at the backend root (two levels above this file).
_MODEL_PATH = Path(__file__).resolve().parents[2] / "yolov8n.pt"
MODEL_NAME = str(_MODEL_PATH) if _MODEL_PATH.exists() else "yolov8n.pt"

CONFIDENCE_THRESHOLD = 0.35
BOX_PADDING_FRACTION = 0.08

# COCO classes that can move between frames and break feature consistency.
DYNAMIC_CLASS_IDS = {
    0,   # person
    1,   # bicycle
    2,   # car
    3,   # motorcycle
    5,   # bus
    6,   # train
    7,   # truck

    14,  # bird
    15,  # cat
    16,  # dog
    17,  # horse
    18,  # sheep
    19,  # cow
    20,  # elephant
    21,  # bear
    22,  # zebra
    23,  # giraffe
}


def load_yolo_model():
    print("Loading YOLO dynamic-object detector:", MODEL_NAME)
    model = YOLO(MODEL_NAME)
    print("YOLO detector loaded.")
    return model


def filter_dynamic_frames(
    frames_dir,
    confidence_threshold=CONFIDENCE_THRESHOLD,
):
    """Mask dynamic objects in every ``frame_*.jpg`` under ``frames_dir``, in place.

    Only frames that actually contain a detection are rewritten, so clean
    frames are not needlessly re-compressed. Returns a summary dict and also
    writes it to ``frames_dir/dynamic_filter_report.json``.
    """
    frames_dir = Path(frames_dir)

    image_paths = sorted(frames_dir.glob("frame_*.jpg"))

    if not image_paths:
        raise RuntimeError(
            f"No frames found for dynamic-object filtering in {frames_dir}"
        )

    model = load_yolo_model()

    print(f"Filtering {len(image_paths)} frames for dynamic objects...")

    processed = []
    total_detections = 0
    frames_with_detections = 0

    for index, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"  [skip] unreadable frame: {image_path.name}")
            continue

        height, width = image.shape[:2]

        result = model.predict(
            source=image,
            conf=confidence_threshold,
            verbose=False,
            device="cpu",
        )[0]

        detections = []

        for box in result.boxes:
            class_id = int(box.cls.item())

            if class_id not in DYNAMIC_CLASS_IDS:
                continue

            confidence = float(box.conf.item())

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Small padding around the box so edge features on the object
            # are suppressed too.
            pad_x = int((x2 - x1) * BOX_PADDING_FRACTION)
            pad_y = int((y2 - y1) * BOX_PADDING_FRACTION)

            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(width, x2 + pad_x)
            y2 = min(height, y2 + pad_y)

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 0, 0),
                thickness=-1,
            )

            detections.append({
                "class_id": class_id,
                "confidence": round(confidence, 4),
                "box": [x1, y1, x2, y2],
            })

        if detections:
            if not cv2.imwrite(str(image_path), image):
                raise RuntimeError(
                    f"Failed to overwrite {image_path.name}"
                )
            frames_with_detections += 1

        total_detections += len(detections)

        processed.append({
            "image": image_path.name,
            "dynamic_objects": len(detections),
            "detections": detections,
        })

        if index % 25 == 0 or index == len(image_paths):
            print(
                f"  [{index}/{len(image_paths)}] "
                f"{total_detections} dynamic objects so far"
            )

    summary = {
        "frames_processed": len(processed),
        "frames_with_detections": frames_with_detections,
        "dynamic_objects_detected": total_detections,
        "confidence_threshold": confidence_threshold,
        "frames": processed,
    }

    report_path = frames_dir / "dynamic_filter_report.json"
    report_path.write_text(json.dumps(summary, indent=2))

    print(
        f"Dynamic-object filtering complete: {total_detections} objects "
        f"across {frames_with_detections}/{len(processed)} frames."
    )
    print("Report:", report_path)

    return summary
