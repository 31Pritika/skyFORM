from pathlib import Path

import cv2
from ultralytics import YOLO


MODEL_NAME = "yolov8n.pt"


# COCO classes that can move and damage
# multi-view feature consistency.
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
    print(
        "Loading YOLO dynamic-object detector..."
    )

    model = YOLO(
        MODEL_NAME
    )

    print(
        "YOLO detector loaded."
    )

    return model


def mask_dynamic_objects(
    input_dir,
    output_dir,
    confidence_threshold=0.35,
):
    input_dir = Path(
        input_dir
    )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_paths = sorted(
        input_dir.glob(
            "*.jpg"
        )
    )

    if not image_paths:
        raise RuntimeError(
            "No keyframes found for "
            "dynamic-object filtering."
        )

    model = load_yolo_model()

    processed = []

    total_detections = 0

    print(
        f"Filtering {len(image_paths)} keyframes..."
    )

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):
        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        height, width = (
            image.shape[:2]
        )

        result = model.predict(
            source=image,
            conf=confidence_threshold,
            verbose=False,
            device="cpu",
        )[0]

        detections = []

        for box in result.boxes:
            class_id = int(
                box.cls.item()
            )

            confidence = float(
                box.conf.item()
            )

            if (
                class_id
                not in DYNAMIC_CLASS_IDS
            ):
                continue

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[
                    0
                ].tolist(),
            )

            # Small padding around detected
            # object to suppress edge features too.
            padding_x = int(
                (x2 - x1)
                * 0.08
            )

            padding_y = int(
                (y2 - y1)
                * 0.08
            )

            x1 = max(
                0,
                x1 - padding_x,
            )

            y1 = max(
                0,
                y1 - padding_y,
            )

            x2 = min(
                width,
                x2 + padding_x,
            )

            y2 = min(
                height,
                y2 + padding_y,
            )

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 0, 0),
                thickness=-1,
            )

            detections.append({
                "class_id":
                    class_id,

                "confidence":
                    round(
                        confidence,
                        4,
                    ),

                "box": [
                    x1,
                    y1,
                    x2,
                    y2,
                ],
            })

        output_path = (
            output_dir
            / image_path.name
        )

        success = cv2.imwrite(
            str(output_path),
            image,
        )

        if not success:
            raise RuntimeError(
                f"Failed to save "
                f"{output_path.name}"
            )

        total_detections += (
            len(detections)
        )

        processed.append({
            "image":
                image_path.name,

            "dynamic_objects":
                len(detections),

            "detections":
                detections,
        })

        print(
            f"[{index}/{len(image_paths)}] "
            f"{image_path.name} | "
            f"{len(detections)} dynamic objects"
        )

    return {
        "frames_processed":
            len(processed),

        "dynamic_objects_detected":
            total_detections,

        "frames":
            processed,
    }