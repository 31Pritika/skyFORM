from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
)


MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_depth_model():
    print("Loading Depth Anything V2...")

    device = get_device()

    try:
        processor = AutoImageProcessor.from_pretrained(
            MODEL_NAME,
            local_files_only=True,
        )

        model = AutoModelForDepthEstimation.from_pretrained(
            MODEL_NAME,
            local_files_only=True,
        )

        print(
            "Loaded Depth Anything V2 "
            "from local cache."
        )

    except Exception:
        print(
            "Local cache unavailable. "
            "Trying normal model load..."
        )

        processor = AutoImageProcessor.from_pretrained(
            MODEL_NAME
        )

        model = AutoModelForDepthEstimation.from_pretrained(
            MODEL_NAME
        )

    model.to(device)
    model.eval()

    print("Depth model loaded.")
    print("Device:", device)

    return processor, model, device


def clean_relative_depth(depth):
    """
    Conservative cleanup for monocular relative depth.

    Important:
    - Depth Anything output remains RELATIVE depth.
    - We do not convert it to metres here.
    - Extreme invalid predictions are rejected.
    - A small bilateral filter reduces pixel-level noise while
      preserving major depth discontinuities.

    Rejected pixels are stored as NaN so the fusion stage can
    automatically ignore them.
    """

    depth = np.asarray(
        depth,
        dtype=np.float32
    )

    finite_positive = (
        np.isfinite(depth)
        & (depth > 0)
    )

    valid_values = depth[
        finite_positive
    ]

    if len(valid_values) < 100:
        return depth

    # Very conservative percentile rejection. This removes only
    # the most extreme monocular-depth predictions.
    low = float(
        np.percentile(
            valid_values,
            0.5
        )
    )

    high = float(
        np.percentile(
            valid_values,
            99.5
        )
    )

    robust_mask = (
        finite_positive
        & (depth >= low)
        & (depth <= high)
    )

    # Bilateral filtering is edge-preserving and tends to reduce
    # small noisy variations better than a normal blur.
    working = depth.copy()

    # Fill invalid pixels temporarily only for filtering.
    median_value = float(
        np.median(
            valid_values
        )
    )

    working[
        ~finite_positive
    ] = median_value

    depth_range = max(
        high - low,
        1e-6
    )

    filtered = cv2.bilateralFilter(
        working.astype(np.float32),
        d=5,
        sigmaColor=depth_range * 0.03,
        sigmaSpace=3.0
    )

    # Restore rejected pixels as NaN. fuse_depth_maps already
    # rejects non-finite depth values.
    filtered[
        ~robust_mask
    ] = np.nan

    return filtered.astype(
        np.float32
    )


def create_depth_preview(depth):
    """
    Converts a relative depth map into an 8-bit preview while
    safely handling NaN / rejected pixels.
    """

    finite = (
        np.isfinite(depth)
        & (depth > 0)
    )

    preview = np.zeros(
        depth.shape,
        dtype=np.uint8
    )

    values = depth[
        finite
    ]

    if len(values) == 0:
        return preview, 0.0, 0.0

    minimum = float(
        np.min(values)
    )

    maximum = float(
        np.max(values)
    )

    normalized = np.zeros(
        depth.shape,
        dtype=np.float32
    )

    normalized[
        finite
    ] = (
        depth[finite] - minimum
    ) / (
        maximum - minimum + 1e-8
    )

    preview[
        finite
    ] = np.clip(
        normalized[
            finite
        ] * 255.0,
        0,
        255
    ).astype(
        np.uint8
    )

    return (
        preview,
        minimum,
        maximum
    )


def generate_depth_maps(
    image_dir,
    output_dir
):
    image_dir = Path(
        image_dir
    )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    processor, model, device = (
        load_depth_model()
    )

    image_paths = sorted(
        image_dir.glob("*.jpg")
    )

    results = []

    print(
        f"Processing {len(image_paths)} images..."
    )

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):
        print(
            f"[{index}/{len(image_paths)}] "
            f"{image_path.name}"
        )

        pil_image = Image.open(
            image_path
        ).convert(
            "RGB"
        )

        inputs = processor(
            images=pil_image,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(device)
            for key, value
            in inputs.items()
        }

        with torch.no_grad():
            outputs = model(
                **inputs
            )

        predicted_depth = (
            outputs.predicted_depth
        )

        prediction = (
            torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=(
                    pil_image.height,
                    pil_image.width
                ),
                mode="bicubic",
                align_corners=False
            )
        )

        raw_depth = (
            prediction.squeeze()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        # Keep the untouched model output for debugging/research.
        raw_path = (
            output_dir
            / f"{image_path.stem}_depth_raw.npy"
        )

        np.save(
            raw_path,
            raw_depth
        )

        # Clean the relative depth before multi-view fusion.
        depth = clean_relative_depth(
            raw_depth
        )

        valid_mask = (
            np.isfinite(depth)
            & (depth > 0)
        )

        valid_pixels = int(
            np.count_nonzero(
                valid_mask
            )
        )

        total_pixels = int(
            depth.size
        )

        valid_ratio = (
            valid_pixels
            / total_pixels
            if total_pixels
            else 0.0
        )

        # Existing fusion code expects this exact filename.
        npy_path = (
            output_dir
            / f"{image_path.stem}_depth.npy"
        )

        np.save(
            npy_path,
            depth.astype(
                np.float32
            )
        )

        (
            preview,
            minimum,
            maximum
        ) = create_depth_preview(
            depth
        )

        preview_path = (
            output_dir
            / f"{image_path.stem}_depth.png"
        )

        cv2.imwrite(
            str(preview_path),
            preview
        )

        print(
            "Valid depth pixels:",
            f"{valid_pixels}/{total_pixels}",
            f"({valid_ratio * 100:.2f}%)"
        )

        results.append({
            "image": image_path.name,
            "depth_file": str(
                npy_path
            ),
            "raw_depth_file": str(
                raw_path
            ),
            "preview": str(
                preview_path
            ),
            "min_depth": minimum,
            "max_depth": maximum,
            "valid_pixels": valid_pixels,
            "total_pixels": total_pixels,
            "valid_ratio": round(
                valid_ratio,
                4
            )
        })

    return results
