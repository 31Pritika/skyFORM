from pathlib import Path
import struct


def get_registered_image_count(model_dir):
    model_dir = Path(model_dir)
    images_file = model_dir / "images.bin"

    if not images_file.exists():
        return 0

    try:
        with open(images_file, "rb") as f:
            data = f.read(8)

        if len(data) != 8:
            return 0

        return struct.unpack("<Q", data)[0]

    except Exception:
        return 0


def find_best_sparse_model(sparse_dir):
    sparse_dir = Path(sparse_dir)

    if not sparse_dir.exists():
        raise RuntimeError(
            f"Sparse directory does not exist: {sparse_dir}"
        )

    candidates = []

    for model_dir in sparse_dir.iterdir():

        if not model_dir.is_dir():
            continue

        required = [
            model_dir / "cameras.bin",
            model_dir / "images.bin",
            model_dir / "points3D.bin",
        ]

        if not all(path.exists() for path in required):
            continue

        image_count = get_registered_image_count(
            model_dir
        )

        if image_count > 0:
            candidates.append(
                (image_count, model_dir)
            )

    if not candidates:
        raise RuntimeError(
            "No valid COLMAP sparse model was found."
        )

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    image_count, best_model = candidates[0]

    print(
        f"Selected COLMAP model: {best_model} "
        f"({image_count} registered images)"
    )

    return best_model
