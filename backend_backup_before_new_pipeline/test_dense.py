from pathlib import Path

from app.services.dense_service import (
    depth_to_point_cloud
)


IMAGE_DIR = Path(
    "outputs/colmap_new/dense/images"
)

DEPTH_DIR = Path(
    "outputs/depth_maps"
)


images = sorted(
    IMAGE_DIR.glob("*.jpg")
)


if not images:
    raise RuntimeError(
        "No COLMAP images found."
    )


image_path = images[0]

depth_path = (
    DEPTH_DIR
    / f"{image_path.stem}_depth.npy"
)


print(
    "Using image:",
    image_path
)

print(
    "Using depth:",
    depth_path
)


result = depth_to_point_cloud(
    image_path=image_path,
    depth_path=depth_path,
    output_path=(
        "outputs/reconstruction/"
        "dense_depth.ply"
    ),
    sample_step=4
)


print(
    "\n--- SkyFORM Dense Reconstruction ---"
)

print(
    "Dense points:",
    result["points"]
)

print(
    "Saved:",
    result["output"]
)