from app.services.fusion_service import (
    fuse_depth_maps
)


result = fuse_depth_maps(
    image_dir=(
        "outputs/colmap_new/dense/images"
    ),
    depth_dir=(
        "outputs/depth_maps"
    ),
    model_dir=(
        "outputs/colmap_new/dense/sparse_txt"
    ),
    output_path=(
        "outputs/reconstruction/"
        "dense_fused.ply"
    ),
    sample_step=12
)


print(
    "\n--- SkyFORM Multi-View Fusion ---"
)

print(
    "Views fused:",
    result["views"]
)

print(
    "Dense points:",
    result["points"]
)

print(
    "Saved:",
    result["output"]
)