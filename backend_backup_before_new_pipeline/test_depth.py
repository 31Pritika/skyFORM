from app.services.depth_service import (
    generate_depth_maps
)


results = generate_depth_maps(
    image_dir="outputs/colmap_new/dense/images",
    output_dir="outputs/depth_maps"
)


print(
    "\n--- SkyFORM Depth Analysis ---"
)

print(
    "Depth maps generated:",
    len(results)
)


for result in results:
    print(
        result["image"],
        "| Min:",
        round(result["min_depth"], 3),
        "| Max:",
        round(result["max_depth"], 3)
    )