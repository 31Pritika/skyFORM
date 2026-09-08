from app.services.mesh_service import (
    create_mesh_from_point_cloud
)


result = create_mesh_from_point_cloud(
    point_cloud_path=(
        "outputs/reconstruction/"
        "dense_fused.ply"
    ),
    mesh_output_path=(
        "outputs/reconstruction/"
        "mesh.ply"
    )
)


print(
    "\n--- SkyFORM Mesh Reconstruction ---"
)

print(
    "Vertices:",
    result["vertices"]
)

print(
    "Triangles:",
    result["triangles"]
)

print(
    "Saved:",
    result["output"]
)