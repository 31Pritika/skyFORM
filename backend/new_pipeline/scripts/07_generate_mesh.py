"""Triangulate measured surface samples without wrapping them in voxel cubes."""
import json
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from pipeline_config import get_cleanup_dir, get_mesh_dir, get_colmap_poses_dir, ensure_job_directories


def reconstruct_surface(cloud, camera_centers, method="poisson"):
    points = np.asarray(cloud.points)
    if len(points) < 30 or not np.isfinite(points).all():
        raise RuntimeError('Surface reconstruction requires at least 30 finite points.')
    spacing = np.asarray(cloud.compute_nearest_neighbor_distance())
    spacing = spacing[spacing > 0]
    if not len(spacing):
        raise RuntimeError('Surface samples have no spatial extent.')
    radius = float(np.median(spacing))
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 8, max_nn=40))
    # Orient normals toward the nearest registered camera, in the same SfM frame.
    if len(camera_centers):
        _, nearest = cKDTree(camera_centers).query(points)
        normals = np.asarray(cloud.normals).copy()
        normals[np.einsum('ij,ij->i', normals, camera_centers[nearest] - points) < 0] *= -1
        cloud.normals = o3d.utility.Vector3dVector(normals)
    else:
        cloud.orient_normals_consistent_tangent_plane(30)
    if method == "poisson":
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            cloud, depth=7, linear_fit=False, n_threads=1)
        distances, _ = cKDTree(points).query(np.asarray(mesh.vertices))
        densities = np.asarray(densities)
        mesh.remove_vertices_by_mask((densities < np.quantile(densities, .03)) |
                                     (distances > radius * 3))
        _, nearest = cKDTree(points).query(np.asarray(mesh.vertices))
        if cloud.has_colors():
            mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(cloud.colors)[nearest])
    else:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            cloud, o3d.utility.DoubleVector([radius * 1.5, radius * 2.5, radius * 4]))
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    if not len(mesh.triangles):
        raise RuntimeError('No supported surface triangles could be reconstructed.')
    labels, counts, _ = mesh.cluster_connected_triangles()
    # Only discard isolated specks; preserve separate visible scene surfaces.
    mesh.remove_triangles_by_mask(np.asarray(counts)[np.asarray(labels)] < 20)
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    if not len(mesh.triangles):
        raise RuntimeError('Only isolated surface fragments were found.')
    return mesh, radius


def generate(method, output):
    ensure_job_directories()
    cloud = o3d.io.read_point_cloud(str(get_cleanup_dir() / 'dense_clean.ply'))
    poses = json.loads((get_colmap_poses_dir() / 'colmap_poses.json').read_text())
    entries = poses if isinstance(poses, list) else poses.get('poses', poses.get('images', []))
    centers = np.asarray([p.get('camera_center', p.get('position')) for p in entries], dtype=float)
    mesh, spacing = reconstruct_surface(cloud, centers, method)
    for extension in ('ply', 'obj'):
        if not o3d.io.write_triangle_mesh(str(output / f'skyform_mesh.{extension}'), mesh):
            raise RuntimeError(f'Failed to export {extension} mesh')
    (output / 'mesh_info.json').write_text(json.dumps({
        'method': method, 'experimental': True,
        'vertices': len(mesh.vertices), 'triangles': len(mesh.triangles),
        'median_spacing': spacing, 'watertight': mesh.is_watertight(),
    }, indent=2))
    print(f'Surface exported: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles')


def main():
    ensure_job_directories()
    # Native Poisson may exit(0) without creating a mesh, or crash. Isolate it
    # and validate artifacts, then fall back to measured-sample triangulation.
    for method in ("poisson", "ball_pivoting"):
        with tempfile.TemporaryDirectory(dir=get_mesh_dir()) as temporary:
            result = subprocess.run([sys.executable, "-u", __file__, method, temporary])
            output = Path(temporary)
            required = [output / name for name in ("skyform_mesh.ply", "skyform_mesh.obj", "mesh_info.json")]
            if result.returncode == 0 and all(path.exists() and path.stat().st_size for path in required):
                for path in required:
                    shutil.copy2(path, get_mesh_dir() / path.name)
                return
            print(f"{method} did not produce a valid mesh; trying fallback.", flush=True)
    raise RuntimeError("Both surface reconstruction methods failed")


if __name__ == '__main__':
    if len(sys.argv) == 3:
        generate(sys.argv[1], Path(sys.argv[2]))
    else:
        main()
