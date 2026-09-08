"""Georeference job outputs using recorded frame times, without claiming survey accuracy."""
import json
import zipfile
from pathlib import Path
import numpy as np
import trimesh
import laspy
from pyproj import CRS, Transformer
from new_pipeline.scripts.format_exports import export_surface_raster, export_fbx


def fit_similarity(source, target):
    source, target = np.asarray(source, float), np.asarray(target, float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError('At least three paired 3D coordinates are required')
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError('Reference coordinates must be finite')
    sm, tm = source.mean(0), target.mean(0)
    x, y = source-sm, target-tm
    for values in (x, y):
        singular = np.linalg.svd(values, compute_uv=False)
        if singular[0] < 1e-8 or singular[1] / singular[0] < .01:
            raise ValueError('Nearly straight or stationary trajectories cannot determine a reliable 3D alignment; supply non-collinear control points')
    u, d, vt = np.linalg.svd(y.T @ x / len(x))
    sign = np.eye(3)
    sign[2,2] = np.linalg.det(u @ vt)
    rotation = u @ sign @ vt
    scale = np.sum(d * np.diag(sign)) / np.mean(np.sum(x*x, axis=1))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid alignment scale')
    translation = tm-scale*(rotation @ sm)
    residual = np.linalg.norm(scale*(source @ rotation.T)+translation-target, axis=1)
    return {'scale': float(scale), 'rotation': rotation.tolist(), 'translation': translation.tolist(),
            'alignment_rmse_m': float(np.sqrt(np.mean(residual**2)))}


def transform_points(points, transform):
    return transform['scale']*(np.asarray(points) @ np.asarray(transform['rotation']).T)+transform['translation']


def validate_checkpoints(source, measured, transform):
    source, measured = np.asarray(source, float), np.asarray(measured, float)
    if source.shape != measured.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError('Supply at least three independent check points, excluded from alignment')
    if not np.isfinite(source).all() or not np.isfinite(measured).all():
        raise ValueError('Check points must be finite')
    errors = np.linalg.norm(transform_points(source, transform)-measured, axis=1)
    return {'checkpoint_count': len(source), 'errors_m': errors.tolist(),
            'rmse_m': float(np.sqrt(np.mean(errors**2))), 'maximum_error_m': float(errors.max()),
            'all_checkpoints_within_1m': bool((errors <= 1).all()),
            'note': 'Applies only to supplied independent checkpoints; does not establish accuracy everywhere.'}


def georeference_job(base_dir, video_id, gps_data):
    base = Path(base_dir)
    job = base/'outputs/jobs'/video_id
    manifest_path = job/'frames/frames.json'
    if not manifest_path.exists():
        raise ValueError('Rerun this video to record exact extraction timestamps before georeferencing')
    times = {row['image']: row['timestamp'] for row in json.loads(manifest_path.read_text())['frames']}
    data = json.loads((job/'colmap_poses/colmap_poses.json').read_text())
    poses = data.get('poses', data.get('images', []))
    gps = sorted(gps_data['points'], key=lambda p: p['timestamp'])
    t = np.asarray([p['timestamp'] for p in gps], float)
    if len(t)<3 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError('GPS needs at least three distinct increasing video-relative timestamps')
    latitude, longitude = gps[0]['latitude'], gps[0]['longitude']
    crs = CRS.from_proj4(f'+proj=aeqd +lat_0={latitude} +lon_0={longitude} +datum=WGS84 +units=m')
    projector = Transformer.from_crs(4326, crs, always_xy=True)
    x, y = projector.transform([p['longitude'] for p in gps], [p['latitude'] for p in gps])
    z = np.asarray([p['altitude'] for p in gps],float)
    target_samples = np.column_stack([x,y,z])
    source, target = [], []
    for pose in poses:
        name = pose.get('image_name', pose.get('image',pose.get('name')))
        timestamp = times.get(name)
        if timestamp is None or not t[0] <= timestamp <= t[-1]: continue
        right = min(np.searchsorted(t,timestamp,side='right'),len(t)-1)
        left = max(0,right-1)
        if t[right]-t[left] > 2: continue
        source.append(pose.get('camera_center',pose.get('position')))
        target.append([np.interp(timestamp,t,target_samples[:,axis]) for axis in range(3)])
    transform = fit_similarity(source,target)
    transform.update(coordinate_system='local_enu_meters', crs_wkt=crs.to_wkt(),
                     matched_cameras=len(source), spatial_accuracy_validated=False,
                     note='GPS alignment residual is not independent reconstruction accuracy. Altitude uses supplied GPS datum.')
    final = job/'final'
    cloud = trimesh.load(final/'skyform_pointcloud.ply',process=False)
    points = transform_points(cloud.vertices,transform)
    trimesh.points.PointCloud(points,colors=cloud.colors).export(final/'skyform_pointcloud_georef.ply')
    export_surface_raster(points,final/'skyform_surface_georef.tif',crs=crs.to_wkt())
    header=laspy.LasHeader(point_format=7,version='1.4')
    header.offsets=points.min(0); header.scales=np.maximum(np.ptp(points,axis=0)/2e9,1e-4)
    header.add_crs(crs)
    las=laspy.LasData(header); las.x,las.y,las.z=points.T
    las.red,las.green,las.blue=(cloud.colors[:,:3].astype(np.uint16)*257).T
    las.write(final/'skyform_pointcloud_georef.las')
    mesh_path=final/'skyform_mesh_experimental.ply'
    if mesh_path.exists():
        mesh=trimesh.load(mesh_path,process=False)
        mesh.vertices=transform_points(mesh.vertices,transform)
        for ext in ('ply','obj','glb'): mesh.export(final/f'skyform_mesh_georef.{ext}')
        export_fbx(final/'skyform_mesh_georef.glb',final/'skyform_mesh_georef.fbx')
    (final/'georeferencing.json').write_text(json.dumps(transform,indent=2))
    report_path = final/'pipeline_report.json'
    if report_path.exists():
        report = json.loads(report_path.read_text())
        report.setdefault('georeferencing', {}).update(georeferenced=True, alignment=transform,
                                                       spatial_accuracy_validated=False)
        report_path.write_text(json.dumps(report,indent=2))
    telemetry=base/'outputs/telemetry'/video_id; telemetry.mkdir(parents=True,exist_ok=True)
    (telemetry/'geospatial_transform.json').write_text(json.dumps(transform,indent=2))
    with zipfile.ZipFile(final/'skyform_results.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(final.iterdir()):
            if path.is_file() and path.suffix!='.zip': archive.write(path,path.name)
    return transform
