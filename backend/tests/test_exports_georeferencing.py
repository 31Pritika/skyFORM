import tempfile
import unittest
from pathlib import Path
import numpy as np
import rasterio
from new_pipeline.scripts.format_exports import export_surface_raster
from app.services.job_georeferencing import fit_similarity, transform_points, validate_checkpoints


class GeospatialTests(unittest.TestCase):
    def test_similarity_and_independent_checkpoints(self):
        source=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float)
        target=source*3+[50,60,70]
        transform=fit_similarity(source,target)
        np.testing.assert_allclose(transform_points(source,transform),target,atol=1e-10)
        check=np.array([[1,1,1],[2,1,0],[2,2,2]],float)
        self.assertTrue(validate_checkpoints(check,check*3+[50,60,70],transform)['all_checkpoints_within_1m'])
        self.assertFalse(validate_checkpoints(check,check*3+[52,60,70],transform)['all_checkpoints_within_1m'])

    def test_straight_trajectory_rejected(self):
        with self.assertRaises(ValueError): fit_similarity([[0,0,0],[1,0,0],[2,0,0]],[[0,0,0],[2,0,0],[4,0,0]])

    def test_geotiff_roundtrip_and_holes(self):
        points=np.array([[0,0,2],[0,0,3],[2,2,8]],float)
        with tempfile.TemporaryDirectory() as folder:
            for crs in (None,'EPSG:32643'):
                path=Path(folder)/'surface.tif'
                export_surface_raster(points,path,crs,max_size=3)
                with rasterio.open(path) as dataset:
                    self.assertEqual(dataset.driver,'GTiff')
                    self.assertEqual(dataset.read(1)[2,0],3)
                    self.assertEqual(dataset.read(1)[0,2],8)
                    self.assertEqual(np.count_nonzero(dataset.read_masks(1)),2)
                    self.assertEqual(dataset.tags()['coordinate_frame'],'projected' if crs else 'relative_unscaled')
                    if crs: self.assertEqual(dataset.crs.to_epsg(),32643)

    def test_job_alignment_uses_recorded_times_and_exports_projected_files(self):
        import json
        import trimesh
        from pyproj import CRS, Transformer
        from app.services.job_georeferencing import georeference_job
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder); job=base/'outputs/jobs/synthetic'
            for name in ['frames','colmap_poses','final']: (job/name).mkdir(parents=True)
            source=np.array([[0,0,0],[10,0,0],[0,10,0],[10,10,2]],float)
            local=CRS.from_proj4('+proj=aeqd +lat_0=12 +lon_0=77 +datum=WGS84 +units=m')
            inverse=Transformer.from_crs(local,4326,always_xy=True)
            lon,lat=inverse.transform(source[:,0]*2,source[:,1]*2)
            # Deliberately not video-frame-index/fps timestamps.
            times=[0,1,2,3]
            (job/'frames/frames.json').write_text(json.dumps({'frames':[{'image':f'frame_{i:04}.jpg','timestamp':t} for i,t in enumerate(times)]}))
            (job/'colmap_poses/colmap_poses.json').write_text(json.dumps({'poses':[{'image':f'frame_{i:04}.jpg','position':p.tolist()} for i,p in enumerate(source)]}))
            trimesh.points.PointCloud(source,colors=np.full((4,4),255,dtype=np.uint8)).export(job/'final/skyform_pointcloud.ply')
            gps={'points':[{'timestamp':t,'latitude':float(lat[i]),'longitude':float(lon[i]),'altitude':float(source[i,2]*2+100)} for i,t in enumerate(times)]}
            result=georeference_job(base,'synthetic',gps)
            self.assertAlmostEqual(result['scale'],2,places=6)
            self.assertFalse(result['spatial_accuracy_validated'])
            with rasterio.open(job/'final/skyform_surface_georef.tif') as dataset:
                self.assertTrue(dataset.crs.is_projected)
                self.assertEqual(dataset.tags()['coordinate_frame'],'projected')
