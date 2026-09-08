"""Interoperable exports with explicit coordinate semantics."""
import shutil
import subprocess
from pathlib import Path
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

LOCAL_CRS = 'LOCAL_CS["SkyFORM relative unscaled",LOCAL_DATUM["Unknown",32767],UNIT["arbitrary",1],AXIS["X",EAST],AXIS["Y",NORTH]]'


def export_surface_raster(points, path, crs=None, max_size=1024):
    points = np.asarray(points, dtype=float)
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        raise ValueError('Cannot rasterize empty geometry')
    low, high = points.min(axis=0), points.max(axis=0)
    extent = high[:2] - low[:2]
    if max(extent) <= 0:
        raise ValueError('Raster requires a two-dimensional spatial extent')
    pixel = float(max(extent) / (max_size - 1))
    width, height = (np.floor(extent / pixel).astype(int) + 1).tolist()
    col = np.clip(np.floor((points[:, 0] - low[0]) / pixel).astype(int), 0, width - 1)
    row = np.clip(np.floor((high[1] - points[:, 1]) / pixel).astype(int), 0, height - 1)
    grid = np.full((height, width), -np.inf, dtype=np.float32)
    np.maximum.at(grid, (row, col), points[:, 2])
    valid = np.isfinite(grid)
    grid[~valid] = -9999
    coordinate_system = CRS.from_user_input(crs) if crs else None
    with rasterio.open(path, 'w', driver='GTiff', width=width, height=height,
                       count=1, dtype='float32', crs=coordinate_system,
                       transform=from_origin(low[0], high[1], pixel, pixel),
                       nodata=-9999, compress='deflate') as dataset:
        dataset.write(grid, 1)
        dataset.write_mask(valid.astype('uint8') * 255)
        dataset.set_band_description(1, 'Maximum measured Z per occupied cell')
        dataset.update_tags(units='metres' if crs else 'arbitrary_unscaled', coordinate_frame='projected' if crs else 'relative_unscaled',
                            accuracy_validated='false',
                            note='No interpolation over missing measurements; local Z is not geographic elevation without alignment.')
    return {'file': Path(path).name, 'coordinate_frame': 'projected' if crs else 'relative_unscaled',
            'width': width, 'height': height, 'occupied_cells': int(valid.sum()),
            'pixel_size': pixel, 'georeferenced': bool(crs)}


def export_fbx(source, destination):
    executable = shutil.which('assimp')
    if not executable:
        raise RuntimeError('FBX export requires Assimp: brew install assimp')
    result = subprocess.run([executable, 'export', str(source), str(destination), '-ffbx'],
                            capture_output=True, text=True, timeout=120)
    if result.returncode or not Path(destination).exists() or Path(destination).stat().st_size < 100:
        raise RuntimeError(f'FBX conversion failed: {result.stdout[-1000:]} {result.stderr[-1000:]}')
    return Path(destination)
