# SkyFORM verification — 8 September 2026

Current verified job: `f55d06ce-38f0-4895-b123-27a52a58c06e`. Short end-to-end run: 42.305 seconds; 20/20 registered cameras, 412,586 cleaned points. Mesh and final report produced. Twelve regression tests, frontend lint and build pass (large viewer bundle warning remains).

## Implemented and checked

- Mesh / point-cloud reconstruction, mesh worker isolation and fallback, job-specific assets, live progress and web visualization.
- Camera Path displays camera bodies, viewing frustums, connecting trajectory, start/end labels, selected-frame preview and scene-context toggle.
- All requested file formats: OBJ, PLY, LAS, GeoTIFF, GLB/glTF, FBX. FBX converted back through Assimp preserves all 33,068 triangles and coordinates to within 0.000000051 reconstruction units. ZIP integrity checked.
- Local GeoTIFF stores sampled maximum Z and explicit missing-data mask. It intentionally has **no geographic CRS**, and uses `units=arbitrary_unscaled`; assigning metres or an EPSG code without alignment would be false. It is not yet a geographically positioned terrain product.
- GPS alignment uses recorded extraction timestamps and projected WGS84 coordinates. It rejects insufficient, stationary or nearly collinear reference trajectories. Geographic GeoTIFF and metric PLY/LAS/mesh exports are generated when valid telemetry is supplied. A synthetic end-to-end test validates the coordinate transform and projected raster; real-world accuracy is not inferred from that test.
- Independent checkpoint validation: POST `/api/geospatial/checkpoints/{video_id}` with `source_points`, `measured_points` in the alignment CRS, and `independent_of_alignment: true`. At least three independent checkpoints are required. Results include individual errors, RMSE and maximum error, and refer only to the supplied checkpoints.
- Fusion reports per-frame supported-pixel fractions; these are diagnostics, not proof of entire-scene coverage. Square MASt3R inputs retain the square field of view. Frame extraction writes actual source timestamps and has a configurable 300-frame default cap (`SKYFORM_MAX_FRAMES`); this cap bounds workload but does not guarantee sufficient overlap or the runtime target.
- Requirement status table distinguishes format/visualization availability from unverified accuracy and coverage.

## Still not certified

Entire-visible-scene reconstruction remains unverified and can be incomplete over water, occlusions and unsupported textures. ≤1 m accuracy requires independent measured reference data; none was supplied. Geographically positioned GeoTIFF needs usable telemetry or references. These cannot truthfully be marked passed from the available video alone.

The 10-minute speed benchmark is on hold at the user's request. The supplied Tower Bridge file was probed at 154.250159 seconds (2m34s), not ten minutes. No ten-minute benchmark was claimed or substituted with repeated footage.

## Runtime dependencies

Assimp 6.0.5 installed with Homebrew for FBX; Rasterio 1.4.3 and PyProj 3.6.1 added to the project virtual environment and requirements. APIs and format behavior were checked against Assimp source and Rasterio documentation.

No commit or deployment performed. Existing work preserved.
