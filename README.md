# SkyFORM

SkyFORM converts a single-pass drone video into a reconstructed 3D scene using computer vision and AI.

The system performs frame selection, dynamic-object filtering, camera reconstruction, monocular depth estimation, multi-view fusion, point-cloud generation, mesh reconstruction, reconstruction-quality analysis, visualization, and 3D model export.

## Tech Stack

### Backend
- Python
- FastAPI
- OpenCV
- PyTorch
- Depth Anything V2
- YOLOv8
- COLMAP
- Open3D

### Frontend
- React
- Vite
- Three.js
- React Three Fiber

---

# Running SkyFORM

## Requirements

Install:

- Python 3
- Node.js + npm
- COLMAP
- Git

On macOS, Homebrew can be used to install the required system packages.

---

## 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd skyFORM
```

---

## 2. Backend Setup

Go to the backend:

```bash
cd backend
```

Create a Python virtual environment:

```bash
python3 -m venv venv
```

Activate it.

### macOS / Linux

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Start the backend:

### macOS / Linux

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uvicorn app.main:app --port 8000
```

### Windows

```bash
uvicorn app.main:app --port 8000
```

The API should now be available at:

```text
http://127.0.0.1:8000
```

Test it by opening:

```text
http://127.0.0.1:8000/health
```

You should receive:

```json
{"status":"healthy"}
```

---

## 3. Frontend Setup

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the URL shown by Vite, normally:

```text
http://localhost:5173
```

---

# Using SkyFORM

1. Upload a drone or moving-camera video.
2. Optionally provide GPS/flight metadata.
3. Start reconstruction.
4. Wait for the reconstruction pipeline to complete.
5. Explore the generated:
   - 3D Point Cloud
   - 3D Mesh
   - Confidence / Quality Map
   - Camera Path
6. Download the reconstructed model.

Current export formats include:

- PLY
- OBJ
- GLB

## Input Data Contract

| Input | Required fields | Purpose |
| --- | --- | --- |
| Drone video | 1080p or 4K video, readable frame rate, one continuous pass | Source imagery for keyframe selection, pose recovery, depth, and appearance |
| GPS coordinates | Latitude, longitude, and preferably altitude for timestamped samples | Geospatial alignment and metric positioning |
| Flight metadata | Frame or timestamp correspondence to the video; camera model or intrinsics when available | Match camera poses to telemetry and reduce scale or calibration uncertainty |
| IMU data | Optional orientation and acceleration samples with timestamps | Improve pose priors during blur or low-texture sections |
| Barometric altitude | Optional timestamped altitude samples | Improve vertical scale and altitude consistency |
| Camera intrinsics | Optional focal length, principal point, and distortion parameters | Improve camera calibration when video metadata is incomplete |
| RTK/PPK corrections | Optional corrected positions and quality indicators | Improve geospatial accuracy beyond ordinary GPS |

GPS telemetry is currently uploaded as CSV with `latitude` and `longitude` columns. `altitude` and `timestamp` are supported; timestamped telemetry is required for the alignment path, which also requires at least three valid timestamped samples.

## Desired Output

For each uploaded single-pass flight, SkyFORM should produce a validated scene package. The package must identify whether the geometry is relative or georeferenced and must not present an unscaled reconstruction as metrically accurate.

| Output | Required content | Intended use |
| --- | --- | --- |
| 3D terrain and structures | Dense colored point cloud and/or triangle mesh covering visible ground, buildings, and infrastructure | Visualization, spatial inspection, and downstream analysis |
| Building facades and rooftops | Reconstructed surfaces from the available viewing angles, with explicit gaps where surfaces are occluded | Building inspection and damage assessment |
| Roads and infrastructure | Visible roads, bridges, towers, utilities, and other man-made structures represented in the scene coordinate system | Mapping, planning, and asset inspection |
| Vegetation and obstacles | Visible vegetation and detected dynamic-object masks or exclusions | Situational awareness and obstacle analysis |
| Textured or colored model | Vertex-colored PLY plus exported OBJ/GLB when mesh conversion succeeds | Interactive viewing and external 3D tools |
| Camera trajectory | Recovered camera poses and flight-path visualization | Coverage review and reconstruction diagnostics |
| Confidence data | Per-point confidence derived from reprojection error and feature-track length, with aggregate quality metrics | Measurement risk assessment and quality filtering |
| Coordinate metadata | Coordinate system, GPS availability, alignment status, source telemetry summary, and scale limitations | Correct interpretation of measurements |

### Coordinate Accuracy Contract

- Without usable GPS, timestamped flight metadata, RTK/PPK, known scale, or ground control, output geometry is relative and must be labeled `relative_unscaled`.
- With valid telemetry and successful alignment, output geometry is exported in a local ENU frame anchored to the reference GPS point and must report alignment status and residual error.
- A single flight path cannot guarantee reconstruction of surfaces that were never observed. Occluded or weakly supported areas must be treated as incomplete rather than fabricated.

## Evaluation Criteria

Evaluation should use held-out scenes with surveyed checkpoints, or independently measured distances where survey data is unavailable. Results should be reported separately for relative reconstruction and georeferenced reconstruction.

| Criterion | Measurement | Target / acceptance rule |
| --- | --- | --- |
| Reconstruction completion | Pipeline reaches mesh and quality-analysis stages without manual intervention | At least 90% of valid test videos complete; failures identify the stage and cause |
| Frame usability | Number of selected keyframes, blur rejection, and usable-frame ratio | At least 5 usable keyframes; no run proceeds while the usable set is empty |
| Camera recovery | Registered-image ratio and COLMAP reprojection error | At least 60% of selected keyframes registered; mean reprojection error below 2 px where reported |
| Geospatial alignment | Checkpoint horizontal and vertical error against surveyed coordinates | Report RMSE and maximum error; target RMSE <= 5 m with ordinary GPS, <= 1 m with RTK/PPK |
| Metric scale | Error in independently measured distances, heights, or areas | Relative-only runs report no metric claim; georeferenced runs target <= 5% distance error |
| Geometry completeness | Percentage of annotated visible surfaces represented by points or mesh faces | At least 70% for the visible evaluation region; report facade, rooftop, terrain, and infrastructure separately |
| Geometry quality | Invalid points, empty mesh checks, triangle count, and connected components | No empty primary artifact; no NaN/inf coordinates; invalid geometry count is zero |
| Dynamic-object handling | Precision and recall of masks for vehicles, people, animals, and other moving objects | Report per-class precision/recall; target mask precision >= 0.80 on the evaluation set |
| Appearance quality | Texture or vertex-color coverage and visual review under varied illumination | At least 95% of exported vertices have valid color data; no severe color corruption |
| Confidence usefulness | Correlation between confidence bands and measured geometric error | Higher-confidence points must have lower median error than lower-confidence points |
| Processing time | Wall-clock time from upload to available outputs, measured by hardware profile | Report median and p90 separately for CPU and GPU; define near-real-time as <= 2x video duration for preview output |
| Export interoperability | Open exported PLY, OBJ, and GLB files in independent viewers | Every claimed format opens and contains non-empty geometry with preserved colors where supported |

### Evaluation Dataset

The acceptance set should include buildings, roads, terrain, vegetation, and dynamic objects captured at 1080p or 4K from a single moving-UAV pass. It should contain varied lighting, shadows, compression levels, motion blur, and GPS quality. Each scene should record video duration, frame rate, camera intrinsics when available, GPS/IMU/barometric metadata, RTK/PPK availability, surveyed checkpoints, and ground-truth distances or a reference model.

---

# Reconstruction Pipeline

```text
Video
  ↓
Frame Selection
  ↓
YOLO Dynamic Object Filtering
  ↓
COLMAP Camera Reconstruction
  ↓
Depth Anything V2
  ↓
Multi-View Fusion
  ↓
Point Cloud Cleaning
  ↓
3D Point Cloud
  ↓
Mesh Generation
  ↓
Quality Analysis
  ↓
Interactive 3D Viewer + Export
```

---

# Notes

- The first run may download the required AI models.
- An internet connection may therefore be required for initial model setup.
- GPU acceleration is used when supported; SkyFORM can also run on CPU.
- COLMAP must be installed separately and accessible from the terminal.
- Without GPS/GCP/known scale information, the reconstruction uses a relative coordinate system.
- GPS-enabled metric/geospatial reconstruction requires suitable timestamped telemetry.

If setup fails on your operating system, provide this README and the error message to an LLM or debugging assistant for OS-specific installation instructions.