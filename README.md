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