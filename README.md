# SkyFORM

SkyFORM turns drone video into a 3D reconstruction.

It can create:

- A camera path
- A point cloud
- A 3D mesh
- Quality and confidence information
- Export files such as PLY, OBJ, GLB, LAS, GeoTIFF, and FBX

The project has:

- A Python/FastAPI backend for video processing
- A React/Vite frontend for the web interface

## Before You Start

You need:

- macOS or Linux
- Python 3.9 or newer
- Node.js 18 or newer
- Git
- FFmpeg, including `ffprobe`
- COLMAP
- Assimp for FBX export

### Install system tools on macOS

Install Homebrew from [brew.sh](https://brew.sh) if it is not already installed. Then run:

```bash
brew install ffmpeg colmap assimp
```

Check that the tools are available:

```bash
python3 --version
node --version
ffprobe -version
colmap -h
assimp version
```

## Download SkyFORM

```bash
git clone https://github.com/joel1701/skyFORM.git
cd skyFORM
```

## Start the Backend

Open a terminal in the project folder and run:

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r external/mast3r/requirements.txt
```

Start the API:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Keep this terminal open.

Test the backend in a browser:

- Health check: <http://127.0.0.1:8000/health>
- API documentation: <http://127.0.0.1:8000/docs>

The health check should return:

```json
{"status":"healthy"}
```

## Start the Frontend

Open a second terminal in the project folder:

```bash
cd frontend
npm install
npm run dev
```

Open the URL printed by Vite, usually:

<http://localhost:5173>

The frontend expects the backend to be running at `http://127.0.0.1:8000`.

## Current Frontend Note

The current frontend entry point imports a stylesheet that is not yet in the repository:

```text
frontend/src/portal/portal.css
```

Because of this, `npm run build` currently fails until that stylesheet is restored or the import in `frontend/src/App.jsx` is updated. The backend can still be started and tested through the API documentation at <http://127.0.0.1:8000/docs>.

## Use the API

The easiest way to explore the backend is the interactive API page:

<http://127.0.0.1:8000/docs>

The main workflow is:

1. Upload a video.
2. Add flight or GPS metadata if available.
3. Start a reconstruction job.
4. Check the job status while it runs.
5. View the camera path, point cloud, mesh, and quality results.
6. Download the exported files.

The API stores temporary input files in:

```text
backend/uploads/
```

Generated results are stored in:

```text
backend/outputs/
```

These folders are local runtime data and are intentionally ignored by Git.

## Run the Tests

With the backend virtual environment activated:

```bash
cd backend
python -m pytest
```

Run the frontend checks from another terminal:

```bash
cd frontend
npm run lint
npm run build
```

The frontend build will continue to report the missing `portal.css` file until the current frontend issue is fixed.

## Reconstruction Pipeline

SkyFORM processes a video in stages:

```text
Video
  -> Video information and frame extraction
  -> Dynamic-object filtering
  -> COLMAP camera reconstruction
  -> MASt3R depth and multi-view fusion
  -> Point-cloud cleanup
  -> Mesh generation
  -> Quality analysis
  -> Export files
```

The default frame limit is 300 frames. To change it:

```bash
SKYFORM_MAX_FRAMES=600 uvicorn app.main:app --host 127.0.0.1 --port 8000
```

More frames can improve coverage but will increase processing time and memory use.

## GPS and Scale

Without GPS, ground-control points, or another known scale reference, the reconstruction uses relative coordinates. It should not be interpreted as a metric survey.

GPS-based alignment requires usable timestamped flight telemetry. Independent checkpoints are required when checking reconstruction accuracy.

## Troubleshooting

### `uvicorn: command not found`

Activate the backend virtual environment and install the requirements again:

```bash
cd backend
source venv/bin/activate
python -m pip install -r requirements.txt
```

### `ffprobe: command not found`

Install FFmpeg:

```bash
brew install ffmpeg
```

### `colmap: command not found`

Install COLMAP:

```bash
brew install colmap
```

### The frontend cannot connect to the backend

Make sure both terminals are running and open:

- Frontend: <http://localhost:5173>
- Backend: <http://127.0.0.1:8000/health>

### A reconstruction is slow or runs out of memory

Use a shorter video or reduce the frame limit:

```bash
SKYFORM_MAX_FRAMES=150 uvicorn app.main:app --host 127.0.0.1 --port 8000
```

GPU acceleration is used when the installed PyTorch environment supports it. CPU processing is also supported but is slower.

## Project Folders

```text
backend/app/                 FastAPI application and services
backend/new_pipeline/        Standalone reconstruction pipeline scripts
backend/external/mast3r/     Vendored MASt3R code
backend/tests/                Backend tests
backend/uploads/              Local uploaded videos
backend/outputs/              Local generated results
frontend/src/                 React application
```

## License and Third-Party Software

SkyFORM includes or uses third-party computer-vision software. See the license and notice files in `backend/external/mast3r/` before redistributing the project.
