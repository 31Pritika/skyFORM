# skyFORM

Generate a georeferenced, textured 3D reconstruction (point cloud + mesh)
from a **single-pass drone video** — no multiple flight passes, no manual
flight planning, no extensive post-processing.

Built around COLMAP (camera pose recovery) + MASt3R (AI-based dense depth
and multi-view fusion), wrapped in a FastAPI backend and a React/Three.js
frontend viewer.

## Table of contents

- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Setup](#setup)
- [Running it](#running-it)
- [Configuration (env vars)](#configuration-env-vars)
- [Viewing a result / sample data](#viewing-a-result--sample-data)
- [Known issues & gotchas](#known-issues--gotchas)
- [Hardware notes (CPU vs GPU)](#hardware-notes-cpu-vs-gpu)
- [Repo structure](#repo-structure)
- [Troubleshooting](#troubleshooting)

## How it works

The pipeline runs in 10 stages, in order:

| # | Stage | What it does |
|---|-------|--------------|
| 1 | Video probe | Reads video metadata (resolution, fps, duration) via ffprobe |
| 2 | Frame extraction | Samples frames from the video at a target rate, capped by `SKYFORM_MAX_FRAMES` |
| 3 | COLMAP SfM | Recovers camera positions/orientations for each frame (feature extraction → matching → mapping) |
| 4 | Read COLMAP poses | Parses COLMAP's output into a usable pose list |
| 5 | COLMAP + MASt3R fusion | Runs MASt3R (a transformer depth model) on adjacent frame pairs, fuses into a dense 3D point cloud using COLMAP's poses as a fixed scaffold |
| 6 | Point-cloud cleanup | Voxel downsampling, statistical + radius outlier removal, optional background/distance filtering |
| 7 | Mesh generation | Poisson surface reconstruction from the cleaned point cloud |
| 8 | Export | Writes PLY, OBJ, GLB, glTF, XYZ, LAS, GeoTIFF (FBX best-effort, needs `assimp`) |
| 9 | Geo metadata | Extracts/validates any GPS/flight metadata found |
| 10 | Telemetry validation | Checks IMU/GPS telemetry if supplied |

**Optional stage** (off by default, see gotchas below): YOLO-based dynamic
object masking (cars/people/animals) runs inside stage 3, before COLMAP, if
`SKYFORM_FILTER_DYNAMIC=1`.

Stage 3 is a **hard gate**: if COLMAP registers fewer than 35% of extracted
frames, the pipeline aborts rather than silently producing a broken,
near-empty reconstruction. If you hit this, your frames are too sparse for
COLMAP to find overlap — see [Troubleshooting](#troubleshooting).

## Requirements

- Python 3.12, Node.js + npm
- `ffmpeg` / `ffprobe`
- `COLMAP` (CLI)
- ~4GB+ free RAM minimum (see [Hardware notes](#hardware-notes-cpu-vs-gpu) —
  this genuinely matters, low-RAM CPU-only boxes need the low-memory profile)
- Ideally an NVIDIA GPU (CUDA) or Apple Silicon (MPS) for MASt3R — CPU works
  but is 10-50x slower

## Setup

```bash
# Backend
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# MASt3R (external, not a normal pip package)
git clone --recursive https://github.com/naver/mast3r external/mast3r
# pinned commit: f5209af

# Frontend
cd ../frontend
npm install
```

**Known first-run fix needed:** `frontend/src/App.jsx` imports
`./portal/portal.css`, which doesn't exist in the repo. Either:
```bash
mkdir -p frontend/src/portal && touch frontend/src/portal/portal.css
```
or remove the import line in `App.jsx`.

## Running it

```bash
# Backend (from backend/, venv active)
HF_HUB_DISABLE_XET=1 QT_QPA_PLATFORM=offscreen uvicorn app.main:app --host 127.0.0.1 --port 8000

# Frontend (from frontend/)
npm run dev
```

Open **`http://localhost:5173`** (not `127.0.0.1:5173` — see CORS note below).

**Persistent sessions (recommended, survives terminal/VS Code disconnects):**
```bash
./run-be.sh   # starts backend in a tmux session `skyform-backend`
./run-fe.sh   # starts frontend in a tmux session `skyform-frontend`
```
Detach without killing: `Ctrl+B` then `D`. Reattach: `tmux attach -t skyform-backend`.
List sessions: `tmux ls`.

## Configuration (env vars)

Set these on the **backend** process before starting it.

| Var | Default | Purpose |
|---|---|---|
| `HF_HUB_DISABLE_XET` | unset | **Set to `1`.** Forces classic HTTP download for the MASt3R checkpoint instead of HuggingFace's Xet backend, which buffers the whole ~2.5GB file into RAM and can stall/OOM on constrained boxes. |
| `QT_QPA_PLATFORM` | unset | **Set to `offscreen`.** COLMAP's CLI tools still try to init a Qt GUI plugin; on a headless/no-display box this crashes without this set. |
| `SKYFORM_MAX_FRAMES` | 300 | Cap on frames sampled from the video. Lower = faster, but too low relative to video length widens frame spacing and can drop below COLMAP's 35% registration gate. |
| `SKYFORM_FUSION_IMAGE_SIZE` | 512 | MASt3R inference resolution. 256 is ~4x faster per pair, coarser depth. **Don't use 224** — dust3r treats it specially and geometry comes out misaligned. |
| `SKYFORM_FUSION_THREADS` | auto (`min(8, cpu_count-2)`) | CPU thread count for MASt3R inference. |
| `SKYFORM_SFM_LOW_MEM` | 1 | Runs COLMAP with a reduced-memory profile (smaller max image size, fewer features, no affine-shape/domain-size-pooling, no guided matching, fewer threads). **Set to `0` only on a machine with plenty of RAM/a real GPU** — the default heavy profile can OOM-kill on <8GB boxes. |
| `SKYFORM_FILTER_DYNAMIC` | 0 | Masks moving cars/people/animals (YOLOv8n) before COLMAP. Off by default — adds memory pressure at the same moment as COLMAP; only enable on a box with headroom, or if your footage genuinely has heavy moving traffic. |
| `SKYFORM_PATH_DIST_MULTIPLIER` | 8.0 | Background/sky point filtering in cleanup — drops points farther than this multiple of median camera spacing from the nearest camera. Very footage-dependent; 8.0 is conservative and self-disables if it would remove too much. Try 15-25 and inspect visually. |
| `SKYFORM_SFM_MIN_IMAGES` / `SKYFORM_SFM_MIN_FRACTION` | 8 / 0.35 | The stage-3 registration hard gate. Raise only if you understand you're allowing sparser reconstructions through. |

## Viewing a result / sample data

Completed jobs live in `backend/outputs/jobs/<job-id>/final/` with exported
PLY/OBJ/GLB/etc — useful for frontend work without running the pipeline.

Check what the backend currently considers "active":
```bash
curl http://127.0.0.1:8000/api/reconstruction/status
```
Note: this reads a **global state file**, not per-job — it only updates
when a job completes through the normal upload flow. A direct/manual script
invocation won't update it.

## Known issues & gotchas

- **CORS**: backend's CORS allow-list must include the exact origin the
  browser uses. `localhost` and `127.0.0.1` are different origins to a
  browser even though they're the same machine — check
  `backend/app/main.py`'s `CORSMiddleware` config if you see CORS errors.
- **Orientation**: `frontend/src/components/ReconstructionViewer.jsx` has an
  `AXIS_CORRECTION` constant (currently `[Math.PI, 0, 0]`) that rotates the
  scene to compensate for COLMAP vs. Three.js using different up-axis
  conventions. **This is footage-dependent** — if a model loads upside down
  or sideways, try `[Math.PI / 2, 0, 0]` or `[-Math.PI / 2, 0, 0]`.
- **FBX export** doesn't work without `assimp` installed system-wide; the
  other 7 export formats are unaffected.
- **No GPS = relative coordinates**: without real flight telemetry, output
  coordinates are `relative_unscaled`, not true-scale/georeferenced.
- **Frame spacing matters more than frame count**: COLMAP needs meaningful
  overlap between consecutive sampled frames. A `SKYFORM_MAX_FRAMES` that's
  too low relative to video length widens spacing and can collapse
  registration to a handful of frames (the stage-3 gate now catches and
  aborts this cleanly rather than silently emitting a near-empty result).

## Hardware notes (CPU vs GPU)

This pipeline is meaningfully faster with a real GPU. On CPU:
- COLMAP SfM: scales with frame count² (exhaustive matching) — minutes to
  tens of minutes depending on frame count and `SKYFORM_SFM_LOW_MEM`.
- MASt3R fusion: ~30s/pair on a modern CPU at 512px, ~7-10s/pair at 256px.
  On CPU-only, low-RAM machines, **this is also a real memory risk** — the
  default COLMAP profile can OOM-kill the process (and in some cases has
  taken down the whole VM on WSL) below ~8GB RAM. `SKYFORM_SFM_LOW_MEM=1`
  (default) mitigates this but doesn't eliminate the need for adequate RAM.

If you have access to any NVIDIA GPU (local or a rented cloud instance —
RunPod/Vast.ai are cheap options), the code auto-detects CUDA and uses it
with no config changes needed; MASt3R inference alone can drop from ~30s/pair
to low single-digit seconds/pair.

## Repo structure

```
backend/
  app/
    main.py                          # FastAPI app, CORS config
    services/
      reconstruction_pipeline.py     # kicks off run_pipeline.py, tracks job state
  new_pipeline/scripts/
    01_video_probe.py … 10_validate_telemetry.py
    dynamic_filter.py                # YOLO masking (optional, see env vars)
    image_geometry.py                # resolution-aware crop/reproject math
    run_pipeline.py                  # orchestrates the 10 stages
  external/mast3r/                   # cloned dependency, not tracked in git
  outputs/jobs/<job-id>/             # per-job artifacts, one folder per stage
  requirements.txt

frontend/
  src/
    App.jsx
    components/
      Dashboard.jsx                  # polls pipeline/reconstruction/geo status
      ReconstructionViewer.jsx       # Three.js viewer, AXIS_CORRECTION
    api/videoApi.js

run-be.sh / run-fe.sh                # tmux-based persistent start scripts
```

## Troubleshooting

**Upload fails with a generic error** → check `curl http://127.0.0.1:8000/health`
first; if the backend's down, nothing else will work.

**CORS error in browser console** → confirm you're using the same origin
(`localhost` vs `127.0.0.1`) that's in the backend's allow-list.

**Reconstruction stuck at ~24%** → that's stage 3 (COLMAP). Check
`ps aux | grep colmap` — if a process is actively using CPU, it's slow, not
hung (CPU-only exhaustive matching genuinely takes a while). If it's not
running at all and the job shows "failed", check `dmesg | tail -50` for an
OOM-kill — see [Hardware notes](#hardware-notes-cpu-vs-gpu).

**Registered frames much lower than extracted frames** → your
`SKYFORM_MAX_FRAMES` is too low relative to video length, spacing frames too
far apart for COLMAP to find overlap. Raise it, or lower target FPS spacing.

**WSL crashes / VS Code disconnects during a run** → almost always memory
pressure from COLMAP/MASt3R on a constrained box. Make sure
`SKYFORM_SFM_LOW_MEM=1` and `SKYFORM_FILTER_DYNAMIC=0` (both are defaults),
consider capping WSL's memory via `.wslconfig` on the Windows side, and
avoid running multiple heavy pipeline jobs concurrently.
