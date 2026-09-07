from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    BackgroundTasks,
)

from fastapi.middleware.cors import (
    CORSMiddleware,
)

from fastapi.staticfiles import (
    StaticFiles,
)

from fastapi.responses import (
    FileResponse,
)

from pydantic import BaseModel
from typing import Optional

from pathlib import Path
import shutil
import uuid

from app.services.video_service import (
    analyze_video,
    extract_preview_frames,
)

from app.services.status_service import (
    get_reconstruction_status,
)

from app.services.pipeline_service import (
    get_pipeline_state,
    try_start_pipeline,
)

from app.services.reconstruction_pipeline import (
    run_reconstruction_pipeline,
)

from app.services.camera_service import (
    get_camera_poses,
)

from app.services.quality_service import (
    get_quality_metrics,
)

from app.services.geospatial_service import (
    parse_gps_csv,
    get_gps_path,
)

from app.services.alignment_service import (
    georeference_reconstruction,
)

from app.services.export_service import (
    create_reconstruction_export,
)

from app.services.flight_metadata_service import (
    save_flight_metadata,
    load_flight_metadata,
)


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="SkyFORM API",
    description=(
        "Single-Pass UAV Video "
        "to 3D Reconstruction System"
    ),
    version="0.5.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DIRECTORIES
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

UPLOAD_DIR = (
    BASE_DIR / "uploads"
)

OUTPUT_DIR = (
    BASE_DIR / "outputs"
)

PREVIEW_DIR = (
    OUTPUT_DIR / "previews"
)

TELEMETRY_DIR = (
    OUTPUT_DIR / "telemetry"
)

EXPORT_DIR = (
    OUTPUT_DIR / "exports"
)

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PREVIEW_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TELEMETRY_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

EXPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# STATIC FILES
# ============================================================

app.mount(
    "/previews",
    StaticFiles(
        directory=str(PREVIEW_DIR)
    ),
    name="previews",
)

app.mount(
    "/outputs",
    StaticFiles(
        directory=str(OUTPUT_DIR)
    ),
    name="outputs",
)


# ============================================================
# MODELS
# ============================================================

class FlightMetadataPayload(BaseModel):
    drone_model: Optional[str] = None
    camera_model: Optional[str] = None

    flight_altitude_m: Optional[float] = None
    speed_mps: Optional[float] = None

    heading_deg: Optional[float] = None
    yaw_deg: Optional[float] = None
    pitch_deg: Optional[float] = None
    roll_deg: Optional[float] = None

    focal_length_mm: Optional[float] = None
    sensor_width_mm: Optional[float] = None
    sensor_height_mm: Optional[float] = None

    capture_start_time: Optional[str] = None
    notes: Optional[str] = None


# ============================================================
# ROOT / HEALTH
# ============================================================

@app.get("/")
def root():
    return {
        "name": "SkyFORM",
        "status": "online",
        "version": "0.5.0",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


# ============================================================
# FLIGHT METADATA
# ============================================================

@app.get(
    "/api/flight-metadata/status/{video_id}"
)
def flight_metadata_status(
    video_id: str,
):
    matching_videos = list(
        UPLOAD_DIR.glob(
            f"{video_id}.*"
        )
    )

    if not matching_videos:
        raise HTTPException(
            status_code=404,
            detail=(
                "Uploaded video "
                "not found."
            ),
        )

    return load_flight_metadata(
        BASE_DIR,
        video_id,
    )


@app.post(
    "/api/flight-metadata/{video_id}"
)
def upload_flight_metadata(
    video_id: str,
    payload: FlightMetadataPayload,
):
    matching_videos = list(
        UPLOAD_DIR.glob(
            f"{video_id}.*"
        )
    )

    if not matching_videos:
        raise HTTPException(
            status_code=404,
            detail=(
                "Uploaded video "
                "not found."
            ),
        )

    if hasattr(
        payload,
        "model_dump",
    ):
        metadata = (
            payload.model_dump(
                exclude_none=True
            )
        )
    else:
        metadata = (
            payload.dict(
                exclude_none=True
            )
        )

    if not metadata:
        raise HTTPException(
            status_code=400,
            detail=(
                "At least one flight "
                "metadata field must "
                "be supplied."
            ),
        )

    try:
        return save_flight_metadata(
            BASE_DIR,
            video_id,
            metadata,
        )

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )


# ============================================================
# RECONSTRUCTION STATUS
# ============================================================

@app.get("/api/reconstruction/status")
def reconstruction_status():
    return get_reconstruction_status(
        BASE_DIR
    )


# ============================================================
# PIPELINE STATUS
# ============================================================

@app.get("/api/pipeline/status")
def pipeline_execution_status():
    return get_pipeline_state(
        BASE_DIR
    )


# ============================================================
# CAMERA POSES
# ============================================================

@app.get("/api/reconstruction/cameras")
def reconstruction_cameras():
    poses = get_camera_poses(
        BASE_DIR
    )

    return {
        "count": len(poses),
        "cameras": poses,
    }


# ============================================================
# QUALITY / CONFIDENCE
# ============================================================

@app.get("/api/reconstruction/quality")
def reconstruction_quality():
    return get_quality_metrics(
        BASE_DIR
    )


# ============================================================
# RECONSTRUCTION EXPORT
# ============================================================

@app.get("/api/reconstruction/export")
def export_reconstruction():

    reconstruction_status_data = (
        get_reconstruction_status(
            BASE_DIR
        )
    )

    quality_data = (
        get_quality_metrics(
            BASE_DIR
        )
    )

    export_path = (
        create_reconstruction_export(
            BASE_DIR,
            reconstruction_status_data,
            quality_data,
        )
    )

    if not export_path.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to create "
                "reconstruction export."
            ),
        )

    return FileResponse(
        path=str(export_path),
        media_type="application/zip",
        filename=(
            "SkyFORM_Reconstruction.zip"
        ),
    )


# ============================================================
# GEOSPATIAL STATUS
# ============================================================

@app.get(
    "/api/geospatial/status/{video_id}"
)
def geospatial_status(
    video_id: str,
):
    matching_videos = list(
        UPLOAD_DIR.glob(
            f"{video_id}.*"
        )
    )

    if not matching_videos:
        raise HTTPException(
            status_code=404,
            detail=(
                "Uploaded video "
                "not found."
            ),
        )

    gps_file = get_gps_path(
        BASE_DIR,
        video_id,
    )

    transform_file = (
        OUTPUT_DIR
        / "telemetry"
        / video_id
        / "geospatial_transform.json"
    )

    # No GPS supplied.
    if not gps_file.exists():
        return {
            "available": False,
            "video_id": video_id,
            "gps_available": False,
            "aligned": False,
            "coordinate_system":
                "relative_unscaled",
            "message":
                "No GPS telemetry supplied.",
        }

    try:
        result = parse_gps_csv(
            gps_file
        )

        aligned = (
            transform_file.exists()
        )

        return {
            "available": True,
            "video_id": video_id,
            "gps_available": True,
            "aligned": aligned,
            "coordinate_system": (
                "local_enu_meters"
                if aligned
                else "relative_unscaled"
            ),
            "telemetry": result,
        }

    except Exception as error:
        return {
            "available": False,
            "video_id": video_id,
            "gps_available": False,
            "aligned": False,
            "coordinate_system":
                "relative_unscaled",
            "message": str(error),
        }


# ============================================================
# GEOSPATIAL UPLOAD
# ============================================================

@app.post(
    "/api/geospatial/upload/{video_id}"
)
async def upload_geospatial_data(
    video_id: str,
    telemetry: UploadFile = File(...),
):
    matching_videos = list(
        UPLOAD_DIR.glob(
            f"{video_id}.*"
        )
    )

    if not matching_videos:
        raise HTTPException(
            status_code=404,
            detail=(
                "Uploaded video "
                "not found."
            ),
        )

    if not telemetry.filename:
        raise HTTPException(
            status_code=400,
            detail=(
                "Telemetry file "
                "has no filename."
            ),
        )

    extension = (
        Path(
            telemetry.filename
        )
        .suffix
        .lower()
    )

    if extension != ".csv":
        raise HTTPException(
            status_code=400,
            detail=(
                "Telemetry must be "
                "a CSV file."
            ),
        )

    saved_path = get_gps_path(
        BASE_DIR,
        video_id,
    )

    transform_file = (
        OUTPUT_DIR
        / "telemetry"
        / video_id
        / "geospatial_transform.json"
    )

    try:
        with open(
            saved_path,
            "wb",
        ) as buffer:
            shutil.copyfileobj(
                telemetry.file,
                buffer,
            )

        result = parse_gps_csv(
            saved_path
        )

        # New GPS telemetry invalidates an older
        # alignment belonging to the same video.
        if transform_file.exists():
            transform_file.unlink()

        return {
            "success": True,
            "video_id": video_id,
            "gps_available": True,
            "aligned": False,
            "coordinate_system":
                "relative_unscaled",
            "telemetry": result,
        }

    except Exception as error:
        if saved_path.exists():
            saved_path.unlink()

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )


# ============================================================
# GEOSPATIAL ALIGNMENT
# ============================================================

@app.post(
    "/api/geospatial/align/{video_id}"
)
def align_reconstruction_to_gps(
    video_id: str,
):
    matching_videos = list(
        UPLOAD_DIR.glob(
            f"{video_id}.*"
        )
    )

    if not matching_videos:
        raise HTTPException(
            status_code=404,
            detail=(
                "Uploaded video "
                "not found."
            ),
        )

    gps_file = get_gps_path(
        BASE_DIR,
        video_id,
    )

    if not gps_file.exists():
        raise HTTPException(
            status_code=400,
            detail=(
                "GPS telemetry not "
                "supplied for this video."
            ),
        )

    try:
        gps_data = parse_gps_csv(
            gps_file
        )

        result = (
            georeference_reconstruction(
                BASE_DIR,
                matching_videos[0],
                gps_data,
            )
        )

        return {
            "success": True,
            "video_id": video_id,
            "gps_available": True,
            "aligned": True,
            "coordinate_system":
                "local_enu_meters",
            "alignment": result,
        }

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )


# ============================================================
# AUTOMATED RECONSTRUCTION PIPELINE
# ============================================================

@app.post(
    "/api/reconstruction/start/{video_id}"
)
def start_reconstruction(
    video_id: str,
    background_tasks: BackgroundTasks,
):
    matching_videos = list(
        UPLOAD_DIR.glob(
            f"{video_id}.*"
        )
    )

    if not matching_videos:
        raise HTTPException(
            status_code=404,
            detail=(
                "Uploaded video not found."
            ),
        )

    video_path = matching_videos[0]

    if (
        video_path.suffix.lower()
        not in {
            ".mp4",
            ".mov",
            ".avi",
            ".mkv",
        }
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid video file."
            ),
        )

    allowed, state = (
        try_start_pipeline(
            BASE_DIR,
            video_id=video_id,
        )
    )

    if not allowed:
        raise HTTPException(
            status_code=409,
            detail=(
                "A reconstruction is "
                "already running."
            ),
        )

    background_tasks.add_task(
        run_reconstruction_pipeline,
        BASE_DIR,
        video_path,
    )

    return {
        "success": True,
        "status": "running",
        "video_id": video_id,
        "progress":
            state["progress"],
        "message": (
            "SkyFORM reconstruction "
            "started."
        ),
    }


# ============================================================
# VIDEO UPLOAD
# ============================================================

@app.post("/api/videos/upload")
async def upload_video(
    video: UploadFile = File(...)
):

    allowed_extensions = {
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
    }

    if not video.filename:
        raise HTTPException(
            status_code=400,
            detail=(
                "Video file has "
                "no filename."
            ),
        )

    original_filename = (
        video.filename
    )

    extension = (
        Path(original_filename)
        .suffix
        .lower()
    )

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported video format."
            ),
        )

    video_id = str(
        uuid.uuid4()
    )

    saved_filename = (
        f"{video_id}{extension}"
    )

    saved_path = (
        UPLOAD_DIR
        / saved_filename
    )

    try:

        # ----------------------------------------
        # SAVE VIDEO
        # ----------------------------------------

        with open(
            saved_path,
            "wb",
        ) as buffer:

            shutil.copyfileobj(
                video.file,
                buffer,
            )

        # ----------------------------------------
        # ANALYZE VIDEO
        # ----------------------------------------

        metadata = analyze_video(
            str(saved_path)
        )

        # ----------------------------------------
        # EXTRACT PREVIEW FRAMES
        # ----------------------------------------

        preview_frames = (
            extract_preview_frames(
                video_path=str(
                    saved_path
                ),
                output_dir=str(
                    PREVIEW_DIR
                ),
                video_id=video_id,
                frame_count_to_extract=5,
            )
        )

        # ----------------------------------------
        # CREATE PREVIEW URLs
        # ----------------------------------------

        for frame in preview_frames:

            frame["url"] = (
                f"/previews/"
                f"{video_id}/"
                f"{frame['filename']}"
            )

        # ----------------------------------------
        # RESPONSE
        # ----------------------------------------

        return {
            "success": True,

            "video": {
                "id":
                    video_id,

                "original_filename":
                    original_filename,

                "stored_filename":
                    saved_filename,

                **metadata,

                "preview_frames":
                    preview_frames,
            },
        }

    except Exception as error:

        if saved_path.exists():
            saved_path.unlink()

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )