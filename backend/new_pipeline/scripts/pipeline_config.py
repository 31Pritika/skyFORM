import os
from pathlib import Path


# ------------------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parents[2]
NEW_PIPELINE_DIR = BACKEND_DIR / "new_pipeline"

MAST3R_ROOT = BACKEND_DIR / "external" / "mast3r"


# ------------------------------------------------------------
# JOB CONFIGURATION
# ------------------------------------------------------------

def get_job_id():
    job_id = os.environ.get("SKYFORM_JOB_ID")

    if not job_id:
        raise RuntimeError(
            "SKYFORM_JOB_ID environment variable is not set."
        )

    # Prevent accidental path traversal.
    if "/" in job_id or "\\" in job_id or ".." in job_id:
        raise RuntimeError(
            f"Invalid SKYFORM_JOB_ID: {job_id}"
        )

    return job_id


def get_job_dir():
    return BACKEND_DIR / "outputs" / "jobs" / get_job_id()


def get_input_dir():
    return get_job_dir() / "input"


def get_frames_dir():
    return get_job_dir() / "frames"


def get_sfm_dir():
    return get_job_dir() / "sfm"


def get_colmap_poses_dir():
    return get_job_dir() / "colmap_poses"


def get_fusion_dir():
    return get_job_dir() / "colmap_mast3r_fusion"


def get_cleanup_dir():
    return get_job_dir() / "pointcloud_cleanup"


def get_mesh_dir():
    return get_job_dir() / "mesh"


def get_georeferencing_dir():
    return get_job_dir() / "georeferencing"


def get_final_dir():
    return get_job_dir() / "final"


def ensure_job_directories():
    directories = [
        get_job_dir(),
        get_input_dir(),
        get_frames_dir(),
        get_colmap_poses_dir(),
        get_fusion_dir(),
        get_cleanup_dir(),
        get_mesh_dir(),
        get_georeferencing_dir(),
        get_final_dir(),
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def get_input_video():
    input_dir = get_input_dir()

    videos = sorted([
        path
        for path in input_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in {
            ".mp4",
            ".mov",
            ".avi",
            ".mkv",
            ".m4v",
        }
    ])

    if not videos:
        raise FileNotFoundError(
            f"No video found in {input_dir}"
        )

    if len(videos) > 1:
        raise RuntimeError(
            f"Multiple videos found in {input_dir}. "
            "Each SkyFORM job must contain exactly one input video."
        )

    return videos[0]
