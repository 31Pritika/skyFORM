from pathlib import Path
from datetime import datetime
import json
import threading


PIPELINE_LOCK = threading.Lock()


DEFAULT_STATE = {
    "status": "idle",
    "video_id": None,
    "progress": 0,
    "current_stage": None,
    "message": "Ready to start reconstruction.",
    "error": None,
    "started_at": None,
    "completed_at": None,
    "stages": {
        "frame_intelligence": "pending",
        "feature_extraction": "pending",
        "camera_reconstruction": "pending",
        "depth_estimation": "pending",
        "depth_fusion": "pending",
        "mesh_generation": "pending",
        "quality_analysis": "pending",
    },
}


def _state_path(base_dir):
    return (
        Path(base_dir)
        / "outputs"
        / "pipeline_state.json"
    )


def _fresh_state():
    return {
        **DEFAULT_STATE,
        "stages":
            DEFAULT_STATE[
                "stages"
            ].copy(),
    }


def _write_state(
    base_dir,
    state,
):
    path = _state_path(
        base_dir
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(path)



def get_pipeline_state(
    base_dir,
):
    path = _state_path(
        base_dir
    )

    if not path.exists():
        return _fresh_state()

    try:
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:
            state = json.load(
                file
            )

        # Backward compatibility with older
        # pipeline_state.json files.
        if "video_id" not in state:
            state["video_id"] = None

        if "stages" not in state:
            state["stages"] = (
                DEFAULT_STATE[
                    "stages"
                ].copy()
            )

        return state

    except Exception:
        return _fresh_state()


def reset_pipeline_state(
    base_dir,
):
    state = _fresh_state()

    _write_state(
        base_dir,
        state,
    )

    return state


def try_start_pipeline(
    base_dir,
    video_id=None,
):
    """
    Atomically reserve the reconstruction
    pipeline for one video.

    Returns:
        (True, state)
        if reconstruction may start.

        (False, state)
        if another reconstruction is
        already running.
    """

    with PIPELINE_LOCK:
        state = get_pipeline_state(
            base_dir
        )

        if (
            state.get("status")
            == "running"
        ):
            return False, state

        state = _fresh_state()

        state["status"] = (
            "running"
        )

        state["video_id"] = (
            str(video_id)
            if video_id
            else None
        )

        state["progress"] = 0

        state["current_stage"] = (
            "frame_intelligence"
        )

        state["message"] = (
            "Reconstruction queued."
        )

        state["error"] = None

        state["started_at"] = (
            datetime.now()
            .isoformat()
        )

        state["completed_at"] = None

        state["stages"][
            "frame_intelligence"
        ] = "running"

        _write_state(
            base_dir,
            state,
        )

        return True, state


def update_pipeline_state(
    base_dir,
    *,
    status=None,
    video_id=None,
    progress=None,
    current_stage=None,
    message=None,
    error=None,
    stage=None,
    stage_status=None,
):
    with PIPELINE_LOCK:
        state = get_pipeline_state(
            base_dir
        )

        if status is not None:
            state["status"] = (
                status
            )

        if video_id is not None:
            state["video_id"] = (
                str(video_id)
            )

        if progress is not None:
            state["progress"] = (
                progress
            )

        if current_stage is not None:
            state[
                "current_stage"
            ] = current_stage

        if message is not None:
            state["message"] = (
                message
            )

        if error is not None:
            state["error"] = (
                error
            )

        if (
            stage is not None
            and stage_status
            is not None
        ):
            state["stages"][
                stage
            ] = stage_status

        _write_state(
            base_dir,
            state,
        )

        return state


def mark_pipeline_started(
    base_dir,
    video_id=None,
):
    """
    Mark execution as started without losing
    the video_id reserved by try_start_pipeline.
    """

    with PIPELINE_LOCK:
        previous_state = (
            get_pipeline_state(
                base_dir
            )
        )

        existing_video_id = (
            previous_state.get(
                "video_id"
            )
        )

        state = _fresh_state()

        state["status"] = (
            "running"
        )

        state["video_id"] = (
            str(video_id)
            if video_id
            else existing_video_id
        )

        state["progress"] = 0

        state["current_stage"] = (
            "frame_intelligence"
        )

        state["message"] = (
            "Reconstruction started."
        )

        state["error"] = None

        state["started_at"] = (
            previous_state.get(
                "started_at"
            )
            or datetime.now()
            .isoformat()
        )

        state["completed_at"] = None

        state["stages"][
            "frame_intelligence"
        ] = "running"

        _write_state(
            base_dir,
            state,
        )

        return state


def mark_pipeline_complete(
    base_dir,
):
    with PIPELINE_LOCK:
        state = get_pipeline_state(
            base_dir
        )

        state["status"] = (
            "completed"
        )

        state["progress"] = 100

        state["current_stage"] = (
            None
        )

        state["message"] = (
            "Reconstruction completed."
        )

        state["error"] = None

        state["completed_at"] = (
            datetime.now()
            .isoformat()
        )

        _write_state(
            base_dir,
            state,
        )

        return state


def mark_pipeline_failed(
    base_dir,
    error_message,
):
    with PIPELINE_LOCK:
        state = get_pipeline_state(
            base_dir
        )

        current_stage = (
            state.get(
                "current_stage"
            )
        )

        state["status"] = (
            "failed"
        )

        state["message"] = (
            "Reconstruction failed."
        )

        state["error"] = (
            str(
                error_message
            )
        )

        if current_stage:
            state["stages"][
                current_stage
            ] = "failed"

        _write_state(
            base_dir,
            state,
        )

        return state


def reconcile_interrupted_pipeline(
    base_dir,
):
    """
    Called once on backend startup.

    The reconstruction pipeline only runs
    in-process (a BackgroundTask that spawns a
    child subprocess). If the backend restarts,
    any state still marked "running" is an
    orphan from a process that no longer exists
    - e.g. the terminal was closed mid-run.

    Clearing it here means the next upload
    starts a fresh run at 0% instead of being
    rejected with "a reconstruction is already
    running" or resuming a dead job's progress.

    Returns the reconciled state, or the
    unchanged state if nothing was running.
    """

    with PIPELINE_LOCK:
        state = get_pipeline_state(
            base_dir
        )

        if (
            state.get("status")
            != "running"
        ):
            return state

        current_stage = state.get(
            "current_stage"
        )

        state["status"] = "failed"

        state["progress"] = 0

        state["message"] = (
            "Reconstruction interrupted "
            "(backend restarted). Upload "
            "again to start a fresh run."
        )

        state["error"] = (
            "Interrupted: backend restarted "
            "while reconstruction was running."
        )

        state["completed_at"] = (
            datetime.now().isoformat()
        )

        if current_stage:
            state["stages"][
                current_stage
            ] = "failed"

        _write_state(
            base_dir,
            state,
        )

        return state