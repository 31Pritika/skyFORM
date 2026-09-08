from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import traceback

from app.services.pipeline_service import (
    mark_pipeline_started,
    mark_pipeline_complete,
    mark_pipeline_failed,
    update_pipeline_state,
)


def copy_or_link(source, destination):
    """
    Prefer a hard link so large uploaded videos are not duplicated.
    Fall back to a normal copy when linking is unavailable.
    """
    source = Path(source)
    destination = Path(destination)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination.exists():
        destination.unlink()

    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def copy_if_exists(source, destination):
    source = Path(source)
    destination = Path(destination)

    if not source.exists():
        return False

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        source,
        destination,
    )

    return True


def load_json(path):
    path = Path(path)

    if not path.exists():
        return {}

    try:
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)
    except Exception:
        return {}


def run_reconstruction_pipeline(
    base_dir,
    video_path,
):
    base_dir = Path(base_dir)
    video_path = Path(video_path)

    video_id = video_path.stem

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video does not exist: {video_path}"
        )

    # =========================================================
    # JOB PATHS
    # =========================================================

    job_dir = (
        base_dir
        / "outputs"
        / "jobs"
        / video_id
    )

    input_dir = job_dir / "input"
    final_dir = job_dir / "final"

    input_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Start every reconstruction from a clean job workspace.
    # Keep the input directory we just created.
    for child in list(job_dir.iterdir()):
        if child == input_dir:
            continue

        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    # Remove stale input from a previous run of this same ID.
    for old_input in input_dir.iterdir():
        if old_input.is_file():
            old_input.unlink()

    job_video = (
        input_dir
        / video_path.name
    )

    copy_or_link(
        video_path,
        job_video,
    )

    # =========================================================
    # OPTIONAL TELEMETRY BRIDGE
    # =========================================================

    # Existing frontend/API stores telemetry here:
    legacy_gps = (
        base_dir
        / "outputs"
        / "telemetry"
        / video_id
        / "gps.csv"
    )

    if legacy_gps.exists():
        shutil.copy2(
            legacy_gps,
            input_dir / "telemetry.csv",
        )

    # =========================================================
    # LEGACY VIEWER PATHS
    # =========================================================

    reconstruction_dir = (
        base_dir
        / "outputs"
        / "reconstruction"
    )

    reconstruction_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    legacy_cloud = (
        reconstruction_dir
        / "dense_fused.ply"
    )

    legacy_mesh = (
        reconstruction_dir
        / "mesh.ply"
    )

    # Never display an older reconstruction while a new one runs.
    for stale in (
        legacy_cloud,
        legacy_mesh,
    ):
        if stale.exists():
            stale.unlink()

    # =========================================================
    # PIPELINE START
    # =========================================================

    mark_pipeline_started(
        base_dir,
        video_id=video_id,
    )

    try:
        update_pipeline_state(
            base_dir,
            progress=5,
            current_stage="frame_intelligence",
            message=(
                "Preparing video and extracting "
                "reconstruction frames."
            ),
            stage="frame_intelligence",
            stage_status="running",
        )

        runner = (
            base_dir
            / "new_pipeline"
            / "scripts"
            / "run_pipeline.py"
        )

        if not runner.exists():
            raise FileNotFoundError(
                f"New pipeline runner not found: {runner}"
            )

        env = os.environ.copy()
        env["SKYFORM_JOB_ID"] = video_id
        env["PYTHONUNBUFFERED"] = "1"

        update_pipeline_state(
            base_dir,
            progress=10,
            message=(
                "SkyFORM reconstruction pipeline started."
            ),
        )

        # -----------------------------------------------------
        # RUN NEW PIPELINE
        # -----------------------------------------------------

        process = subprocess.Popen(
            [
                sys.executable,
                str(runner),
            ],
            cwd=str(
                base_dir
                / "new_pipeline"
                / "scripts"
            ),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        if process.stdout is not None:
            for line in process.stdout:
                if line.startswith("SKYFORM_STAGE "):
                    event = json.loads(line.removeprefix("SKYFORM_STAGE "))
                    stage_map = {1: "frame_intelligence", 2: "frame_intelligence",
                                 3: "feature_extraction", 4: "camera_reconstruction",
                                 5: "depth_estimation", 6: "depth_fusion",
                                 7: "mesh_generation", 8: "quality_analysis",
                                 9: "quality_analysis", 10: "quality_analysis"}
                    stage = stage_map[event["index"]]
                    update_pipeline_state(base_dir, progress=min(88, event["index"] * 8),
                                          current_stage=stage, stage=stage,
                                          stage_status=event["status"], message=event["name"])
                print(
                    "[SkyFORM]",
                    line.rstrip(),
                    flush=True,
                )

        return_code = process.wait()

        if return_code != 0:
            raise RuntimeError(
                "New SkyFORM reconstruction "
                f"pipeline failed with code {return_code}."
            )

        # =====================================================
        # VERIFY PRIMARY RESULT
        # =====================================================

        primary_cloud = (
            final_dir
            / "skyform_pointcloud.ply"
        )

        if not primary_cloud.exists():
            raise RuntimeError(
                "Pipeline completed but final "
                "point cloud was not produced."
            )

        # =====================================================
        # COMPATIBILITY BRIDGE
        # =====================================================

        update_pipeline_state(
            base_dir,
            progress=90,
            current_stage="depth_fusion",
            message=(
                "Publishing reconstructed geometry "
                "to the SkyFORM viewer."
            ),
            stage="depth_fusion",
            stage_status="running",
        )

        shutil.copy2(
            primary_cloud,
            legacy_cloud,
        )

        # Mesh is OPTIONAL.
        mesh_candidates = [
            final_dir
            / "skyform_mesh_experimental.ply",

            job_dir
            / "mesh"
            / "skyform_mesh.ply",
        ]

        mesh_published = False

        for candidate in mesh_candidates:
            if candidate.exists():
                shutil.copy2(
                    candidate,
                    legacy_mesh,
                )
                mesh_published = True
                break

        update_pipeline_state(
            base_dir,
            progress=94,
            message=(
                "Dense point cloud published."
            ),
            stage="depth_fusion",
            stage_status="completed",
        )

        # =====================================================
        # CAMERA / COLMAP COMPATIBILITY
        # =====================================================

        # status_service historically searches colmap_new/sparse.
        # Mirror the selected job sparse model there so existing
        # metrics continue to work without model_converter.
        job_sparse_root = (
            job_dir
            / "sfm"
            / "sparse"
        )

        legacy_sparse_root = (
            base_dir
            / "outputs"
            / "colmap_new"
            / "sparse"
        )

        if legacy_sparse_root.exists():
            shutil.rmtree(
                legacy_sparse_root
            )

        if job_sparse_root.exists():
            shutil.copytree(
                job_sparse_root,
                legacy_sparse_root,
            )

        # Store new camera pose JSON in a stable legacy location.
        job_pose_json = (
            job_dir
            / "colmap_poses"
            / "colmap_poses.json"
        )

        legacy_pose_json = (
            reconstruction_dir
            / "camera_poses.json"
        )

        copy_if_exists(
            job_pose_json,
            legacy_pose_json,
        )

        # =====================================================
        # FINAL STATUS
        # =====================================================

        update_pipeline_state(
            base_dir,
            progress=97,
            current_stage="mesh_generation",
            message=(
                "Experimental mesh generated."
                if mesh_published
                else
                "Point cloud complete; "
                "experimental mesh unavailable."
            ),
            stage="mesh_generation",
            stage_status=(
                "completed"
                if mesh_published
                else "skipped"
            ),
        )

        update_pipeline_state(
            base_dir,
            progress=99,
            current_stage="quality_analysis",
            message=(
                "Final reconstruction report generated."
            ),
            stage="quality_analysis",
            stage_status="completed",
        )

        report = load_json(
            final_dir
            / "pipeline_report.json"
        )

        runtime = load_json(
            final_dir
            / "pipeline_runtime.json"
        )

        mark_pipeline_complete(
            base_dir
        )

        return {
            "success": True,
            "video": str(video_path),
            "video_id": video_id,
            "job_dir": str(job_dir),
            "mesh_available": mesh_published,
            "report": report,
            "runtime": runtime,
            "artifacts": {
                "dense_point_cloud":
                    str(legacy_cloud),
                "mesh": (
                    str(legacy_mesh)
                    if mesh_published
                    else None
                ),
                "job_point_cloud":
                    str(primary_cloud),
                "results_zip":
                    str(
                        final_dir
                        / "skyform_results.zip"
                    ),
            },
        }

    except Exception as error:
        print()
        print("SKYFORM PIPELINE FAILED")
        traceback.print_exc()

        mark_pipeline_failed(
            base_dir,
            str(error),
        )

        raise
