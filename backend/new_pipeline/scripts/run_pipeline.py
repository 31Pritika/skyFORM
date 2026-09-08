import json
import os
from pathlib import Path
import subprocess
import sys
import time

from pipeline_config import (
    get_job_id,
    get_final_dir,
    ensure_job_directories,
)


SCRIPTS_DIR = Path(__file__).resolve().parent


STAGES = [
    ("Video probe", "01_video_probe.py", True),
    ("Frame extraction", "02_extract_frames.py", True),
    ("COLMAP SfM", "03_sfm.py", True),
    ("Read COLMAP poses", "04_read_colmap_poses.py", True),
    ("COLMAP + MASt3R fusion", "05_colmap_mast3r_fusion.py", True),
    ("Point cloud cleanup", "06_clean_pointcloud.py", True),
    ("Mesh generation", "07_generate_mesh.py", False),
    ("Final export", "08_export_results.py", True),
    ("Geo metadata inspection", "09_extract_geo_metadata.py", False),
    ("Telemetry validation", "10_validate_telemetry.py", False),
]


def format_time(seconds):
    minutes = int(seconds // 60)
    remaining = int(seconds % 60)

    if minutes:
        return f"{minutes}m {remaining}s"

    return f"{remaining}s"


def run_stage(index, name, script, critical):
    path = SCRIPTS_DIR / script

    print("\n" + "=" * 62)
    print(
        f" STAGE {index}/{len(STAGES)} - {name}"
    )
    print("=" * 62)

    start = time.perf_counter()

    print("SKYFORM_STAGE " + json.dumps({"index": index, "status": "running", "name": name}), flush=True)

    result = subprocess.run(
        [
            sys.executable,
            str(path),
        ],
        cwd=str(SCRIPTS_DIR),
        env=os.environ.copy(),
    )

    elapsed = time.perf_counter() - start

    status = (
        "PASS"
        if result.returncode == 0
        else "FAILED"
    )

    print(
        f"\n{name}: {status} "
        f"({format_time(elapsed)})"
    )

    print("SKYFORM_STAGE " + json.dumps({"index": index, "status": "completed" if result.returncode == 0 else "failed", "name": name}), flush=True)

    return {
        "name": name,
        "script": script,
        "critical": critical,
        "status": status,
        "return_code": result.returncode,
        "seconds": round(elapsed, 3),
    }


def main():
    ensure_job_directories()

    job_id = get_job_id()
    final_dir = get_final_dir()

    print("\n" + "=" * 62)
    print(" SKYFORM - FULL RECONSTRUCTION PIPELINE")
    print("=" * 62)
    print("Job:", job_id)
    print("=" * 62)

    pipeline_start = time.perf_counter()

    results = []
    critical_failure = None

    for index, stage in enumerate(
        STAGES,
        start=1,
    ):
        name, script, critical = stage

        result = run_stage(
            index,
            name,
            script,
            critical,
        )

        results.append(result)

        if (
            result["status"] == "FAILED"
            and critical
        ):
            critical_failure = name
            break

    total_seconds = (
        time.perf_counter()
        - pipeline_start
    )

    runtime = {
        "job_id": job_id,
        "runtime_recorded": True,
        "total_seconds": round(
            total_seconds,
            3,
        ),
        "formatted": format_time(
            total_seconds
        ),
        "stages": results,
        "critical_failure":
            critical_failure,
    }

    final_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    runtime_path = (
        final_dir
        / "pipeline_runtime.json"
    )

    with open(runtime_path, "w") as f:
        json.dump(
            runtime,
            f,
            indent=2,
        )

    print("\n" + "=" * 62)
    print(" PIPELINE SUMMARY")
    print("=" * 62)

    for result in results:
        print(
            f"{result['name']:<30} "
            f"{result['status']:<8} "
            f"{format_time(result['seconds'])}"
        )

    print("-" * 62)
    print(
        "Total:",
        format_time(total_seconds),
    )

    if critical_failure:
        print(
            "Critical failure:",
            critical_failure,
        )
        print("=" * 62)
        sys.exit(1)

    # Generate final report AFTER runtime has been persisted.
    report_script = (
        SCRIPTS_DIR
        / "11_pipeline_report.py"
    )

    report_result = subprocess.run(
        [
            sys.executable,
            str(report_script),
        ],
        cwd=str(SCRIPTS_DIR),
        env=os.environ.copy(),
    )

    if report_result.returncode != 0:
        print(
            "Pipeline report generation failed."
        )

    if report_result.returncode != 0:
        sys.exit(report_result.returncode)

    # Package reports and metadata only after their stages have finished.
    import zipfile
    with zipfile.ZipFile(final_dir / "skyform_results.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for artifact in sorted(final_dir.iterdir()):
            if artifact.is_file() and artifact.suffix != ".zip":
                archive.write(artifact, artifact.name)

    print("\nPIPELINE COMPLETE")
    print("Job:", job_id)
    print("Final output:", final_dir)
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
