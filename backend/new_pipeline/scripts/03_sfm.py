import shutil
import subprocess

from pipeline_config import (
    get_frames_dir,
    get_sfm_dir,
    ensure_job_directories,
)


def run(command):
    print(
        "\n>",
        " ".join(str(x) for x in command),
    )

    subprocess.run(
        [str(x) for x in command],
        check=True,
    )


def main():
    ensure_job_directories()

    frames_dir = get_frames_dir()
    sfm_dir = get_sfm_dir()

    database = sfm_dir / "database.db"
    sparse_dir = sfm_dir / "sparse"

    # ---------------------------------------------------------
    # VALIDATE FRAMES
    # ---------------------------------------------------------

    frames = sorted(
        path
        for path in frames_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in {
            ".jpg",
            ".jpeg",
            ".png",
        }
    )

    if len(frames) < 2:
        raise RuntimeError(
            f"Not enough frames for COLMAP: {len(frames)}"
        )

    # ---------------------------------------------------------
    # CLEAN ONLY THIS JOB'S PREVIOUS SFM RESULT
    # ---------------------------------------------------------

    if sfm_dir.exists():
        shutil.rmtree(sfm_dir)

    sparse_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n==============================")
    print(" SKYFORM - COLMAP SFM")
    print("==============================")
    print("Frames:", len(frames))
    print("Frame directory:", frames_dir)
    print("SfM directory:", sfm_dir)

    # ---------------------------------------------------------
    # 1. FEATURE EXTRACTION
    # ---------------------------------------------------------

    print("\n[1/3] Extracting features...")

    run([
        "colmap",
        "feature_extractor",

        "--database_path",
        database,

        "--image_path",
        frames_dir,

        "--ImageReader.single_camera",
        "1",

        "--ImageReader.camera_model",
        "SIMPLE_RADIAL",
    ])

    # ---------------------------------------------------------
    # 2. SEQUENTIAL MATCHING
    # ---------------------------------------------------------

    print(
        "\n[2/3] Matching sequential video frames..."
    )

    run([
        "colmap",
        "sequential_matcher",

        "--database_path",
        database,
    ])

    # ---------------------------------------------------------
    # 3. SPARSE RECONSTRUCTION
    # ---------------------------------------------------------

    print(
        "\n[3/3] Reconstructing cameras..."
    )

    run([
        "colmap",
        "mapper",

        "--database_path",
        database,

        "--image_path",
        frames_dir,

        "--output_path",
        sparse_dir,
    ])

    # ---------------------------------------------------------
    # FIND GENERATED MODELS
    # ---------------------------------------------------------

    models = sorted(
        path
        for path in sparse_dir.iterdir()
        if path.is_dir()
        and (path / "cameras.bin").exists()
        and (path / "images.bin").exists()
        and (path / "points3D.bin").exists()
    )

    if not models:
        raise RuntimeError(
            "COLMAP did not produce a valid sparse model."
        )

    print("\n==============================")
    print(" SFM COMPLETE")
    print("==============================")

    print(
        "Sparse models produced:",
        len(models),
    )

    for model in models:
        print("Model:", model)

    print("==============================\n")


if __name__ == "__main__":
    main()
