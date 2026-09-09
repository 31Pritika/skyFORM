import os
import shutil
import struct
import subprocess

from pipeline_config import (
    get_frames_dir,
    get_sfm_dir,
    ensure_job_directories,
)


# ============================================================
# CONFIG
# ============================================================

# At or below this frame count we can afford exhaustive matching,
# which is dramatically more robust than sequential-only matching
# for shaky / low-overlap drone video. Above it we fall back to a
# generous sequential window.
EXHAUSTIVE_MATCH_LIMIT = int(
    os.environ.get("SKYFORM_SFM_EXHAUSTIVE_LIMIT", "250")
)

# A reconstruction that registers fewer frames than this is treated
# as a hard failure rather than silently feeding 2-4 frames into the
# rest of the pipeline.
MIN_REGISTERED_IMAGES = int(
    os.environ.get("SKYFORM_SFM_MIN_IMAGES", "8")
)

# ...and it must also cover at least this fraction of the frames.
MIN_REGISTERED_FRACTION = float(
    os.environ.get("SKYFORM_SFM_MIN_FRACTION", "0.35")
)

# Below this fraction we keep going but print a prominent warning.
WARN_REGISTERED_FRACTION = 0.60


# Robust feature extraction. affine-shape estimation + domain-size
# pooling produce far more repeatable descriptors on low-resolution
# / low-texture footage, at the cost of CPU time (this is never the
# pipeline bottleneck - MASt3R fusion is). The two expensive options
# are skipped for very large frame sets to keep extraction bounded.
HEAVY_SIFT_FRAME_LIMIT = int(
    os.environ.get("SKYFORM_SFM_HEAVY_SIFT_LIMIT", "400")
)

# --- Low-memory profile -------------------------------------------------
#
# This COLMAP is built without CUDA, so SIFT extraction and matching run
# on the CPU. On a small box (a few GB of RAM - typical for WSL2 and
# entry cloud instances) the "robust" defaults below - 16k features per
# image, affine-shape estimation, domain-size pooling, guided matching,
# 32k matches per pair, one worker thread per core - drive a single
# colmap process well past 5 GB of virtual memory. The kernel OOM-killer
# then kills it partway through this stage and the whole pipeline dies at
# ~24%.
#
# SKYFORM_SFM_LOW_MEM (default on) caps every one of those knobs to
# something that fits in ~2 GB. Set SKYFORM_SFM_LOW_MEM=0 to restore the
# previous, heavier behaviour on a machine that can afford it.
LOW_MEM = os.environ.get("SKYFORM_SFM_LOW_MEM", "1") != "0"

# Longest image edge handed to the SIFT extractor. Drone frames are
# usually 1920x1080; 1024 still gives COLMAP plenty of texture to solve
# camera poses (the dense detail comes from stage 05 MASt3R fusion, which
# has its own SKYFORM_FUSION_IMAGE_SIZE knob) while cutting the
# extractor's pyramid memory to roughly a third of the 1920px cost.
SFM_MAX_IMAGE_SIZE = os.environ.get(
    "SKYFORM_SFM_MAX_IMAGE_SIZE", "1024" if LOW_MEM else "3200"
)

# Worker threads for the CPU extractor / matcher. Each thread carries its
# own image pyramid + descriptor buffers, so "-1" (all cores) multiplies
# peak RSS by the core count. On a ~4 GB box shared with an editor /
# language servers, 2 is the ceiling that reliably avoids the OOM killer.
SFM_NUM_THREADS = os.environ.get(
    "SKYFORM_SFM_NUM_THREADS", "2" if LOW_MEM else "-1"
)

MAX_NUM_FEATURES = "4096" if LOW_MEM else "16384"
MAX_NUM_MATCHES = "8192" if LOW_MEM else "32768"


def feature_extractor_args(frame_count):
    args = [
        "--ImageReader.single_camera", "1",
        # NOTE: stage 05 fusion (make_camera_rays) currently requires
        # SIMPLE_RADIAL. Do not change this without updating stage 05.
        "--ImageReader.camera_model", "SIMPLE_RADIAL",
        # Built without CUDA - be explicit so COLMAP never tries to spin
        # up a GL/GPU SIFT context under offscreen Qt.
        "--SiftExtraction.use_gpu", "0",
        "--SiftExtraction.num_threads", SFM_NUM_THREADS,
        "--SiftExtraction.max_image_size", SFM_MAX_IMAGE_SIZE,
        "--SiftExtraction.max_num_features", MAX_NUM_FEATURES,
        "--SiftExtraction.edge_threshold", "16",
        "--SiftExtraction.peak_threshold", "0.00333",
    ]

    # affine-shape estimation + domain-size pooling roughly triple the
    # extractor's memory and CPU cost. Worth it on a big machine; on a
    # low-mem CPU box they are exactly what tips COLMAP into the OOM
    # killer, so they stay off whenever the low-mem profile is active.
    if not LOW_MEM and frame_count <= HEAVY_SIFT_FRAME_LIMIT:
        args += [
            "--SiftExtraction.estimate_affine_shape", "1",
            "--SiftExtraction.domain_size_pooling", "1",
        ]

    return args

MATCHER_COMMON_ARGS = [
    "--SiftMatching.use_gpu", "0",
    "--SiftMatching.num_threads", SFM_NUM_THREADS,
    "--SiftMatching.guided_matching", "0" if LOW_MEM else "1",
    "--SiftMatching.max_ratio", "0.85",
    "--SiftMatching.max_num_matches", MAX_NUM_MATCHES,
]

SEQUENTIAL_MATCHER_ARGS = [
    "--SequentialMatching.overlap", "20",
    "--SequentialMatching.quadratic_overlap", "1",
]

# Mapper passes, applied in order. Each pass runs the incremental
# mapper again over the same database; COLMAP appends any new
# sub-models it can build. The first pass uses mildly relaxed
# thresholds, later passes get progressively more permissive so
# stragglers still get registered.
MAPPER_PASSES = [
    [
        "--Mapper.init_min_num_inliers", "30",
        "--Mapper.abs_pose_min_num_inliers", "20",
        "--Mapper.abs_pose_min_inlier_ratio", "0.10",
        "--Mapper.min_num_matches", "10",
        "--Mapper.filter_max_reproj_error", "5",
        "--Mapper.max_reg_trials", "4",
        "--Mapper.ba_global_max_num_iterations", "75",
    ],
    [
        "--Mapper.init_min_num_inliers", "15",
        "--Mapper.abs_pose_min_num_inliers", "12",
        "--Mapper.abs_pose_min_inlier_ratio", "0.05",
        "--Mapper.min_num_matches", "6",
        "--Mapper.filter_max_reproj_error", "6",
        "--Mapper.max_reg_trials", "6",
        "--Mapper.init_max_reg_trials", "5",
        "--Mapper.ba_global_max_num_iterations", "100",
    ],
]


# ============================================================
# HELPERS
# ============================================================

def run(command, check=True):
    printable = " ".join(str(x) for x in command)
    print("\n>", printable)

    result = subprocess.run(
        [str(x) for x in command],
        check=False,
    )

    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {printable}"
        )

    return result.returncode


def registered_image_count(model_dir):
    images_bin = model_dir / "images.bin"

    if not images_bin.exists():
        return 0

    try:
        with open(images_bin, "rb") as f:
            head = f.read(8)

        if len(head) != 8:
            return 0

        return struct.unpack("<Q", head)[0]

    except Exception:
        return 0


def valid_models(sparse_dir):
    models = []

    for model_dir in sorted(sparse_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        # Scratch directories used while running the mapper passes.
        if model_dir.name.startswith("_"):
            continue

        required = [
            model_dir / "cameras.bin",
            model_dir / "images.bin",
            model_dir / "points3D.bin",
        ]

        if not all(path.exists() for path in required):
            continue

        count = registered_image_count(model_dir)

        if count > 0:
            models.append((count, model_dir))

    models.sort(key=lambda item: item[0], reverse=True)

    return models


def merge_models(sparse_dir, models):
    """Fold every sub-model into the largest one via colmap model_merger.

    COLMAP's incremental mapper often fragments shaky video into several
    partial reconstructions. Picking only the biggest one throws away
    every frame in the others. Merging keeps them when they share enough
    overlap; when a merge does not help we simply keep the best input.
    """
    if len(models) < 2:
        return

    merged_dir = sparse_dir / "merged"

    if merged_dir.exists():
        shutil.rmtree(merged_dir)

    merged_dir.mkdir(parents=True, exist_ok=True)

    # Seed with the largest model.
    best_count, best_dir = models[0]

    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        shutil.copy2(best_dir / name, merged_dir / name)

    current_count = best_count

    for count, model_dir in models[1:]:
        code = run(
            [
                "colmap", "model_merger",
                "--input_path1", merged_dir,
                "--input_path2", model_dir,
                "--output_path", merged_dir,
                "--max_reproj_error", "8",
            ],
            check=False,
        )

        if code != 0:
            continue

        new_count = registered_image_count(merged_dir)

        if new_count > current_count:
            print(
                f"  merged {model_dir.name}: "
                f"{current_count} -> {new_count} images"
            )
            current_count = new_count
        else:
            print(
                f"  {model_dir.name} did not merge cleanly; keeping it separate."
            )

    if current_count <= best_count:
        # Nothing was gained - drop the merged copy so it does not
        # masquerade as a distinct model downstream.
        shutil.rmtree(merged_dir)
        print("  model merge produced no improvement.")


# ============================================================
# MAIN
# ============================================================

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

    frame_count = len(frames)

    if frame_count < 2:
        raise RuntimeError(
            f"Not enough frames for COLMAP: {frame_count}"
        )

    # ---------------------------------------------------------
    # OPTIONAL: DYNAMIC-OBJECT FILTERING
    # ---------------------------------------------------------

    # With SKYFORM_FILTER_DYNAMIC=1, mask moving objects (vehicles,
    # people, animals) out of the frames before feature extraction so
    # they cannot pollute multi-view matches / camera poses. Frames are
    # edited in place, so stage 05 fusion (same frame dir) benefits too.
    # Non-fatal by design: a detector failure must not sink the run.

    if os.environ.get("SKYFORM_FILTER_DYNAMIC") == "1":
        print("\n[dynamic-object filtering enabled]")
        try:
            from dynamic_filter import filter_dynamic_frames
            filter_dynamic_frames(frames_dir)
        except Exception as exc:
            print(f"[WARNING] Dynamic-object filtering failed: {exc}")
            print("Continuing with unfiltered frames.")
    else:
        print(
            "\n[dynamic-object filtering disabled "
            "(set SKYFORM_FILTER_DYNAMIC=1 to enable)]"
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
    print("Frames:", frame_count)
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

        *feature_extractor_args(frame_count),
    ])

    # ---------------------------------------------------------
    # 2. MATCHING
    # ---------------------------------------------------------

    if frame_count <= EXHAUSTIVE_MATCH_LIMIT:
        print(
            f"\n[2/3] Exhaustive matching ({frame_count} frames)..."
        )

        run([
            "colmap",
            "exhaustive_matcher",

            "--database_path",
            database,

            *MATCHER_COMMON_ARGS,
        ])

    else:
        print(
            f"\n[2/3] Sequential matching ({frame_count} frames)..."
        )

        run([
            "colmap",
            "sequential_matcher",

            "--database_path",
            database,

            *SEQUENTIAL_MATCHER_ARGS,
            *MATCHER_COMMON_ARGS,
        ])

    # ---------------------------------------------------------
    # 3. SPARSE RECONSTRUCTION
    # ---------------------------------------------------------

    print("\n[3/3] Reconstructing cameras...")

    best_count = 0
    collected = 0

    for index, mapper_args in enumerate(MAPPER_PASSES, start=1):
        print(
            f"\n  Mapper pass {index}/{len(MAPPER_PASSES)}"
        )

        # Give each pass its own output directory so COLMAP always
        # starts numbering models from 0 and never clobbers or skips
        # a previous pass's results.
        pass_dir = sparse_dir / f"_pass_{index}"

        if pass_dir.exists():
            shutil.rmtree(pass_dir)

        pass_dir.mkdir(parents=True, exist_ok=True)

        run(
            [
                "colmap",
                "mapper",

                "--database_path",
                database,

                "--image_path",
                frames_dir,

                "--output_path",
                pass_dir,

                "--Mapper.multiple_models",
                "1",

                # Bound bundle-adjustment memory on small CPU boxes (see
                # the low-mem profile notes near the top of this file).
                "--Mapper.num_threads",
                SFM_NUM_THREADS,

                *mapper_args,
            ],
            check=False,
        )

        # Promote every model this pass produced into a flat,
        # sequentially numbered slot under sparse/.
        for produced in sorted(pass_dir.iterdir()):
            if not produced.is_dir():
                continue

            if not all(
                (produced / name).exists()
                for name in (
                    "cameras.bin",
                    "images.bin",
                    "points3D.bin",
                )
            ):
                continue

            destination = sparse_dir / str(collected)
            shutil.move(str(produced), str(destination))
            collected += 1

        shutil.rmtree(pass_dir, ignore_errors=True)

        models = valid_models(sparse_dir)
        best_count = models[0][0] if models else 0

        print(
            f"  After pass {index}: "
            f"{len(models)} model(s), "
            f"best = {best_count}/{frame_count} images"
        )

        if best_count >= max(
            MIN_REGISTERED_IMAGES,
            int(WARN_REGISTERED_FRACTION * frame_count),
        ):
            # Good enough - no need to relax further.
            break

    # ---------------------------------------------------------
    # MERGE FRAGMENTS
    # ---------------------------------------------------------

    models = valid_models(sparse_dir)

    if not models:
        raise RuntimeError(
            "COLMAP did not produce a valid sparse model."
        )

    if len(models) > 1:
        print(
            f"\nCOLMAP produced {len(models)} sub-models; "
            "attempting to merge them..."
        )
        merge_models(sparse_dir, models)
        models = valid_models(sparse_dir)

    best_count, best_model = models[0]
    registered_fraction = best_count / frame_count

    # ---------------------------------------------------------
    # REPORT + SANITY GATE
    # ---------------------------------------------------------

    print("\n==============================")
    print(" SFM COMPLETE")
    print("==============================")
    print("Sparse models produced:", len(models))

    for count, model in models:
        marker = " <- selected" if model == best_model else ""
        print(f"  {model.name}: {count} images{marker}")

    print(
        f"\nRegistered {best_count}/{frame_count} frames "
        f"({registered_fraction * 100:.1f}%)"
    )

    if (
        best_count < MIN_REGISTERED_IMAGES
        or registered_fraction < MIN_REGISTERED_FRACTION
    ):
        raise RuntimeError(
            "COLMAP camera reconstruction is too sparse: only "
            f"{best_count}/{frame_count} frames "
            f"({registered_fraction * 100:.1f}%) were registered. "
            "The rest of the pipeline would reflect only those few "
            "frames. This usually means the input video has too "
            "little parallax / overlap, heavy motion blur, or very "
            "low resolution. Try a slower, wider drone pass or a "
            "higher-resolution video."
        )

    if registered_fraction < WARN_REGISTERED_FRACTION:
        print(
            "\n[WARNING] Fewer than "
            f"{int(WARN_REGISTERED_FRACTION * 100)}% of frames were "
            "registered. Downstream stages will only reflect the "
            f"{best_count} registered frames."
        )

    print("==============================\n")


if __name__ == "__main__":
    main()
