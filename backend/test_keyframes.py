from app.services.frame_service import (
    select_keyframes,
    save_selected_keyframes
)


# NEW video you just uploaded
video_path = (
    "uploads/"
    "8fbbff64-ada0-4d46-88ee-2fc5cf4b6d25.mp4"
)


# Analyze the new video
result = select_keyframes(
    video_path=video_path,
    sample_interval=5,
    sharpness_threshold=50.0,
    difference_threshold=5.0
)


print("\n--- SkyFORM Keyframe Analysis ---")

print(
    "Total frames:",
    result["total_frames"]
)

print(
    "Analyzed frames:",
    result["analyzed_frames"]
)

print(
    "Selected keyframes:",
    result["selected_count"]
)


# Save NEW video's keyframes separately
saved = save_selected_keyframes(
    video_path=video_path,
    keyframes=result["keyframes"],
    output_dir="outputs/keyframes/new_test"
)


print(
    "Saved keyframes:",
    len(saved)
)

print(
    "Location: outputs/keyframes/new_test"
)