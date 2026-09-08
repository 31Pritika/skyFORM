from app.services.feature_service import (
    analyze_keyframe_sequence
)


KEYFRAME_DIR = (
    "outputs/keyframes/test"
)


results = analyze_keyframe_sequence(
    KEYFRAME_DIR
)


print(
    "\n--- SkyFORM Feature Matching ---"
)

print(
    "Keyframe pairs:",
    len(results)
)


total_matches = 0


for result in results:

    total_matches += result[
        "matches"
    ]

    print(
        f"{result['image_a']} -> "
        f"{result['image_b']} | "
        f"Features: "
        f"{result['features_a']} / "
        f"{result['features_b']} | "
        f"Matches: "
        f"{result['matches']}"
    )


average_matches = (
    total_matches / len(results)
    if results
    else 0
)


print(
    "\nAverage matches per pair:",
    round(average_matches, 2)
)