import csv
import json
import math
import sys
from pathlib import Path

from pipeline_config import (
    get_input_dir,
    get_georeferencing_dir,
    ensure_job_directories,
)


REQUIRED_COLUMNS = {
    "timestamp",
    "latitude",
    "longitude",
    "altitude",
}


def validate_rows(rows):
    previous_time = None
    for index, row in enumerate(rows, start=2):
        try:
            timestamp, latitude, longitude, altitude = [float(row[key]) for key in
                                                       ("timestamp", "latitude", "longitude", "altitude")]
        except (ValueError, TypeError, KeyError) as error:
            raise RuntimeError(f"Invalid numeric telemetry at row {index}") from error
        if not all(math.isfinite(value) for value in (timestamp, latitude, longitude, altitude)):
            raise RuntimeError(f"Non-finite telemetry at row {index}")
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise RuntimeError(f"Coordinates out of range at row {index}")
        if previous_time is not None and timestamp <= previous_time:
            raise RuntimeError(f"Timestamps must increase at row {index}")
        previous_time = timestamp



def main():
    ensure_job_directories()

    telemetry_path = get_input_dir() / "telemetry.csv"
    output_path = (
        get_georeferencing_dir()
        / "telemetry_validation.json"
    )

    result = {
        "telemetry_present": False,
        "schema_valid": False,
        "samples": 0,
        "georeferencing_ready": False,
    }

    print("\n========================================")
    print(" SKYFORM - TELEMETRY VALIDATION")
    print("========================================")

    if not telemetry_path.exists():
        print("No telemetry.csv supplied.")
        print("Georeferencing remains disabled.")

        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        return

    result["telemetry_present"] = True

    with open(telemetry_path, newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise RuntimeError(
                "Telemetry file has no header."
            )

        columns = {
            column.strip()
            for column in reader.fieldnames
        }

        missing = REQUIRED_COLUMNS - columns

        if missing:
            raise RuntimeError(
                "Missing telemetry columns: "
                + ", ".join(sorted(missing))
            )

        rows = [{key.strip(): value for key, value in row.items()} for row in reader]

    validate_rows(rows)

    result["schema_valid"] = True
    result["samples"] = len(rows)
    result["georeferencing_ready"] = len(rows) >= 3
    if len(rows) >= 3:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from app.services.job_georeferencing import georeference_job
        from pipeline_config import BACKEND_DIR, get_job_id
        try:
            alignment = georeference_job(BACKEND_DIR, get_job_id(), {"points": [{key: float(row[key]) for key in REQUIRED_COLUMNS} for row in rows]})
            result["aligned"] = True
            result["alignment_rmse_m"] = alignment["alignment_rmse_m"]
        except (ValueError, RuntimeError, OSError) as error:
            result["aligned"] = False
            result["georeferencing_error"] = str(error)
            print("Georeferencing unavailable:", error)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print("Telemetry schema: VALID")
    print("Telemetry samples:", len(rows))

    if rows:
        print(
            "Telemetry inspection and alignment attempt completed; see validation JSON."
        )
    else:
        print(
            "No GPS samples supplied. "
            "Georeferencing remains disabled."
        )

    print("========================================\n")


if __name__ == "__main__":
    main()
