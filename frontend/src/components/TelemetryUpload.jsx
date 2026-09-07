import {
  useEffect,
  useRef,
  useState,
} from "react";

import {
  MapPin,
  Upload,
  Check,
} from "lucide-react";

import {
  getGeospatialStatus,
  uploadTelemetry,
} from "../api/videoApi";


export default function TelemetryUpload({
  videoId,
}) {
  const inputRef = useRef(null);

  const [status, setStatus] =
    useState(null);

  const [uploading, setUploading] =
    useState(false);

  const [error, setError] =
    useState("");


  const loadStatus = async () => {
    if (!videoId) {
      setStatus(null);
      return;
    }

    try {
      const data =
        await getGeospatialStatus(
          videoId
        );

      setStatus(data);
    } catch (err) {
      console.error(
        "GPS status error:",
        err
      );

      setStatus(null);
    }
  };


  useEffect(() => {
    setStatus(null);
    setError("");

    loadStatus();
  }, [videoId]);


  const handleFile = async (
    event
  ) => {
    const file =
      event.target.files?.[0];

    if (!file) {
      return;
    }

    if (!videoId) {
      setError(
        "Upload a video before adding GPS telemetry."
      );

      return;
    }

    try {
      setUploading(true);
      setError("");

      await uploadTelemetry(
        videoId,
        file
      );

      await loadStatus();

    } catch (err) {
      setError(
        err.response?.data?.detail ||
          "Telemetry upload failed."
      );

    } finally {
      setUploading(false);

      if (inputRef.current) {
        inputRef.current.value =
          "";
      }
    }
  };


  const ready =
    Boolean(
      status?.available
    );

  const aligned =
    status?.coordinate_system ===
    "local_enu_meters";


  return (
    <div className="telemetry-upload">

      <div className="telemetry-header">

        <MapPin size={17} />

        <div>
          <strong>
            Geospatial alignment
          </strong>

          <span>
            {!videoId
              ? "Upload a video before adding GPS telemetry"

              : ready
                ? aligned
                  ? `${status.telemetry?.point_count ?? 0} GPS samples · metric ENU aligned`

                  : `${status.telemetry?.point_count ?? 0} GPS samples loaded · alignment pending`

                : "Relative coordinate frame — GPS not supplied"}
          </span>
        </div>

      </div>


      {ready ? (
        <div className="telemetry-ready">
          <Check size={14} />

          {aligned
            ? "GPS ALIGNED"
            : "TELEMETRY AVAILABLE"}
        </div>

      ) : (
        <>
          <input
            ref={inputRef}
            type="file"
            accept=".csv,text/csv"
            hidden
            onChange={
              handleFile
            }
          />

          <button
            className="telemetry-button"
            disabled={
              uploading ||
              !videoId
            }
            onClick={() =>
              inputRef.current?.click()
            }
          >
            <Upload size={13} />

            {uploading
              ? "Uploading..."
              : "Add GPS CSV"}
          </button>
        </>
      )}


      {error && (
        <span className="telemetry-error">
          {error}
        </span>
      )}

    </div>
  );
}