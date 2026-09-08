import {
  useEffect,
  useCallback,
  useState,
} from "react";

import {
  Upload,
  Plus,
  ShieldCheck,
  Play,
  LoaderCircle,
  Check,
  X,
} from "lucide-react";

import Sidebar from "../components/Sidebar";
import MetricCard from "../components/MetricCard";
import PipelineStatus from "../components/PipelineStatus";
import VideoUpload from "../components/VideoUpload";
import ReconstructionViewer from "../components/ReconstructionViewer";
import TelemetryUpload from "../components/TelemetryUpload";
import RequirementsStatus from "../components/RequirementsStatus";

import {
  uploadVideo,
  getPreviewUrl,
  getReconstructionStatus,
  startReconstruction,
  getPipelineExecutionStatus,
  getGeospatialStatus,
  getFlightMetadataStatus,
  saveFlightMetadata,
} from "../api/videoApi";


function Dashboard() {
  const [videoData, setVideoData] =
    useState(null);

  const [uploading, setUploading] =
    useState(false);

  const [
    uploadProgress,
    setUploadProgress,
  ] = useState(0);

  const [
    uploadError,
    setUploadError,
  ] = useState("");

  const [
    reconstruction,
    setReconstruction,
  ] = useState(null);

  const [
    pipelineExecution,
    setPipelineExecution,
  ] = useState(null);

  const [
    startingReconstruction,
    setStartingReconstruction,
  ] = useState(false);

  const [
    reconstructionError,
    setReconstructionError,
  ] = useState("");

  const [
    newProjectMode,
    setNewProjectMode,
  ] = useState(false);

  const [
    currentRunStarted,
    setCurrentRunStarted,
  ] = useState(false);


  const [
    geospatialStatus,
    setGeospatialStatus,
  ] = useState(null);

  const [
    flightMetadataStatus,
    setFlightMetadataStatus,
  ] = useState(null);

  const [
    savingFlightMetadata,
    setSavingFlightMetadata,
  ] = useState(false);

  const [
    flightMetadataMessage,
    setFlightMetadataMessage,
  ] = useState("");

  const [
    flightMetadata,
    setFlightMetadata,
  ] = useState({
    drone_model: "",
    camera_model: "",
    flight_altitude_m: "",
    speed_mps: "",
    heading_deg: "",
    yaw_deg: "",
    pitch_deg: "",
    roll_deg: "",
    focal_length_mm: "",
    sensor_width_mm: "",
    sensor_height_mm: "",
    capture_start_time: "",
    notes: "",
  });


  // =========================================================
  // LOAD EXISTING RECONSTRUCTION
  // =========================================================

  const loadReconstructionStatus =
    async () => {
      try {
        const data =
          await getReconstructionStatus();

        setReconstruction(
          data
        );
      } catch (error) {
        console.error(
          "Could not load reconstruction status:",
          error
        );
      }
    };


  const loadPipelineStatus =
    async () => {
      try {
        const data =
          await getPipelineExecutionStatus();

        setPipelineExecution(
          data
        );

        return data;
      } catch (error) {
        console.error(
          "Could not load pipeline status:",
          error
        );

        return null;
      }
    };


  const loadInputReadiness = useCallback(() => {
    const id = videoData?.id || (!newProjectMode && reconstruction?.video_id);
    if (!id) return Promise.resolve();
    return Promise.all([getGeospatialStatus(id), getFlightMetadataStatus(id)])
      .then(([geo, flight]) => {
        setGeospatialStatus(geo);
        setFlightMetadataStatus(flight);
        if (flight?.available && flight.metadata) {
          setFlightMetadata((previous) => ({ ...previous, ...flight.metadata }));
        }
      }).catch(console.error);
  }, [videoData, newProjectMode, reconstruction?.video_id]);



  useEffect(() => {
    getReconstructionStatus().then(setReconstruction).catch(console.error);
    getPipelineExecutionStatus().then(setPipelineExecution).catch(console.error);
    loadInputReadiness();

    const readinessInterval =
      setInterval(
        loadInputReadiness,
        3000
      );

    return () =>
      clearInterval(
        readinessInterval
      );
  }, [loadInputReadiness]);


  // =========================================================
  // LIVE PIPELINE POLLING
  // =========================================================

  useEffect(() => {
    if (
      pipelineExecution?.status !==
      "running"
    ) {
      return;
    }

    const interval =
      setInterval(
        async () => {
          const data =
            await loadPipelineStatus();

          if (!data) {
            return;
          }

          if (
            data.status ===
            "completed"
          ) {
            await loadReconstructionStatus();
            setNewProjectMode(false);
          }

          if (
            data.status ===
            "failed"
          ) {
            setReconstructionError(
              data.error ||
                "Reconstruction failed."
            );
          }
        },
        1000
      );

    return () =>
      clearInterval(interval);

  }, [pipelineExecution?.status]);


  // =========================================================
  // VIDEO UPLOAD
  // =========================================================

  const handleVideoUpload =
    async (file) => {
      try {
        setUploading(true);
        setUploadProgress(0);
        setUploadError("");
        setReconstructionError("");
        setNewProjectMode(true);
        setCurrentRunStarted(false);

        const result =
          await uploadVideo(
            file,
            setUploadProgress
          );

        setVideoData(
          result.video
        );

        console.log(
          "SkyFORM video:",
          result.video
        );

      } catch (error) {
        console.error(error);

        setUploadError(
          error.response?.data
            ?.detail ||
            "Video upload failed."
        );

      } finally {
        setUploading(false);
      }
    };


  // =========================================================
  // START FULL RECONSTRUCTION
  // =========================================================

  const handleStartReconstruction =
    async () => {
      if (!videoData?.id) {
        return;
      }

      try {
        setStartingReconstruction(
          true
        );

        setReconstructionError("");

        await startReconstruction(
          videoData.id
        );

        setCurrentRunStarted(true);

        // The background task may take a
        // moment to update pipeline_state.json.
        // Set local running state immediately.
        setPipelineExecution(
          (previous) => ({
            ...(previous || {}),
            status: "running",
            progress: 0,
            current_stage:
              "frame_intelligence",
            message:
              "Starting SkyFORM reconstruction...",
            error: null,
            stages: {
              frame_intelligence:
                "running",
              feature_extraction:
                "pending",
              camera_reconstruction:
                "pending",
              depth_estimation:
                "pending",
              depth_fusion:
                "pending",
              mesh_generation:
                "pending",
              quality_analysis:
                "pending",
            },
          })
        );

      } catch (error) {
        console.error(error);

        setReconstructionError(
          error.response?.data
            ?.detail ||
            "Could not start reconstruction."
        );

      } finally {
        setStartingReconstruction(
          false
        );
      }
    };


  // =========================================================
  // FLIGHT METADATA
  // =========================================================

  const handleFlightMetadataChange =
    (event) => {
      const {
        name,
        value,
      } = event.target;

      setFlightMetadata(
        (previous) => ({
          ...previous,
          [name]: value,
        })
      );
    };


  const handleSaveFlightMetadata =
    async (event) => {
      event.preventDefault();

      try {
        setSavingFlightMetadata(true);
        setFlightMetadataMessage("");

        const payload = {};

        Object.entries(
          flightMetadata
        ).forEach(
          ([key, value]) => {
            if (
              value !== "" &&
              value !== null &&
              value !== undefined
            ) {
              payload[key] = value;
            }
          }
        );

        if (!videoData?.id) {
          throw new Error(
            "Upload a video before saving flight metadata."
          );
        }

        await saveFlightMetadata(
          videoData.id,
          payload
        );

        await loadInputReadiness();

        setFlightMetadataMessage(
          "Flight metadata saved."
        );
      } catch (error) {
        setFlightMetadataMessage(
          error.response?.data?.detail ||
            "Could not save flight metadata."
        );
      } finally {
        setSavingFlightMetadata(false);
      }
    };


  // =========================================================
  // HELPERS
  // =========================================================

  const formatDuration =
    (seconds) => {
      if (!seconds) {
        return "—";
      }

      const mins =
        Math.floor(
          seconds / 60
        );

      const secs =
        Math.round(
          seconds % 60
        );

      return mins > 0
        ? `${mins}m ${secs}s`
        : `${secs}s`;
    };


  const reconstructionReady =
    reconstruction?.status ===
      "ready" &&
    !newProjectMode;


  const pipelineRunning =
    pipelineExecution?.status ===
      "running";


  const pipelineFailed =
    currentRunStarted &&
    pipelineExecution?.status ===
      "failed";


  const pipelineCompleted =
    currentRunStarted &&
    pipelineExecution?.status ===
      "completed";


  const progress =
    pipelineExecution?.progress ?? 0;


  const gpsReady =
    Boolean(
      geospatialStatus?.available
    );

  const flightMetadataReady =
    Boolean(
      flightMetadataStatus?.available
    );

  const coordinateFrame =
    geospatialStatus?.coordinate_system ===
    "local_enu_meters"
      ? "Metric ENU"
      : gpsReady
        ? "GPS supplied"
        : "Relative / unscaled";


  // =========================================================
  // RENDER
  // =========================================================

  return (
    <div className="app-shell">

      <Sidebar exportUrl={reconstructionReady && !pipelineRunning ? "http://127.0.0.1:8000/api/reconstruction/export" : null} />


      <main className="main-content">

        {/* TOP BAR */}

        <header className="topbar">

          <div>

            <p className="eyebrow">
              RECONSTRUCTION WORKSPACE
            </p>

            <h1>
              Overview
            </h1>

          </div>


          <button
            className="new-project-button"
            onClick={() => {
              setNewProjectMode(true);
              setCurrentRunStarted(false);
              setPipelineExecution(null);
              setVideoData(null);
              setUploadError("");
              setReconstructionError("");
              setGeospatialStatus(null);
              setFlightMetadataStatus(null);
              setFlightMetadataMessage("");
            }}
          >
            <Plus size={16} />

            New reconstruction
          </button>

        </header>


        {/* ACTIVE PROJECT */}

        <section
          className="project-header"
        >

          <div>

            <p className="section-label">
              ACTIVE PROJECT
            </p>


            <h2>
              {pipelineRunning
                ? videoData
                    ?.original_filename ||
                  "SkyFORM Reconstruction"
                : reconstructionReady
                  ? "SkyFORM Reconstruction"
                  : videoData
                    ? videoData
                        .original_filename
                    : "No reconstruction loaded"}
            </h2>


            <p
              className="project-subtitle"
            >
              {pipelineRunning
                ? pipelineExecution
                    ?.message ||
                  "Reconstruction in progress."
                : reconstructionReady
                  ? "Multi-view reconstruction completed successfully."
                  : videoData
                    ? "Video ingested successfully and ready for reconstruction."
                    : "Upload drone footage to begin a reconstruction."}
            </p>

          </div>


          {(reconstructionReady ||
            videoData ||
            pipelineRunning) && (

            <div
              className="project-status"
            >

              <span
                className="status-dot"
              />

              {pipelineRunning
                ? `Processing ${progress}%`
                : reconstructionReady
                  ? "Reconstruction ready"
                  : "Video ingested"}

            </div>
          )}

        </section>


        {/* MAIN WORKSPACE */}

        <section
          className="workspace-grid"
        >

          {/* VIEWPORT */}

          <div
            className="viewport-panel"
          >

            <div
              className="viewport-toolbar"
            >

              <span>
                3D VIEWPORT
              </span>

              <span>
                {pipelineRunning
                  ? `PROCESSING ${progress}%`
                  : reconstructionReady
                    ? "MODEL READY"
                    : videoData
                      ? videoData
                          .resolution
                      : "NO DATA"}
              </span>

            </div>


            <div
              className="viewport-area"
            >

              {reconstructionReady && (
                <ReconstructionViewer key={reconstruction.video_id} jobId={reconstruction.video_id} />
                )}


              {!videoData &&
                !pipelineRunning &&
                !reconstructionReady && (

                  <div
                    className="viewport-empty"
                  >

                    <div
                      className="viewport-icon"
                    >
                      <Upload
                        size={22}
                      />
                    </div>


                    <h3>
                      No reconstruction
                      loaded
                    </h3>


                    <p>
                      Upload UAV footage
                      to initialize the
                      SkyFORM processing
                      pipeline.
                    </p>


                    <VideoUpload
                      onFileSelected={
                        handleVideoUpload
                      }
                      uploading={
                        uploading
                      }
                      progress={
                        uploadProgress
                      }
                    />


                    {uploadError && (
                      <p
                        className="upload-error"
                      >
                        {uploadError}
                      </p>
                    )}

                  </div>
                )}


              {videoData &&
                !pipelineRunning &&
                !pipelineCompleted && (

                  <div
                    className="video-loaded-content"
                  >

                    <div
                      className="video-loaded-badge"
                    >
                      VIDEO INGESTED
                    </div>


                    <h3>
                      {
                        videoData
                          .original_filename
                      }
                    </h3>


                    <p>
                      Footage is ready for
                      automatic multi-view
                      reconstruction.
                    </p>


                    <button
                      className="reconstruction-start-button"
                      onClick={
                        handleStartReconstruction
                      }
                      disabled={
                        startingReconstruction
                      }
                    >

                      {startingReconstruction
                        ? (
                          <LoaderCircle
                            size={15}
                            className="spin-icon"
                          />
                        )
                        : (
                          <Play
                            size={15}
                          />
                        )}

                      {startingReconstruction
                        ? "Starting..."
                        : "Start Reconstruction"}

                    </button>


                    <VideoUpload
                      onFileSelected={
                        handleVideoUpload
                      }
                      uploading={
                        uploading
                      }
                      progress={
                        uploadProgress
                      }
                    />


                    {reconstructionError && (
                      <p
                        className="upload-error"
                      >
                        {reconstructionError}
                      </p>
                    )}

                  </div>
                )}


              {pipelineRunning && (

                  <div
                    className="reconstruction-progress-overlay"
                  >

                    <LoaderCircle
                      size={25}
                      className="spin-icon"
                    />


                    <div
                      className="reconstruction-progress-title"
                    >
                      Building 3D reconstruction
                    </div>


                    <div
                      className="reconstruction-progress-message"
                    >
                      {pipelineExecution
                        ?.message}
                    </div>


                    <div
                      className="reconstruction-progress-track"
                    >

                      <div
                        className="reconstruction-progress-fill"
                        style={{
                          width:
                            `${progress}%`,
                        }}
                      />

                    </div>


                    <div
                      className="reconstruction-progress-value"
                    >
                      {progress}%
                    </div>

                  </div>
                )}


              {pipelineFailed && (
                <div
                  className="reconstruction-failed-overlay"
                >

                  <X size={24} />

                  <h3>
                    Reconstruction failed
                  </h3>

                  <p>
                    {reconstructionError ||
                      pipelineExecution
                        ?.error ||
                      "An unexpected processing error occurred."}
                  </p>

                </div>
              )}

            </div>

          </div>


          {/* PIPELINE */}

          <div
            className="pipeline-panel"
          >

            <div
              className="panel-heading"
            >

              <div>

                <p
                  className="section-label"
                >
                  PROCESSING
                </p>

                <h3>
                  Pipeline status
                </h3>

              </div>

            </div>


            {pipelineRunning ||
            pipelineCompleted ||
            pipelineFailed ? (

              <div
                className="execution-stage-list"
              >

                {[
                  [
                    "frame_intelligence",
                    "Frame Intelligence",
                  ],
                  [
                    "feature_extraction",
                    "Feature Extraction",
                  ],
                  [
                    "camera_reconstruction",
                    "Camera Reconstruction",
                  ],
                  [
                    "depth_estimation",
                    "AI Depth Estimation",
                  ],
                  [
                    "depth_fusion",
                    "Multi-View Fusion",
                  ],
                  [
                    "mesh_generation",
                    "Mesh Generation",
                  ],
                  [
                    "quality_analysis",
                    "Quality Analysis",
                  ],
                ].map(
                  ([key, label]) => {

                    const status =
                      pipelineExecution
                        ?.stages?.[
                          key
                        ] ||
                      "pending";

                    return (
                      <div
                        className={`execution-stage execution-stage-${status}`}
                        key={key}
                      >

                        <div
                          className="execution-stage-icon"
                        >
                          {status ===
                          "completed" ? (
                            <Check
                              size={12}
                            />
                          ) : status ===
                            "running" ? (
                            <LoaderCircle
                              size={12}
                              className="spin-icon"
                            />
                          ) : status ===
                            "failed" ? (
                            <X
                              size={12}
                            />
                          ) : (
                            <span />
                          )}
                        </div>


                        <span>
                          {label}
                        </span>

                      </div>
                    );
                  }
                )}


                <div
                  className="pipeline-progress-summary"
                >

                  <div>
                    <span>
                      Overall progress
                    </span>

                    <strong>
                      {progress}%
                    </strong>
                  </div>


                  <div
                    className="pipeline-progress-track"
                  >

                    <div
                      className="pipeline-progress-fill"
                      style={{
                        width:
                          `${progress}%`,
                      }}
                    />

                  </div>

                </div>

              </div>

            ) : (

              <PipelineStatus
                pipeline={
                  reconstructionReady
                    ? reconstruction?.pipeline
                    : null
                }
              />

            )}

          </div>

        </section>


        {/* VIDEO METADATA */}

        {videoData && (
          <section
            className="video-info-section"
          >

            <div
              className="section-heading-row"
            >

              <div>

                <p
                  className="section-label"
                >
                  INPUT ANALYSIS
                </p>

                <h3>
                  Video metadata
                </h3>

              </div>


              <span
                className="video-id"
              >
                ID:{" "}
                {videoData.id.slice(
                  0,
                  8
                )}
              </span>

            </div>


            <div
              className="metadata-grid"
            >

              <MetricCard
                label="RESOLUTION"
                value={
                  videoData.resolution
                }
                detail={`${videoData.width} × ${videoData.height}`}
              />


              <MetricCard
                label="FRAME RATE"
                value={`${videoData.fps} FPS`}
                detail="Detected stream rate"
              />


              <MetricCard
                label="TOTAL FRAMES"
                value={
                  videoData.frame_count
                }
                detail="Frames available"
              />


              <MetricCard
                label="DURATION"
                value={
                  formatDuration(
                    videoData
                      .duration_seconds
                  )
                }
                detail={`${videoData.duration_seconds}s`}
              />


              <MetricCard
                label="CODEC"
                value={
                  videoData.codec
                    ?.toUpperCase() ||
                  "—"
                }
                detail="Video encoding"
              />


              <MetricCard
                label="FILE SIZE"
                value={`${videoData.file_size_mb} MB`}
                detail="Stored locally"
              />

            </div>

          </section>
        )}


        {/* INPUT READINESS */}

        <section className="metrics-section">

          <div className="section-heading-row">

            <div>
              <p className="section-label">
                INPUT READINESS
              </p>

              <h3>
                Reconstruction inputs
              </h3>
            </div>

            <span className="metrics-status">
              {gpsReady &&
              flightMetadataReady
                ? "Required metadata supplied"
                : "Metadata incomplete"}
            </span>

          </div>


          <div className="metrics-grid">

            <MetricCard
              label="VIDEO FOOTAGE"
              value={
                videoData
                  ? "Loaded"
                  : "Missing"
              }
              detail={
                videoData
                  ? videoData.original_filename
                  : "Upload UAV footage"
              }
            />

            <MetricCard
              label="GPS TELEMETRY"
              value={
                gpsReady
                  ? "Loaded"
                  : "Missing"
              }
              detail={
                gpsReady
                  ? `${geospatialStatus?.telemetry?.point_count ?? "—"} GPS samples`
                  : "Required for georeferencing"
              }
            />

            <MetricCard
              label="FLIGHT METADATA"
              value={
                flightMetadataReady
                  ? "Loaded"
                  : "Missing"
              }
              detail={
                flightMetadataReady
                  ? "Metadata saved locally"
                  : "Required SIH input"
              }
            />

            <MetricCard
              label="COORDINATE FRAME"
              value={coordinateFrame}
              detail={
                coordinateFrame ===
                "Metric ENU"
                  ? "Geospatial alignment available"
                  : gpsReady
                    ? "GPS available; metric alignment pending"
                    : "SfM coordinates are not metric"
              }
            />

          </div>

        </section>


        {/* FLIGHT METADATA */}

        <section className="video-info-section">

          <div className="section-heading-row">

            <div>
              <p className="section-label">
                FLIGHT DATA
              </p>

              <h3>
                Flight metadata
              </h3>
            </div>

            <span className="video-id">
              {flightMetadataReady
                ? "SAVED"
                : "NOT SUPPLIED"}
            </span>

          </div>


          <form
            onSubmit={
              handleSaveFlightMetadata
            }
            style={{
              display: "grid",
              gap: "14px",
            }}
          >

            <div
              style={{
                display: "grid",
                gridTemplateColumns:
                  "repeat(auto-fit, minmax(180px, 1fr))",
                gap: "12px",
              }}
            >

              {[
                [
                  "drone_model",
                  "Drone model",
                  "text",
                ],
                [
                  "camera_model",
                  "Camera model",
                  "text",
                ],
                [
                  "flight_altitude_m",
                  "Altitude (m)",
                  "number",
                ],
                [
                  "speed_mps",
                  "Speed (m/s)",
                  "number",
                ],
                [
                  "heading_deg",
                  "Heading (°)",
                  "number",
                ],
                [
                  "focal_length_mm",
                  "Focal length (mm)",
                  "number",
                ],
              ].map(
                ([
                  name,
                  label,
                  type,
                ]) => (

                  <label
                    key={name}
                    style={{
                      display: "grid",
                      gap: "6px",
                    }}
                  >

                    <span
                      style={{
                        fontSize: "11px",
                        letterSpacing:
                          "0.08em",
                        opacity: 0.65,
                      }}
                    >
                      {label.toUpperCase()}
                    </span>

                    <input
                      name={name}
                      type={type}
                      step={
                        type === "number"
                          ? "any"
                          : undefined
                      }
                      value={
                        flightMetadata[
                          name
                        ] ?? ""
                      }
                      onChange={
                        handleFlightMetadataChange
                      }
                      style={{
                        width: "100%",
                        boxSizing:
                          "border-box",
                        padding:
                          "10px 12px",
                        border:
                          "1px solid rgba(255,255,255,0.12)",
                        borderRadius:
                          "8px",
                        background:
                          "rgba(255,255,255,0.03)",
                        color:
                          "inherit",
                        outline:
                          "none",
                      }}
                    />

                  </label>
                )
              )}

            </div>


            <label
              style={{
                display: "grid",
                gap: "6px",
              }}
            >

              <span
                style={{
                  fontSize: "11px",
                  letterSpacing:
                    "0.08em",
                  opacity: 0.65,
                }}
              >
                NOTES
              </span>

              <textarea
                name="notes"
                rows={3}
                value={
                  flightMetadata.notes
                }
                onChange={
                  handleFlightMetadataChange
                }
                placeholder="Optional flight or capture notes"
                style={{
                  width: "100%",
                  boxSizing:
                    "border-box",
                  padding:
                    "10px 12px",
                  border:
                    "1px solid rgba(255,255,255,0.12)",
                  borderRadius:
                    "8px",
                  background:
                    "rgba(255,255,255,0.03)",
                  color:
                    "inherit",
                  resize:
                    "vertical",
                  outline:
                    "none",
                }}
              />

            </label>


            <div
              style={{
                display: "flex",
                alignItems:
                  "center",
                gap: "12px",
              }}
            >

              <button
                type="submit"
                className="reconstruction-start-button"
                disabled={
                  savingFlightMetadata
                }
              >

                {savingFlightMetadata
                  ? (
                    <LoaderCircle
                      size={15}
                      className="spin-icon"
                    />
                  )
                  : (
                    <Check
                      size={15}
                    />
                  )}

                {savingFlightMetadata
                  ? "Saving..."
                  : "Save Flight Metadata"}

              </button>


              {flightMetadataMessage && (
                <span
                  style={{
                    fontSize:
                      "12px",
                    opacity:
                      0.72,
                  }}
                >
                  {
                    flightMetadataMessage
                  }
                </span>
              )}

            </div>

          </form>

        </section>


        {/* PREVIEW FRAMES */}

        {videoData
          ?.preview_frames
          ?.length > 0 && (

          <section
            className="preview-section"
          >

            <div
              className="section-heading-row"
            >

              <div>

                <p
                  className="section-label"
                >
                  FRAME SAMPLING
                </p>

                <h3>
                  Video preview
                </h3>

              </div>


              <span
                className="preview-note"
              >
                Evenly sampled frames
              </span>

            </div>


            <div
              className="preview-grid"
            >

              {videoData
                .preview_frames
                .map(
                  (frame) => (

                    <div
                      className="preview-card"
                      key={
                        frame.filename
                      }
                    >

                      <img
                        src={
                          getPreviewUrl(
                            frame.url
                          )
                        }
                        alt={`Frame ${frame.frame_number}`}
                      />


                      <div
                        className="preview-card-footer"
                      >

                        <span>
                          Frame{" "}
                          {
                            frame
                              .frame_number
                          }
                        </span>


                        <span>
                          {
                            frame
                              .filename
                          }
                        </span>

                      </div>

                    </div>
                  )
                )}

            </div>

          </section>
        )}


        {/* RECONSTRUCTION METRICS */}

        <section
          className="metrics-section"
        >

          <div
            className="section-heading-row"
          >

            <div>

              <p
                className="section-label"
              >
                RECONSTRUCTION METRICS
              </p>

              <h3>
                Spatial statistics
              </h3>

            </div>


            <span
              className="metrics-status"
            >
              {pipelineRunning
                ? `Processing ${progress}%`
                : reconstructionReady
                  ? "Reconstruction ready"
                  : "Awaiting reconstruction"}
            </span>

          </div>


          <div
            className="metrics-grid"
          >

            <MetricCard
              label="KEYFRAMES"
              value={
                reconstructionReady
                  ? reconstruction?.metrics?.keyframes ?? "—"
                  : "—"
              }
              detail="Selected reconstruction frames"
            />


            <MetricCard
              label="CAMERAS REGISTERED"
              value={
                reconstructionReady &&
                reconstruction?.metrics
                  ? `${reconstruction.metrics.registered_cameras}/${reconstruction.metrics.keyframes}`
                  : "—"
              }
              detail="COLMAP camera registration"
            />


            <MetricCard
              label="FUSED 3D POINTS"
              value={
                reconstructionReady &&
                reconstruction?.metrics?.fused_points != null
                  ? reconstruction.metrics.fused_points.toLocaleString()
                  : "—"
              }
              detail={
                reconstructionReady &&
                reconstruction?.metrics
                  ? `${reconstruction.metrics.sparse_points.toLocaleString()} sparse SfM points`
                  : "Dense reconstruction pending"
              }
            />


            <MetricCard
              label="REPROJECTION ERROR"
              value={
                reconstructionReady &&
                reconstruction?.metrics?.reprojection_error_px != null
                  ? `${reconstruction.metrics.reprojection_error_px.toFixed(
                      3
                    )} px`
                  : "—"
              }
              detail="Mean COLMAP reprojection error"
            />

          </div>

        </section>


        {reconstructionReady && <RequirementsStatus key={reconstruction.video_id} jobId={reconstruction.video_id} />}

        {/* CAPABILITIES */}

        <section
          className="capability-strip"
        >

          <div
            className="capability-item"
          >
            <TelemetryUpload
              key={videoData?.id || (!newProjectMode && reconstruction?.video_id) || "no-video"}
              videoId={videoData?.id || (!newProjectMode && reconstruction?.video_id)}
            />
          </div>


          <div
            className="capability-divider"
          />


          <div
            className="capability-item"
          >

            <ShieldCheck
              size={17}
            />


            <div>

              <strong>
                Local-first processing
              </strong>


              <span>
                Uploaded footage
                remains within the
                local SkyFORM
                processing
                environment.
              </span>

            </div>

          </div>

        </section>


      </main>

    </div>
  );
}


export default Dashboard;