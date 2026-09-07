const stages = [
  {
    key: "video_ingestion",
    number: "01",
    title: "Video ingestion",
    pending: "Awaiting footage",
    completed: "Footage validated + ingested",
  },
  {
    key: "frame_intelligence",
    number: "02",
    title: "Frame intelligence",
    pending: "Quality + keyframe analysis",
    completed: "Keyframes selected",
  },
  {
    key: "spatial_computation",
    number: "03",
    title: "Spatial computation",
    pending: "Feature matching + camera poses",
    completed: "Camera poses registered",
  },
  {
    key: "reconstruction",
    number: "04",
    title: "3D reconstruction",
    pending: "Point cloud + mesh",
    completed: "Dense cloud + mesh generated",
  },
  {
    key: "geospatial_alignment",
    number: "05",
    title: "Geospatial alignment",
    pending: "Relative coordinate frame",
    completed: "GPS aligned",
  },
];


function PipelineStatus({ pipeline }) {
  return (
    <div className="pipeline-list">

      {stages.map(
        (stage, index) => {

          const isCompleted =
            Boolean(
              pipeline?.[
                stage.key
              ]
            );

          return (
            <div
              className={`pipeline-item ${
                isCompleted
                  ? "completed"
                  : ""
              }`}
              key={stage.key}
            >

              <div className="pipeline-marker">

                <span>
                  {isCompleted
                    ? "✓"
                    : stage.number}
                </span>

                {index !==
                  stages.length - 1 && (
                  <div
                    className="pipeline-connector"
                  />
                )}

              </div>


              <div className="pipeline-copy">

                <strong>
                  {stage.title}
                </strong>

                <span>
                  {isCompleted
                    ? stage.completed
                    : stage.pending}
                </span>

              </div>

            </div>
          );
        }
      )}

    </div>
  );
}


export default PipelineStatus;