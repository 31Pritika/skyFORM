import { useEffect, useState } from "react";

export default function RequirementsStatus({ jobId }) {
  const [report, setReport] = useState(null);
  useEffect(() => {
    const controller = new AbortController();
    fetch(`http://127.0.0.1:8000/outputs/jobs/${encodeURIComponent(jobId)}/final/pipeline_report.json`, { signal: controller.signal, cache: "no-store" })
      .then((response) => response.ok ? response.json() : null).then(setReport)
      .catch((error) => { if (error.name !== "AbortError") console.error(error); });
    return () => controller.abort();
  }, [jobId]);
  if (!report) return null;
  const files = report.requirements?.formats_available || [];
  const formats = [".obj", ".ply", ".las", ".tif", ".glb", ".gltf", ".fbx"];
  const completeFormats = formats.every((extension) => files.some((file) => file.endsWith(extension)));
  const rows = [
    ["Mesh / point cloud", report.dense_reconstruction?.mesh_available ? "Available · mesh is experimental" : "Point cloud available; mesh unavailable"],
    ["10-minute video in <15 minutes", `Not verified · this run took ${Math.round(report.runtime?.total_seconds || 0)} seconds`],
    ["Spatial accuracy ≤1 m", report.checkpoint_validation ? `${report.checkpoint_validation.all_checkpoints_within_1m ? "Within 1 m" : "Exceeds 1 m"} at supplied checkpoints only` : "Unverified · independent measurements needed"],
    ["Entire visible scene", "Unverified · unsupported regions may be missing"],
    ["Output formats", completeFormats ? "OBJ, PLY, LAS, GeoTIFF, GLB, glTF, FBX available" : `Available: ${formats.filter((extension) => files.some((file) => file.endsWith(extension))).join(", ")}`],
    ["GeoTIFF coordinates", report.georeferencing?.georeferenced ? "GPS-aligned · accuracy not independently validated" : "Local, unscaled coordinates · no geographic location"],
    ["Web visualization", "Point Cloud, Mesh, Confidence, Camera Path"],
  ];
  return <section className="video-info-section"><div className="section-heading-row"><div><p className="section-label">VALIDATION</p><h3>Requirement status</h3></div></div>
    <table className="requirements-table"><tbody>{rows.map(([requirement,status]) => <tr key={requirement}><th scope="row">{requirement}</th><td>{status}</td></tr>)}</tbody></table>
  </section>;
}
