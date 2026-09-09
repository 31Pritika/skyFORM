import { Canvas } from "@react-three/fiber";
import { OrbitControls, Line, Html } from "@react-three/drei";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";
import { useEffect, useMemo, useState } from "react";
import * as THREE from "three";

const API = "http://localhost:8000";
const modes = { points: "Point Cloud", mesh: "Mesh", confidence: "Confidence", cameras: "Camera Path" };

// COLMAP reconstructs in its own world frame: the camera convention (+X right,
// +Y down, +Z forward) is propagated to the world by the first registered view,
// so for down-looking drone footage the scene comes out roughly Z-down / Y-back.
// Three.js is Y-up, so the raw model loads upside down / on its side. A +90°
// roll about X re-seats "down" onto -Y for every source at once (point cloud,
// mesh, confidence points, camera markers + frustums all share this frame).
// If a capture instead reconstructs Y-down, use [Math.PI, 0, 0]; the group below
// is the single place to adjust.
const AXIS_CORRECTION = [Math.PI , 0, 0];

function useGeometry(file, jobId) {
  const [result, setResult] = useState({ geometry: null, error: null });
  useEffect(() => {
    let disposed = false;
    let geometry;
    const artifact = file === "mesh.ply" ? "skyform_mesh_experimental.ply" : "skyform_pointcloud.ply";
    const url = jobId ? `${API}/outputs/jobs/${encodeURIComponent(jobId)}/final/${artifact}` : `${API}/outputs/reconstruction/${file}`;
    new PLYLoader().load(url, (loaded) => {
      if (disposed) { loaded.dispose(); return; }
      geometry = loaded;
      for (const [name, attribute] of Object.entries(loaded.attributes)) {
        if (attribute.array instanceof Float64Array) loaded.setAttribute(name, new THREE.Float32BufferAttribute(attribute.array, attribute.itemSize));
      }
      if (!loaded.getAttribute("position")?.count) {
        setResult({ geometry: null, error: "This file contains no geometry." });
        return;
      }
      loaded.computeBoundingBox();
      if (file === "mesh.ply") loaded.computeVertexNormals();
      setResult({ geometry: loaded, error: null });
    }, undefined, () => {
      if (!disposed) setResult({ geometry: null, error: `Could not load ${file}. Check that reconstruction has completed.` });
    });
    return () => { disposed = true; geometry?.dispose(); };
  }, [file, jobId]);
  return result;
}

function useDiagnostic(endpoint) {
  const [result, setResult] = useState({ data: null, error: null });
  useEffect(() => {
    const controller = new AbortController();
    fetch(`${API}/api/reconstruction/${endpoint}`, { signal: controller.signal })
      .then((response) => { if (!response.ok) throw new Error(); return response.json(); })
      .then((data) => setResult({ data, error: null }))
      .catch((error) => { if (error.name !== "AbortError") setResult({ data: null, error: `Could not load ${endpoint} data.` }); });
    return () => controller.abort();
  }, [endpoint]);
  return result;
}

function CameraMarker({ camera, index, count, selected, onSelect }) {
  const quaternion = useMemo(() => {
    const r = camera.rotation_matrix;
    if (r?.length !== 3) return new THREE.Quaternion();
    // The camera API returns camera-to-world rotations (COLMAP's +Z looks forward).
    return new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().set(
      ...r[0], 0, ...r[1], 0, ...r[2], 0, 0, 0, 0, 1,
    ));
  }, [camera]);
  const color = selected ? "#ffffff" : index === 0 ? "#34d399" : index === count - 1 ? "#fb923c" : "#38bdf8";
  const corners = [[-.035,-.022,.06],[.035,-.022,.06],[.035,.022,.06],[-.035,.022,.06]];
  return <group position={camera.position}>
    <mesh onClick={(event) => { event.stopPropagation(); onSelect(index); }}>
      <sphereGeometry args={[selected ? .018 : .012, 12, 12]} />
      <meshBasicMaterial color={color} depthTest={false} />
    </mesh>
    <group quaternion={quaternion} scale={selected ? 1.5 : .55}>
      <mesh><boxGeometry args={[.027,.019,.019]} /><meshBasicMaterial color={color} depthTest={false} /></mesh>
      {corners.map((corner, i) => <Line key={i} points={[[0,0,0], corner, corners[(i+1)%4]]} color={color} lineWidth={selected ? 2 : 1} depthTest={false} />)}
    </group>
    {(selected || index === 0 || index === count - 1) && <Html center position={[0,.055,0]} style={{ pointerEvents: "none" }}><span className="camera-label">{index === 0 ? "Start · " : index === count - 1 ? "End · " : ""}{index + 1}</span></Html>}
  </group>;
}

function Scene({ geometry, cameras, mode, autoRotate, selected, onSelect, meshStyle, reset, pathOnly }) {
  const { normalized, normalizedCameras } = useMemo(() => {
    const box = mode === "cameras" && pathOnly ? new THREE.Box3() : geometry?.boundingBox?.clone() || new THREE.Box3();
    if (mode === "cameras") cameras.forEach((camera) => box.expandByPoint(new THREE.Vector3(...camera.position)));
    if (box.isEmpty()) box.setFromCenterAndSize(new THREE.Vector3(), new THREE.Vector3(1,1,1));
    const center = box.getCenter(new THREE.Vector3());
    const scale = (mode === "cameras" && pathOnly ? 1.7 : 2.2) / Math.max(box.getSize(new THREE.Vector3()).length(), .000001);
    const copy = geometry?.clone();
    if (copy) { copy.translate(-center.x, -center.y, -center.z); copy.scale(scale, scale, scale); }
    return { normalized: copy, normalizedCameras: cameras.map((camera) => ({ ...camera, position: new THREE.Vector3(...camera.position).sub(center).multiplyScalar(scale).toArray() })) };
  }, [geometry, cameras, mode, pathOnly]);
  useEffect(() => () => normalized?.dispose(), [normalized]);
  return <>
    <color attach="background" args={["#090e16"]} />
    <ambientLight intensity={.6} />
    <directionalLight position={[3,5,4]} intensity={2.2} />
    <directionalLight position={[-4,1,-2]} intensity={1} />
    <group rotation={AXIS_CORRECTION}>
    {normalized && !(mode === "cameras" && pathOnly) && (mode === "mesh" ? <mesh geometry={normalized}>
      <meshStandardMaterial color={meshStyle === "color" && normalized.hasAttribute("color") ? "#ffffff" : "#a9bdce"} vertexColors={meshStyle === "color" && normalized.hasAttribute("color")} roughness={.85} metalness={0} side={THREE.DoubleSide} wireframe={meshStyle === "wireframe"} />
    </mesh> : <points geometry={normalized}>
      <pointsMaterial size={mode === "confidence" ? 3 : 2} sizeAttenuation={false} vertexColors={mode !== "cameras" && normalized.hasAttribute("color")} color={mode === "cameras" ? "#64748b" : "#ffffff"} transparent={mode === "cameras"} opacity={mode === "cameras" ? .25 : 1} depthWrite={mode !== "cameras"} />
    </points>)}
    {mode === "cameras" && <group>
      {normalizedCameras.length > 1 && <Line points={normalizedCameras.map((camera) => camera.position)} color="#38bdf8" lineWidth={2} depthTest={false} />}
      {normalizedCameras.map((camera, index) => <CameraMarker key={camera.id ?? camera.image} camera={camera} index={index} count={cameras.length} selected={index === selected} onSelect={onSelect} />)}
    </group>}
    </group>
    <OrbitControls key={`${mode}-${reset}`} makeDefault enableDamping autoRotate={autoRotate} autoRotateSpeed={.6} />
  </>;
}

const number = (value, digits = 0) => Number.isFinite(value) ? value.toLocaleString(undefined, { maximumFractionDigits: digits }) : "Unavailable";

export default function ReconstructionViewer({ jobId }) {
  const [mode, setMode] = useState("points");
  const [autoRotate, setAutoRotate] = useState(false);
  const [selected, setSelected] = useState(0);
  const [pathOnly, setPathOnly] = useState(true);
  const [meshStyle, setMeshStyle] = useState("solid");
  const [reset, setReset] = useState(0);
  const cloud = useGeometry("dense_fused.ply", jobId);
  const mesh = useGeometry("mesh.ply", jobId);
  const quality = useDiagnostic("quality");
  const poses = useDiagnostic("cameras");
  const cameras = useMemo(() => (poses.data?.cameras || []).filter((camera) => camera.position?.length === 3 && camera.position.every(Number.isFinite)).sort((a,b) => a.image.localeCompare(b.image, undefined, { numeric: true })), [poses.data]);
  const confidence = useMemo(() => {
    const points = (quality.data?.points || []).filter((point) => point.position?.length === 3 && point.position.every(Number.isFinite) && Number.isFinite(point.confidence));
    if (!points.length) return null;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(points.flatMap((point) => point.position), 3));
    geometry.setAttribute("color", new THREE.Float32BufferAttribute(points.flatMap((point) => point.confidence >= .75 ? [.2,.95,.52] : point.confidence >= .45 ? [1,.72,.18] : [1,.25,.25]), 3));
    geometry.computeBoundingBox();
    return geometry;
  }, [quality.data]);
  useEffect(() => () => confidence?.dispose(), [confidence]);
  const geometry = mode === "mesh" ? mesh.geometry : mode === "confidence" ? confidence || cloud.geometry : cloud.geometry;
  const error = mode === "mesh" ? mesh.error : mode === "confidence" ? quality.error || cloud.error : mode === "cameras" ? poses.error || cloud.error : cloud.error;
  const summary = quality.data?.summary;
  return <div className="reconstruction-viewer">
    <div className="viewer-toolbar">
      <div className="viewer-mode-group">{Object.entries(modes).map(([key,label]) => <button key={key} className={`viewer-button ${mode === key ? "active" : ""}`} onClick={() => setMode(key)}>{label}</button>)}</div>
      <div className="viewer-mode-group"><button className="viewer-button" onClick={() => setReset((value) => value + 1)}>Reset view</button><button className="viewer-button" onClick={() => setAutoRotate((value) => !value)}>{autoRotate ? "Stop Rotation" : "Auto Rotate"}</button></div>
    </div>
    <div style={{ position: "absolute", left: 0, right: mode === "cameras" && pathOnly ? 250 : 0, top: mode === "cameras" && pathOnly ? 48 : 0, bottom: 0 }}>
    <Canvas key={reset} camera={{ position: [0,.5,3.4], fov: 45 }} dpr={[1,1.5]} gl={{ antialias: true }}>
      <Scene geometry={geometry} cameras={cameras} mode={mode} autoRotate={autoRotate} selected={selected} onSelect={setSelected} meshStyle={meshStyle} reset={reset} pathOnly={pathOnly} />
    </Canvas>
    </div>
    {(error || !geometry) && <div className="viewer-message" role="status">{error || "Loading reconstruction…"}</div>}
    <div className="viewer-diagnostics" style={mode === "cameras" && pathOnly ? { left: "auto", right: 12, width: 220, top: 65, bottom: "auto" } : undefined}>
      {mode === "mesh" && <><strong>Surface inspection</strong><div className="viewer-mode-group">{["solid","color","wireframe"].map((style) => <button key={style} className={`viewer-button ${meshStyle === style ? "active" : ""}`} onClick={() => setMeshStyle(style)}>{style}</button>)}</div><span>{number(mesh.geometry ? (mesh.geometry.index?.count || mesh.geometry.attributes.position.count) / 3 : null)} triangles · experimental triangulated surface</span></>}
      {mode === "confidence" && <><strong>Reconstruction confidence</strong>{confidence ? <><span>🟢 High ≥ 0.75 · 🟡 Medium ≥ 0.45 · 🔴 Low</span><span>{number(summary?.point_count)} sparse points · {number(summary?.average_reprojection_error, 2)} px mean error</span></> : <span>{quality.data ? "Per-point confidence unavailable. Showing the point cloud for context." : "Loading quality measurements…"}</span>}<span>{summary?.confidence_note}</span></>}
      {mode === "cameras" && <><strong>{cameras.length} registered cameras · capture order</strong><span>Green: start · Orange: end · Camera bodies + frustums: viewing direction</span><button className="viewer-button" onClick={() => setPathOnly((value) => !value)}>{pathOnly ? "Show path in scene" : "Focus camera path"}</button>{cameras.length ? <><label>Frame {selected + 1} / {cameras.length}<input aria-label="Selected camera" type="range" min="0" max={cameras.length - 1} value={selected} onChange={(event) => setSelected(Number(event.target.value))} /></label><span>{cameras[selected]?.image} · click a marker to select</span>{cameras[selected]?.image_url && <img className="camera-preview" src={`${API}${cameras[selected].image_url}`} alt={`Selected camera frame ${selected + 1}`} />}</> : <span>{poses.data ? "No camera poses are available." : "Loading camera poses…"}</span>}</>}
    </div>
    <div className="viewer-overlay"><div>{modes[mode]}</div><span>Drag to orbit · Scroll to zoom · Right-drag to pan</span></div>
  </div>;
}
