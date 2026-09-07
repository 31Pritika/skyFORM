import { Canvas, useFrame } from "@react-three/fiber";

import {
  OrbitControls,
  PerspectiveCamera,
  Grid,
} from "@react-three/drei";

import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";

import {
  Suspense,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import * as THREE from "three";


const API = "http://localhost:8000";


/* =========================================================
   POINT CLOUD
   ========================================================= */

function PointCloud({
  visible,
  onSceneTransform,
}) {
  const [geometry, setGeometry] =
    useState(null);

  useEffect(() => {
    const loader = new PLYLoader();

    loader.load(
      `${API}/outputs/reconstruction/dense_fused.ply`,

      (loadedGeometry) => {
        loadedGeometry.computeBoundingBox();

        const box =
          loadedGeometry.boundingBox;

        if (!box) {
          return;
        }

        const center =
          new THREE.Vector3();

        const size =
          new THREE.Vector3();

        box.getCenter(center);
        box.getSize(size);

        const maxDimension =
          Math.max(
            size.x,
            size.y,
            size.z
          );

        const scale =
          maxDimension > 0
            ? 2.2 / maxDimension
            : 1;

        onSceneTransform({
          center: [
            center.x,
            center.y,
            center.z,
          ],
          scale,
        });

        loadedGeometry.setDrawRange(
          0,
          0
        );

        setGeometry(
          loadedGeometry
        );
      },

      undefined,

      (error) => {
        console.error(
          "Point cloud load error:",
          error
        );
      }
    );

    return () => {
      setGeometry(
        (current) => {
          if (current) {
            current.dispose();
          }

          return null;
        }
      );
    };
  }, [onSceneTransform]);


  useEffect(() => {
    if (
      geometry &&
      visible
    ) {
      geometry.setDrawRange(
        0,
        0
      );
    }
  }, [
    geometry,
    visible,
  ]);


  useFrame(() => {
    if (
      !geometry ||
      !visible
    ) {
      return;
    }

    const position =
      geometry.getAttribute(
        "position"
      );

    if (!position) {
      return;
    }

    const current =
      geometry.drawRange.count;

    if (
      current >= position.count
    ) {
      return;
    }

    geometry.setDrawRange(
      0,
      Math.min(
        current + 900,
        position.count
      )
    );
  });


  if (
    !geometry ||
    !visible
  ) {
    return null;
  }


  const hasColors =
    Boolean(
      geometry.getAttribute(
        "color"
      )
    );


  return (
    <points geometry={geometry}>
      <pointsMaterial
        size={0.012}
        sizeAttenuation
        vertexColors={hasColors}
        color={
          hasColors
            ? undefined
            : "#38bdf8"
        }
      />
    </points>
  );
}


/* =========================================================
   MESH
   Lazy-loaded only when the user selects Mesh.
   ========================================================= */

function MeshModel({
  visible,
  onSceneTransform,
}) {
  const [geometry, setGeometry] =
    useState(null);

  const [error, setError] =
    useState(false);

  const requestedRef =
    useRef(false);


  useEffect(() => {
    if (
      !visible ||
      geometry ||
      requestedRef.current
    ) {
      return;
    }

    requestedRef.current = true;

    console.log(
      "Loading reconstruction mesh..."
    );

    const loader =
      new PLYLoader();

    loader.load(
      `${API}/outputs/reconstruction/mesh.ply`,

      (loadedGeometry) => {
        console.log(
          "Mesh loaded:",
          loadedGeometry
            .getAttribute("position")
            ?.count,
          "vertices"
        );

        // Sanitize geometry for WebGL.
        // PLYLoader can produce Float64 attributes, which Three.js/WebGL
        // cannot upload directly. Convert every Float64 attribute to Float32.
        Object.keys(loadedGeometry.attributes).forEach(
          (attributeName) => {
            const attribute =
              loadedGeometry.getAttribute(
                attributeName
              );

            if (
              attribute &&
              attribute.array instanceof Float64Array
            ) {
              console.log(
                `Converting ${attributeName} from Float64Array to Float32Array`
              );

              loadedGeometry.setAttribute(
                attributeName,
                new THREE.BufferAttribute(
                  new Float32Array(
                    attribute.array
                  ),
                  attribute.itemSize,
                  attribute.normalized
                )
              );
            }
          }
        );

        // MeshBasicMaterial does not need normals.
        // Remove them completely to avoid any unnecessary GPU attribute upload.
        loadedGeometry.deleteAttribute("normal");

        // The mesh has more than 65,535 vertices, so its indexed form uses
        // Uint32 indices. Convert to non-indexed triangles so rendering does
        // not depend on 32-bit element-index support in the browser.
        if (loadedGeometry.index) {
          const nonIndexedGeometry =
            loadedGeometry.toNonIndexed();

          loadedGeometry.dispose();
          loadedGeometry =
            nonIndexedGeometry;
        }

        loadedGeometry.computeBoundingBox();
        loadedGeometry.computeBoundingSphere();

        const box =
          loadedGeometry.boundingBox;

        if (box) {
          const center =
            new THREE.Vector3();

          const size =
            new THREE.Vector3();

          box.getCenter(center);
          box.getSize(size);

          const maxDimension =
            Math.max(
              size.x,
              size.y,
              size.z
            );

          const scale =
            maxDimension > 0
              ? 2.2 / maxDimension
              : 1;

          onSceneTransform({
            center: [
              center.x,
              center.y,
              center.z,
            ],
            scale,
          });
        }

        setGeometry(
          loadedGeometry
        );
      },

      undefined,

      (loadError) => {
        console.error(
          "Mesh load error:",
          loadError
        );

        setError(true);
        requestedRef.current = false;
      }
    );
  }, [
    visible,
    geometry,
    onSceneTransform,
  ]);


  useEffect(() => {
    return () => {
      if (geometry) {
        geometry.dispose();
      }
    };
  }, [geometry]);


  if (
    !visible ||
    !geometry ||
    error
  ) {
    return null;
  }


  const hasColors =
    Boolean(
      geometry.getAttribute(
        "color"
      )
    );


  return (
    <mesh
      geometry={geometry}
      frustumCulled={false}
    >
      <meshBasicMaterial
        vertexColors={hasColors}
        color={
          hasColors
            ? undefined
            : "#94a3b8"
        }
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}


/* =========================================================
   CONFIDENCE MAP
   ========================================================= */

function ConfidenceMap({
  visible,
  onSummary,
}) {
  const [geometry, setGeometry] =
    useState(null);


  useEffect(() => {
    const loadQuality =
      async () => {
        try {
          const response =
            await fetch(
              `${API}/api/reconstruction/quality`
            );

          const data =
            await response.json();

          if (
            !data.available ||
            !data.points?.length
          ) {
            return;
          }

          onSummary(
            data.summary
          );

          const positions = [];
          const colors = [];

          data.points.forEach(
            (point) => {
              positions.push(
                point.position[0],
                point.position[1],
                point.position[2]
              );

              if (
                point.confidence >= 0.75
              ) {
                colors.push(
                  0.20,
                  0.95,
                  0.52
                );

              } else if (
                point.confidence >= 0.45
              ) {
                colors.push(
                  1.0,
                  0.72,
                  0.18
                );

              } else {
                colors.push(
                  1.0,
                  0.25,
                  0.25
                );
              }
            }
          );

          const confidenceGeometry =
            new THREE.BufferGeometry();

          confidenceGeometry.setAttribute(
            "position",
            new THREE.Float32BufferAttribute(
              positions,
              3
            )
          );

          confidenceGeometry.setAttribute(
            "color",
            new THREE.Float32BufferAttribute(
              colors,
              3
            )
          );

          confidenceGeometry.setDrawRange(
            0,
            0
          );

          setGeometry(
            confidenceGeometry
          );

        } catch (error) {
          console.error(
            "Confidence data load error:",
            error
          );
        }
      };

    loadQuality();
  }, [onSummary]);


  useEffect(() => {
    return () => {
      if (geometry) {
        geometry.dispose();
      }
    };
  }, [geometry]);


  useEffect(() => {
    if (
      geometry &&
      visible
    ) {
      geometry.setDrawRange(
        0,
        0
      );
    }
  }, [
    geometry,
    visible,
  ]);


  useFrame(() => {
    if (
      !geometry ||
      !visible
    ) {
      return;
    }

    const position =
      geometry.getAttribute(
        "position"
      );

    if (!position) {
      return;
    }

    const current =
      geometry.drawRange.count;

    if (
      current >= position.count
    ) {
      return;
    }

    geometry.setDrawRange(
      0,
      Math.min(
        current + 40,
        position.count
      )
    );
  });


  if (
    !geometry ||
    !visible
  ) {
    return null;
  }


  return (
    <points geometry={geometry}>
      <pointsMaterial
        size={0.035}
        sizeAttenuation
        vertexColors
      />
    </points>
  );
}


/* =========================================================
   CAMERA TRAJECTORY
   ========================================================= */

function CameraTrajectory({
  visible,
}) {
  const [cameras, setCameras] =
    useState([]);

  const [
    visibleCount,
    setVisibleCount,
  ] =
    useState(0);

  const timerRef =
    useRef(0);


  useEffect(() => {
    const loadCameras =
      async () => {
        try {
          const response =
            await fetch(
              `${API}/api/reconstruction/cameras`
            );

          const data =
            await response.json();

          setCameras(
            data.cameras || []
          );

        } catch (error) {
          console.error(
            "Camera pose load error:",
            error
          );
        }
      };

    loadCameras();
  }, []);


  useEffect(() => {
    if (visible) {
      setVisibleCount(0);
      timerRef.current = 0;
    }
  }, [visible]);


  useFrame((_, delta) => {
    if (
      !visible ||
      cameras.length === 0
    ) {
      return;
    }

    timerRef.current += delta;

    if (
      timerRef.current > 0.12
    ) {
      timerRef.current = 0;

      setVisibleCount(
        (current) =>
          Math.min(
            current + 1,
            cameras.length
          )
      );
    }
  });


  const visibleCameras =
    cameras.slice(
      0,
      visibleCount
    );


  const trajectoryGeometry =
    useMemo(() => {
      if (
        visibleCameras.length < 2
      ) {
        return null;
      }

      const points =
        visibleCameras.map(
          (camera) =>
            new THREE.Vector3(
              camera.position[0],
              camera.position[1],
              camera.position[2]
            )
        );

      return new THREE.BufferGeometry()
        .setFromPoints(points);

    }, [visibleCameras]);


  useEffect(() => {
    return () => {
      if (trajectoryGeometry) {
        trajectoryGeometry.dispose();
      }
    };
  }, [trajectoryGeometry]);


  if (!visible) {
    return null;
  }


  return (
    <>
      {trajectoryGeometry && (
        <line
          geometry={
            trajectoryGeometry
          }
        >
          <lineBasicMaterial
            color="#38bdf8"
            transparent
            opacity={0.8}
          />
        </line>
      )}


      {visibleCameras.map(
        (camera, index) => (
          <group
            key={camera.id}
            position={[
              camera.position[0],
              camera.position[1],
              camera.position[2],
            ]}
          >

            <mesh>
              <sphereGeometry
                args={[
                  0.07,
                  12,
                  12,
                ]}
              />

              <meshBasicMaterial
                color={
                  index ===
                  visibleCameras.length - 1
                    ? "#ffffff"
                    : "#38bdf8"
                }
              />
            </mesh>


            <mesh
              rotation={[
                Math.PI / 2,
                0,
                0,
              ]}
            >
              <coneGeometry
                args={[
                  0.08,
                  0.18,
                  4,
                ]}
              />

              <meshBasicMaterial
                color="#0ea5e9"
                wireframe
                transparent
                opacity={0.8}
              />
            </mesh>

          </group>
        )
      )}
    </>
  );
}


/* =========================================================
   SCENE
   ========================================================= */

function Scene({
  mode,
  autoRotate,
  onQualitySummary,
}) {
  const [
    sceneTransform,
    setSceneTransform,
  ] = useState({
    center: [0, 0, 0],
    scale: 1,
  });


  const handleTransform =
    useMemo(
      () =>
        ({ center, scale }) => {
          setSceneTransform({
            center,
            scale,
          });
        },
      []
    );


  const [
    cx,
    cy,
    cz,
  ] =
    sceneTransform.center;


  return (
    <>
      <color
        attach="background"
        args={["#07090d"]}
      />


      <ambientLight
        intensity={1.2}
      />


      <directionalLight
        position={[4, 6, 5]}
        intensity={1.8}
      />


      <directionalLight
        position={[-4, 2, -4]}
        intensity={0.6}
      />


      <PerspectiveCamera
        makeDefault
        position={[0, 0.8, 3.5]}
        fov={45}
      />


      <Suspense fallback={null}>

        <group
          scale={
            sceneTransform.scale
          }
          position={[
            -cx *
              sceneTransform.scale,

            -cy *
              sceneTransform.scale,

            -cz *
              sceneTransform.scale,
          ]}
        >

          <PointCloud
            visible={
              mode === "points" ||
              mode === "cameras"
            }
            onSceneTransform={
              handleTransform
            }
          />


          <MeshModel
            visible={
              mode === "mesh"
            }
            onSceneTransform={
              handleTransform
            }
          />


          <ConfidenceMap
            visible={
              mode === "confidence"
            }
            onSummary={
              onQualitySummary
            }
          />


          <CameraTrajectory
            visible={
              mode === "cameras"
            }
          />

        </group>

      </Suspense>


      <Grid
        position={[0, -1, 0]}
        args={[10, 10]}
        cellSize={0.25}
        cellThickness={0.4}
        sectionSize={1}
        sectionThickness={0.8}
        fadeDistance={8}
        fadeStrength={1}
        infiniteGrid
      />


      <OrbitControls
        makeDefault
        enableDamping
        dampingFactor={0.08}
        autoRotate={autoRotate}
        autoRotateSpeed={0.6}
      />

    </>
  );
}


/* =========================================================
   VIEWER
   ========================================================= */

export default function ReconstructionViewer() {
  const [mode, setMode] =
    useState("points");

  const [
    autoRotate,
    setAutoRotate,
  ] =
    useState(true);

  const [
    qualitySummary,
    setQualitySummary,
  ] =
    useState(null);


  const handleQualitySummary =
    useMemo(
      () =>
        (summary) => {
          setQualitySummary(
            summary
          );
        },
      []
    );


  return (
    <div className="reconstruction-viewer">

      <div className="viewer-toolbar">

        <div className="viewer-mode-group">

          <button
            className={
              mode === "points"
                ? "viewer-button active"
                : "viewer-button"
            }
            onClick={() =>
              setMode("points")
            }
          >
            Point Cloud
          </button>


          <button
            className={
              mode === "mesh"
                ? "viewer-button active"
                : "viewer-button"
            }
            onClick={() =>
              setMode("mesh")
            }
          >
            Mesh
          </button>


          <button
            className={
              mode === "confidence"
                ? "viewer-button active"
                : "viewer-button"
            }
            onClick={() =>
              setMode("confidence")
            }
          >
            Confidence
          </button>


          <button
            className={
              mode === "cameras"
                ? "viewer-button active"
                : "viewer-button"
            }
            onClick={() =>
              setMode("cameras")
            }
          >
            Camera Path
          </button>

        </div>


        <button
          className="viewer-button"
          onClick={() =>
            setAutoRotate(
              (value) => !value
            )
          }
        >
          {autoRotate
            ? "Stop Rotation"
            : "Auto Rotate"}
        </button>

      </div>


      <Canvas
        dpr={[1, 1.5]}
        gl={{
          antialias: false,
          alpha: false,
          powerPreference:
            "high-performance",
        }}
      >
        <Scene
          mode={mode}
          autoRotate={
            autoRotate
          }
          onQualitySummary={
            handleQualitySummary
          }
        />
      </Canvas>


      {mode === "confidence" && (
        <div
          style={{
            position: "absolute",
            left: "18px",
            bottom: "52px",
            padding: "12px 14px",
            background:
              "rgba(8, 11, 15, 0.92)",
            border:
              "1px solid #242b33",
            borderRadius: "8px",
            fontSize: "10px",
            lineHeight: "1.8",
            color: "#94a3b8",
            pointerEvents: "none",
          }}
        >
          <div
            style={{
              fontWeight: 600,
              color: "#e2e8f0",
              marginBottom: "5px",
              letterSpacing: "0.05em",
            }}
          >
            RECONSTRUCTION CONFIDENCE
          </div>

          <div>
            🟢 High ≥ 0.75
            {qualitySummary &&
              ` · ${qualitySummary.high_confidence_points}`}
          </div>

          <div>
            🟡 Medium 0.45–0.75
            {qualitySummary &&
              ` · ${qualitySummary.medium_confidence_points}`}
          </div>

          <div>
            🔴 Low &lt; 0.45
            {qualitySummary &&
              ` · ${qualitySummary.low_confidence_points}`}
          </div>

          {qualitySummary && (
            <div
              style={{
                marginTop: "5px",
                borderTop:
                  "1px solid #242b33",
                paddingTop: "5px",
              }}
            >
              Mean heuristic score:{" "}
              {(
                qualitySummary.average_confidence *
                100
              ).toFixed(1)}
              %
            </div>
          )}
        </div>
      )}


      <div className="viewer-overlay">

        <div>
          <span className="viewer-status-dot" />

          {mode === "confidence"
            ? "QUALITY ANALYSIS ACTIVE"
            : mode === "cameras"
            ? "CAMERA TRAJECTORY"
            : "RECONSTRUCTION READY"}
        </div>

        <span>
          {mode === "confidence" &&
          qualitySummary
            ? `${qualitySummary.point_count} validated SfM points · ${qualitySummary.average_reprojection_error.toFixed(
                2
              )} px mean reprojection error`

            : mode === "cameras"
            ? "18 registered camera positions · relative coordinate frame"

            : "18 registered views · 72,576 fused points · relative coordinate frame"}
        </span>

      </div>

    </div>
  );
}