import axios from "axios";

const API_BASE_URL = "http://127.0.0.1:8000";


export async function uploadVideo(
  file,
  onProgress
) {
  const formData = new FormData();

  formData.append("video", file);

  const response = await axios.post(
    `${API_BASE_URL}/api/videos/upload`,
    formData,
    {
      headers: {
        "Content-Type": "multipart/form-data",
      },

      onUploadProgress: (progressEvent) => {
        if (!progressEvent.total) {
          return;
        }

        const percent = Math.round(
          (progressEvent.loaded * 100) /
            progressEvent.total
        );

        if (onProgress) {
          onProgress(percent);
        }
      },
    }
  );

  return response.data;
}


export function getPreviewUrl(
  relativeUrl
) {
  return `${API_BASE_URL}${relativeUrl}`;
}


export async function getReconstructionStatus() {
  const response = await axios.get(
    `${API_BASE_URL}/api/reconstruction/status`
  );

  return response.data;
}


export async function getCameraPoses() {
  const response = await axios.get(
    `${API_BASE_URL}/api/reconstruction/cameras`
  );

  return response.data;
}


// ============================================================
// VIDEO-SPECIFIC GPS STATUS
// ============================================================

export async function getGeospatialStatus(
  videoId
) {
  if (!videoId) {
    return {
      available: false,
      coordinate_system:
        "relative_unscaled",
      message: "No video selected.",
    };
  }

  const response = await axios.get(
    `${API_BASE_URL}/api/geospatial/status/${videoId}`
  );

  return response.data;
}


// ============================================================
// VIDEO-SPECIFIC GPS UPLOAD
// ============================================================

export async function uploadTelemetry(
  videoId,
  file
) {
  if (!videoId) {
    throw new Error(
      "A video must be uploaded before GPS telemetry."
    );
  }

  const formData = new FormData();

  formData.append(
    "telemetry",
    file
  );

  const response = await axios.post(
    `${API_BASE_URL}/api/geospatial/upload/${videoId}`,
    formData,
    {
      headers: {
        "Content-Type":
          "multipart/form-data",
      },
    }
  );

  return response.data;
}


// ============================================================
// VIDEO-SPECIFIC FLIGHT METADATA STATUS
// ============================================================

export async function getFlightMetadataStatus(
  videoId
) {
  if (!videoId) {
    return {
      available: false,
      metadata: {},
    };
  }

  const response = await axios.get(
    `${API_BASE_URL}/api/flight-metadata/status/${videoId}`
  );

  return response.data;
}


// ============================================================
// VIDEO-SPECIFIC FLIGHT METADATA UPLOAD
// ============================================================

export async function saveFlightMetadata(
  videoId,
  metadata
) {
  if (!videoId) {
    throw new Error(
      "A video must be uploaded before flight metadata."
    );
  }

  const response = await axios.post(
    `${API_BASE_URL}/api/flight-metadata/${videoId}`,
    metadata,
    {
      headers: {
        "Content-Type":
          "application/json",
      },
    }
  );

  return response.data;
}


// ============================================================
// GPS ALIGNMENT
// ============================================================

export async function alignReconstructionToGps(
  videoId
) {
  if (!videoId) {
    throw new Error(
      "A video ID is required for GPS alignment."
    );
  }

  const response = await axios.post(
    `${API_BASE_URL}/api/geospatial/align/${videoId}`
  );

  return response.data;
}


// ============================================================
// START RECONSTRUCTION
// ============================================================

export async function startReconstruction(
  videoId
) {
  const response = await axios.post(
    `${API_BASE_URL}/api/reconstruction/start/${videoId}`
  );

  return response.data;
}


// ============================================================
// PIPELINE STATUS
// ============================================================

export async function getPipelineExecutionStatus() {
  const response = await axios.get(
    `${API_BASE_URL}/api/pipeline/status`
  );

  return response.data;
}


// ============================================================
// EXPORT
// ============================================================

export function downloadReconstructionExport() {
  window.location.href =
    `${API_BASE_URL}/api/reconstruction/export`;
}