import { extractString, fallbackBasename } from "../utils/format.js";

export async function checkCaptureReadiness(options) {
  const formData = new FormData();
  formData.set("frame", options.frame, options.frame.name || "preview.jpg");
  formData.set("guide_scale", String(options.guideScale || 1));

  const response = await fetch(`${options.baseUrl}/capture-readiness`, {
    method: "POST",
    body: formData,
  });

  const payload = await readJsonResponse(
    response,
    "The backend did not return a valid readiness JSON response.",
  );

  if (!response.ok) {
    throw new Error(payload.error || `The backend returned status ${response.status}.`);
  }

  return {
    success: Boolean(payload.success),
    score: Number(payload.score || 0),
    reason: extractString(payload.reason) || null,
    checks: payload.checks && typeof payload.checks === "object" ? payload.checks : {},
    metrics: payload.metrics && typeof payload.metrics === "object" ? payload.metrics : {},
    latencyMs: Number(payload.latency_ms || 0),
  };
}

export async function uploadVideo(options) {
  const formData = new FormData();
  formData.set("video", options.file, options.file.name || "capture.mp4");
  formData.set("response_mode", "json");
  formData.set("keep_artifacts", String(options.keepArtifacts));
  formData.set("clip_duration_ms", String(options.clipDurationMs));
  formData.set("frame_count", String(options.frameCount));

  const response = await fetch(`${options.baseUrl}/frames`, {
    method: "POST",
    body: formData,
  });

  const payload = await readJsonResponse(
    response,
    "The backend did not return a valid JSON response.",
  );

  if (!response.ok) {
    throw new Error(payload.error || `The backend returned status ${response.status}.`);
  }

  return normalizeResultPayload(payload);
}

export async function uploadAnalyzeImage(options) {
  const formData = new FormData();
  formData.set("image", options.file, options.file.name || "capture.jpg");
  formData.set("response_mode", "binary");
  formData.set("keep_artifacts", String(options.keepArtifacts));

  const response = await fetch(`${options.baseUrl}/analyze`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw await readApiError(response, "Backend nije prihvatio upload slike.");
  }

  const contentType = (response.headers.get("Content-Type") || "").toLowerCase();
  if (!contentType.startsWith("image/png")) {
    let payload = null;
    try {
      payload = await response.json();
    } catch (error) {
      throw new Error("Backend je vratio neocekivan odgovor umesto PNG overlay-a.");
    }

    return normalizeResultPayload(payload);
  }

  const overlayBlob = await response.blob();
  const overlayObjectUrl = URL.createObjectURL(overlayBlob);

  return {
    sourceName:
      extractString(response.headers.get("X-Image-Name")) ||
      fallbackBasename(options.file.name) ||
      "capture.jpg",
    processingTimeMs: Number(response.headers.get("X-Processing-Time-Ms") || 0),
    overlayDataUrl: overlayObjectUrl,
    overlayObjectUrl,
    sourcePath: extractString(response.headers.get("X-Input-Image-Path")),
    artifactsDir: extractString(response.headers.get("X-Artifacts-Dir")),
    overlayOutputPath: extractString(response.headers.get("X-Overlay-Output-Path")),
    metadataOutputPath: extractString(response.headers.get("X-Metadata-Output-Path")),
    frameCount: 1,
    selectedFrameIndex: null,
    selectedTimestampMs: null,
    frames: [],
  };
}

function normalizeResultPayload(payload) {
  const selectedFrameIndex = toFiniteNumber(payload.selected_frame_index);
  const normalizedFrames = normalizeFrames(payload.frames, selectedFrameIndex);
  const nestedFrameAnalysis =
    normalizedFrames.length > 0
      ? normalizedFrames.find((frame) => frame.selected)?.analysis ||
        normalizedFrames[0].analysis
      : null;

  const overlayDataUrl =
    extractString(payload.overlay_data_url) ||
    extractString(nestedFrameAnalysis && nestedFrameAnalysis.overlay_data_url);

  if (!overlayDataUrl) {
    throw new Error("The backend response is missing overlay_data_url.");
  }

  const sourceName =
    extractString(payload.video_name) ||
    extractString(payload.image_name) ||
    extractString(payload.source_name) ||
    extractString(nestedFrameAnalysis && nestedFrameAnalysis.image_name) ||
    fallbackBasename(
      extractString(payload.video_path) ||
        extractString(payload.input_video_path) ||
        extractString(payload.source_path),
    ) ||
    "capture.mp4";

  return {
    sourceName,
    processingTimeMs: Number(payload.processing_time_ms || 0),
    overlayDataUrl,
    sourcePath:
      extractString(payload.input_video_path) ||
      extractString(payload.video_path) ||
      extractString(payload.source_path) ||
      extractString(nestedFrameAnalysis && nestedFrameAnalysis.input_image_path),
    artifactsDir: extractString(payload.artifacts_dir),
    overlayOutputPath:
      extractString(payload.overlay_output_path) ||
      extractString(nestedFrameAnalysis && nestedFrameAnalysis.overlay_output_path),
    metadataOutputPath:
      extractString(payload.metadata_output_path) ||
      extractString(nestedFrameAnalysis && nestedFrameAnalysis.metadata_output_path),
    overlayObjectUrl: null,
    frameCount:
      Number.isFinite(payload.frame_count) && payload.frame_count > 0
        ? Number(payload.frame_count)
        : Array.isArray(payload.frames)
          ? payload.frames.length
          : null,
    selectedFrameIndex,
    selectedTimestampMs: toFiniteNumber(payload.selected_timestamp_ms),
    frames: normalizedFrames.map((frame) => ({
      frameIndex: frame.frameIndex,
      timestampMs: frame.timestampMs,
      imageName: frame.imageName,
      overlayDataUrl: frame.overlayDataUrl,
      overlayOutputPath: frame.overlayOutputPath,
      metadataOutputPath: frame.metadataOutputPath,
      selected: frame.selected,
    })),
  };
}

function normalizeFrames(rawFrames, selectedFrameIndex) {
  if (!Array.isArray(rawFrames)) {
    return [];
  }

  return rawFrames
    .map((frame, index) => normalizeFrame(frame, index, selectedFrameIndex))
    .filter(Boolean);
}

function normalizeFrame(rawFrame, index, selectedFrameIndex) {
  if (!rawFrame || typeof rawFrame !== "object") {
    return null;
  }

  const analysis =
    rawFrame.analysis && typeof rawFrame.analysis === "object" ? rawFrame.analysis : {};
  const overlayDataUrl = extractString(analysis.overlay_data_url);
  if (!overlayDataUrl) {
    return null;
  }

  const frameIndex = toFiniteNumber(rawFrame.frame_index);
  const imageName =
    extractString(analysis.image_name) ||
    `frame_${String(index + 1).padStart(2, "0")}.png`;

  return {
    frameIndex,
    timestampMs: toFiniteNumber(rawFrame.timestamp_ms),
    imageName,
    overlayDataUrl,
    overlayOutputPath: extractString(analysis.overlay_output_path),
    metadataOutputPath: extractString(analysis.metadata_output_path),
    selected:
      frameIndex !== null &&
      selectedFrameIndex !== null &&
      frameIndex === selectedFrameIndex,
    analysis,
  };
}

function toFiniteNumber(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

async function readJsonResponse(response, fallbackMessage) {
  const rawBody = await response.text();
  if (!rawBody) {
    throw new Error(`${fallbackMessage} Empty response body.`);
  }

  try {
    return JSON.parse(rawBody);
  } catch (error) {
    const contentType = extractString(response.headers.get("Content-Type")) || "unknown";
    const snippet = rawBody.replace(/\s+/g, " ").trim().slice(0, 220);
    const detail = snippet ? ` Response started with: ${snippet}` : "";
    throw new Error(
      `${fallbackMessage} Status ${response.status}, content type ${contentType}.${detail}`,
    );
  }
}

async function readApiError(response, fallbackMessage) {
  try {
    const payload = await response.json();
    return new Error(payload.error || fallbackMessage);
  } catch (error) {
    return new Error(fallbackMessage);
  }
}
