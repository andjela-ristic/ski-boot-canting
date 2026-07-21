import { extractString, fallbackBasename } from "../utils/format.js";

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

  let payload = null;
  try {
    payload = await response.json();
  } catch (error) {
    throw new Error("Backend nije vratio validan JSON odgovor.");
  }

  if (!response.ok) {
    throw new Error(payload.error || `Backend je vratio status ${response.status}.`);
  }

  return normalizeResultPayload(payload);
}

function normalizeResultPayload(payload) {
  const nestedFrameAnalysis =
    Array.isArray(payload.frames) && payload.frames.length > 0
      ? payload.frames[0].analysis || null
      : null;

  const overlayDataUrl =
    extractString(payload.overlay_data_url) ||
    extractString(nestedFrameAnalysis && nestedFrameAnalysis.overlay_data_url);

  if (!overlayDataUrl) {
    throw new Error("Backend odgovor nema overlay_data_url.");
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
    frameCount:
      Number.isFinite(payload.frame_count) && payload.frame_count > 0
        ? Number(payload.frame_count)
        : Array.isArray(payload.frames)
          ? payload.frames.length
          : null,
  };
}
