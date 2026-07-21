import {
  deriveDefaultBaseUrl,
  getClipDurationMs,
  getFrameCount,
  getGuideScale,
  normalizedBaseUrlValue,
  persistForm,
} from "../state/app-state.js";
import { checkCaptureReadiness, uploadVideo } from "../services/api-client.js";
import {
  canUseLiveCamera,
  initializeCamera,
  recordClip,
} from "../services/camera-recorder.js";
import { formatFileSize, normalizeError } from "../utils/format.js";

const READINESS_POLL_INTERVAL_MS = 250;
const READINESS_SUCCESS_STREAK = 2;
const READINESS_FAILURE_STREAK = 3;
const NON_READY_GUIDE_DETAIL = "";
const BASE_GUIDE_WIDTH_RATIO = 0.6;
const BASE_GUIDE_HEIGHT_RATIO = 0.8;

export function bindCapturePanel(options) {
  options.elements.captureNote.textContent = buildCaptureNote();
  syncGuideScaleUi(options);
  startReadinessLoop(options);

  options.elements.resetBaseUrl.addEventListener("click", () => {
    options.elements.baseUrl.value = deriveDefaultBaseUrl();
    persistForm(options.elements);
    options.setStatus("Advanced settings were reset to their default values.", "info");
  });

  options.elements.baseUrl.addEventListener("change", () => {
    persistForm(options.elements);
  });
  options.elements.keepArtifacts.addEventListener("change", () => {
    persistForm(options.elements);
  });
  options.elements.frameCount.addEventListener("change", () => {
    persistForm(options.elements);
  });
  options.elements.clipDuration.addEventListener("change", () => {
    persistForm(options.elements);
  });
  options.elements.guideScale.addEventListener("input", () => {
    syncGuideScaleUi(options);
    persistForm(options.elements);
  });

  options.elements.toggleCamera.addEventListener("click", async () => {
    if (!canUseLiveCamera()) {
      options.setStatus("Camera switching is only available when live preview is running in the browser.", "warning");
      return;
    }

    options.state.facingMode =
      options.state.facingMode === "environment" ? "user" : "environment";
    await initializeCamera({
      elements: options.elements,
      state: options.state,
      setStatus: options.setStatus,
    });
  });

  options.elements.recordButton.addEventListener("click", async () => {
    if (options.state.busy) {
      return;
    }

    if (!canUseLiveCamera()) {
      options.setStatus("Use the Record or choose video option for this session.", "warning");
      return;
    }

    try {
      options.state.busy = true;
      options.state.activeOperation = "recording";
      options.refreshChrome();

      const file = await recordClip({
        elements: options.elements,
        state: options.state,
        setStatus: options.setStatus,
        durationMs: getClipDurationMs(options.elements),
      });

      setSelectedVideo(options, file, "Clip is ready for analysis.");
      options.setStatus("The clip was saved. You can run the analysis right away.", "success");
    } catch (error) {
      console.error(error);
      options.setStatus(normalizeError(error, "Recording failed."), "error");
    } finally {
      options.state.busy = false;
      options.state.activeOperation = null;
      options.refreshChrome();
    }
  });

  options.elements.videoFile.addEventListener("change", () => {
    const file = options.elements.videoFile.files && options.elements.videoFile.files[0];
    if (!file) {
      return;
    }

    setSelectedVideo(options, file, "Video is selected and ready for analysis.");
    options.setStatus("The video is ready. Run the analysis whenever you want.", "success");
  });

  options.elements.uploadButton.addEventListener("click", async () => {
    if (options.state.busy) {
      return;
    }

    if (!options.state.selectedVideoFile) {
      options.setStatus("Record or choose a video file first.", "warning");
      return;
    }

    try {
      options.state.busy = true;
      options.state.activeOperation = "uploading";
      options.refreshChrome();
      options.setStatus("Uploading the original video for analysis...", "info");
      persistForm(options.elements);

      const requestOptions = {
        file: options.state.selectedVideoFile,
        baseUrl: normalizedBaseUrlValue(options.elements),
        keepArtifacts: options.elements.keepArtifacts.checked,
        clipDurationMs: getClipDurationMs(options.elements),
        frameCount: getFrameCount(options.elements),
      };

      const result = await uploadVideo(requestOptions);

      rememberResultOverlay(options, result);
      options.renderResult(result);
      options.setStatus("Analysis finished. The result is shown below.", "success");
    } catch (error) {
      console.error(error);
      options.setStatus(normalizeError(error, "Upload or processing failed."), "error");
    } finally {
      options.state.busy = false;
      options.state.activeOperation = null;
      options.refreshChrome();
    }
  });
}

function setSelectedVideo(options, file, label) {
  options.state.selectedVideoFile = file;
  options.elements.selectedFileLabel.textContent = label;
  options.elements.selectedFileMeta.textContent = `${file.name} - ${formatFileSize(file.size)}`;

  if (options.state.clipPreviewUrl) {
    URL.revokeObjectURL(options.state.clipPreviewUrl);
  }

  options.state.clipPreviewUrl = URL.createObjectURL(file);
  options.elements.clipPreview.src = options.state.clipPreviewUrl;
  options.elements.clipShell.classList.remove("is-hidden");
}

function startReadinessLoop(options) {
  if (options.state.readinessLoopHandle) {
    window.clearTimeout(options.state.readinessLoopHandle);
  }

  const tick = async () => {
    options.state.readinessLoopHandle = window.setTimeout(tick, READINESS_POLL_INTERVAL_MS);

    if (
      options.state.busy ||
      !options.state.currentStream ||
      options.elements.cameraPreview.readyState < HTMLMediaElement.HAVE_CURRENT_DATA
    ) {
      if (!options.state.currentStream) {
        options.state.readinessConsecutiveSuccess = 0;
        options.state.readinessConsecutiveFailure = 0;
        options.state.readinessLastLatencyMs = null;
        options.state.readinessLastScore = null;
        options.state.readinessLastReason = null;
        setGuideState(options, "idle", "Waiting for live preview");
      }
      return;
    }

    if (options.state.readinessRequestInFlight) {
      return;
    }

    try {
      options.state.readinessRequestInFlight = true;
      const frame = await capturePreviewFrame(options.elements.cameraPreview);
      const readiness = await checkCaptureReadiness({
        frame,
        baseUrl: normalizedBaseUrlValue(options.elements),
        guideScale: getGuideScale(options.elements),
      });

      options.state.readinessLastLatencyMs = readiness.latencyMs;
      options.state.readinessLastScore = Number.isFinite(readiness.score) ? readiness.score : null;
      options.state.readinessLastReason = readiness.reason;

      if (readiness.success) {
        options.state.readinessConsecutiveSuccess += 1;
        options.state.readinessConsecutiveFailure = 0;
      } else {
        options.state.readinessConsecutiveFailure += 1;
        options.state.readinessConsecutiveSuccess = 0;
      }

      if (options.state.readinessConsecutiveSuccess >= READINESS_SUCCESS_STREAK) {
        setGuideState(options, "ready", formatReadinessMeta(options));
        return;
      }

      if (options.state.readinessConsecutiveFailure >= READINESS_FAILURE_STREAK) {
        setGuideState(options, "not-ready", NON_READY_GUIDE_DETAIL);
        return;
      }

      setGuideState(options, "pending", NON_READY_GUIDE_DETAIL);
    } catch (error) {
      options.state.readinessConsecutiveSuccess = 0;
      options.state.readinessConsecutiveFailure = 0;
      options.state.readinessLastLatencyMs = null;
      options.state.readinessLastScore = null;
      options.state.readinessLastReason = null;
      setGuideState(options, "idle", "Check the backend endpoint");
    } finally {
      options.state.readinessRequestInFlight = false;
    }
  };

  setGuideState(options, "idle", "Waiting for live preview");
  options.state.readinessLoopHandle = window.setTimeout(tick, READINESS_POLL_INTERVAL_MS);
}

async function capturePreviewFrame(videoElement) {
  const sourceWidth = videoElement.videoWidth || 0;
  const sourceHeight = videoElement.videoHeight || 0;
  if (sourceWidth <= 0 || sourceHeight <= 0) {
    throw new Error("Preview frame is not ready.");
  }

  const viewportWidth = Math.max(1, Math.round(videoElement.clientWidth || sourceWidth));
  const viewportHeight = Math.max(1, Math.round(videoElement.clientHeight || sourceHeight));
  const coverScale = Math.max(
    viewportWidth / sourceWidth,
    viewportHeight / sourceHeight,
  );
  const visibleSourceWidth = sourceWidth / coverScale;
  const visibleSourceHeight = sourceHeight / coverScale;
  const visibleSourceX = Math.max(0, (sourceWidth - visibleSourceWidth) * 0.5);
  const visibleSourceY = Math.max(0, (sourceHeight - visibleSourceHeight) * 0.5);

  const targetWidth = Math.min(720, Math.max(480, viewportWidth));
  const targetHeight = Math.max(
    1,
    Math.round(targetWidth * (visibleSourceHeight / visibleSourceWidth)),
  );

  const canvas = document.createElement("canvas");
  canvas.width = targetWidth;
  canvas.height = targetHeight;

  const context = canvas.getContext("2d", { alpha: false });
  if (!context) {
    throw new Error("Canvas context is not available.");
  }

  context.drawImage(
    videoElement,
    visibleSourceX,
    visibleSourceY,
    visibleSourceWidth,
    visibleSourceHeight,
    0,
    0,
    targetWidth,
    targetHeight,
  );

  const blob = await new Promise((resolve, reject) => {
    canvas.toBlob(
      (value) => {
        if (value) {
          resolve(value);
          return;
        }
        reject(new Error("JPEG frame encoding failed."));
      },
      "image/jpeg",
      0.72,
    );
  });

  return new File([blob], `preview-${Date.now()}.jpg`, {
    type: "image/jpeg",
    lastModified: Date.now(),
  });
}

function rememberResultOverlay(options, result) {
  if (options.state.resultOverlayObjectUrl) {
    URL.revokeObjectURL(options.state.resultOverlayObjectUrl);
  }

  options.state.resultOverlayObjectUrl = result.overlayObjectUrl || null;
}

function syncGuideScaleUi(options) {
  const guideScale = getGuideScale(options.elements);
  const widthRatio = Math.min(0.96, BASE_GUIDE_WIDTH_RATIO * guideScale);
  const heightRatio = Math.min(0.96, BASE_GUIDE_HEIGHT_RATIO * guideScale);

  options.elements.readinessGuide.style.setProperty("--guide-width-ratio", widthRatio.toFixed(4));
  options.elements.readinessGuide.style.setProperty("--guide-height-ratio", heightRatio.toFixed(4));
  options.elements.guideScaleValue.textContent = `${Math.round(guideScale * 100)}%`;
}

function setGuideState(options, stateName, detail = "") {
  options.elements.readinessGuide.className = `readiness-guide is-${stateName}`;
  if (options.elements.readinessGuideDetail) {
    options.elements.readinessGuideDetail.textContent = detail;
  }
  options.state.readinessLastOutcome = stateName;
}

function formatReadinessMeta(options) {
  const score =
    Number.isFinite(options.state.readinessLastScore)
      ? `${Math.round(options.state.readinessLastScore * 100)}%`
      : "n/a";
  const latency =
    Number.isFinite(options.state.readinessLastLatencyMs)
      ? `${Math.round(options.state.readinessLastLatencyMs)} ms`
      : "n/a";
  return `Score ${score} - ${latency}`;
}

function buildCaptureNote() {
  if (isLikelyIosDevice()) {
    return "For the sharpest iPhone result, prefer Record or choose video. The app now uploads the original video for analysis instead of extracting a browser-compressed frame.";
  }

  return "For maximum quality, video analysis uploads the original clip instead of a browser-extracted preview frame.";
}

function isLikelyIosDevice() {
  const userAgent = navigator.userAgent || "";
  const platform = navigator.platform || "";
  return (
    /iPad|iPhone|iPod/i.test(userAgent) ||
    (platform === "MacIntel" && Number(navigator.maxTouchPoints || 0) > 1)
  );
}
