import {
  deriveDefaultBaseUrl,
  getClipDurationMs,
  getFrameCount,
  getGuideScale,
  normalizedBaseUrlValue,
  persistForm,
} from "../state/app-state.js";
import {
  checkCaptureReadiness,
  uploadAnalyzeImage,
  uploadVideo,
} from "../services/api-client.js";
import {
  canUseLiveCamera,
  initializeCamera,
  recordClip,
  stopCurrentStream,
} from "../services/camera-recorder.js";
import { normalizeError } from "../utils/format.js";

const READINESS_POLL_INTERVAL_MS = 250;
const READINESS_SUCCESS_STREAK = 1;
const READINESS_FAILURE_STREAK = 3;
const READINESS_READY_HOLD_MS = 2000;
const NON_READY_GUIDE_DETAIL = "";
const BASE_GUIDE_WIDTH_RATIO = 0.6;
const BASE_GUIDE_HEIGHT_RATIO = 0.8;
const QUICK_CAPTURE_TARGET_DURATION_MS = 2000;
const QUICK_CAPTURE_RECORD_DURATION_MS = 2200;

export function bindCapturePanel(options) {
  options.elements.captureNote.textContent = buildCaptureNote();
  syncGuideScaleUi(options);
  clearCapturedClipPreview(options);
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
      options.setStatus(
        "Camera switching is only available when live preview is running in the browser.",
        "warning",
      );
      return;
    }

    options.state.facingMode =
      options.state.facingMode === "environment" ? "user" : "environment";
    await initializeCamera({
      elements: options.elements,
      state: options.state,
      setStatus: options.setStatus,
      refreshChrome: options.refreshChrome,
    });
  });

  options.elements.recordButton.addEventListener("click", async () => {
    if (options.state.busy) {
      return;
    }

    if (!canUseLiveCamera()) {
      options.setStatus("Quick capture requires live camera access in the browser.", "warning");
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
        durationMs: QUICK_CAPTURE_RECORD_DURATION_MS,
        refreshChrome: options.refreshChrome,
      });
      await showCapturedClipPreview(options, file);

      stopCurrentStream({
        elements: options.elements,
        state: options.state,
        refreshChrome: options.refreshChrome,
      });

      options.state.activeOperation = "uploading";
      options.refreshChrome();
      options.setStatus("Extracting a sharp frame from the captured clip...", "info");
      persistForm(options.elements);

      const frameFile = await extractFrameFromVideoFile(file);
      options.setStatus("Uploading the extracted frame for analysis...", "info");

      const result = await uploadCapturedFrame(options, frameFile);
      rememberResultOverlay(options, result);
      options.renderResult(result);
      options.setStatus("Analysis finished. The captured clip was reduced to one frame and processed.", "success");
    } catch (error) {
      console.error(error);
      options.setStatus(normalizeError(error, "Recording failed."), "error");
    } finally {
      stopCurrentStream({
        elements: options.elements,
        state: options.state,
        refreshChrome: options.refreshChrome,
      });
      options.state.busy = false;
      options.state.activeOperation = null;
      options.refreshChrome();
    }
  });

  options.elements.uploadButton.addEventListener("click", () => {
    if (options.state.busy) {
      return;
    }

    options.elements.videoUploadInput.value = "";
    options.elements.videoUploadInput.click();
  });

  options.elements.videoUploadInput.addEventListener("change", async (event) => {
    const file = event.target.files && event.target.files[0];
    if (!file || options.state.busy) {
      return;
    }

    const clipDurationMs = getClipDurationMs(options.elements);

    try {
      options.state.busy = true;
      options.state.activeOperation = "uploading";
      options.refreshChrome();
      options.setStatus(`Uploading ${file.name} for analysis...`, "info");
      persistForm(options.elements);

      const result = await uploadSelectedVideo(options, file, clipDurationMs);
      rememberResultOverlay(options, result);
      options.renderResult(result);
      options.setStatus(`Analysis finished. ${file.name} was processed.`, "success");
    } catch (error) {
      console.error(error);
      options.setStatus(normalizeError(error, "Video upload failed."), "error");
    } finally {
      options.elements.videoUploadInput.value = "";
      options.state.busy = false;
      options.state.activeOperation = null;
      options.refreshChrome();
    }
  });
}

async function uploadSelectedVideo(options, file, clipDurationMs) {
  return uploadVideo({
    file,
    baseUrl: normalizedBaseUrlValue(options.elements),
    keepArtifacts: options.elements.keepArtifacts.checked,
    clipDurationMs,
    frameCount: getFrameCount(options.elements),
  });
}

async function uploadCapturedFrame(options, file) {
  return uploadAnalyzeImage({
    file,
    baseUrl: normalizedBaseUrlValue(options.elements),
    keepArtifacts: options.elements.keepArtifacts.checked,
  });
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

    if (
      options.state.readinessReadyHoldUntil &&
      Date.now() < options.state.readinessReadyHoldUntil
    ) {
      setGuideState(options, "ready", formatReadinessMeta(options));
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
        options.state.readinessReadyHoldUntil = Date.now() + READINESS_READY_HOLD_MS;
        setGuideState(options, "ready", formatReadinessMeta(options));
        return;
      }

      if (options.state.readinessConsecutiveFailure >= READINESS_FAILURE_STREAK) {
        options.state.readinessReadyHoldUntil = null;
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
      options.state.readinessReadyHoldUntil = null;
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
  const coverScale = Math.max(viewportWidth / sourceWidth, viewportHeight / sourceHeight);
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

async function showCapturedClipPreview(options, file) {
  if (options.state.capturedClipObjectUrl) {
    URL.revokeObjectURL(options.state.capturedClipObjectUrl);
  }

  const objectUrl = URL.createObjectURL(file);
  options.state.capturedClipObjectUrl = objectUrl;
  options.elements.capturedClipPreview.src = objectUrl;
  options.elements.capturedClipShell.classList.remove("is-hidden");
  options.elements.capturedClipNote.textContent =
    "Preview of the exact clip being sent to the backend.";
  options.elements.capturedClipPreview.load();
  const previewLoaded = await waitForVideoPreview(options.elements.capturedClipPreview);
  if (!previewLoaded) {
    options.elements.capturedClipNote.textContent =
      "Preview could not be loaded in this browser, but the original clip will still be sent.";
    return;
  }
  options.elements.capturedClipPreview.currentTime = 0;
  options.elements.capturedClipPreview.play().catch(() => undefined);
}

async function extractFrameFromVideoFile(file) {
  const objectUrl = URL.createObjectURL(file);
  const videoElement = document.createElement("video");
  videoElement.preload = "metadata";
  videoElement.muted = true;
  videoElement.playsInline = true;
  videoElement.src = objectUrl;

  try {
    await waitForVideoPreview(videoElement);

    if (!Number.isFinite(videoElement.duration) || videoElement.duration <= 0) {
      throw new Error("Captured clip metadata is not available.");
    }

    const targetTime = Math.max(
      0,
      Math.min(videoElement.duration - 0.05, videoElement.duration * 0.5),
    );
    await seekVideo(videoElement, targetTime);

    const sourceWidth = videoElement.videoWidth || 0;
    const sourceHeight = videoElement.videoHeight || 0;
    if (sourceWidth <= 0 || sourceHeight <= 0) {
      throw new Error("Captured clip frame dimensions are not available.");
    }

    const canvas = document.createElement("canvas");
    canvas.width = sourceWidth;
    canvas.height = sourceHeight;

    const context = canvas.getContext("2d", { alpha: false });
    if (!context) {
      throw new Error("Canvas context is not available.");
    }

    context.drawImage(videoElement, 0, 0, sourceWidth, sourceHeight);
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
        0.92,
      );
    });

    const frameName = (file.name || "quick-capture.mp4").replace(/\.[^.]+$/, "") + ".jpg";
    return new File([blob], frameName, {
      type: "image/jpeg",
      lastModified: Date.now(),
    });
  } finally {
    URL.revokeObjectURL(objectUrl);
    videoElement.removeAttribute("src");
    videoElement.load();
  }
}

function clearCapturedClipPreview(options) {
  options.elements.capturedClipPreview.pause();
  options.elements.capturedClipPreview.removeAttribute("src");
  options.elements.capturedClipPreview.load();
  options.elements.capturedClipShell.classList.add("is-hidden");
  options.elements.capturedClipNote.textContent =
    "Preview of the exact clip being sent to the backend.";
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
    return "For the sharpest iPhone result, use Quick 2-second capture. The app uploads the original recorded clip for analysis.";
  }

  return "Quick 2-second capture uploads the original recorded clip for maximum quality.";
}

function isLikelyIosDevice() {
  const userAgent = navigator.userAgent || "";
  const platform = navigator.platform || "";
  return (
    /iPad|iPhone|iPod/i.test(userAgent) ||
    (platform === "MacIntel" && Number(navigator.maxTouchPoints || 0) > 1)
  );
}

async function waitForVideoPreview(videoElement) {
  if (videoElement.readyState >= HTMLMediaElement.HAVE_METADATA) {
    return true;
  }

  return new Promise((resolve) => {
    const handleLoadedMetadata = () => {
      cleanup();
      resolve(true);
    };
    const handleError = () => {
      cleanup();
      resolve(false);
    };
    const cleanup = () => {
      videoElement.removeEventListener("loadedmetadata", handleLoadedMetadata);
      videoElement.removeEventListener("error", handleError);
    };

    videoElement.addEventListener("loadedmetadata", handleLoadedMetadata, { once: true });
    videoElement.addEventListener("error", handleError, { once: true });
  });
}

async function seekVideo(videoElement, timeSeconds) {
  if (Math.abs((videoElement.currentTime || 0) - timeSeconds) < 0.02) {
    return;
  }

  await new Promise((resolve, reject) => {
    const handleSeeked = () => {
      cleanup();
      resolve();
    };
    const handleError = () => {
      cleanup();
      reject(new Error("Video seek failed while extracting the frame."));
    };
    const cleanup = () => {
      videoElement.removeEventListener("seeked", handleSeeked);
      videoElement.removeEventListener("error", handleError);
    };

    videoElement.addEventListener("seeked", handleSeeked, { once: true });
    videoElement.addEventListener("error", handleError, { once: true });
    videoElement.currentTime = timeSeconds;
  });
}
