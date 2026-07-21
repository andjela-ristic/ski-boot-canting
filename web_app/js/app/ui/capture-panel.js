import {
  deriveDefaultBaseUrl,
  getClipDurationMs,
  getFrameCount,
  normalizedBaseUrlValue,
  persistForm,
} from "../state/app-state.js";
import { checkCaptureReadiness, uploadAnalyzeImage, uploadVideo } from "../services/api-client.js";
import {
  canUseLiveCamera,
  initializeCamera,
  recordClip,
} from "../services/camera-recorder.js";
import { formatFileSize, normalizeError } from "../utils/format.js";

const READINESS_POLL_INTERVAL_MS = 250;
const READINESS_SUCCESS_STREAK = 3;
const READINESS_FAILURE_STREAK = 2;
const NON_READY_GUIDE_LABEL = "Place the boot in frame";
const NON_READY_GUIDE_DETAIL = "The frame will turn green when it is good";

export function bindCapturePanel(options) {
  options.elements.captureNote.textContent = buildCaptureNote();
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
      options.setStatus("Preparing a representative frame for analysis...", "info");
      persistForm(options.elements);

      const requestOptions = {
        file: options.state.selectedVideoFile,
        baseUrl: normalizedBaseUrlValue(options.elements),
        keepArtifacts: options.elements.keepArtifacts.checked,
        clipDurationMs: getClipDurationMs(options.elements),
        frameCount: getFrameCount(options.elements),
      };

      let result;

      try {
        const analysisFrame = await extractAnalysisFrameFromClip(
          options.elements.clipPreview,
          options.state.selectedVideoFile,
        );
        options.setStatus("Running analysis on the extracted frame...", "info");
        result = await uploadAnalyzeImage({
          file: analysisFrame,
          baseUrl: requestOptions.baseUrl,
          keepArtifacts: requestOptions.keepArtifacts,
        });
        result.sourceName = options.state.selectedVideoFile.name;
      } catch (error) {
        if (!isFrameExtractionError(error)) {
          throw error;
        }

        console.warn(error);
        options.setStatus(
          "Local frame extraction failed, so the app is falling back to the slower video upload path.",
          "warning",
        );
        result = await uploadVideo(requestOptions);
      }

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
        setGuideState(options, "idle", NON_READY_GUIDE_LABEL, "Waiting for live preview");
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
        setGuideState(options, "ready", "Frame is ready", formatReadinessMeta(options));
        return;
      }

      if (options.state.readinessConsecutiveFailure >= READINESS_FAILURE_STREAK) {
        setGuideState(options, "not-ready", NON_READY_GUIDE_LABEL, NON_READY_GUIDE_DETAIL);
        return;
      }

      setGuideState(options, "pending", NON_READY_GUIDE_LABEL, NON_READY_GUIDE_DETAIL);
    } catch (error) {
      options.state.readinessConsecutiveSuccess = 0;
      options.state.readinessConsecutiveFailure = 0;
      options.state.readinessLastLatencyMs = null;
      options.state.readinessLastScore = null;
      options.state.readinessLastReason = null;
      setGuideState(options, "idle", "Readiness check unavailable", "Check the backend endpoint");
    } finally {
      options.state.readinessRequestInFlight = false;
    }
  };

  setGuideState(options, "idle", NON_READY_GUIDE_LABEL, "Waiting for live preview");
  options.state.readinessLoopHandle = window.setTimeout(tick, READINESS_POLL_INTERVAL_MS);
}

async function capturePreviewFrame(videoElement) {
  const sourceWidth = videoElement.videoWidth || 0;
  const sourceHeight = videoElement.videoHeight || 0;
  if (sourceWidth <= 0 || sourceHeight <= 0) {
    throw new Error("Preview frame is not ready.");
  }

  const targetWidth = Math.min(720, Math.max(480, sourceWidth));
  const scale = targetWidth / sourceWidth;
  const targetHeight = Math.max(1, Math.round(sourceHeight * scale));

  const canvas = document.createElement("canvas");
  canvas.width = targetWidth;
  canvas.height = targetHeight;

  const context = canvas.getContext("2d", { alpha: false });
  if (!context) {
    throw new Error("Canvas context is not available.");
  }

  context.drawImage(videoElement, 0, 0, targetWidth, targetHeight);

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

async function extractAnalysisFrameFromClip(videoElement, sourceFile) {
  try {
    await ensureClipPreviewReady(videoElement);

    const sourceWidth = videoElement.videoWidth || 0;
    const sourceHeight = videoElement.videoHeight || 0;
    if (sourceWidth <= 0 || sourceHeight <= 0) {
      throw new Error("Clip preview dimensions are not available.");
    }

    const duration = Number.isFinite(videoElement.duration) ? videoElement.duration : 0;
    if (duration > 0.1) {
      const targetTime = Math.max(0, Math.min(duration * 0.5, duration - 0.05));
      await seekClipPreview(videoElement, targetTime);
    } else if (videoElement.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      await waitForVideoEvent(videoElement, "loadeddata");
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
          reject(new Error("JPEG analysis frame encoding failed."));
        },
        "image/jpeg",
        0.92,
      );
    });

    return new File([blob], `${deriveFrameFilename(sourceFile)}.jpg`, {
      type: "image/jpeg",
      lastModified: Date.now(),
    });
  } catch (error) {
    throw createFrameExtractionError("Local frame extraction failed.", error);
  }
}

async function ensureClipPreviewReady(videoElement) {
  if (
    videoElement.readyState >= HTMLMediaElement.HAVE_METADATA &&
    videoElement.videoWidth > 0 &&
    videoElement.videoHeight > 0
  ) {
    return;
  }

  if (videoElement.error) {
    throw new Error("Clip preview reported a decode error.");
  }

  await waitForVideoEvent(videoElement, "loadedmetadata");
}

async function seekClipPreview(videoElement, targetTime) {
  if (!Number.isFinite(targetTime) || targetTime <= 0) {
    return;
  }

  if (
    videoElement.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
    Math.abs(videoElement.currentTime - targetTime) < 0.03
  ) {
    return;
  }

  const seekPromise = waitForVideoEvent(videoElement, "seeked");
  videoElement.currentTime = targetTime;
  await seekPromise;
}

function waitForVideoEvent(videoElement, eventName) {
  return new Promise((resolve, reject) => {
    const handleSuccess = () => {
      cleanup();
      resolve();
    };

    const handleError = () => {
      cleanup();
      reject(new Error(`Video event failed while waiting for ${eventName}.`));
    };

    const cleanup = () => {
      videoElement.removeEventListener(eventName, handleSuccess);
      videoElement.removeEventListener("error", handleError);
    };

    videoElement.addEventListener(eventName, handleSuccess, { once: true });
    videoElement.addEventListener("error", handleError, { once: true });
  });
}

function deriveFrameFilename(sourceFile) {
  const sourceName =
    sourceFile && typeof sourceFile.name === "string" && sourceFile.name.trim()
      ? sourceFile.name.trim()
      : `capture-${Date.now()}`;
  const extensionIndex = sourceName.lastIndexOf(".");
  return extensionIndex > 0 ? sourceName.slice(0, extensionIndex) : sourceName;
}

function createFrameExtractionError(message, cause) {
  const error = new Error(message);
  error.name = "FrameExtractionError";
  error.cause = cause;
  return error;
}

function isFrameExtractionError(error) {
  return error instanceof Error && error.name === "FrameExtractionError";
}

function rememberResultOverlay(options, result) {
  if (options.state.resultOverlayObjectUrl) {
    URL.revokeObjectURL(options.state.resultOverlayObjectUrl);
  }

  options.state.resultOverlayObjectUrl = result.overlayObjectUrl || null;
}

function setGuideState(options, stateName, label, detail = "") {
  options.elements.readinessGuide.className = `readiness-guide is-${stateName}`;
  options.elements.readinessGuideBadge.textContent = label;
  options.elements.readinessGuideDetail.textContent = detail;
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
    return "For the sharpest iPhone result, prefer Record or choose video. Live preview is mainly for alignment and can look softer in Safari.";
  }

  return "If the live camera is not available, the app remains fully usable through video upload.";
}

function isLikelyIosDevice() {
  const userAgent = navigator.userAgent || "";
  const platform = navigator.platform || "";
  return (
    /iPad|iPhone|iPod/i.test(userAgent) ||
    (platform === "MacIntel" && Number(navigator.maxTouchPoints || 0) > 1)
  );
}
