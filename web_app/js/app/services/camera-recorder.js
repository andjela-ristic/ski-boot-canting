import { wait } from "../utils/format.js";

const MAX_IDEAL_WIDTH = 7680;
const MAX_IDEAL_HEIGHT = 4320;
const RECORDER_TIMESLICE_MS = 250;
const RETRY_RECORDING_WAIT_MS = 120;
const SAFE_RECORDING_MIN_BITRATE = 6_000_000;
const SAFE_RECORDING_MAX_BITRATE = 20_000_000;
const RESOLUTION_FALLBACKS = [
  { width: 7680, height: 4320 },
  { width: 4032, height: 3024 },
  { width: 3840, height: 2160 },
  { width: 3264, height: 2448 },
  { width: 2560, height: 1440 },
  { width: 1920, height: 1440 },
  { width: 1920, height: 1080 },
  { width: 1280, height: 720 },
];

export function canUseLiveCamera() {
  return Boolean(
    window.isSecureContext &&
      navigator.mediaDevices &&
      typeof navigator.mediaDevices.getUserMedia === "function" &&
      typeof window.MediaRecorder === "function",
  );
}

export async function initializeCamera(options) {
  options.setStatus("Preparing the camera...", "info");
  stopCurrentStream(options);

  const stream = await navigator.mediaDevices.getUserMedia({
    video: buildPreferredVideoConstraints(options.state.facingMode),
    audio: false,
  });

  await upgradeVideoTrack(stream);

  options.state.currentStream = stream;
  options.elements.cameraPreview.srcObject = stream;
  options.elements.previewPlaceholder.classList.add("is-hidden");
  await options.elements.cameraPreview.play().catch(() => undefined);
  options.refreshChrome?.();

  const streamDetails = describeStream(stream);
  const iosHint = isLikelyIosDevice()
    ? " On iPhone, Record or choose video is usually sharper than live preview."
    : "";

  options.setStatus(
    `The camera is ready${streamDetails}. You can record a clip directly in the app.${iosHint}`,
    "success",
  );
}

export async function ensureCameraStream(options) {
  if (options.state.currentStream) {
    return options.state.currentStream;
  }

  await initializeCamera(options);
  return options.state.currentStream;
}

export function stopCurrentStream(options) {
  if (!options.state.currentStream) {
    return;
  }

  for (const track of options.state.currentStream.getTracks()) {
    track.stop();
  }

  options.state.currentStream = null;
  options.elements.cameraPreview.srcObject = null;
  options.elements.previewPlaceholder.classList.remove("is-hidden");
  options.refreshChrome?.();
}

export async function recordClip(options) {
  const stream = await ensureCameraStream(options);
  options.setStatus(`Recording a ${Math.round(options.durationMs / 100) / 10}s clip...`, "info");

  try {
    return await recordClipOnce(stream, options.durationMs, { conservative: false });
  } catch (error) {
    if (!isEmptyRecordingError(error)) {
      throw error;
    }

    options.setStatus(
      "The browser returned an empty first take. Retrying with safer recording settings...",
      "warning",
    );
    await softenStreamForRecording(stream);
    return recordClipOnce(stream, options.durationMs, { conservative: true });
  }
}

async function recordClipOnce(stream, durationMs, recorderConfig = {}) {
  const recorderOptions = chooseRecorderOptions(stream, recorderConfig);
  const recorder =
    Object.keys(recorderOptions).length > 0
      ? new MediaRecorder(stream, recorderOptions)
      : new MediaRecorder(stream);
  const chunks = [];

  const recordingFailed = new Promise((_, reject) => {
    recorder.addEventListener(
      "error",
      (event) => {
        reject(event.error || new Error("MediaRecorder error."));
      },
      { once: true },
    );
  });

  const started = new Promise((resolve) => {
    recorder.addEventListener("start", resolve, { once: true });
  });

  const stopped = new Promise((resolve) => {
    recorder.addEventListener("stop", resolve, { once: true });
  });

  recorder.addEventListener("dataavailable", (event) => {
    if (event.data && event.data.size > 0) {
      chunks.push(event.data);
    }
  });

  recorder.start(RECORDER_TIMESLICE_MS);
  await Promise.race([started, recordingFailed]);
  await wait(durationMs);

  if (recorder.state !== "inactive" && typeof recorder.requestData === "function") {
    recorder.requestData();
    await wait(RETRY_RECORDING_WAIT_MS);
  }

  if (recorder.state !== "inactive") {
    recorder.stop();
  }
  await Promise.race([stopped, recordingFailed]);

  if (chunks.length === 0) {
    throw new Error("The browser returned an empty recording.");
  }

  const firstChunk = chunks.find((chunk) => chunk && chunk.size > 0) || chunks[0];
  const mimeType = recorder.mimeType || firstChunk.type || "video/mp4";
  const extension = mimeType.includes("webm") ? "webm" : "mp4";
  const filename = `canting-scan-${Date.now()}.${extension}`;
  const blob = new Blob(chunks, { type: mimeType });
  return new File([blob], filename, {
    type: mimeType,
    lastModified: Date.now(),
  });
}

function chooseRecorderOptions(stream, recorderConfig = {}) {
  const recorderOptions = {};
  const candidates = [
    'video/mp4;codecs="avc1.42E01E,mp4a.40.2"',
    "video/mp4",
    "video/webm;codecs=vp8,opus",
    "video/webm",
  ];

  const targetVideoBitrate = calculateTargetVideoBitrate(stream, recorderConfig.conservative);
  if (Number.isFinite(targetVideoBitrate) && targetVideoBitrate > 0) {
    recorderOptions.videoBitsPerSecond = targetVideoBitrate;
  }

  if (typeof MediaRecorder.isTypeSupported === "function") {
    const supported = candidates.find((candidate) => {
      return MediaRecorder.isTypeSupported(candidate);
    });

    if (supported) {
      recorderOptions.mimeType = supported;
    }
  }

  return recorderOptions;
}

async function softenStreamForRecording(stream) {
  const videoTrack =
    stream && typeof stream.getVideoTracks === "function" ? stream.getVideoTracks()[0] : null;
  if (!videoTrack || typeof videoTrack.applyConstraints !== "function") {
    return;
  }

  const candidates = [
    { width: { ideal: 1920 }, height: { ideal: 1080 }, frameRate: { ideal: 24 } },
    { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 24 } },
    { frameRate: { ideal: 24 } },
  ];

  for (const candidate of candidates) {
    try {
      await videoTrack.applyConstraints(candidate);
      return;
    } catch (error) {
      // Keep trying lower-impact fallbacks until one sticks.
    }
  }
}

function buildPreferredVideoConstraints(facingMode) {
  const supportedConstraints =
    navigator.mediaDevices &&
    typeof navigator.mediaDevices.getSupportedConstraints === "function"
      ? navigator.mediaDevices.getSupportedConstraints()
      : {};

  const constraints = {
    facingMode: { ideal: facingMode },
  };

  if (supportedConstraints.width) {
    constraints.width = { ideal: MAX_IDEAL_WIDTH };
  }
  if (supportedConstraints.height) {
    constraints.height = { ideal: MAX_IDEAL_HEIGHT };
  }
  if (supportedConstraints.frameRate) {
    constraints.frameRate = { ideal: 30 };
  }
  if (supportedConstraints.resizeMode) {
    constraints.resizeMode = "none";
  }
  if (supportedConstraints.width && supportedConstraints.height) {
    constraints.advanced = RESOLUTION_FALLBACKS.map((resolution) => ({
      width: resolution.width,
      height: resolution.height,
    }));
  }

  return constraints;
}

async function upgradeVideoTrack(stream) {
  const videoTrack =
    stream && typeof stream.getVideoTracks === "function" ? stream.getVideoTracks()[0] : null;
  if (!videoTrack || typeof videoTrack.applyConstraints !== "function") {
    return;
  }

  const capabilities =
    typeof videoTrack.getCapabilities === "function" ? videoTrack.getCapabilities() : null;
  const constraintCandidates = buildTrackConstraintCandidates(capabilities);

  for (const candidate of constraintCandidates) {
    try {
      await videoTrack.applyConstraints(candidate);
      return;
    } catch (error) {
      // Ignore individual upgrade attempts and keep the working stream.
    }
  }
}

function buildTrackConstraintCandidates(capabilities) {
  const candidates = [];
  const maxWidth =
    capabilities && capabilities.width && Number.isFinite(capabilities.width.max)
      ? capabilities.width.max
      : null;
  const maxHeight =
    capabilities && capabilities.height && Number.isFinite(capabilities.height.max)
      ? capabilities.height.max
      : null;
  const resizeMode =
    capabilities &&
    Array.isArray(capabilities.resizeMode) &&
    capabilities.resizeMode.includes("none")
      ? "none"
      : null;

  if (Number.isFinite(maxWidth) && Number.isFinite(maxHeight)) {
    candidates.push(
      withOptionalResizeMode(
        {
          width: { exact: maxWidth },
          height: { exact: maxHeight },
        },
        resizeMode,
      ),
      withOptionalResizeMode(
        {
          width: { ideal: maxWidth },
          height: { ideal: maxHeight },
        },
        resizeMode,
      ),
    );
  }

  for (const resolution of RESOLUTION_FALLBACKS) {
    if (
      (Number.isFinite(maxWidth) && resolution.width > maxWidth) ||
      (Number.isFinite(maxHeight) && resolution.height > maxHeight)
    ) {
      continue;
    }

    candidates.push(
      withOptionalResizeMode(
        {
          width: { exact: resolution.width },
          height: { exact: resolution.height },
        },
        resizeMode,
      ),
      withOptionalResizeMode(
        {
          width: { ideal: resolution.width },
          height: { ideal: resolution.height },
        },
        resizeMode,
      ),
    );
  }

  return candidates;
}

function withOptionalResizeMode(constraint, resizeMode) {
  if (!resizeMode) {
    return constraint;
  }

  return {
    ...constraint,
    resizeMode,
  };
}

function calculateTargetVideoBitrate(stream, conservative = false) {
  const videoTrack =
    stream && typeof stream.getVideoTracks === "function" ? stream.getVideoTracks()[0] : null;
  if (!videoTrack || typeof videoTrack.getSettings !== "function") {
    return conservative ? SAFE_RECORDING_MIN_BITRATE : 12_000_000;
  }

  const settings = videoTrack.getSettings();
  const width = Number.isFinite(settings.width) ? settings.width : 1920;
  const height = Number.isFinite(settings.height) ? settings.height : 1080;
  const frameRate =
    Number.isFinite(settings.frameRate) && settings.frameRate > 0
      ? settings.frameRate
      : 30;
  const ratio = conservative ? 0.06 : 0.08;
  const rawTarget = Math.round(width * height * frameRate * ratio);
  return clamp(rawTarget, SAFE_RECORDING_MIN_BITRATE, SAFE_RECORDING_MAX_BITRATE);
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function describeStream(stream) {
  const videoTrack =
    stream && typeof stream.getVideoTracks === "function" ? stream.getVideoTracks()[0] : null;
  if (!videoTrack || typeof videoTrack.getSettings !== "function") {
    return "";
  }

  const settings = videoTrack.getSettings();
  const parts = [];

  if (Number.isFinite(settings.width) && Number.isFinite(settings.height)) {
    parts.push(`${settings.width}x${settings.height}`);
  }
  if (Number.isFinite(settings.frameRate) && settings.frameRate > 0) {
    parts.push(`${Math.round(settings.frameRate)} fps`);
  }

  return parts.length > 0 ? ` at ${parts.join(", ")}` : "";
}

function isLikelyIosDevice() {
  const userAgent = navigator.userAgent || "";
  const platform = navigator.platform || "";
  return (
    /iPad|iPhone|iPod/i.test(userAgent) ||
    (platform === "MacIntel" && Number(navigator.maxTouchPoints || 0) > 1)
  );
}

function isEmptyRecordingError(error) {
  return error instanceof Error && error.message === "The browser returned an empty recording.";
}
