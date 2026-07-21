import { wait } from "../utils/format.js";

const MAX_IDEAL_WIDTH = 7680;
const MAX_IDEAL_HEIGHT = 4320;
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
}

export async function recordClip(options) {
  const stream = await ensureCameraStream(options);
  const recorderOptions = chooseRecorderOptions(stream);
  const recorder =
    Object.keys(recorderOptions).length > 0
      ? new MediaRecorder(stream, recorderOptions)
      : new MediaRecorder(stream);
  const chunks = [];

  const stopped = new Promise((resolve, reject) => {
    recorder.addEventListener("dataavailable", (event) => {
      if (event.data && event.data.size > 0) {
        chunks.push(event.data);
      }
    });

    recorder.addEventListener("stop", resolve, { once: true });
    recorder.addEventListener(
      "error",
      (event) => {
        reject(event.error || new Error("MediaRecorder error."));
      },
      { once: true },
    );
  });

  options.setStatus(`Recording a ${Math.round(options.durationMs / 100) / 10}s clip...`, "info");
  recorder.start();
  await wait(options.durationMs);

  if (typeof recorder.requestData === "function") {
    recorder.requestData();
  }

  recorder.stop();
  await stopped;

  if (chunks.length === 0) {
    throw new Error("The browser returned an empty recording.");
  }

  const mimeType = recorder.mimeType || chunks[0].type || "video/mp4";
  const extension = mimeType.includes("webm") ? "webm" : "mp4";
  const filename = `canting-scan-${Date.now()}.${extension}`;
  const blob = new Blob(chunks, { type: mimeType });
  return new File([blob], filename, {
    type: mimeType,
    lastModified: Date.now(),
  });
}

function chooseRecorderOptions(stream) {
  const options = {};
  const candidates = [
    'video/mp4;codecs="avc1.42E01E,mp4a.40.2"',
    "video/mp4",
    "video/webm;codecs=vp8,opus",
    "video/webm",
  ];

  const targetVideoBitrate = calculateTargetVideoBitrate(stream);
  if (Number.isFinite(targetVideoBitrate) && targetVideoBitrate > 0) {
    options.videoBitsPerSecond = targetVideoBitrate;
  }

  if (typeof MediaRecorder.isTypeSupported === "function") {
    const supported = candidates.find((candidate) => {
      return MediaRecorder.isTypeSupported(candidate);
    });

    if (supported) {
      options.mimeType = supported;
    }
  }

  return options;
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

function calculateTargetVideoBitrate(stream) {
  const videoTrack =
    stream && typeof stream.getVideoTracks === "function" ? stream.getVideoTracks()[0] : null;
  if (!videoTrack || typeof videoTrack.getSettings !== "function") {
    return 12_000_000;
  }

  const settings = videoTrack.getSettings();
  const width = Number.isFinite(settings.width) ? settings.width : 1920;
  const height = Number.isFinite(settings.height) ? settings.height : 1080;
  const frameRate = Number.isFinite(settings.frameRate) && settings.frameRate > 0 ? settings.frameRate : 30;
  const rawTarget = Math.round(width * height * frameRate * 0.18);
  return clamp(rawTarget, 12_000_000, 85_000_000);
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
