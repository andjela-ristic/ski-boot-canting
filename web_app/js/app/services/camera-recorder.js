import { wait } from "../utils/format.js";

export function canUseLiveCamera() {
  return Boolean(
    window.isSecureContext &&
      navigator.mediaDevices &&
      typeof navigator.mediaDevices.getUserMedia === "function" &&
      typeof window.MediaRecorder === "function",
  );
}

export async function initializeCamera(options) {
  options.setStatus("Pripremamo kameru...", "info");
  stopCurrentStream(options);

  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: { ideal: options.state.facingMode },
    },
    audio: false,
  });

  options.state.currentStream = stream;
  options.elements.cameraPreview.srcObject = stream;
  options.elements.previewPlaceholder.classList.add("is-hidden");
  await options.elements.cameraPreview.play().catch(() => undefined);
  options.setStatus("Kamera je spremna. Mozes snimiti kadar direktno u aplikaciji.", "success");
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
  const recorderOptions = chooseRecorderOptions();
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

  options.setStatus(`Snimamo ${Math.round(options.durationMs / 100) / 10}s kadar...`, "info");
  recorder.start();
  await wait(options.durationMs);

  if (typeof recorder.requestData === "function") {
    recorder.requestData();
  }

  recorder.stop();
  await stopped;

  if (chunks.length === 0) {
    throw new Error("Browser je vratio prazan snimak.");
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

function chooseRecorderOptions() {
  const candidates = [
    'video/mp4;codecs="avc1.42E01E,mp4a.40.2"',
    "video/mp4",
    "video/webm;codecs=vp8,opus",
    "video/webm",
  ];

  if (typeof MediaRecorder.isTypeSupported !== "function") {
    return {};
  }

  const supported = candidates.find((candidate) => {
    return MediaRecorder.isTypeSupported(candidate);
  });

  return supported ? { mimeType: supported } : {};
}
