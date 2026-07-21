const STORAGE_KEYS = {
  baseUrl: "canting.baseUrl",
  keepArtifacts: "canting.keepArtifacts",
  frameCount: "canting.frameCount",
  clipDurationMs: "canting.clipDurationMs",
  guideScale: "canting.guideScale",
};

export function createAppState() {
  return {
    currentStream: null,
    facingMode: "environment",
    resultOverlayObjectUrl: null,
    readinessLoopHandle: null,
    readinessRequestInFlight: false,
    readinessConsecutiveSuccess: 0,
    readinessConsecutiveFailure: 0,
    readinessLastOutcome: "idle",
    readinessLastLatencyMs: null,
    readinessLastScore: null,
    readinessLastReason: null,
    busy: false,
    activeOperation: null,
  };
}

export function hydrateForm(elements) {
  elements.baseUrl.value =
    window.localStorage.getItem(STORAGE_KEYS.baseUrl) || deriveDefaultBaseUrl();

  const keepArtifacts = window.localStorage.getItem(STORAGE_KEYS.keepArtifacts);
  elements.keepArtifacts.checked = keepArtifacts === "true";

  const frameCount = window.localStorage.getItem(STORAGE_KEYS.frameCount);
  if (frameCount) {
    elements.frameCount.value = frameCount;
  }

  const clipDuration = window.localStorage.getItem(STORAGE_KEYS.clipDurationMs);
  if (clipDuration) {
    elements.clipDuration.value = clipDuration;
  }

  const guideScale = window.localStorage.getItem(STORAGE_KEYS.guideScale);
  if (guideScale) {
    elements.guideScale.value = String(normalizeGuideScaleValue(guideScale));
  }
}

export function persistForm(elements) {
  window.localStorage.setItem(STORAGE_KEYS.baseUrl, normalizedBaseUrlValue(elements));
  window.localStorage.setItem(
    STORAGE_KEYS.keepArtifacts,
    String(elements.keepArtifacts.checked),
  );
  window.localStorage.setItem(STORAGE_KEYS.frameCount, String(getFrameCount(elements)));
  window.localStorage.setItem(
    STORAGE_KEYS.clipDurationMs,
    String(getClipDurationMs(elements)),
  );
  window.localStorage.setItem(STORAGE_KEYS.guideScale, String(getGuideScale(elements)));
}

export function deriveDefaultBaseUrl() {
  if (window.location.origin && window.location.origin.startsWith("http")) {
    return window.location.origin;
  }
  return "http://127.0.0.1:8000";
}

export function normalizedBaseUrlValue(elements) {
  const raw = (elements.baseUrl.value || "").trim();
  if (!raw) {
    return deriveDefaultBaseUrl();
  }

  return raw.endsWith("/") ? raw.slice(0, -1) : raw;
}

export function getFrameCount(elements) {
  const parsed = Number.parseInt(elements.frameCount.value, 10);
  if (!Number.isFinite(parsed)) {
    return 6;
  }
  return Math.min(30, Math.max(1, parsed));
}

export function getClipDurationMs(elements) {
  const parsed = Number.parseInt(elements.clipDuration.value, 10);
  if (!Number.isFinite(parsed)) {
    return 2000;
  }
  return Math.min(10000, Math.max(500, parsed));
}

export function getGuideScale(elements) {
  return normalizeGuideScaleValue(elements.guideScale.value);
}

function normalizeGuideScaleValue(rawValue) {
  const parsed = Number.parseFloat(rawValue);
  if (!Number.isFinite(parsed)) {
    return 1;
  }
  return Math.min(1.2, Math.max(0.9, parsed));
}
