export function extractString(value) {
  return typeof value === "string" && value.trim() ? value.trim() : "";
}

export function fallbackBasename(path) {
  if (!path) {
    return "";
  }

  const normalized = path.replaceAll("\\", "/");
  const segments = normalized.split("/");
  return segments[segments.length - 1] || "";
}

export function formatFileSize(size) {
  if (!Number.isFinite(size) || size <= 0) {
    return "nepoznata velicina";
  }

  const units = ["B", "KB", "MB", "GB"];
  let value = size;
  let unitIndex = 0;

  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }

  const digits = value >= 10 || unitIndex === 0 ? 0 : 1;
  return `${value.toFixed(digits)} ${units[unitIndex]}`;
}

export function normalizeError(error, fallbackMessage) {
  if (error instanceof Error && error.message) {
    return error.message;
  }
  return fallbackMessage;
}

export function wait(durationMs) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, durationMs);
  });
}
