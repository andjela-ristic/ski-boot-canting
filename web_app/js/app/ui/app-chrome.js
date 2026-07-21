export function renderChrome(options) {
  renderCapabilityPills(options);
  renderCaptureNote(options);
  renderPreviewPlaceholder(options);
  syncBusyState(options);
}

function renderCapabilityPills(options) {
  if (window.isSecureContext) {
    options.elements.securePill.textContent = "Secure connection: ready";
    options.elements.securePill.className = "pill success";
  } else {
    options.elements.securePill.textContent = "Secure connection: limited";
    options.elements.securePill.className = "pill warning";
  }

  if (options.liveAvailable) {
    options.elements.cameraPill.textContent = options.state.currentStream
      ? "In-app camera: active"
      : "In-app camera: available";
    options.elements.cameraPill.className = "pill success";
  } else {
    options.elements.cameraPill.textContent = "In-app camera: upload mode";
    options.elements.cameraPill.className = "pill warning";
  }
}

function renderCaptureNote(options) {
  if (window.isSecureContext) {
    options.elements.captureNote.textContent =
      "Tap Quick 2-second capture to record a short clip and upload it immediately for analysis.";
    return;
  }

  options.elements.captureNote.textContent =
    "Quick 2-second capture requires browser camera access. Open the app through HTTPS for this session.";
}

function renderPreviewPlaceholder(options) {
  if (options.state.currentStream) {
    options.elements.previewPlaceholder.textContent =
      "Live capture is running.";
    return;
  }

  if (options.liveAvailable) {
    options.elements.previewPlaceholder.textContent =
      "Camera preview will appear here when the browser allows live access.";
    return;
  }

  options.elements.previewPlaceholder.textContent =
    "Open the HTTPS version of the app for live camera access, then use Quick 2-second capture.";
}

function syncBusyState(options) {
  const liveUnavailable = !options.liveAvailable;

  options.elements.recordButton.disabled = options.state.busy || liveUnavailable;
  options.elements.toggleCamera.disabled = options.state.busy || liveUnavailable;
  options.elements.resetBaseUrl.disabled = options.state.busy;
  options.elements.toggleCamera.classList.toggle("is-hidden", liveUnavailable);

  if (options.state.activeOperation === "recording") {
    options.elements.recordButton.textContent = "Recording...";
  } else if (options.state.activeOperation === "uploading") {
    options.elements.recordButton.textContent = "Uploading...";
  } else if (liveUnavailable) {
    options.elements.recordButton.textContent = "Quick capture unavailable";
  } else {
    options.elements.recordButton.textContent = "Quick 2-second capture";
  }
}
