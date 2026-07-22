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

  if (options.previewAvailable) {
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
  if (options.previewAvailable && options.quickCaptureAvailable) {
    options.elements.captureNote.textContent =
      "Tap Quick 2-second capture to record in-app, or choose a saved video clip for upload.";
    return;
  }

  if (options.previewAvailable) {
    options.elements.captureNote.textContent =
      "Live preview is available. If Quick capture stays unavailable in this browser, Choose video still works.";
    return;
  }

  options.elements.captureNote.textContent =
    "Live preview needs HTTPS camera access, but Choose video can still upload a saved clip.";
}

function renderPreviewPlaceholder(options) {
  if (options.state.currentStream) {
    options.elements.previewPlaceholder.textContent =
      "Live capture is running.";
    return;
  }

  if (options.previewAvailable) {
    options.elements.previewPlaceholder.textContent =
      "Camera preview will appear here when the browser allows live access.";
    return;
  }

  options.elements.previewPlaceholder.textContent =
    "Open the HTTPS version of the app for live camera access, then use Quick 2-second capture.";
}

function syncBusyState(options) {
  const previewUnavailable = !options.previewAvailable;
  const quickCaptureUnavailable = !options.quickCaptureAvailable;

  options.elements.recordButton.disabled = options.state.busy || quickCaptureUnavailable;
  options.elements.uploadButton.disabled = options.state.busy;
  options.elements.toggleCamera.disabled = options.state.busy || previewUnavailable;
  options.elements.resetBaseUrl.disabled = options.state.busy;
  options.elements.toggleCamera.classList.toggle("is-hidden", previewUnavailable);

  if (options.state.activeOperation === "recording") {
    options.elements.recordButton.textContent = "Recording...";
  } else if (options.state.activeOperation === "uploading") {
    options.elements.recordButton.textContent = "Uploading...";
  } else if (quickCaptureUnavailable) {
    options.elements.recordButton.textContent = "Quick capture unavailable";
  } else {
    options.elements.recordButton.textContent = "Quick 2-second capture";
  }

  options.elements.uploadButton.textContent =
    options.state.activeOperation === "uploading" ? "Uploading..." : "Choose video";
}
