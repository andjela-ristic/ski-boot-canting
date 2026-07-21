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
    options.elements.cameraPill.textContent = "In-app camera: active";
    options.elements.cameraPill.className = "pill success";
  } else {
    options.elements.cameraPill.textContent = "In-app camera: upload mode";
    options.elements.cameraPill.className = "pill warning";
  }
}

function renderCaptureNote(options) {
  if (window.isSecureContext) {
    options.elements.captureNote.textContent =
      "If the browser allows camera access, you can record a clip directly here. Video upload remains available at all times.";
    return;
  }

  options.elements.captureNote.textContent =
    "Video upload is recommended for this session. If you want live camera access in the browser, open the app through an HTTPS link.";
}

function renderPreviewPlaceholder(options) {
  if (options.liveAvailable) {
    options.elements.previewPlaceholder.textContent =
      "Camera preview will appear here when the browser allows live access.";
    return;
  }

  options.elements.previewPlaceholder.textContent =
    "Open the HTTPS version of the app for live camera access, or use Record or choose video right away.";
}

function syncBusyState(options) {
  const liveUnavailable = !options.liveAvailable;

  options.elements.recordButton.disabled = options.state.busy || liveUnavailable;
  options.elements.toggleCamera.disabled = options.state.busy || liveUnavailable;
  options.elements.uploadButton.disabled = options.state.busy;
  options.elements.resetBaseUrl.disabled = options.state.busy;
  options.elements.toggleCamera.classList.toggle("is-hidden", liveUnavailable);

  if (options.state.activeOperation === "recording") {
    options.elements.recordButton.textContent = "Recording...";
  } else if (liveUnavailable) {
    options.elements.recordButton.textContent = "Quick capture unavailable";
  } else {
    options.elements.recordButton.textContent = "Quick 2-second capture";
  }

  if (options.state.activeOperation === "uploading") {
    options.elements.uploadButton.textContent = "Analysis running...";
  } else {
    options.elements.uploadButton.textContent = "Run Analysis";
  }
}
