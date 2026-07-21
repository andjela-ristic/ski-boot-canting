export function renderChrome(options) {
  renderCapabilityPills(options);
  renderCaptureNote(options);
  renderPreviewPlaceholder(options);
  syncBusyState(options);
}

function renderCapabilityPills(options) {
  if (window.isSecureContext) {
    options.elements.securePill.textContent = "Bezbedna veza: spremna";
    options.elements.securePill.className = "pill success";
  } else {
    options.elements.securePill.textContent = "Bezbedna veza: ogranicena";
    options.elements.securePill.className = "pill warning";
  }

  if (options.liveAvailable) {
    options.elements.cameraPill.textContent = "Kamera u aplikaciji: aktivna";
    options.elements.cameraPill.className = "pill success";
  } else {
    options.elements.cameraPill.textContent = "Kamera u aplikaciji: upload rezim";
    options.elements.cameraPill.className = "pill warning";
  }
}

function renderCaptureNote(options) {
  if (window.isSecureContext) {
    options.elements.captureNote.textContent =
      "Ako browser dozvoli kameru, mozes snimiti kadar direktno ovde. Uvek ostaje dostupna i opcija video uploada.";
    return;
  }

  options.elements.captureNote.textContent =
    "Za ovo otvaranje preporucen je video upload. Ako zelis live kameru u browseru, otvori aplikaciju preko HTTPS linka.";
}

function renderPreviewPlaceholder(options) {
  if (options.liveAvailable) {
    options.elements.previewPlaceholder.textContent =
      "Kamera ce se pojaviti ovde kada browser dozvoli live pristup.";
    return;
  }

  options.elements.previewPlaceholder.textContent =
    "Otvori HTTPS verziju aplikacije za live kameru ili odmah koristi snimi/izaberi video.";
}

function syncBusyState(options) {
  const liveUnavailable = !options.liveAvailable;

  options.elements.recordButton.disabled = options.state.busy || liveUnavailable;
  options.elements.toggleCamera.disabled = options.state.busy || liveUnavailable;
  options.elements.uploadButton.disabled = options.state.busy;
  options.elements.resetBaseUrl.disabled = options.state.busy;
  options.elements.toggleCamera.classList.toggle("is-hidden", liveUnavailable);

  if (options.state.activeOperation === "recording") {
    options.elements.recordButton.textContent = "Snimamo...";
  } else if (liveUnavailable) {
    options.elements.recordButton.textContent = "Brzo snimanje nije dostupno";
  } else {
    options.elements.recordButton.textContent = "Brzo snimi 2 sekunde";
  }

  if (options.state.activeOperation === "uploading") {
    options.elements.uploadButton.textContent = "Analiza u toku...";
  } else {
    options.elements.uploadButton.textContent = "Pokreni analizu";
  }
}
