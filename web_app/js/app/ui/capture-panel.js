import {
  deriveDefaultBaseUrl,
  getClipDurationMs,
  getFrameCount,
  normalizedBaseUrlValue,
  persistForm,
} from "../state/app-state.js";
import { uploadVideo } from "../services/api-client.js";
import {
  canUseLiveCamera,
  initializeCamera,
  recordClip,
} from "../services/camera-recorder.js";
import { formatFileSize, normalizeError } from "../utils/format.js";

export function bindCapturePanel(options) {
  options.elements.resetBaseUrl.addEventListener("click", () => {
    options.elements.baseUrl.value = deriveDefaultBaseUrl();
    persistForm(options.elements);
    options.setStatus("Napredna podesavanja su vracena na podrazumevane vrednosti.", "info");
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
      options.setStatus("Promena kamere je dostupna samo kada live preview radi u browseru.", "warning");
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
      options.setStatus("U ovom otvaranju koristi opciju Snimi ili izaberi video.", "warning");
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

      setSelectedVideo(options, file, "Snimak je spreman za analizu.");
      options.setStatus("Snimak je sacuvan. Mozes odmah da pokrenes analizu.", "success");
    } catch (error) {
      console.error(error);
      options.setStatus(normalizeError(error, "Snimanje nije uspelo."), "error");
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

    setSelectedVideo(options, file, "Video je izabran i spreman za analizu.");
    options.setStatus("Video je spreman. Pokreni analizu kada zelis.", "success");
  });

  options.elements.uploadButton.addEventListener("click", async () => {
    if (options.state.busy) {
      return;
    }

    if (!options.state.selectedVideoFile) {
      options.setStatus("Prvo snimi ili izaberi video fajl.", "warning");
      return;
    }

    try {
      options.state.busy = true;
      options.state.activeOperation = "uploading";
      options.refreshChrome();
      options.setStatus("Analiza je u toku. Ovo moze trajati nekoliko sekundi...", "info");
      persistForm(options.elements);

      const result = await uploadVideo({
        file: options.state.selectedVideoFile,
        baseUrl: normalizedBaseUrlValue(options.elements),
        keepArtifacts: options.elements.keepArtifacts.checked,
        clipDurationMs: getClipDurationMs(options.elements),
        frameCount: getFrameCount(options.elements),
      });

      options.renderResult(result);
      options.setStatus("Analiza je zavrsena. Rezultat je prikazan ispod.", "success");
    } catch (error) {
      console.error(error);
      options.setStatus(normalizeError(error, "Upload ili obrada nisu uspeli."), "error");
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
  options.elements.selectedFileMeta.textContent = `${file.name} • ${formatFileSize(file.size)}`;

  if (options.state.clipPreviewUrl) {
    URL.revokeObjectURL(options.state.clipPreviewUrl);
  }

  options.state.clipPreviewUrl = URL.createObjectURL(file);
  options.elements.clipPreview.src = options.state.clipPreviewUrl;
  options.elements.clipShell.classList.remove("is-hidden");
}
