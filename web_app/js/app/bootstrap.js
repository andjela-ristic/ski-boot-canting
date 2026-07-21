import { getAppElements } from "./dom/elements.js";
import { loadPanelFragments } from "./layout/load-panel-fragments.js";
import {
  createAppState,
  hydrateForm,
} from "./state/app-state.js";
import {
  canUseLiveCamera,
  initializeCamera,
  stopCurrentStream,
} from "./services/camera-recorder.js";
import { bindCapturePanel } from "./ui/capture-panel.js";
import { createStatusPresenter } from "./ui/status-panel.js";
import { renderChrome } from "./ui/app-chrome.js";
import { renderResult } from "./ui/result-panel.js";

export async function bootstrapApp() {
  const shell = document.querySelector(".shell");

  try {
    await loadPanelFragments();

    const elements = getAppElements();
    const state = createAppState();
    const statusPresenter = createStatusPresenter(elements);

    hydrateForm(elements);

    const refreshChrome = () => {
      renderChrome({
        elements,
        state,
        liveAvailable: canUseLiveCamera(),
      });
    };

    bindCapturePanel({
      elements,
      state,
      setStatus: statusPresenter.setStatus,
      refreshChrome,
      renderResult: (result) => {
        renderResult({
          elements,
          result,
        });
      },
    });

    refreshChrome();
    statusPresenter.setStatus("Preparing the app and checking the camera...", "info");

    if (canUseLiveCamera()) {
      initializeCamera({
        elements,
        state,
        setStatus: statusPresenter.setStatus,
        refreshChrome,
      }).catch((error) => {
        console.error(error);
        statusPresenter.setStatus(
          "The browser camera is not available right now. Use Record or choose video.",
          "warning",
        );
        refreshChrome();
      });
    } else {
      statusPresenter.setStatus(
        "The app is ready. For this session, use Record or choose video.",
        "warning",
      );
    }

    registerServiceWorker();

    window.addEventListener("pagehide", () => {
      stopCurrentStream({ elements, state });
      if (state.resultOverlayObjectUrl) {
        URL.revokeObjectURL(state.resultOverlayObjectUrl);
        state.resultOverlayObjectUrl = null;
      }
    });
  } catch (error) {
    console.error(error);
    if (shell) {
      shell.innerHTML = `
        <div class="card app-fatal">
          <div>
            <h2>App loading failed.</h2>
            <p>Check that all static files are available and refresh the page.</p>
          </div>
        </div>
      `;
    }
  }
}

function registerServiceWorker() {
  if (!("serviceWorker" in navigator)) {
    return;
  }

  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/service-worker.js").catch((error) => {
      console.error("Service worker registration failed:", error);
    });
  });
}
