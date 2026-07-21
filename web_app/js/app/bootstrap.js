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
    statusPresenter.setStatus("Pripremamo aplikaciju i proveravamo kameru...", "info");

    if (canUseLiveCamera()) {
      initializeCamera({
        elements,
        state,
        setStatus: statusPresenter.setStatus,
      }).catch((error) => {
        console.error(error);
        statusPresenter.setStatus(
          "Kamera u browseru trenutno nije dostupna. Koristi opciju Snimi ili izaberi video.",
          "warning",
        );
        refreshChrome();
      });
    } else {
      statusPresenter.setStatus(
        "Aplikacija je spremna. Za ovo otvaranje koristi opciju Snimi ili izaberi video.",
        "warning",
      );
    }

    registerServiceWorker();

    window.addEventListener("pagehide", () => {
      stopCurrentStream({ elements, state });
    });
  } catch (error) {
    console.error(error);
    if (shell) {
      shell.innerHTML = `
        <div class="card app-fatal">
          <div>
            <h2>Ucitivanje aplikacije nije uspelo.</h2>
            <p>Proveri da li su svi staticki fajlovi dostupni i osvezi stranicu.</p>
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
