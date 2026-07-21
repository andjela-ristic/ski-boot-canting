import { normalizedBaseUrlValue } from "../state/app-state.js";

export function renderResult(options) {
  options.elements.resultPlaceholder.classList.add("is-hidden");
  options.elements.resultContent.classList.remove("is-hidden");
  options.elements.overlayImage.src = options.result.overlayDataUrl;
  options.elements.overlayImage.alt = `Overlay rezultat za ${options.result.sourceName}`;
  options.elements.resultTitle.textContent = options.result.sourceName;
  options.elements.resultSummary.textContent =
    `Analiza je zavrsena za ${options.result.processingTimeMs.toFixed(2)} ms. ` +
    "Overlay ispod prikazuje vraceni vizuelni rezultat.";

  options.elements.resultMeta.innerHTML = "";
  addMetaChip(options.elements.resultMeta, "Snimak", options.result.sourceName);
  addMetaChip(
    options.elements.resultMeta,
    "Obrada",
    `${options.result.processingTimeMs.toFixed(2)} ms`,
  );
  if (options.result.frameCount) {
    addMetaChip(options.elements.resultMeta, "Frejmovi", String(options.result.frameCount));
  }

  const details = [
    ["Izvorni put", options.result.sourcePath],
    ["Artefakti", options.result.artifactsDir],
    ["Overlay fajl", options.result.overlayOutputPath],
    ["Metadata fajl", options.result.metadataOutputPath],
    ["Server", normalizedBaseUrlValue(options.elements)],
  ];

  options.elements.resultDetails.innerHTML = "";
  let detailCount = 0;
  for (const [label, value] of details) {
    if (!value) {
      continue;
    }

    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = value;
    options.elements.resultDetails.append(dt, dd);
    detailCount += 1;
  }

  options.elements.technicalDetails.classList.toggle("is-hidden", detailCount === 0);
  options.elements.resultContent.scrollIntoView({
    behavior: "smooth",
    block: "start",
  });
}

function addMetaChip(container, label, value) {
  const chip = document.createElement("span");
  chip.className = "meta-chip";
  chip.textContent = `${label}: ${value}`;
  container.appendChild(chip);
}
