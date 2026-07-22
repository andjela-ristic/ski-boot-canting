import { fallbackBasename } from "../utils/format.js";
import { normalizedBaseUrlValue } from "../state/app-state.js";

export function renderResult(options) {
  const frames = Array.isArray(options.result.frames) ? options.result.frames : [];
  const selectedFrame = frames.find((frame) => frame.selected) || null;

  options.elements.resultPlaceholder.classList.add("is-hidden");
  options.elements.resultContent.classList.remove("is-hidden");
  options.elements.overlayImage.src = options.result.overlayDataUrl;
  options.elements.overlayImage.alt = `Overlay result for ${options.result.sourceName}`;
  options.elements.downloadOverlay.href = options.result.overlayDataUrl;
  options.elements.downloadOverlay.download = buildMainOverlayDownloadName(options.result);
  options.elements.resultTitle.textContent = options.result.sourceName;
  options.elements.resultSummary.textContent =
    buildResultSummary(options.result, frames.length, selectedFrame);

  options.elements.resultMeta.innerHTML = "";
  addMetaChip(options.elements.resultMeta, "Clip", options.result.sourceName);
  addMetaChip(
    options.elements.resultMeta,
    "Processing",
    `${options.result.processingTimeMs.toFixed(2)} ms`,
  );
  if (options.result.frameCount) {
    addMetaChip(options.elements.resultMeta, "Frames", String(options.result.frameCount));
  }
  if (selectedFrame) {
    addMetaChip(
      options.elements.resultMeta,
      "Best frame",
      buildFrameDisplayName(selectedFrame),
    );
  }

  renderFrameGallery(options, frames);

  const details = [
    ["Source path", options.result.sourcePath],
    ["Artifacts", options.result.artifactsDir],
    ["Overlay file", options.result.overlayOutputPath],
    ["Metadata file", options.result.metadataOutputPath],
    [
      "Selected frame index",
      Number.isFinite(options.result.selectedFrameIndex)
        ? String(options.result.selectedFrameIndex)
        : null,
    ],
    [
      "Selected frame time",
      Number.isFinite(options.result.selectedTimestampMs)
        ? formatTimestampMs(options.result.selectedTimestampMs)
        : null,
    ],
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

function buildResultSummary(result, frameCount, selectedFrame) {
  const intro = `Analysis finished in ${result.processingTimeMs.toFixed(2)} ms. `;
  if (frameCount > 0) {
    const selectedText = selectedFrame
      ? `The best sampled overlay is shown first (${buildFrameDisplayName(selectedFrame)}). `
      : "The best sampled overlay is shown first. ";
    const closableText =
      frameCount > 1 ? 'Extra frames start closed and can be opened with "Show all frames".' : "";
    return `${intro}${selectedText}All returned frame overlays are listed below. ${closableText}`.trim();
  }

  return `${intro}The overlay below shows the returned visual result.`;
}

function renderFrameGallery(options, frames) {
  options.elements.frameGallery.innerHTML = "";
  options.elements.frameGallerySection.classList.toggle("is-hidden", frames.length === 0);
  options.elements.restoreFramesButton.classList.add("is-hidden");
  options.elements.frameGalleryStatus.classList.add("is-hidden");
  options.elements.frameGalleryStatus.textContent = "";

  if (frames.length === 0) {
    options.elements.restoreFramesButton.onclick = null;
    return;
  }

  const closableCards = [];

  for (const frame of frames) {
    const card = document.createElement("article");
    card.className = "frame-card";
    if (frame.selected) {
      card.classList.add("is-selected");
    }

    const header = document.createElement("div");
    header.className = "frame-card-header";

    const title = document.createElement("h5");
    title.className = "frame-card-title";
    title.textContent = buildFrameDisplayName(frame);
    header.appendChild(title);

    if (frame.selected) {
      const badge = document.createElement("span");
      badge.className = "frame-card-badge";
      badge.textContent = "Best";
      header.appendChild(badge);
    } else {
      const closeButton = document.createElement("button");
      closeButton.type = "button";
      closeButton.className = "ghost-button ghost-button-small frame-card-close";
      closeButton.textContent = "Close";
      closeButton.setAttribute("aria-label", `Close ${buildFrameDisplayName(frame)}`);
      card.classList.add("is-hidden");
      closeButton.addEventListener("click", () => {
        card.classList.add("is-hidden");
        updateFrameGalleryControls(options, closableCards);
      });
      header.appendChild(closeButton);
      closableCards.push(card);
    }

    const figure = document.createElement("div");
    figure.className = "frame-card-figure";
    const image = document.createElement("img");
    image.src = frame.overlayDataUrl;
    image.alt = `Overlay for ${buildFrameDisplayName(frame)}`;
    image.loading = "lazy";
    figure.appendChild(image);

    const footer = document.createElement("div");
    footer.className = "frame-card-footer";

    const meta = document.createElement("div");
    meta.className = "frame-card-meta";
    meta.textContent = buildFrameMeta(frame);
    footer.appendChild(meta);

    const downloadLink = document.createElement("a");
    downloadLink.className = "ghost-button ghost-button-small frame-card-download";
    downloadLink.href = frame.overlayDataUrl;
    downloadLink.download = buildFrameOverlayDownloadName(frame);
    downloadLink.textContent = "Download";
    footer.appendChild(downloadLink);

    card.append(header, figure, footer);
    options.elements.frameGallery.appendChild(card);
  }

  options.elements.restoreFramesButton.onclick = () => {
    for (const card of closableCards) {
      card.classList.remove("is-hidden");
    }
    updateFrameGalleryControls(options, closableCards);
  };
  updateFrameGalleryControls(options, closableCards);
}

function updateFrameGalleryControls(options, closableCards) {
  if (closableCards.length === 0) {
    options.elements.restoreFramesButton.classList.add("is-hidden");
    options.elements.frameGalleryStatus.classList.add("is-hidden");
    options.elements.frameGalleryStatus.textContent = "";
    return;
  }

  const hiddenCount = closableCards.filter((card) => card.classList.contains("is-hidden")).length;
  const hasHiddenCards = hiddenCount > 0;
  options.elements.restoreFramesButton.classList.toggle("is-hidden", !hasHiddenCards);
  options.elements.frameGalleryStatus.classList.toggle("is-hidden", !hasHiddenCards);
  options.elements.frameGalleryStatus.textContent = hasHiddenCards
    ? hiddenCount === 1
      ? '1 extra frame is hidden. Use "Show all frames" to restore it.'
      : `${hiddenCount} extra frames are hidden. Use "Show all frames" to restore them.`
    : "";
}

function buildFrameDisplayName(frame) {
  if (Number.isFinite(frame.frameIndex)) {
    return `Frame ${frame.frameIndex + 1}`;
  }
  return frame.imageName || "Frame";
}

function buildFrameMeta(frame) {
  const parts = [];
  if (Number.isFinite(frame.frameIndex)) {
    parts.push(`idx ${frame.frameIndex}`);
  }
  if (Number.isFinite(frame.timestampMs)) {
    parts.push(formatTimestampMs(frame.timestampMs));
  }
  return parts.join(" | ") || frame.imageName || "Overlay";
}

function formatTimestampMs(value) {
  return `${value.toFixed(0)} ms`;
}

function buildMainOverlayDownloadName(result) {
  const sourceName =
    fallbackBasename(result.sourceName) ||
    fallbackBasename(result.sourcePath) ||
    "canting-result";
  return `${stripExtension(sourceName)}-overlay.png`;
}

function buildFrameOverlayDownloadName(frame) {
  const frameName = frame.imageName || "frame";
  return `${stripExtension(frameName)}-overlay.png`;
}

function stripExtension(fileName) {
  return String(fileName).replace(/\.[^.]+$/, "");
}
