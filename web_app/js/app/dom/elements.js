export function getAppElements() {
  return {
    baseUrl: requireElement("base-url"),
    keepArtifacts: requireElement("keep-artifacts"),
    frameCount: requireElement("frame-count"),
    clipDuration: requireElement("clip-duration"),
    resetBaseUrl: requireElement("reset-base-url"),
    securePill: requireElement("secure-pill"),
    cameraPill: requireElement("camera-pill"),
    cameraPreview: requireElement("camera-preview"),
    readinessGuide: requireElement("readiness-guide"),
    readinessGuideDetail: getOptionalElement("readiness-guide-detail"),
    guideScale: requireElement("guide-scale"),
    guideScaleValue: requireElement("guide-scale-value"),
    previewPlaceholder: requireElement("preview-placeholder"),
    capturedClipShell: requireElement("captured-clip-shell"),
    capturedClipPreview: requireElement("captured-clip-preview"),
    capturedClipNote: requireElement("captured-clip-note"),
    recordButton: requireElement("record-button"),
    uploadButton: requireElement("upload-button"),
    videoUploadInput: requireElement("video-upload-input"),
    toggleCamera: requireElement("toggle-camera"),
    captureNote: requireElement("capture-note"),
    statusPanel: requireElement("status-panel"),
    resultPlaceholder: requireElement("result-placeholder"),
    resultContent: requireElement("result-content"),
    resultMeta: requireElement("result-meta"),
    downloadOverlay: requireElement("download-overlay"),
    overlayImage: requireElement("overlay-image"),
    frameGallerySection: requireElement("frame-gallery-section"),
    restoreFramesButton: requireElement("restore-frames-button"),
    frameGalleryStatus: requireElement("frame-gallery-status"),
    frameGallery: requireElement("frame-gallery"),
    resultTitle: requireElement("result-title"),
    resultSummary: requireElement("result-summary"),
    resultDetails: requireElement("result-details"),
    technicalDetails: requireElement("technical-details"),
  };
}

function requireElement(id) {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error(`Missing required DOM element: #${id}`);
  }
  return element;
}

function getOptionalElement(id) {
  return document.getElementById(id);
}
