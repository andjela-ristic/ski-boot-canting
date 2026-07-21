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
    readinessGuideBadge: requireElement("readiness-guide-badge"),
    readinessGuideDetail: requireElement("readiness-guide-detail"),
    previewPlaceholder: requireElement("preview-placeholder"),
    recordButton: requireElement("record-button"),
    toggleCamera: requireElement("toggle-camera"),
    videoFile: requireElement("video-file"),
    selectedFileLabel: requireElement("selected-file-label"),
    selectedFileMeta: requireElement("selected-file-meta"),
    uploadButton: requireElement("upload-button"),
    clipShell: requireElement("clip-shell"),
    clipPreview: requireElement("clip-preview"),
    captureNote: requireElement("capture-note"),
    statusPanel: requireElement("status-panel"),
    resultPlaceholder: requireElement("result-placeholder"),
    resultContent: requireElement("result-content"),
    resultMeta: requireElement("result-meta"),
    overlayImage: requireElement("overlay-image"),
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
