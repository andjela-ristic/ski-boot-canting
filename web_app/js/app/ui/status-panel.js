export function createStatusPresenter(elements) {
  return {
    setStatus(message, tone) {
      elements.statusPanel.textContent = message;
      elements.statusPanel.className = "status-panel";

      if (tone === "success") {
        elements.statusPanel.classList.add("success");
      } else if (tone === "warning") {
        elements.statusPanel.classList.add("warning");
      } else if (tone === "error") {
        elements.statusPanel.classList.add("error");
      }
    },
  };
}
