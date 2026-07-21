const PANEL_SLOTS = [
  { slotId: "topbar-slot", path: "/panels/topbar.html" },
  { slotId: "hero-slot", path: "/panels/hero.html" },
  { slotId: "capture-slot", path: "/panels/capture-panel.html" },
  { slotId: "status-slot", path: "/panels/status-panel.html" },
  { slotId: "result-slot", path: "/panels/result-panel.html" },
];

export async function loadPanelFragments() {
  await Promise.all(
    PANEL_SLOTS.map(async (panel) => {
      const slot = document.getElementById(panel.slotId);
      if (!slot) {
        throw new Error(`Missing slot container: #${panel.slotId}`);
      }

      const response = await fetch(panel.path, {
        headers: {
          Accept: "text/html",
        },
      });

      if (!response.ok) {
        throw new Error(`Could not load panel fragment: ${panel.path}`);
      }

      slot.innerHTML = await response.text();
    }),
  );
}
