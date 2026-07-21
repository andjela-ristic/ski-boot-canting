const CACHE_NAME = "canting-pwa-shell-v2";
const STATIC_ASSETS = [
  "/",
  "/panels/topbar.html",
  "/panels/hero.html",
  "/panels/capture-panel.html",
  "/panels/status-panel.html",
  "/panels/result-panel.html",
  "/styles/base.css",
  "/styles/layout.css",
  "/styles/panels/topbar.css",
  "/styles/panels/hero.css",
  "/styles/panels/capture.css",
  "/styles/panels/status.css",
  "/styles/panels/result.css",
  "/js/main.js",
  "/js/app/bootstrap.js",
  "/js/app/dom/elements.js",
  "/js/app/layout/load-panel-fragments.js",
  "/js/app/state/app-state.js",
  "/js/app/services/api-client.js",
  "/js/app/services/camera-recorder.js",
  "/js/app/ui/status-panel.js",
  "/js/app/ui/app-chrome.js",
  "/js/app/ui/result-panel.js",
  "/js/app/ui/capture-panel.js",
  "/js/app/utils/format.js",
  "/manifest.webmanifest",
  "/favicon.png",
  "/icons/icon-192.png",
  "/icons/icon-512.png",
  "/icons/icon-maskable-192.png",
  "/icons/icon-maskable-512.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(CACHE_NAME)
      .then((cache) => cache.addAll(STATIC_ASSETS))
      .then(() => self.skipWaiting()),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter((key) => key !== CACHE_NAME)
            .map((key) => caches.delete(key)),
        ),
      )
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (event) => {
  const { request } = event;

  if (request.method !== "GET") {
    return;
  }

  const requestUrl = new URL(request.url);
  if (requestUrl.origin !== self.location.origin) {
    return;
  }

  if (
    requestUrl.pathname.startsWith("/frames") ||
    requestUrl.pathname.startsWith("/analyze") ||
    requestUrl.pathname.startsWith("/health") ||
    requestUrl.pathname.startsWith("/api/")
  ) {
    return;
  }

  event.respondWith(
    caches.match(request).then((cachedResponse) => {
      if (cachedResponse) {
        return cachedResponse;
      }

      return fetch(request).then((networkResponse) => {
        if (!networkResponse.ok) {
          return networkResponse;
        }

        const clonedResponse = networkResponse.clone();
        caches.open(CACHE_NAME).then((cache) => {
          cache.put(request, clonedResponse);
        });
        return networkResponse;
      });
    }),
  );
});
