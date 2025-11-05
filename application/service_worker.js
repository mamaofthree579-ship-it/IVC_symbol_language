self.addEventListener("install", event => {
  console.log("Service Worker installing.");
  self.skipWaiting();
});

self.addEventListener("activate", event => {
  console.log("Service Worker activated.");
});

self.addEventListener("fetch", event => {
  event.respondWith(fetch(event.request));
});

const CACHE_NAME = "ivc-cache-v1";
const urlsToCache = ["/", "/manifest.json"];

self.addEventListener("install", event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(urlsToCache))
  );
});

self.addEventListener("fetch", event => {
  event.respondWith(
    caches.match(event.request).then(response => {
      return response || fetch(event.request);
    })
  );
});
