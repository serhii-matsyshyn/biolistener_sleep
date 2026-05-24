// WebSocket subscription manager - feeds live telemetry into React Query's cache.

import type { QueryClient } from "@tanstack/react-query";
import type { TelemetrySnapshot } from "./client";

export function subscribeLive(
  sessionId: string,
  queryClient: QueryClient,
): () => void {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  // /api/live/... matches the daemon's WebSocket route (under /api/ so it
  // doesn't collide with the React `/live` route on hard reload).
  const url = `${proto}//${window.location.host}/api/live/${sessionId}`;
  const ws = new WebSocket(url);

  ws.onmessage = (evt) => {
    try {
      const payload = JSON.parse(evt.data) as TelemetrySnapshot;
      queryClient.setQueryData(["telemetry", sessionId], payload);
    } catch {
      // ignore malformed frame
    }
  };

  // Keep socket alive with a heartbeat - daemon expects receive_text()
  const heartbeat = window.setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) ws.send("ping");
  }, 5000);

  return () => {
    window.clearInterval(heartbeat);
    ws.close();
  };
}
