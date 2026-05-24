import { useEffect } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import type { TelemetrySnapshot } from "../api/client";
import { subscribeLive } from "../api/ws";

export function useLiveTelemetry(sessionId: string | null) {
  const qc = useQueryClient();
  useEffect(() => {
    if (!sessionId) return;
    return subscribeLive(sessionId, qc);
  }, [sessionId, qc]);
  return useQuery<TelemetrySnapshot | undefined>({
    queryKey: ["telemetry", sessionId],
    queryFn: async () => undefined, // populated by WS
    enabled: !!sessionId,
    staleTime: Infinity,
  });
}
