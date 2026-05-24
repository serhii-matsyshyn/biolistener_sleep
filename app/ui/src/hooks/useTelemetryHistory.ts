/**
 * Ring buffer over the live-telemetry stream.
 *
 * The WS feed pushes one TelemetrySnapshot per tick (4 Hz). This hook
 * subscribes to the same cache key as ``useLiveTelemetry`` and accumulates
 * the last N samples into local state so the page can render time-series
 * plots without each <Plot> remounting on every update.
 *
 * The buffer is per-session: switching sessions resets it.
 */
import { useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import type { TelemetrySnapshot } from "../api/client";

export interface HistorySample {
  t_s: number;        // elapsed seconds since session start
  alpha: number;
  theta: number;
  beta: number;
  hypno_theta: number;
  sigma: number;
  theta_alpha_ratio: number;
  plv: number;
  restfulness: number;
  freq_hz: number;
  brightness: number;
  fill: number;
  reward: number;
}

function toSample(snap: TelemetrySnapshot): HistorySample | null {
  if (!snap.features) return null;
  return {
    t_s:                snap.elapsed_s,
    alpha:              snap.features.alpha_pwr,
    theta:              snap.features.theta_pwr,
    beta:               snap.features.beta_pwr,
    hypno_theta:        snap.features.hypnagogic_theta_pwr,
    sigma:              snap.features.sigma_pwr,
    theta_alpha_ratio:  snap.features.theta_alpha_ratio,
    plv:                snap.plv,
    restfulness:        snap.restfulness,
    freq_hz:            snap.stim?.frequency_hz ?? 0,
    brightness:         snap.stim?.brightness ?? 0,
    fill:               snap.stim?.fill ?? 0,
    reward:             snap.reward,
  };
}

export function useTelemetryHistory(
  sessionId: string | null,
  maxSamples = 240,            // ≈ 60 s at 4 Hz
): HistorySample[] {
  const qc = useQueryClient();
  const [history, setHistory] = useState<HistorySample[]>([]);

  // Reset buffer when sessionId changes.
  useEffect(() => { setHistory([]); }, [sessionId]);

  useEffect(() => {
    if (!sessionId) return;
    const key = ["telemetry", sessionId];
    // Subscribe to the same cache slot the WS writes into.
    const unsub = qc.getQueryCache().subscribe((evt) => {
      if (evt.type !== "updated") return;
      if (JSON.stringify(evt.query.queryKey) !== JSON.stringify(key)) return;
      const snap = evt.query.state.data as TelemetrySnapshot | undefined;
      if (!snap) return;
      const s = toSample(snap);
      if (!s) return;
      setHistory((prev) => {
        // Skip duplicate ticks (the WS can re-emit terminal state).
        if (prev.length && prev[prev.length - 1].t_s === s.t_s) return prev;
        const next = prev.length >= maxSamples
          ? prev.slice(prev.length - maxSamples + 1)
          : prev.slice();
        next.push(s);
        return next;
      });
    });
    return unsub;
  }, [sessionId, qc, maxSamples]);

  return history;
}
