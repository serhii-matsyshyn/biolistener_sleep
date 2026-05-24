import Plot from "react-plotly.js";
import type { TelemetrySnapshot } from "../api/client";

interface Props {
  snap: TelemetrySnapshot | null | undefined;
}

// Band powers only — PLV lives in its own card (PlvGauge) because mixing
// PSD-integral values (~0.001–0.2) with a normalized [0, 1] coherence on
// one axis always lets PLV dominate and the bands collapse to invisible.
export function BandPowerBars({ snap }: Props) {
  const f = snap?.features;
  if (!f) return null;
  return (
    <Plot
      data={[
        {
          type: "bar",
          x: ["α (8-13)", "θ (4-8)", "β (13-30)", "hypnoθ (4-7)", "σ (12-16)"],
          y: [
            f.alpha_pwr,
            f.theta_pwr,
            f.beta_pwr,
            f.hypnagogic_theta_pwr,
            f.sigma_pwr,
          ],
          marker: { color: ["#26C6DA", "#7E57C2", "#FFA726", "#5C6BC0", "#26A69A"] },
        },
      ]}
      layout={{
        autosize: true,
        height: 280,
        margin: { l: 50, r: 10, t: 30, b: 40 },
        title: { text: "Band powers" },
        yaxis: { title: { text: "power" } },
        plot_bgcolor: "transparent",
        paper_bgcolor: "transparent",
        font: { color: "#cfd8dc" },
      }}
      useResizeHandler
      style={{ width: "100%" }}
    />
  );
}
