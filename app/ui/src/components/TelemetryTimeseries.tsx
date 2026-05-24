/**
 * Live time-series plots — the moving counterpart of the BandPowerBars
 * + PlvGauge static panels. Rolling 60 s window at 4 Hz tick rate.
 *
 * Four stacked subplots, one scale per metric group:
 *   1. Band powers - log y so values across ~3 decades stay visible
 *      (alpha can sit at 0.4 while hypnagogic-theta sits at 0.005).
 *   2. Coherence  - PLV + RESTFULNESS, linear fixed [0, 1]. Both are
 *      normalized [0, 1] quantities, so a constant reference scale lets
 *      the operator eyeball "is it improving?" tick-to-tick.
 *   3. Drowsiness - theta/alpha ratio alone, linear auto-scale (it's
 *      unbounded and routinely exceeds 1.0 once the policy descends).
 *   4. Stimulation - frequency_hz on the left axis, brightness/fill %
 *      overlaid on a separate right axis so Hz and % aren't compared
 *      against the same gridlines.
 */
import Plot from "react-plotly.js";
import { useTranslation } from "react-i18next";
import type { HistorySample } from "../hooks/useTelemetryHistory";

interface Props {
  history: HistorySample[];
}

export function TelemetryTimeseries({ history }: Props) {
  const { t } = useTranslation();
  const xs = history.map((s) => s.t_s);
  const xrange = xs.length
    ? [Math.max(0, xs[xs.length - 1] - 60), xs[xs.length - 1] + 0.5]
    : undefined;

  const line = (
    name: string,
    y: number[],
    color: string,
    yaxis: string,
  ): Partial<Plotly.PlotData> => ({
    type: "scatter",
    mode: "lines",
    name,
    x: xs,
    y,
    line: { color, width: 1.6 },
    yaxis,
    // No `legendgroup` on purpose: with grouping, Plotly's horizontal
    // legend stacks each group as a vertical column and lines them up
    // left-to-right - which clusters the band-powers group into a tall
    // strip rather than flowing all 11 items as one horizontal row.
  });

  return (
    <Plot
      data={[
        line(t("live.alpha"),      history.map((s) => s.alpha),       "#26C6DA", "y"),
        line(t("live.theta"),      history.map((s) => s.theta),       "#7E57C2", "y"),
        line(t("live.beta"),       history.map((s) => s.beta),        "#FFA726", "y"),
        line(t("live.hypnagogic"), history.map((s) => s.hypno_theta), "#5C6BC0", "y"),
        line(t("live.spindle"),    history.map((s) => s.sigma),       "#26A69A", "y"),

        line(t("live.plv"),         history.map((s) => s.plv),         "#EC407A", "y2"),
        line(t("live.restfulness"), history.map((s) => s.restfulness), "#66BB6A", "y2"),

        line(t("live.thetaAlphaRatio"), history.map((s) => s.theta_alpha_ratio), "#FFCA28", "y3"),

        line(t("live.frequency"),  history.map((s) => s.freq_hz),          "#EF5350", "y4"),
        line(t("live.brightness"), history.map((s) => s.brightness * 100), "#FF7043", "y5"),
        line(t("live.fill"),       history.map((s) => s.fill * 100),       "#AB47BC", "y5"),
      ]}
      layout={{
        autosize: true,
        height: 680,
        // Bottom margin large enough for the horizontal legend to live
        // BELOW the chart in a single row (or wrap once at most) rather
        // than fighting the top-of-chart space with 11 long Ukrainian
        // labels.
        margin: { l: 60, r: 60, t: 24, b: 96 },
        showlegend: true,
        // Pin the legend to span the FULL chart width as a single horizontal
        // strip, NOT a centered cluster. Without `entrywidth*` Plotly's "h"
        // orientation lays out items column-first when they don't fit on one
        // row - which clusters everything in the middle and leaves both
        // edges of the chart empty. `entrywidthmode: "fraction"` +
        // `entrywidth: 1 / N` gives each of the 11 traces an equal slot, so
        // they distribute evenly across the canvas left-to-right.
        legend: {
          orientation: "h",
          x: 0,
          xanchor: "left",
          y: -0.12,
          yanchor: "top",
          font: { size: 11 },
          traceorder: "normal",
          // No `entrywidth` on purpose: each entry takes the width its
          // label actually needs, and Plotly wraps to a new row when the
          // total exceeds chart width. A fixed slot (1/11) caused long
          // Ukrainian labels to overlap their neighbours on narrow viewports.
        },
        plot_bgcolor: "transparent",
        paper_bgcolor: "transparent",
        font: { color: "#cfd8dc", size: 11 },
        xaxis: {
          domain: [0, 1],
          range: xrange,
          gridcolor: "rgba(255,255,255,0.06)",
          title: { text: "t (s)", standoff: 6 },
          anchor: "y4",
        },
        // Panel 1 - band powers, log y so small values stay visible.
        yaxis: {
          domain: [0.78, 1.00],
          type: "log",
          title: { text: t("live.bandPowers"), standoff: 6 },
          gridcolor: "rgba(255,255,255,0.06)",
          zerolinecolor: "rgba(255,255,255,0.12)",
        },
        // Panel 2 - PLV + restfulness, both unit-less in [0, 1].
        yaxis2: {
          domain: [0.54, 0.74],
          range: [0, 1.05],
          title: { text: "PLV / rest", standoff: 6 },
          gridcolor: "rgba(255,255,255,0.06)",
          zerolinecolor: "rgba(255,255,255,0.12)",
        },
        // Panel 3 - theta/alpha ratio, unbounded; let it auto-scale.
        yaxis3: {
          domain: [0.30, 0.50],
          title: { text: "θ / α", standoff: 6 },
          gridcolor: "rgba(255,255,255,0.06)",
          zerolinecolor: "rgba(255,255,255,0.12)",
        },
        // Panel 4 - stim. Hz on left, % on right (overlaid).
        yaxis4: {
          domain: [0.04, 0.26],
          title: { text: t("live.frequency"), standoff: 6 },
          gridcolor: "rgba(255,255,255,0.06)",
          zerolinecolor: "rgba(255,255,255,0.12)",
        },
        yaxis5: {
          domain: [0.04, 0.26],
          range: [0, 100],
          title: { text: "%", standoff: 6 },
          overlaying: "y4",
          side: "right",
          showgrid: false,
        },
      }}
      config={{ displayModeBar: false, responsive: true }}
      useResizeHandler
      style={{ width: "100%" }}
    />
  );
}
