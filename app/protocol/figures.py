"""
Figures for the thesis.

Every function takes a loaded :class:`~app.protocol.analysis.Session`, returns a
matplotlib figure, and draws only what the recording supports - a panel with no
data says so on the axes instead of rendering an empty box that looks like a
result.

``save_all`` writes a numbered set to a directory ready to be referenced from
the thesis text.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

# No backend is forced here: savefig works under any of them, and forcing Agg
# would kill inline display when these are called from a notebook. Headless
# scripts already get Agg by default.
import matplotlib.pyplot as plt

from app.protocol import analysis as an
from app.protocol import quality as q

logger = logging.getLogger("protocol.figures")

FIGSIZE = (7.0, 4.0)
DPI = 200


def _empty(ax, msg: str) -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes,
            fontsize=9, color="0.4")
    ax.set_xticks([])
    ax.set_yticks([])


# ---- validation session --------------------------------------------------

def fig_alpha_blocking(sess: an.Session):
    """Eyes-open vs eyes-closed spectra - the electrodes-see-cortex figure."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fs = sess.sampling_rate
    try:
        pairs = [("A2a_eyes_open", "A2b_eyes_closed"),
                 ("A2c_eyes_open", "A2d_eyes_closed")]
        for i, (o, c) in enumerate(pairs):
            for name, style in ((o, "--"), (c, "-")):
                amps, freqs = q.psd_welch(sess.segment(name), fs)
                sel = (freqs >= 1) & (freqs <= 30)
                ax.semilogy(freqs[sel], amps[sel], style,
                            color=f"C{i}", lw=1.4,
                            label=f"{'closed' if 'closed' in name else 'open'} "
                                  f"(repeat {i + 1})")
        ax.axvspan(*q.ALPHA_BAND, color="0.85", zorder=0, label="alpha band")
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("power spectral density")
        ax.set_title("Alpha blocking on eye closure")
        ax.legend(fontsize=7, ncol=2)
    except Exception as err:
        _empty(ax, f"alpha-blocking blocks missing\n{err}")
    fig.tight_layout()
    return fig


def fig_ssvep_and_artifact(sess: an.Session):
    """
    The brightness ladder on shared axes.

    A response that grows with commanded brightness is being driven by light. One
    that sits at the same height with brightness at zero is the amplifier picking
    up the drive electronics.
    """
    from app.protocol.spec import BRIGHTNESS_LADDER, SSVEP_CHECK_HZ
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fs = sess.sampling_rate
    try:
        for i, b in enumerate(sorted(BRIGHTNESS_LADDER)):
            name = f"A4_bright_{int(b * 100):03d}"
            amps, freqs = q.psd_welch(sess.segment(name), fs)
            sel = (freqs >= 1) & (freqs <= 30)
            ax.semilogy(freqs[sel], amps[sel], "--" if b == 0 else "-",
                        color=f"C{i}", lw=1.4,
                        label=f"{int(b * 100)} % brightness"
                              + (" (control)" if b == 0 else ""))
        for h, lab in ((1, "f"), (2, "2f")):
            ax.axvline(SSVEP_CHECK_HZ * h, color="C3", lw=0.8, ls=":")
            ax.text(SSVEP_CHECK_HZ * h, ax.get_ylim()[1], f" {lab}",
                    va="top", fontsize=8, color="C3")
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("power spectral density")
        ax.set_title(f"Photic driving at {SSVEP_CHECK_HZ:g} Hz "
                     f"against commanded brightness")
        ax.legend(fontsize=8)
    except Exception as err:
        _empty(ax, f"brightness-ladder blocks missing\n{err}")
    fig.tight_layout()
    return fig


def fig_raw_traces(sess: an.Session, seconds: float = 5.0):
    """A few seconds of raw signal, eyes open against eyes closed."""
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.0), sharex=True, sharey=True)
    fs = sess.sampling_rate
    for ax, name in zip(axes, ("A2a_eyes_open", "A2b_eyes_closed")):
        try:
            seg = sess.segment(name)[: int(seconds * fs)]
            ax.plot(np.arange(seg.size) / fs, seg, lw=0.6, color="C0")
            ax.set_ylabel(name.split("_", 1)[1].replace("_", " "))
        except Exception as err:
            _empty(ax, f"{name} missing\n{err}")
    axes[-1].set_xlabel("time (s)")
    axes[0].set_title("Raw signal, eyes open against eyes closed")
    fig.tight_layout()
    return fig


def fig_spectrogram(sess: an.Session, fmax: float = 30.0):
    """Whole session as a spectrogram, with the block boundaries marked."""
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    if sess.raw is None:
        _empty(ax, "no raw recording")
        return fig
    fs = sess.sampling_rate
    x = sess.raw[sess.primary_channel]
    nper = int(4 * fs)
    if x.size < nper * 2:
        _empty(ax, "recording too short for a spectrogram")
        return fig
    # Plain Welch slices rather than a scipy dependency: one column per window,
    # which is also the resolution the features are computed at.
    cols, times = [], []
    for i in range(0, x.size - nper, nper // 2):
        amps, freqs = q.psd_welch(x[i:i + nper], fs)
        sel = freqs <= fmax
        cols.append(np.log10(np.maximum(amps[sel], 1e-12)))
        times.append((i + nper / 2) / fs / 60.0)
    img = np.array(cols).T
    ax.imshow(img, aspect="auto", origin="lower", cmap="magma",
              extent=[times[0], times[-1], 0, fmax])
    for m in sess.markers:
        if m.get("event") == "block_start":
            ax.axvline(m["t_s"] / 60.0, color="w", lw=0.5, alpha=0.6)
    ax.set_xlabel("session time (min)")
    ax.set_ylabel("frequency (Hz)")
    ax.set_title("Session spectrogram (log power), block boundaries marked")
    fig.tight_layout()
    return fig


def fig_gate_table(gates: Sequence[q.GateResult]):
    fig, ax = plt.subplots(figsize=(7.0, 0.5 + 0.42 * max(1, len(gates))))
    ax.axis("off")
    rows = [[g.name, g.label, f"{g.value:.3g}", f"{g.threshold:.3g}",
             "pass" if g.passed else ("FAIL" if g.blocking else "warn")]
            for g in gates]
    if not rows:
        _empty(ax, "no gates evaluated")
        return fig
    t = ax.table(cellText=rows,
                 colLabels=["gate", "measure", "value", "limit", "result"],
                 loc="center", cellLoc="left")
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1, 1.3)
    for r, g in enumerate(gates, start=1):
        colour = "#d8f0d8" if g.passed else ("#f6d6d6" if g.blocking else "#faf0d0")
        for c in range(5):
            t[(r, c)].set_facecolor(colour)
    ax.set_title("Signal-quality gates", pad=12)
    fig.tight_layout()
    return fig


# ---- experimental session ------------------------------------------------

def fig_session_overview(sess: an.Session):
    """Drowsiness, commanded frequency and KSS against block structure."""
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 5.0), sharex=True)
    tk = sess.ticks
    if tk is None or tk.empty:
        _empty(axes[0], "no ticks recorded")
        return fig
    t = tk["t_s"] / 60.0

    axes[0].plot(t, tk["theta_alpha_ratio"], lw=1.0, color="C0")
    axes[0].set_ylabel("theta/alpha")
    axes[1].plot(t, tk["act_freq_hz"].where(tk["act_brightness"] > 0), lw=1.0,
                 color="C1")
    axes[1].set_ylabel("delivered (Hz)")
    axes[1].set_xlabel("session time (min)")

    edges, seen = [], set()
    for m in sess.markers:
        if m.get("event") == "block_start" and m.get("block") not in seen:
            seen.add(m["block"])
            edges.append((m["t_s"] / 60.0, m["block"]))
    for ax in axes:
        for x, name in edges:
            ax.axvline(x, color="0.8", lw=0.7, zorder=0)
    for x, name in edges:
        axes[0].text(x, axes[0].get_ylim()[1], name.replace("_", " "),
                     rotation=90, fontsize=5.5, va="top", color="0.35")

    if sess.kss is not None and not sess.kss.empty and "t_s" in sess.kss:
        ax2 = axes[0].twinx()
        ax2.plot(sess.kss["t_s"] / 60.0, sess.kss["value"], "o-", color="C3",
                 ms=4, lw=1.0)
        ax2.set_ylabel("KSS", color="C3")
        ax2.set_ylim(0.5, 9.5)
    axes[0].set_title("Session overview")
    fig.tight_layout()
    return fig


def fig_arm_contrast(sess: an.Session):
    """
    The primary outcome: reward per segment, by arm, in the order they ran.

    Plotted against segment position rather than as two bars, because the
    position balance is what makes the contrast interpretable - a reader should
    be able to see that each arm sits once early and once late.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    tab = an.arm_summary(sess)
    if tab.empty:
        _empty(ax, "no descent segments recorded")
        return fig
    colours = {"ramp": "C0", "agent": "C1"}
    for arm, grp in tab.groupby("arm"):
        ax.plot(grp["position"], grp["mean_reward"], "o-", ms=9,
                color=colours.get(arm, "C2"), label=arm, lw=1.4)
    ax.set_xticks(sorted(tab["position"]))
    ax.set_xlabel("segment position in the session")
    ax.set_ylabel("mean reward from real EEG")
    c = an.arm_contrast(sess, n_boot=400)
    if c.get("ok"):
        title = (f"Ramp - agent {c['ramp_minus_agent']:+.3f} "
                 f"[{c['ci_lo']:+.3f}, {c['ci_hi']:+.3f}]")
        if not c.get("balanced", True):
            title += "  - POSITIONS UNBALANCED"
        if not c.get("quality_comparable", True):
            title += "  - QUALITY NOT COMPARABLE"
        ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def fig_arm_dose(sess: an.Session):
    """
    Delivered light per arm, which the contrast cannot be read without.

    The agent controls brightness and the ramp does not, so the contrast in
    F8 is read together with this panel.
    """
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    tab = an.arm_summary(sess)
    if tab.empty:
        _empty(ax, "no descent segments recorded")
        return fig
    means = tab.groupby("arm")["mean_dose"].mean()
    ax.bar(means.index, means.values,
           color=[{"ramp": "C0", "agent": "C1"}.get(a, "C2") for a in means.index])
    for i, (a, v) in enumerate(means.items()):
        ax.text(i, v, f" {v:.3f}", va="bottom", ha="center", fontsize=9)
    ratio = (means.get("agent", float("nan")) / means["ramp"]
             if means.get("ramp") else float("nan"))
    ax.set_ylabel("delivered dose (duty x brightness)")
    ax.set_title(f"Agent delivered {ratio:.0%} of the ramp's light", fontsize=10)
    fig.tight_layout()
    return fig


def fig_response_curves(sess: an.Session, blocks: Optional[Sequence[str]] = None):
    """Every sweep's entrainment curve on one axis."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    blocks = list(blocks or an.sweep_blocks(sess))
    drawn = 0
    for i, b in enumerate(blocks):
        try:
            c = an.response_curve(sess, b)
            if not c.probes:
                continue
            ax.plot(c.freqs, c.amplitude, "o-", color=f"C{i}", lw=1.3, ms=4,
                    label=b.replace("_", " "))
            drawn += 1
        except Exception as err:
            logger.warning("curve %s: %s", b, err)
    if not drawn:
        _empty(ax, "no sweep data")
    else:
        ax.set_xlabel("probe frequency (Hz)")
        ax.set_ylabel("response amplitude at probe")
        ax.set_title("Entrainment response by sweep")
        ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def fig_peak_shift(sess: an.Session, blocks: Optional[Sequence[str]] = None,
                   seed: int = 0):
    """Fitted peak per sweep with bootstrap intervals - the headline panel."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    blocks = list(blocks or an.sweep_blocks(sess))
    tab = an.peak_table(sess, blocks, seed=seed)
    ok = tab.dropna(subset=["peak_hz"])
    if ok.empty:
        _empty(ax, "no peak could be fitted")
        return fig
    x = np.arange(len(ok))
    ax.errorbar(x, ok["peak_hz"],
                yerr=[ok["peak_hz"] - ok["ci_lo"], ok["ci_hi"] - ok["peak_hz"]],
                fmt="o", capsize=4, ms=6, lw=1.4, color="C0")
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace("_", "\n") for b in ok["block"]], fontsize=7)
    ax.set_ylabel("fitted response peak (Hz)")
    ax.set_title("Response peak per sweep (95 % bootstrap interval)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def fig_safety_events(sess: an.Session):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ev = sess.safety_events
    if not ev:
        _empty(ax, "no safety events - nothing was clamped")
        return fig
    kinds: Dict[str, int] = {}
    for e in ev:
        k = e.get("kind") or e.get("field_name") or "unknown"
        kinds[k] = kinds.get(k, 0) + 1
    names = sorted(kinds, key=kinds.get, reverse=True)
    ax.barh(names, [kinds[n] for n in names], color="C2")
    ax.set_xlabel("count")
    ax.set_title(f"Safety supervisor interventions ({len(ev)} total)")
    fig.tight_layout()
    return fig


def fig_descent_arms(sess: an.Session):
    """Delivered frequency during each descent block, overlaid."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    tk = sess.ticks
    names = [b["name"] for b in sess.manifest.get("blocks", [])
             if b.get("kind", "").startswith("descent")]
    drawn = 0
    for i, name in enumerate(names):
        seg = tk[tk["block"] == name]
        if seg.empty:
            continue
        t0 = seg["t_s"].min()
        ax.plot((seg["t_s"] - t0) / 60.0, seg["act_freq_hz"], lw=1.2,
                color=f"C{i}", label=name.replace("_", " "))
        drawn += 1
    if not drawn:
        _empty(ax, "no descent blocks recorded")
    else:
        ax.set_xlabel("time within block (min)")
        ax.set_ylabel("delivered frequency (Hz)")
        ax.set_title("Descent arms, equal duration")
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---- driver --------------------------------------------------------------

VALIDATION_FIGS = [
    ("F1_raw_traces", fig_raw_traces),
    ("F2_alpha_blocking", fig_alpha_blocking),
    ("F3_spectrogram", fig_spectrogram),
    # One figure covers the driving check and its control, because they are only
    # interpretable together: the ladder shows both on shared axes.
    ("F5_brightness_ladder", fig_ssvep_and_artifact),
]
# Primary outcome first. The order here is the order they appear in the results
# chapter, and it follows the pre-registered outcome order rather than the
# history of what was built.
SESSION_FIGS = [
    ("F7_overview", fig_session_overview),
    ("F8_arm_contrast", fig_arm_contrast),        # primary
    ("F9_arm_dose", fig_arm_dose),                # primary - read with F8
    ("F10_descent_arms", fig_descent_arms),
    ("F11_safety", fig_safety_events),
    ("F12_response_curves", fig_response_curves),  # secondary
    ("F13_peak_shift", fig_peak_shift),            # secondary
]


def save_all(sess: an.Session, out_dir: Path, kind: str = "auto") -> List[Path]:
    """Render the figure set for a session and return the files written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if kind == "auto":
        kind = "validation" if "validation" in sess.manifest.get("protocol", "") \
            else "session"
    specs = VALIDATION_FIGS if kind == "validation" else SESSION_FIGS

    written = []
    for name, fn in specs:
        try:
            fig = fn(sess)
        except Exception:
            logger.warning("figure %s failed", name, exc_info=True)
            continue
        path = out_dir / f"{name}.png"
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    if kind == "validation":
        try:
            fig = fig_gate_table(an.validation_gates(sess))
            path = out_dir / "F6_gates.png"
            fig.savefig(path, dpi=DPI, bbox_inches="tight")
            plt.close(fig)
            written.append(path)
        except Exception:
            logger.warning("gate table failed", exc_info=True)
    return written
