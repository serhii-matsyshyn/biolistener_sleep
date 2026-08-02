"""
Shared figure style for the arm-comparison figures.

The categorical palette is Okabe-Ito, reordered so that every adjacent
pair clears the colour-vision-deficiency separation check rather than
just most of them (the pink/green adjacency in the canonical order fails
at deutan ΔE 7.6; the order below lifts the worst adjacent pair to 11.0
deutan / 8.5 tritan, with a normal-vision floor of 16.4).

Colour is never the only channel: every arm also carries a distinct
marker and dash pattern, so the figures survive greyscale printing and
the lighter hues' sub-3:1 contrast against white.

Colour follows the *entity*, not its rank - an arm keeps its hue in every
figure it appears in, whether or not the other arms are present.
"""
from __future__ import annotations

# Validated adjacent-safe order, checked in light mode.
PALETTE = ("#0072B2", "#009E73", "#D55E00", "#CC79A7", "#E69F00", "#56B4E9")
NEUTRAL = "#6B7280"
INK = "#1F2328"
MUTED = "#6E7781"
GRID = "#DFE3E8"

ARM_STYLE = {
    # arm            colour        marker  dash             label
    "nostim":       (NEUTRAL,     "x", (0, (1, 1)),  "no stimulation"),
    "fixed10":      ("#E69F00",   "s", (0, (3, 2)),  "fixed 10 Hz open-loop"),
    "gen_e4_r1":    ("#D55E00",   "^", (0, (5, 2)),  "scripted ramp 11→4 Hz"),
    "gen_best":     ("#D55E00",   "^", "solid",      "best generic open-loop"),
    "orc_e4_r1":    ("#CC79A7",   "v", (0, (5, 2)),  "oracle ramp α→4 Hz"),
    "orc_best":     ("#CC79A7",   "v", "solid",      "best oracle open-loop"),
    "yoked":        ("#009E73",   "D", (0, (4, 1, 1, 1)), "yoked replay of SAC"),
    "sac":          ("#0072B2",   "o", "solid",      "SAC closed-loop"),
    # Which training distribution each SAC arm came from is decided by the
    # policy-selection step, not fixed in advance, so the reporting code
    # overwrites these labels from the run manifest.
    "sac_alt":      ("#56B4E9",   "P", (0, (2, 2)),  "SAC (coupling-randomised)"),
    "sac_paper1":   ("#56B4E9",   "P", (0, (2, 2)),  "SAC (coupling-randomised)"),
}

CONTRAST_STYLE = {
    "total_closed_loop_value":      ("#0072B2", "o", "solid",
                                     "total closed-loop value  (SAC − yoked)"),
    "subject_identification_value": ("#D55E00", "^", (0, (5, 2)),
                                     "subject identification  (oracle − generic)"),
    "target_tracking_value":        ("#009E73", "D", (0, (4, 1, 1, 1)),
                                     "target tracking  (SAC − oracle)"),
}

MODE_LABEL = {"endo": "endogenous coupling", "exo": "exogenous drift"}


def apply(plt) -> None:
    """Recessive grid, thin marks, text in ink tokens rather than series hues."""
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlecolor": INK,
        "axes.labelsize": 9.5,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.9,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "legend.labelcolor": INK,
        "lines.linewidth": 2.0,
        "lines.markersize": 5.5,
        "font.size": 9.5,
    })


def arm_style(arm: str):
    return ARM_STYLE.get(arm, (PALETTE[-1], "o", "solid", arm))
