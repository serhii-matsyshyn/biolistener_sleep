"""
Arm-comparison analysis: when does closing the loop pay?

Turns the sweep's per-tick cell parquets into (a) one tidy per-episode
metric table and (b) the paired contrasts between stimulation arms. Also
provides the delivery masking that keeps frequency-derived metrics honest
when an arm stops emitting light, and the entrainment/contamination
frequency metrics that qualify any θ/α comparison.

Everything here is deliberately *paired*. Every arm at every coupling
level ran on the same 100 simulated subjects, seeded identically, so the
right unit of analysis is the within-subject difference, not two
independent group means. That is what makes a 100-subject simulation
informative about an effect that is small relative to between-subject
variance - and between-subject variance here is enormous, because
``cortical_gain`` spans a 35x range.

The three contrasts:

    SAC        − yoked        = total closed-loop value
    oracle-OL  − generic-OL   = value of subject identification
    SAC        − oracle-OL    = value of target tracking

Statistics: Wilcoxon signed-rank on the paired differences (the
distributions are bimodal - roughly a third of subjects never descend -
so a t-test's normality assumption is not available), bias-corrected
bootstrap CIs on the paired mean, and the matched-pairs rank-biserial
correlation as effect size. Across the coupling levels of one contrast
family, p-values are Holm-corrected.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy import stats

TICK_DT_S = 0.25
FLOOR_HZ = 4.0            # PhoticInputConfig.min_resonance_hz
# Ticks to skip before reading a "baseline" band power: the 4 s Welch
# window needs 16 ticks at 4 Hz to fill, and until it does the features
# come from a shorter, non-comparable window.
BASELINE_SKIP_TICKS = 16
CELL_RE = re.compile(r"^(?P<mode>[a-z0-9]+)__(?:c(?P<coupling>[-\d.]+)__)?(?P<arm>.+)$")

# metric -> is a larger value the better outcome?
METRIC_DIRECTION = {
    "final_theta_alpha": True,
    "theta_alpha_last30s": True,
    "mean_theta_alpha": True,
    "auc_theta_alpha": True,
    "final_hypno_theta": True,
    "mean_hypno_theta": True,
    "alpha_suppression": True,     # 1 - alpha_end/alpha_start
    "final_strobe_hz": False,
    "mean_plv": True,
    "mean_plv_2f": True,
    "time_to_first_below_8hz": False,
    "time_to_first_below_6hz": False,
    "total_reward": True,
}
PRIMARY_METRIC = "final_theta_alpha"

# Static per-subject columns, constant within an episode.
SUBJECT_COLS = ("alpha_hz", "cortical_gain", "resonance_width_hz",
                "responder_type", "shift_per_unit_ratio", "exogenous_shift")


# --------------------------------------------------------------- loading

def parse_cell_name(stem: str) -> dict:
    """``endo__c0.25__sac`` -> mode/coupling/arm."""
    m = CELL_RE.match(stem)
    if not m:
        raise ValueError(f"unparseable cell name: {stem}")
    c = m.group("coupling")
    return {"mode": m.group("mode"),
            "coupling": float(c) if c is not None else np.nan,
            "arm": m.group("arm")}


def _first_crossing_s(t_s: np.ndarray, freq: np.ndarray, thresh: float
                      ) -> float:
    """Simulated seconds until the strobe first drops below ``thresh``.

    NaN when it never does. Reported alongside the censoring rate rather
    than imputed, because ~20-30 % of episodes never descend at all and
    filling those with the episode length would silently invent data.
    """
    idx = np.flatnonzero(freq < thresh)
    return float(t_s[idx[0]]) if idx.size else np.nan


def episode_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse a per-tick cell frame to one row per episode."""
    out = []
    for ep, g in df.groupby("episode", sort=True):
        g = g.sort_values("tick")
        f = g["strobe_freq_hz"].to_numpy(np.float64)
        t = g["t_s"].to_numpy(np.float64)
        ratio = g["theta_alpha_ratio"].to_numpy(np.float64)
        alpha = g["alpha"].to_numpy(np.float64)
        res = g["effective_resonance_hz"].to_numpy(np.float64)
        d = np.diff(f)

        # Ratchet accounting: the policy is paid for *descending* but never
        # for *staying*, so it can collect the descent bonus repeatedly by
        # oscillating. Total travel vs net descent measures exactly that.
        up, down = float(d[d > 0].sum()), float(-d[d < 0].sum())
        first8 = _first_crossing_s(t, f, 8.0)
        if np.isnan(first8):
            max_reascent = np.nan
        else:
            # Largest rise above the running minimum after the first
            # crossing - "how far back up did it climb once it got there".
            after = f[t >= first8]
            run_min = np.minimum.accumulate(after)
            max_reascent = float(np.max(after - run_min)) if after.size else 0.0

        # Baseline α is taken *after* the 4 s feature window has filled
        # (16 ticks at 4 Hz). Before that, extract_features runs on a
        # growing partial window and the band powers are not comparable to
        # the steady-state ones.
        b0 = min(BASELINE_SKIP_TICKS, max(0, alpha.size - 4))
        a0 = float(np.mean(alpha[b0:b0 + 4])) if alpha.size > b0 \
            else float(alpha[0])
        aN = float(np.mean(alpha[-4:])) if alpha.size >= 4 else float(alpha[-1])

        # Frequency-derived metrics are masked to ticks where light was
        # actually delivered: the actuator register keeps holding a frequency
        # while the strobe is off, so an unmasked arm could score a perfect
        # descent on zero photic energy. An arm that delivers nothing gets
        # NaN, not a win.
        delivered = (
            g["enabled"].to_numpy().astype(bool)
            & (g["fill"].to_numpy() > 0.0)
            & (g["brightness"].to_numpy() > 0.0)
        ) if "enabled" in g.columns else (
            (g["fill"].to_numpy() > 0.0) & (g["brightness"].to_numpy() > 0.0)
        )
        n_delivered = int(delivered.sum())
        fd = f[delivered] if n_delivered else np.array([], dtype=float)
        td = t[delivered] if n_delivered else np.array([], dtype=float)

        def _fd(fn, default=np.nan):
            return float(fn(fd)) if n_delivered else float(default)

        row = {
            "episode": int(ep),
            "n_ticks": int(len(g)),
            "frac_ticks_delivering": float(delivered.mean()),
            # --- outcomes
            "final_theta_alpha": float(ratio[-1]),
            # Robustness companion to the single-tick endpoint: the mean
            # over the last 30 s. The endpoint is the primary outcome, but a
            # policy that oscillates can land its last tick anywhere, so
            # both are reported.
            "theta_alpha_last30s": float(ratio[-min(len(ratio), 120):].mean()),
            "mean_theta_alpha": float(ratio.mean()),
            "auc_theta_alpha": float(ratio.sum() * TICK_DT_S),
            "final_hypno_theta": float(g["hypno_theta"].to_numpy()[-1]),
            "mean_hypno_theta": float(g["hypno_theta"].mean()),
            "alpha_suppression": float(1.0 - aN / a0) if a0 > 0 else np.nan,
            "final_strobe_hz": _fd(lambda a: a[-1]),
            "mean_strobe_hz": _fd(np.mean),
            "min_strobe_hz": _fd(np.min),
            "mean_plv": float(g["plv"].mean()),
            "mean_plv_2f": float(g["plv_2f"].mean()),
            # Stimulus-contamination controls. θ/α is a *spectral* measure,
            # and a strobe inside the 4-8 Hz theta band deposits power
            # directly into the numerator by driving an SSVEP at the drive
            # frequency. Any arm comparison on θ/α therefore has to be read
            # against how long each arm actually sat in the theta band, and
            # against how strongly the EEG was phase-locked to it.
            "frac_ticks_in_theta_band": float(
                ((f >= 4.0) & (f < 8.0) & delivered).mean()),
            "frac_ticks_in_alpha_band": float(
                ((f >= 8.0) & (f < 12.0) & delivered).mean()),
            "theta_band_dose": float(
                ((f >= 4.0) & (f < 8.0)).mean()
                * (g["fill"] * g["brightness"]).mean()),
            "total_reward": float(g["reward"].sum()),
            "time_to_first_below_8hz": (
                _first_crossing_s(td, fd, 8.0) if n_delivered else np.nan),
            "time_to_first_below_6hz": (
                _first_crossing_s(td, fd, 6.0) if n_delivered else np.nan),
            "ever_below_6hz": (
                bool((fd < 6.0).any()) if n_delivered else False),
            # --- dose audit (the arms must differ in schedule, not energy)
            "mean_fill": float(g["fill"].mean()),
            "mean_brightness": float(g["brightness"].mean()),
            "photic_energy": float((g["fill"] * g["brightness"]).mean()),
            # --- realised plant state: drift is an OUTCOME, not a setting
            "resonance_start_hz": float(res[0]),
            "resonance_min_hz": float(res.min()),
            "resonance_final_hz": float(res[-1]),
            "resonance_excursion_hz": float(res[0] - res.min()),
            "frac_ticks_at_floor": float((res <= FLOOR_HZ + 1e-9).mean()),
            # --- ratchet
            "up_travel_hz": up,
            "down_travel_hz": down,
            "net_descent_hz": float(f[0] - f[-1]),
            "travel_ratio": (up / down) if down > 0 else np.nan,
            "max_reascent_after_8hz": max_reascent,
            # --- safety accounting
            "n_safety_events": int(g["n_safety_events"].sum()),
            "frac_ramp_limited": float(g["ramp_limited_freq"].mean()),
            # Electrode-disconnect dropouts. The simulated EEG occasionally
            # rails (BrainFlow's get_railed_percentage > 80 %), which trips
            # the supervisor's disconnect guard: stimulation is disabled for
            # that tick and the reward pays k_disconnect = -10. Panic never
            # latches here and 300 s is far inside the 45 min session
            # timeout, so `not enabled` identifies exactly these ticks.
            #
            # This is a property of the simulator rather than of any one
            # arm - the same rate (19/100 episodes, 1.5 % of ticks) shows up
            # in earlier 100-subject rollouts. It matters twice over: it
            # suppresses stimulation for those ticks, and it makes
            # `total_reward` a poor efficacy measure, because a single
            # railing episode costs more return than a whole successful
            # descent earns. Reported per arm; never used for selection.
            "frac_ticks_disconnected": float((~g["enabled"]).mean()),
            "any_disconnect": bool((~g["enabled"]).any()),
        }
        for c in SUBJECT_COLS:
            if c in g.columns:
                row[c] = g[c].iloc[0]
        out.append(row)
    return pd.DataFrame(out)


def load_all_cells(cells_dir: Path,
                   include: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Per-episode metrics for every cell parquet, tagged mode/coupling/arm."""
    cells_dir = Path(cells_dir)
    frames = []
    for p in sorted(cells_dir.glob("*.parquet")):
        meta = parse_cell_name(p.stem)
        if include is not None and meta["arm"] not in include:
            continue
        m = episode_metrics(pd.read_parquet(p))
        for k, v in meta.items():
            m[k] = v
        m["cell"] = p.stem
        frames.append(m)
    if not frames:
        raise FileNotFoundError(f"no cell parquets in {cells_dir}")
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------ statistics

@dataclass
class PairedResult:
    contrast: str
    metric: str
    mode: str
    coupling: float
    n: int
    mean_diff: float
    median_diff: float
    ci_lo: float
    ci_hi: float
    p_wilcoxon: float
    rank_biserial: float
    win_rate: float
    a_mean: float
    b_mean: float
    p_holm: float = np.nan

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _bootstrap_ci(x: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05,
                  seed: int = 20260728) -> tuple:
    """Percentile bootstrap CI on the paired mean difference."""
    if x.size == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))


def paired_contrast(a: pd.Series, b: pd.Series, *, metric: str,
                    contrast: str, mode: str, coupling: float,
                    higher_is_better: Optional[bool] = None) -> PairedResult:
    """Within-subject a − b, oriented so positive always means "a is better"."""
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    hib = METRIC_DIRECTION.get(metric, True) if higher_is_better is None \
        else higher_is_better
    sign = 1.0 if hib else -1.0
    d = sign * (joined["a"].to_numpy(np.float64)
                - joined["b"].to_numpy(np.float64))
    n = int(d.size)
    if n == 0:
        return PairedResult(contrast, metric, mode, coupling, 0, *([np.nan] * 9))
    nz = d[d != 0]
    if nz.size >= 1:
        try:
            p = float(stats.wilcoxon(nz, alternative="two-sided").pvalue)
        except ValueError:
            p = 1.0
        # matched-pairs rank-biserial: (W+ − W−) / total rank sum
        ranks = stats.rankdata(np.abs(nz))
        total = ranks.sum()
        rb = float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / total) \
            if total > 0 else 0.0
    else:
        p, rb = 1.0, 0.0
    lo, hi = _bootstrap_ci(d)
    return PairedResult(
        contrast=contrast, metric=metric, mode=mode, coupling=coupling, n=n,
        mean_diff=float(d.mean()), median_diff=float(np.median(d)),
        ci_lo=lo, ci_hi=hi, p_wilcoxon=p, rank_biserial=rb,
        win_rate=float((d > 0).mean()),
        a_mean=float(joined["a"].mean()), b_mean=float(joined["b"].mean()),
    )


def mcnemar_paired(a: pd.Series, b: pd.Series) -> dict:
    """Exact McNemar test for a paired binary outcome (e.g. reached < 6 Hz).

    Used instead of a paired difference for the descent-success and
    time-to-descent outcomes, where ~20-30 % of episodes never cross at
    all: dropping those pairs would silently condition on success.
    """
    j = pd.concat([a.rename("a").astype(bool), b.rename("b").astype(bool)],
                  axis=1).dropna()
    n01 = int((~j["a"] & j["b"]).sum())      # b only
    n10 = int((j["a"] & ~j["b"]).sum())      # a only
    n = n01 + n10
    p = float(stats.binomtest(n10, n, 0.5).pvalue) if n else 1.0
    return {"n_pairs": int(len(j)), "a_only": n10, "b_only": n01,
            "a_rate": float(j["a"].mean()), "b_rate": float(j["b"].mean()),
            "rate_diff": float(j["a"].mean() - j["b"].mean()),
            "p_mcnemar": p}


def holm_correct(results: list[PairedResult]) -> list[PairedResult]:
    """Holm-Bonferroni within a family of results (in place, returns list).

    Results with no usable p-value (an empty stratum, say) stay NaN and are
    excluded from the family size. Letting a NaN through the running-max
    would silently assign it p = 0.
    """
    usable = [i for i, r in enumerate(results)
              if r.p_wilcoxon is not None and np.isfinite(r.p_wilcoxon)]
    for i, r in enumerate(results):
        if i not in usable:
            r.p_holm = np.nan
    order = sorted(usable, key=lambda i: results[i].p_wilcoxon)
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * results[i].p_wilcoxon)
        running = max(running, adj)          # enforce monotonicity
        results[i].p_holm = running
    return results


# ------------------------------------------------------------ arm picking

def series_for(metrics: pd.DataFrame, mode: str, coupling: float, arm: str,
               metric: str) -> pd.Series:
    """Per-episode metric series for one cell, indexed by episode."""
    sel = metrics[(metrics["mode"] == mode) & (metrics["arm"] == arm)]
    if not np.isnan(coupling):
        sel = sel[np.isclose(sel["coupling"], coupling)]
    if sel.empty:
        raise KeyError(f"no cell for mode={mode} coupling={coupling} arm={arm}")
    return sel.set_index("episode")[metric].sort_index()


def coupling_invariant_series(metrics: pd.DataFrame, arm: str, metric: str
                              ) -> pd.Series:
    sel = metrics[metrics["arm"] == arm]
    if sel.empty:
        raise KeyError(f"no cell for arm={arm}")
    return sel.set_index("episode")[metric].sort_index()


def family_arms_in(metrics: pd.DataFrame, anchor: str) -> list[str]:
    return sorted(a for a in metrics["arm"].unique()
                  if a.startswith(f"{anchor}_e"))


def best_generic_arm(metrics: pd.DataFrame, mode: str, coupling: float,
                     metric: str) -> str:
    """Population-best generic schedule: one protocol, shipped to everyone.

    Selected on the population mean at this coupling level, which is the
    choice a manufacturer could actually make - it needs no per-subject
    information, only the population.
    """
    hib = METRIC_DIRECTION.get(metric, True)
    scores = {}
    for arm in family_arms_in(metrics, "gen"):
        try:
            scores[arm] = float(series_for(metrics, mode, coupling, arm,
                                           metric).mean())
        except KeyError:
            continue
    if not scores:
        raise KeyError("no generic family arms present")
    return (max if hib else min)(scores, key=scores.get)


def best_oracle_series(metrics: pd.DataFrame, mode: str, coupling: float,
                       metric: str) -> tuple:
    """Per-subject best oracle schedule - the non-adaptive upper bound.

    Each subject gets the family member that turned out best *for that
    subject*, anchored on its true α. No real protocol could do this: it
    needs both the subject's α and the realised outcome. It is therefore a
    deliberately generous upper bound on identification value, which makes
    the residual attributed to tracking conservative.
    """
    hib = METRIC_DIRECTION.get(metric, True)
    cols = {}
    for arm in family_arms_in(metrics, "orc"):
        try:
            cols[arm] = series_for(metrics, mode, coupling, arm, metric)
        except KeyError:
            continue
    if not cols:
        raise KeyError("no oracle family arms present")
    wide = pd.DataFrame(cols)
    best = wide.max(axis=1) if hib else wide.min(axis=1)
    chosen = wide.idxmax(axis=1) if hib else wide.idxmin(axis=1)
    return best, chosen


# --------------------------------------------------------- decomposition

def decomposition(metrics: pd.DataFrame, *, mode: str, metric: str,
                  couplings: Iterable[float],
                  use_family: bool = True) -> list[PairedResult]:
    """The three value contrasts at each coupling level, Holm-corrected."""
    results: list[PairedResult] = []
    for c in couplings:
        sac = series_for(metrics, mode, c, "sac", metric)
        yoked = series_for(metrics, mode, c, "yoked", metric)
        gen_arm = best_generic_arm(metrics, mode, c, metric) if use_family \
            else "gen_e4_r1"
        generic = series_for(metrics, mode, c, gen_arm, metric)
        if use_family:
            oracle, _ = best_oracle_series(metrics, mode, c, metric)
        else:
            oracle = series_for(metrics, mode, c, "orc_e4_r1", metric)

        r1 = paired_contrast(sac, yoked, metric=metric,
                             contrast="total_closed_loop_value",
                             mode=mode, coupling=c)
        r2 = paired_contrast(oracle, generic, metric=metric,
                             contrast="subject_identification_value",
                             mode=mode, coupling=c)
        r3 = paired_contrast(sac, oracle, metric=metric,
                             contrast="target_tracking_value",
                             mode=mode, coupling=c)
        for r in (r1, r2, r3):
            r.__dict__["generic_arm"] = gen_arm
        results.extend([r1, r2, r3])

    for name in {r.contrast for r in results}:
        holm_correct([r for r in results if r.contrast == name])
    return results


def coupling_trend(metrics: pd.DataFrame, *, mode: str, metric: str,
                   couplings, arm_a: str = "sac", arm_b: str = "yoked",
                   contrast: str = "total_closed_loop_value") -> dict:
    """Does the paired advantage *grow* with the coupling gain?

    The level of the advantage answers "does closing the loop pay here?".
    This function answers the dose-response question instead: "how does the
    advantage scale with the strength of the state->plant coupling?" That
    is a claim about the slope, and it needs its own test rather than an
    eyeball comparison of five separately-tested levels.

    Two within-subject tests, because every coupling level ran on the same
    simulated subjects:

    * **endpoint contrast** - paired Wilcoxon of the per-subject advantage
      at the highest gain against the same subject's advantage at the
      lowest. Assumption-free and directly interpretable.
    * **per-subject slope** - ordinary least squares of that subject's
      advantage on the coupling gain, then a one-sample Wilcoxon on the
      slopes. Uses all five levels instead of only the ends.
    """
    hib = METRIC_DIRECTION.get(metric, True)
    sign = 1.0 if hib else -1.0
    cols = {}
    for c in couplings:
        a = series_for(metrics, mode, c, arm_a, metric)
        b = series_for(metrics, mode, c, arm_b, metric)
        j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
        cols[float(c)] = sign * (j["a"] - j["b"])
    wide = pd.DataFrame(cols).dropna()
    xs = np.asarray(sorted(cols), dtype=float)
    lo, hi = xs[0], xs[-1]

    d_end = (wide[hi] - wide[lo]).to_numpy(np.float64)
    try:
        p_end = float(stats.wilcoxon(d_end[d_end != 0]).pvalue) \
            if np.any(d_end != 0) else 1.0
    except ValueError:
        p_end = 1.0

    xc = xs - xs.mean()
    denom = float((xc ** 2).sum())
    slopes = ((wide[xs].to_numpy(np.float64) * xc).sum(axis=1) / denom)
    try:
        p_slope = float(stats.wilcoxon(slopes[slopes != 0]).pvalue) \
            if np.any(slopes != 0) else 1.0
    except ValueError:
        p_slope = 1.0

    return {
        "mode": mode, "metric": metric, "contrast": contrast,
        "arm_a": arm_a, "arm_b": arm_b,
        "n": int(len(wide)),
        "coupling_lo": float(lo), "coupling_hi": float(hi),
        "advantage_at_lo": float(wide[lo].mean()),
        "advantage_at_hi": float(wide[hi].mean()),
        "endpoint_change": float(d_end.mean()),
        "endpoint_ci_lo": _bootstrap_ci(d_end)[0],
        "endpoint_ci_hi": _bootstrap_ci(d_end)[1],
        "p_endpoint": p_end,
        "frac_subjects_improving": float((d_end > 0).mean()),
        "mean_slope_per_unit_gain": float(slopes.mean()),
        "slope_ci_lo": _bootstrap_ci(slopes)[0],
        "slope_ci_hi": _bootstrap_ci(slopes)[1],
        "p_slope": p_slope,
        "frac_subjects_positive_slope": float((slopes > 0).mean()),
    }


def _per_subject_slopes(metrics, mode, metric, couplings, arm_a, arm_b):
    hib = METRIC_DIRECTION.get(metric, True)
    sign = 1.0 if hib else -1.0
    cols = {}
    for c in couplings:
        a = series_for(metrics, mode, c, arm_a, metric)
        b = series_for(metrics, mode, c, arm_b, metric)
        j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
        cols[float(c)] = sign * (j["a"] - j["b"])
    wide = pd.DataFrame(cols).dropna()
    xs = np.asarray(sorted(cols), dtype=float)
    xc = xs - xs.mean()
    return pd.Series(
        (wide[xs].to_numpy(np.float64) * xc).sum(axis=1) / float((xc ** 2).sum()),
        index=wide.index)


def mode_interaction(metrics: pd.DataFrame, *, metric: str, couplings,
                     endo_arms=("sac", "yoked"),
                     exo_arms=("sac_xmode", "yoked_xmode")) -> dict:
    """Does the dose-response slope differ between the two plant families?

    Both slopes are computed on the **same simulated subjects** and, when
    ``exo_arms`` names the cross-mode pair, with the **same policy** - so
    the difference is attributable to the plant rather than to which
    controller happened to be selected for each condition. Only coupling
    levels present in both conditions are used, so the two slopes are
    estimated over an identical design.
    """
    se = _per_subject_slopes(metrics, "endo", metric, couplings, *endo_arms)
    sx = _per_subject_slopes(metrics, "exo", metric, couplings, *exo_arms)
    j = pd.concat([se.rename("endo"), sx.rename("exo")], axis=1).dropna()
    d = (j["endo"] - j["exo"]).to_numpy(np.float64)
    try:
        p = float(stats.wilcoxon(d[d != 0]).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        p = 1.0
    lo, hi = _bootstrap_ci(d)
    return {
        "metric": metric, "couplings": [float(c) for c in couplings],
        "endo_arms": list(endo_arms), "exo_arms": list(exo_arms),
        "n": int(len(j)),
        "mean_slope_endo": float(j["endo"].mean()),
        "mean_slope_exo": float(j["exo"].mean()),
        "slope_difference": float(d.mean()),
        "ci_lo": lo, "ci_hi": hi, "p_wilcoxon": p,
        "frac_subjects_endo_steeper": float((d > 0).mean()),
    }


def gain_terciles(metrics: pd.DataFrame) -> pd.Series:
    """Per-episode cortical-gain tercile label.

    Stratify on ``cortical_gain``, NOT on ``responder_type``. Only 10 % of
    subjects carry the non-responder label but ~29 % of episodes fail to
    descend, so the label does not explain the bimodality; the gain does.
    """
    ref = (metrics.drop_duplicates("episode")
                  .set_index("episode")["cortical_gain"].sort_index())
    q = ref.quantile([1 / 3, 2 / 3]).to_numpy()
    labels = pd.cut(ref, [-np.inf, q[0], q[1], np.inf],
                    labels=["low", "mid", "high"])
    return labels
