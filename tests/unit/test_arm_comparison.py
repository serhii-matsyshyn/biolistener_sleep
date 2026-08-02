"""
Tests for the closed-loop analysis layer.

The statistics decide how the experiment reads, so the parts that are
easy to get silently wrong are pinned here: the sign convention on
lower-is-better metrics, Holm monotonicity, the ratchet accounting, and
the "best of family" selection rules that define the oracle and generic
comparators.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.analysis import arm_comparison as P


def _ticks(episode: int, freqs, ratio=None, res=None, **statics) -> pd.DataFrame:
    n = len(freqs)
    ratio = np.linspace(0.5, 3.0, n) if ratio is None else ratio
    res = np.full(n, 10.0) if res is None else res
    return pd.DataFrame({
        "episode": episode, "tick": np.arange(n),
        "t_s": (np.arange(n) + 1) * 0.25,
        "strobe_freq_hz": np.asarray(freqs, dtype=float),
        "fill": 0.4, "brightness": 0.6,
        "enabled": statics.get("enabled", True),
        "theta_alpha_ratio": ratio, "alpha": np.linspace(2.0, 1.0, n),
        "hypno_theta": np.linspace(0.2, 0.9, n),
        "plv": 0.3, "plv_2f": 0.1, "reward": 1.0,
        "effective_resonance_hz": res, "shift_drive": ratio,
        "n_safety_events": 0, "ramp_limited_freq": False,
        "responder_type": statics.get("responder_type", "moderate"),
        "alpha_hz": statics.get("alpha_hz", 10.0),
        "cortical_gain": statics.get("cortical_gain", 400.0),
        "resonance_width_hz": 3.0,
        "shift_per_unit_ratio": statics.get("shift", 2.0),
        "exogenous_shift": False,
    })


def test_cell_name_parsing_extracts_mode_coupling_and_arm():
    assert P.parse_cell_name("endo__c0.25__gen_e3_r0.5") == {
        "mode": "endo", "coupling": 0.25, "arm": "gen_e3_r0.5"}
    assert P.parse_cell_name("endo__c0__sac")["coupling"] == 0.0
    inv = P.parse_cell_name("endo__nostim")
    assert inv["arm"] == "nostim" and np.isnan(inv["coupling"])


def test_episode_metrics_ratchet_accounting():
    """Travel up/down must be measured, not inferred from the endpoints."""
    freqs = [11, 10, 9, 12, 11, 7, 5, 9, 4]
    m = P.episode_metrics(_ticks(0, freqs)).iloc[0]
    d = np.diff(freqs)
    assert m["up_travel_hz"] == pytest.approx(d[d > 0].sum())
    assert m["down_travel_hz"] == pytest.approx(-d[d < 0].sum())
    assert m["net_descent_hz"] == pytest.approx(11 - 4)
    # first crossing of 8 Hz is at index 5 (freq 7), i.e. t = 1.5 s
    assert m["time_to_first_below_8hz"] == pytest.approx(1.5)
    # after that it goes 7,5,9,4 -> running min 7,5,5,4 -> worst rise 9-5 = 4
    assert m["max_reascent_after_8hz"] == pytest.approx(4.0)


def test_episode_metrics_never_crossing_is_nan_not_imputed():
    m = P.episode_metrics(_ticks(0, [12.0] * 8)).iloc[0]
    assert np.isnan(m["time_to_first_below_8hz"])
    assert np.isnan(m["max_reascent_after_8hz"])
    assert m["ever_below_6hz"] is False or m["ever_below_6hz"] == False  # noqa: E712


def test_episode_metrics_counts_stimulation_dropouts():
    """`not enabled` is the electrode-disconnect guard firing, and it must
    be surfaced: it suppresses stimulation and pays -10 reward per tick."""
    g = _ticks(0, [10.0] * 10)
    g.loc[3:5, "enabled"] = False
    m = P.episode_metrics(g).iloc[0]
    assert m["frac_ticks_disconnected"] == pytest.approx(0.3)
    assert bool(m["any_disconnect"]) is True
    clean = P.episode_metrics(_ticks(0, [10.0] * 10)).iloc[0]
    assert clean["frac_ticks_disconnected"] == 0.0
    assert bool(clean["any_disconnect"]) is False


def test_episode_metrics_floor_saturation():
    res = np.array([10.0, 8.0, 4.0, 4.0, 4.0, 4.0])
    m = P.episode_metrics(_ticks(0, [10.0] * 6, res=res)).iloc[0]
    assert m["frac_ticks_at_floor"] == pytest.approx(4 / 6)
    assert m["resonance_excursion_hz"] == pytest.approx(6.0)


def test_paired_contrast_sign_convention_lower_is_better():
    """final_strobe_hz is better when lower; positive must still mean 'a wins'."""
    a = pd.Series([4.0, 5.0, 6.0])       # a descends further
    b = pd.Series([9.0, 9.0, 9.0])
    r = P.paired_contrast(a, b, metric="final_strobe_hz", contrast="t",
                          mode="endo", coupling=1.0)
    assert r.mean_diff > 0
    assert r.win_rate == 1.0
    # ... and the reverse orientation flips it
    r2 = P.paired_contrast(b, a, metric="final_strobe_hz", contrast="t",
                           mode="endo", coupling=1.0)
    assert r2.mean_diff == pytest.approx(-r.mean_diff)


def test_paired_contrast_higher_is_better_default():
    a = pd.Series([3.0, 4.0, 5.0])
    b = pd.Series([1.0, 1.0, 1.0])
    r = P.paired_contrast(a, b, metric="final_theta_alpha", contrast="t",
                          mode="endo", coupling=0.0)
    assert r.mean_diff == pytest.approx(3.0)
    assert 0.99 < r.rank_biserial <= 1.0
    assert r.ci_lo < r.mean_diff < r.ci_hi


def test_paired_contrast_identical_series_is_null():
    s = pd.Series([1.0, 2.0, 3.0])
    r = P.paired_contrast(s, s.copy(), metric="final_theta_alpha",
                          contrast="t", mode="endo", coupling=0.0)
    assert r.mean_diff == 0.0
    assert r.p_wilcoxon == 1.0
    assert r.rank_biserial == 0.0


def test_holm_is_monotone_and_never_shrinks_a_p_value():
    res = [P.PairedResult("c", "m", "endo", c, 10, 0.1, 0.1, 0, 0, p, 0, 0.5,
                          0, 0)
           for c, p in zip(range(4), [0.001, 0.02, 0.04, 0.9])]
    P.holm_correct(res)
    adj = [r.p_holm for r in res]
    assert adj == sorted(adj)
    assert all(a >= r.p_wilcoxon for a, r in zip(adj, res))
    assert adj[0] == pytest.approx(0.004)


def test_mcnemar_counts_discordant_pairs_only():
    a = pd.Series([True, True, False, False, True])
    b = pd.Series([True, False, True, False, False])
    out = P.mcnemar_paired(a, b)
    assert out["a_only"] == 2 and out["b_only"] == 1
    assert out["rate_diff"] == pytest.approx(0.2)


def _fake_metrics() -> pd.DataFrame:
    """Two arms per anchor with a known winner, plus sac/yoked."""
    rows = []
    spec = {
        "gen_e4_r1": [1.0, 1.0, 1.0, 1.0],
        "gen_e3_r1": [2.0, 2.0, 2.0, 2.0],      # population-best generic
        "orc_e4_r1": [3.0, 1.0, 3.0, 1.0],
        "orc_e3_r1": [1.0, 4.0, 1.0, 4.0],      # per-subject best alternates
        "sac": [5.0, 5.0, 5.0, 5.0],
        "yoked": [2.0, 2.0, 2.0, 2.0],
    }
    for arm, vals in spec.items():
        for ep, v in enumerate(vals):
            rows.append({"mode": "endo", "coupling": 1.0, "arm": arm,
                         "episode": ep, "final_theta_alpha": v,
                         "cortical_gain": 100.0 * (ep + 1),
                         "responder_type": "moderate"})
    return pd.DataFrame(rows)


def test_best_generic_arm_picks_population_best():
    m = _fake_metrics()
    assert P.best_generic_arm(m, "endo", 1.0, "final_theta_alpha") == "gen_e3_r1"


def test_best_oracle_series_picks_per_subject_best():
    m = _fake_metrics()
    best, chosen = P.best_oracle_series(m, "endo", 1.0, "final_theta_alpha")
    assert list(best) == [3.0, 4.0, 3.0, 4.0]
    assert list(chosen) == ["orc_e4_r1", "orc_e3_r1",
                            "orc_e4_r1", "orc_e3_r1"]


def test_decomposition_splits_total_into_identification_and_tracking():
    """total = identification + tracking only when the comparators line up;
    here we check each component is computed from the intended pair."""
    m = _fake_metrics()
    res = P.decomposition(m, mode="endo", metric="final_theta_alpha",
                          couplings=[1.0])
    by = {r.contrast: r for r in res}
    assert by["total_closed_loop_value"].mean_diff == pytest.approx(3.0)
    assert by["subject_identification_value"].mean_diff == pytest.approx(1.5)
    assert by["target_tracking_value"].mean_diff == pytest.approx(1.5)


def test_gain_terciles_split_roughly_evenly():
    rows = [{"episode": e, "cortical_gain": float(e), "arm": "sac"}
            for e in range(30)]
    labels = P.gain_terciles(pd.DataFrame(rows))
    counts = labels.value_counts()
    assert set(counts.index) == {"low", "mid", "high"}
    assert counts.min() >= 9
