"""
Unit tests for the reward-ablation registry + the new RewardWeights
fields (``k_alpha_pwr``, ``flat_plv``) and the adversarial shield test.

These exist primarily to catch regressions in the variant definitions -
the thesis demo's headline plots rely on these specific numbers and
behaviours, so freezing them in tests prevents accidental edits.
"""
from __future__ import annotations

import pytest

from app.agent.reward import RewardWeights, band_weight, compute_reward
from app.agent.reward_variants import REWARD_VARIANTS, resolve
from app.agent.safety_adversarial import run_shield_attack
from app.eeg.features import FeatureSnapshot


def _fs(**kw) -> FeatureSnapshot:
    base = dict(t_ms=0, alpha_pwr=0.0, theta_pwr=0.0, beta_pwr=0.0,
                hypnagogic_theta_pwr=0.0, sigma_pwr=0.0,
                theta_alpha_ratio=0.0, emg_rms=0.0, eog_energy=0.0,
                railed_pct=0.0, n_samples=500)
    base.update(kw)
    return FeatureSnapshot(**base)


def test_registry_has_four_named_variants():
    assert set(REWARD_VARIANTS) == {
        "naive_alpha_max", "naive_plv_only",
        "theta_alpha_delta", "full_shaped",
    }


def test_resolve_by_name():
    rw = resolve("full_shaped")
    assert rw is REWARD_VARIANTS["full_shaped"]


def test_resolve_passes_through_RewardWeights_instance():
    custom = RewardWeights(k_theta_alpha=2.0)
    assert resolve(custom) is custom


def test_resolve_none_returns_defaults():
    rw = resolve(None)
    assert isinstance(rw, RewardWeights)
    assert rw.k_theta_alpha == RewardWeights().k_theta_alpha


def test_resolve_unknown_raises():
    with pytest.raises(ValueError, match="Unknown reward variant"):
        resolve("not_a_real_variant")


def test_naive_alpha_max_only_rewards_alpha_power():
    """The naive baseline must produce reward purely as a function of
    α-power. Anything else (PLV, descent, drowsiness biomarkers) is
    zeroed out so the exploit-style policy emerges cleanly."""
    rw = REWARD_VARIANTS["naive_alpha_max"]
    # Strong α, no other features, no PLV, no descent.
    r = compute_reward(prev=_fs(), curr=_fs(alpha_pwr=0.42),
                       plv_strobe=0.9, strobe_freq_hz=10.0,
                       prev_strobe_freq_hz=10.0, weights=rw)
    # k_alpha_pwr = 1.0 -> r ≈ α-power; tiny float wobble allowed.
    assert r == pytest.approx(0.42, abs=1e-9)


def test_flat_plv_skips_band_weighting():
    """``flat_plv=True`` makes PLV at any strobe frequency earn the full
    coefficient - this is THE exploit that the naive_plv_only variant
    demonstrates in the ablation."""
    rw = RewardWeights(k_theta_alpha=0.0, k_theta_alpha_level=0.0,
                       k_hypnagogic=0.0, k_plv=1.0, k_descent=0.0,
                       k_alpha_pwr=0.0, flat_plv=True)
    # 15 Hz is in the β/wake band - band_weight(15) = 0.
    assert band_weight(15.0) == 0.0
    r = compute_reward(prev=_fs(), curr=_fs(), plv_strobe=0.8,
                       strobe_freq_hz=15.0, prev_strobe_freq_hz=15.0,
                       weights=rw)
    assert r == pytest.approx(0.8, abs=1e-9)


def test_band_weight_still_applies_when_flat_plv_false():
    """Sanity: the same PLV at 15 Hz earns zero when band-weighting is on."""
    rw = RewardWeights(k_theta_alpha=0.0, k_theta_alpha_level=0.0,
                       k_hypnagogic=0.0, k_plv=1.0, k_descent=0.0,
                       k_alpha_pwr=0.0, flat_plv=False)
    r = compute_reward(prev=_fs(), curr=_fs(), plv_strobe=0.8,
                       strobe_freq_hz=15.0, prev_strobe_freq_hz=15.0,
                       weights=rw)
    assert r == 0.0


def test_shield_attack_blocks_all_pse_requests():
    """The headline test for the safety supervisor: under 500 adversarial
    PSE-band requests, NONE may leak through. If this ever fails we have
    a real safety bug, not a test flake."""
    r = run_shield_attack(n_requests=500, seed=42)
    assert r.n_requested == 500
    assert r.n_pse_leaked == 0
    assert r.n_pse_blocked == 500
    assert r.n_brightness_clamped == 500
    assert r.n_fill_clamped == 500
