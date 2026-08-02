"""
Tests for the experimental arms and the machinery that runs them.

Grouped by the failure they exist to prevent:

* **Simulated clock** - a time-driven open-loop policy that reads the wall
  clock finishes a 300 s ramp in two seconds of simulated time, silently
  invalidating its arm.
* **Dose equivalence** - the yoked control is only dose-matched if it
  replays the donor's exact physical stimulus, so the physical ->
  normalised inverse must match ``SleepOnsetEnv._decode_action`` exactly.
* **Subject identity** - every arm must see bit-identical subjects, and
  the coupling override must not perturb the RNG draw sequence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.agent.env import EnvConfig, SleepOnsetEnv
from app.agent.eval_rollout import bind_sim_clock, rollout_episodes
from app.agent.policies import (
    BaselineNoStimPolicy, FixedFrequencyPolicy, LinearRampDownPolicy,
    ScheduledRampPolicy, YokedReplayPolicy, assert_dose_equivalence,
    build_yoked_factory, normalise_clamped_actions, oracle_ramp_factory,
)
from app.agent.safety import FREQ_MAX_HZ, FREQ_MIN_HZ, SafetySupervisor, StimAction
from app.brain_sim.domain_random import RandomizationRanges
from app.brain_sim.photic_input import PhoticInputConfig
from app.open_loop_track.policy import OpenLoopTrackPolicy


def _decode_freq(a0: float) -> float:
    return FREQ_MIN_HZ + (float(a0) + 1.0) / 2.0 * (FREQ_MAX_HZ - FREQ_MIN_HZ)


# clocks

def test_linear_ramp_tick_clock_ignores_wall_time():
    """With tick_hz set, elapsed time comes from predict() calls only."""
    pol = LinearRampDownPolicy(start_hz=11.0, end_hz=4.0, duration_s=300.0,
                               tick_hz=4.0)
    pol.reset()
    obs = np.zeros(20, dtype=np.float32)
    first, _ = pol.predict(obs)
    assert _decode_freq(first[0]) == pytest.approx(11.0, abs=1e-6)
    # 300 s at 4 Hz = 1200 ticks; one already consumed.
    for _ in range(1199):
        pol.predict(obs)
    last, _ = pol.predict(obs)
    assert _decode_freq(last[0]) == pytest.approx(4.0, abs=1e-6)


def test_linear_ramp_hits_midpoint_frequency_at_half_duration():
    pol = LinearRampDownPolicy(start_hz=11.0, end_hz=4.0, duration_s=100.0,
                               tick_hz=4.0)
    pol.reset()
    obs = np.zeros(20, dtype=np.float32)
    for _ in range(200):            # 50 s of simulated time
        action, _ = pol.predict(obs)
    assert _decode_freq(action[0]) == pytest.approx(7.5, abs=0.05)


def test_linear_ramp_defaults_to_wall_clock():
    """tick_hz=None keeps live-session semantics for the daemon."""
    pol = LinearRampDownPolicy(duration_s=60.0)
    assert pol.tick_hz is None
    pol.reset()
    pol._start_t -= 30.0
    action, _ = pol.predict(np.zeros(20, dtype=np.float32))
    assert _decode_freq(action[0]) == pytest.approx(7.5, abs=0.2)


def test_reset_restarts_ramp_at_start_frequency():
    """Each episode must begin a fresh ramp, not resume the previous one."""
    pol = LinearRampDownPolicy(start_hz=11.0, end_hz=4.0, duration_s=10.0,
                               tick_hz=4.0)
    pol.reset()
    obs = np.zeros(20, dtype=np.float32)
    for _ in range(40):
        pol.predict(obs)
    pol.reset()
    action, _ = pol.predict(obs)
    assert _decode_freq(action[0]) == pytest.approx(11.0, abs=1e-6)


def test_open_loop_track_elapsed_time_follows_simulated_clock():
    pol = OpenLoopTrackPolicy(playback=None, tick_hz=4.0)
    pol.reset()
    assert pol.elapsed_s() == 0.0
    pol._ticks = 400
    assert pol.elapsed_s() == pytest.approx(100.0)


def test_bind_sim_clock_binds_and_respects_explicit_value():
    pol = LinearRampDownPolicy()
    bind_sim_clock(pol, 4.0)
    assert pol.tick_hz == 4.0
    pol2 = LinearRampDownPolicy(tick_hz=2.0)
    bind_sim_clock(pol2, 4.0)
    assert pol2.tick_hz == 2.0          # explicit setting wins


def test_bind_sim_clock_rejects_unconvertible_wall_clock_policy():
    """A time-driven policy with no tick_hz must fail loudly, not silently."""
    class LegacyWallClockPolicy:
        def __init__(self):
            self._start_t = 0.0

        def predict(self, obs, deterministic=True):
            return np.zeros(5, dtype=np.float32), None

    with pytest.raises(TypeError, match="tick_hz"):
        bind_sim_clock(LegacyWallClockPolicy(), 4.0)


def test_bind_sim_clock_ignores_stateless_policies():
    bind_sim_clock(FixedFrequencyPolicy(), 4.0)
    bind_sim_clock(BaselineNoStimPolicy(), 4.0)


# safety sim clock

def test_safety_event_timestamps_use_simulated_clock():
    sv = SafetySupervisor(tick_dt_s=0.25)
    for _ in range(8):
        sv.clamp(StimAction(frequency_hz=20.0))   # PSE band -> logs every tick
    stamps = [e.t_ms for e in sv.events if e.kind == "pse_guard"]
    assert stamps[:4] == [0, 250, 500, 750]


def test_safety_supervisor_defaults_to_wall_clock():
    sv = SafetySupervisor()
    assert sv.tick_dt_s is None
    sv.clamp(StimAction(frequency_hz=10.0))
    assert sv.session_elapsed_s() < 1.0        # wall time, not tick count


def test_safety_session_timeout_fires_on_simulated_time():
    sv = SafetySupervisor(tick_dt_s=1.0, session_timeout_s=5.0)
    for _ in range(8):      # elapsed only exceeds 5.0 s on the 7th tick
        out = sv.clamp(StimAction(frequency_hz=10.0, brightness=0.3))
    assert out.enabled is False
    assert any(e.kind == "session_timeout" for e in sv.events)


# schedule family

def test_scheduled_ramp_reaches_end_at_ramp_end_frac():
    pol = ScheduledRampPolicy(start_hz=10.0, end_hz=4.0, duration_s=100.0,
                              ramp_end_frac=0.5, tick_hz=4.0)
    assert pol.target_hz(0.0) == pytest.approx(10.0)
    assert pol.target_hz(25.0) == pytest.approx(7.0)
    assert pol.target_hz(50.0) == pytest.approx(4.0)
    assert pol.target_hz(90.0) == pytest.approx(4.0)   # holds after ramp


def test_scheduled_ramp_hold_before_ramp_start():
    pol = ScheduledRampPolicy(start_hz=10.0, end_hz=4.0, duration_s=100.0,
                              ramp_start_frac=0.5, ramp_end_frac=1.0)
    assert pol.target_hz(10.0) == pytest.approx(10.0)
    assert pol.target_hz(75.0) == pytest.approx(7.0)


def test_oracle_factory_anchors_on_true_alpha():
    factory = oracle_ramp_factory(end_hz=4.0, duration_s=300.0)
    pol = factory(0, {"alpha_hz": 9.25, "cortical_gain": 300.0})
    assert pol.start_hz == pytest.approx(9.25)
    assert pol.target_hz(0.0) == pytest.approx(9.25)
    assert pol.target_hz(300.0) == pytest.approx(4.0)


# dose equivalence

def test_normalise_clamped_actions_inverts_env_decode_exactly():
    """The yoked encoder must be the exact inverse of the env's decoder."""
    env = SleepOnsetEnv(EnvConfig(episode_s=1.0, include_restfulness=False))
    rng = np.random.default_rng(0)
    raw = rng.uniform(-1.0, 1.0, size=(64, 5))
    decoded = [env._decode_action(a) for a in raw]
    df = pd.DataFrame({
        "strobe_freq_hz": [d.frequency_hz for d in decoded],
        "fill": [d.fill for d in decoded],
        "brightness": [d.brightness for d in decoded],
        "audio_depth_pct": [d.audio_strobe_depth_pct for d in decoded],
        "audio_vol_pct": [d.audio_master_vol_pct for d in decoded],
    })
    back = normalise_clamped_actions(df)
    assert np.max(np.abs(back - raw)) < 1e-12
    env.close()


def test_yoked_policy_replays_trajectory_in_order():
    traj = np.arange(15, dtype=np.float64).reshape(3, 5)
    pol = YokedReplayPolicy(trajectory=traj)
    pol.reset()
    seen = [pol.predict(None)[0] for _ in range(5)]
    assert np.allclose(seen[0], traj[0])
    assert np.allclose(seen[2], traj[2])
    # past the end it holds the final command rather than raising
    assert np.allclose(seen[4], traj[2])


def test_build_yoked_factory_deranges_subjects():
    df = pd.DataFrame({
        "episode": np.repeat(np.arange(4), 3),
        "tick": np.tile(np.arange(3), 4),
        "strobe_freq_hz": np.linspace(4.0, 13.0, 12),
        "fill": 0.4, "brightness": 0.6,
        "audio_depth_pct": 20.0, "audio_vol_pct": 50.0,
    })
    factory = build_yoked_factory(df, offset=1)
    assert [factory(e, {}).donor_episode for e in range(4)] == [3, 0, 1, 2]


def test_build_yoked_factory_rejects_self_replay():
    df = pd.DataFrame({
        "episode": np.repeat(np.arange(3), 2), "tick": np.tile([0, 1], 3),
        "strobe_freq_hz": 8.0, "fill": 0.4, "brightness": 0.6,
        "audio_depth_pct": 20.0, "audio_vol_pct": 50.0,
    })
    with pytest.raises(ValueError, match="own"):
        build_yoked_factory(df, offset=3)


def test_yoked_rollout_is_dose_equivalent_end_to_end():
    """On real rollouts, the yoked arm must deliver the donor's exact
    stimulus while the recipient is a different simulated subject."""
    rand = RandomizationRanges(shift_per_unit_ratio_override=1.0)
    donor = rollout_episodes(
        policy=ScheduledRampPolicy(start_hz=11.0, end_hz=4.0, duration_s=8.0,
                                   fill=0.4, brightness=0.6),
        n_episodes=3, episode_s=8.0, randomization=rand, base_seed=7,
    )
    yoked = rollout_episodes(
        policy_factory=build_yoked_factory(donor, offset=1),
        n_episodes=3, episode_s=8.0, randomization=rand, base_seed=7,
    )
    worst = assert_dose_equivalence(donor, yoked, offset=1, tol=1e-9)
    assert worst["n_exact"] == 3, "no dropouts expected in a short episode"
    deltas = [v for k, v in worst.items() if k.startswith("max_abs_delta_")
              and not k.endswith("_divergent")]
    assert max(deltas) < 1e-9
    # ... and the recipient really is a different subject from the donor.
    assert not np.allclose(
        donor[donor.episode == 0]["theta_alpha_ratio"].to_numpy(),
        yoked[yoked.episode == 1]["theta_alpha_ratio"].to_numpy(),
    )


# coupling / subjects

def test_coupling_range_draw_is_conditional_and_last():
    """Leaving the range unset must not perturb the historical RNG draws."""
    plain = RandomizationRanges().sample(seed=123)
    override = RandomizationRanges(
        shift_per_unit_ratio_override=0.5).sample(seed=123)
    assert plain.alpha_hz == override.alpha_hz
    assert plain.photic_config.cortical_gain == override.photic_config.cortical_gain
    assert plain.nmm_params.p_mean == override.nmm_params.p_mean
    assert override.photic_config.shift_per_unit_ratio == 0.5
    assert plain.photic_config.shift_per_unit_ratio == 2.0


def test_coupling_range_samples_within_bounds_and_varies():
    ranges = RandomizationRanges(shift_per_unit_ratio_range=(0.0, 2.0))
    vals = [ranges.sample(seed=s).photic_config.shift_per_unit_ratio
            for s in range(30)]
    assert all(0.0 <= v <= 2.0 for v in vals)
    assert len(set(vals)) > 20


def test_coupling_override_takes_precedence_over_range():
    ranges = RandomizationRanges(shift_per_unit_ratio_range=(0.0, 2.0),
                                 shift_per_unit_ratio_override=1.25)
    assert ranges.sample(seed=3).photic_config.shift_per_unit_ratio == 1.25


def test_every_arm_sees_identical_subjects():
    """Within-subject design: same base_seed -> same simulated people."""
    rand = RandomizationRanges(shift_per_unit_ratio_override=0.5)
    a = rollout_episodes(policy=FixedFrequencyPolicy(frequency_hz=10.0),
                         n_episodes=2, episode_s=4.0, randomization=rand,
                         base_seed=11)
    b = rollout_episodes(policy=BaselineNoStimPolicy(),
                         n_episodes=2, episode_s=4.0, randomization=rand,
                         base_seed=11)
    for col in ("alpha_hz", "cortical_gain", "resonance_width_hz",
                "responder_type"):
        assert list(a.groupby("episode")[col].first()) == \
               list(b.groupby("episode")[col].first())


# --------------------------------------------------------- exogenous mode

def test_exogenous_drive_depends_only_on_elapsed_time():
    # The drift runs on the subject's OWN timescale (sol_scale_s), not on
    # episode length, so the expected values are pinned to sol_scale_s.
    pc = PhoticInputConfig(base_resonance_hz=10.0, shift_per_unit_ratio=1.0,
                           exogenous_shift=True, exogenous_span=3.8,
                           sol_scale_s=300.0)
    assert pc.exogenous_drive(0.0) == pytest.approx(0.5)
    assert pc.exogenous_drive(150.0) == pytest.approx(0.5 + 1.9)
    assert pc.exogenous_drive(300.0) == pytest.approx(0.5 + 3.8)
    assert pc.exogenous_drive(600.0) == pytest.approx(0.5 + 3.8)


def test_exogenous_drive_scales_with_subject_timescale():
    """A slow subject must be correspondingly earlier in its drift at time t."""
    fast = PhoticInputConfig(exogenous_shift=True, exogenous_span=3.8,
                             sol_scale_s=150.0)
    slow = PhoticInputConfig(exogenous_shift=True, exogenous_span=3.8,
                             sol_scale_s=600.0)
    assert fast.exogenous_drive(150.0) == pytest.approx(0.5 + 3.8)   # done
    assert slow.exogenous_drive(150.0) == pytest.approx(0.5 + 0.95)  # quarter
    assert slow.exogenous_drive(150.0) < fast.exogenous_drive(150.0)


def test_exogenous_mode_ignores_measured_theta_alpha():
    """The controller must not be able to move its own target here."""
    env = SleepOnsetEnv(EnvConfig(
        episode_s=4.0, include_restfulness=False,
        randomization=RandomizationRanges(exogenous_shift=True,
                                          shift_per_unit_ratio_override=1.0),
    ))
    env.reset(seed=5)
    action = np.zeros(5, dtype=np.float32)
    drives, ratios = [], []
    for _ in range(8):
        _, _, _, trunc, info = env.step(action)
        drives.append(info["shift_drive"])
        ratios.append(info["features"].theta_alpha_ratio)
        if trunc:
            break
    # The drive is a straight function of session progress...
    assert drives == sorted(drives)
    # ...and is not the measured biomarker, which is what closes the loop
    # in the endogenous condition.
    assert not np.allclose(drives, ratios, atol=1e-6)
    env.close()


def test_endogenous_mode_still_tracks_theta_alpha():
    env = SleepOnsetEnv(EnvConfig(episode_s=4.0, include_restfulness=False))
    env.reset(seed=5)
    _, _, _, _, info = env.step(np.zeros(5, dtype=np.float32))
    _, _, _, _, info2 = env.step(np.zeros(5, dtype=np.float32))
    assert info2["shift_drive"] == pytest.approx(
        info["features"].theta_alpha_ratio, abs=1e-9)
    env.close()


def test_nostim_arm_is_coupling_invariant():
    """With stimulation off, coupling strength cannot reach the subject, so
    the no-stim arm only has to be run once per design."""
    frames = [
        rollout_episodes(
            policy=BaselineNoStimPolicy(), n_episodes=1, episode_s=4.0,
            randomization=RandomizationRanges(
                shift_per_unit_ratio_override=c, exogenous_shift=exo),
            base_seed=3)
        for c, exo in ((0.0, False), (2.0, False), (2.0, True))
    ]
    ref = frames[0][["theta_alpha_ratio", "plv", "reward"]].to_numpy(np.float64)
    for f in frames[1:]:
        other = f[["theta_alpha_ratio", "plv", "reward"]].to_numpy(np.float64)
        assert np.array_equal(ref, other)


# rollout API

def test_rollout_requires_exactly_one_policy_source():
    with pytest.raises(ValueError):
        rollout_episodes(n_episodes=1, episode_s=2.0)
    with pytest.raises(ValueError):
        rollout_episodes(checkpoint_path="x.zip",
                         policy=FixedFrequencyPolicy(), n_episodes=1)


def test_rollout_records_resonance_and_subject_columns():
    df = rollout_episodes(
        policy=FixedFrequencyPolicy(frequency_hz=10.0, fill=0.4,
                                    brightness=0.6),
        n_episodes=1, episode_s=4.0, base_seed=17,
        randomization=RandomizationRanges(shift_per_unit_ratio_override=2.0),
    )
    for col in ("effective_resonance_hz", "shift_drive", "responder_type",
                "cortical_gain", "alpha_hz", "shift_per_unit_ratio",
                "audio_depth_pct", "audio_vol_pct", "ramp_limited_freq",
                "n_safety_events", "tick"):
        assert col in df.columns, col
    assert (df["effective_resonance_hz"] >= 4.0).all()
    assert (df["effective_resonance_hz"] <= df["alpha_hz"] + 1e-9).all()
    assert (df["shift_per_unit_ratio"] == 2.0).all()


def test_policy_factory_receives_subject_info():
    seen: list[dict] = []

    def factory(ep: int, subject: dict):
        seen.append(subject)
        return FixedFrequencyPolicy(frequency_hz=10.0)

    rollout_episodes(policy_factory=factory, n_episodes=2, episode_s=2.0,
                     base_seed=21)
    assert len(seen) == 2
    assert {"alpha_hz", "responder", "cortical_gain"} <= set(seen[0])
    assert seen[0]["alpha_hz"] != seen[1]["alpha_hz"]
