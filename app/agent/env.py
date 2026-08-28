"""
Gymnasium env for the closed-loop sleep-onset RL agent.

Wraps:
- ``InProcessBrainSim`` - domain-randomized Jansen-Rit NMM (sim-only)
- ``StimAction`` -> ``SafetySupervisor`` -> applied to brain + audio (no-op
  on the training path; pyo isn't booted)
- Per-tick :func:`extract_features` + optional RESTFULNESS auxiliary
- :func:`compute_reward` on the manual features only

The observation built here is reproduced element for element by
``app.protocol.runner`` on the deployment path; ``app.agent.obs_state``
owns the smoothing both use.

Observation space (float32):
    [alpha_pwr, theta_pwr, beta_pwr, hypnagogic_theta_pwr, sigma_pwr,
     theta_alpha_ratio, emg_rms, eog_energy, railed_pct,
     plv_strobe,
     restfulness (auxiliary; only when include_restfulness),
     last_strobe_freq, last_strobe_fill, last_strobe_brightness]

Action space (continuous, Box):
    [strobe_freq_hz_norm (-1..1), strobe_fill_norm (-1..1),
     strobe_brightness_norm (-1..1),
     audio_strobe_depth_pct_norm (-1..1),
     audio_master_vol_pct_norm (-1..1)]

The norm-space is decoded to physical units before clamping. The
SafetySupervisor's clamps are unconditional - the agent can never produce
an unsafe physical action.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from app.agent.obs_state import ObsState
from app.agent.reward import RewardWeights, compute_reward
from app.agent.safety import (
    FREQ_MAX_HZ, FREQ_MIN_HZ, SafetySupervisor, StimAction,
)
from app.audio.controller import AudioController
from app.brain_sim.brainflow_board import InProcessBrainSim
from app.brain_sim.domain_random import RandomizationRanges, SubjectSample
from app.brain_sim.nmm import MultiChannelBrain
from app.brain_sim.photic_input import StrobeParams, strobe_envelope
from app.eeg.features import (
    FEATURE_NAMES, FeatureSnapshot, extract_features, feature_vector,
)
from app.eeg.montage import OCCIPITAL_ALPHA, Montage
from app.eeg.plv import plv_strobe_to_eeg, plv_strobe_harmonic

logger = logging.getLogger("agent.env")


CONTROL_TICK_HZ = 4.0      # 250 ms control loop tick - proposal default
WINDOW_SECONDS = 4.0       # rolling EEG window for feature extraction
EPISODE_SECONDS = 60.0     # short by default to make training tractable
SAMPLING_RATE = 500        # BIOLISTENER default


@dataclass
class EnvConfig:
    sampling_rate: int = SAMPLING_RATE
    tick_hz: float = CONTROL_TICK_HZ
    window_s: float = WINDOW_SECONDS
    episode_s: float = EPISODE_SECONDS
    montage: Montage = OCCIPITAL_ALPHA
    include_restfulness: bool = True
    randomization: RandomizationRanges = None
    reward_weights: RewardWeights = None
    # Stimulus-artifact diagnostic. When True the feature extractor
    # additionally reports band powers with the strobe harmonics excised
    # from the PSD, plus a bin-count-matched decoy control, so an apparent
    # band-power change can be separated from the photic driving artifact.
    # Purely additive - the observation vector and the reward are unchanged,
    # so a deterministic rollout produces bit-identical actions either way.
    diagnostics: bool = False


class SleepOnsetEnv(gym.Env):
    """
    Single-subject simulator env for sleep-onset RL training.

    Each ``reset()`` samples a new subject from ``RandomizationRanges``,
    instantiates a fresh NMM, and runs an episode of ``episode_s`` seconds.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[EnvConfig] = None):
        super().__init__()
        self.config = config or EnvConfig()
        if self.config.randomization is None:
            self.config.randomization = RandomizationRanges()
        if self.config.reward_weights is None:
            self.config.reward_weights = RewardWeights()

        # Observation: 9 manual features + plv + restfulness + 3 last-action
        # + 2 temporal context (smoothed PLV, Δsmoothed θ/α over 2s).
        # The temporal pair makes the agent explicitly aware of trends so
        # it can learn to downweight transient noise spikes instead of
        # relying on a hardcoded threshold.
        obs_dim = len(FEATURE_NAMES) + 1 + (1 if self.config.include_restfulness else 0) + 3 + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32,
        )

        self._sim: Optional[InProcessBrainSim] = None
        self._supervisor: Optional[SafetySupervisor] = None
        self._audio: Optional[AudioController] = None
        self._subject: Optional[SubjectSample] = None
        self._strobe: StrobeParams = StrobeParams()
        self._buf: np.ndarray = np.zeros(
            (1, int(self.config.window_s * self.config.sampling_rate)),
            dtype=np.float64,
        )
        self._buf_pos: int = 0
        self._buf_filled: bool = False
        self._t: float = 0.0
        # Deterministic episode stream for unseeded resets (see reset()).
        # set_episode_seed_base() is called per subprocess rank by make_env.
        self._episode_seed_base: int = 0
        self._episode_idx: int = 0
        self._prev_features: Optional[FeatureSnapshot] = None
        self._prev_strobe_freq_hz: Optional[float] = None
        # Temporal smoothing state
        # Full noise-reduction stack at 500 SPS / 4 Hz tick:
        #   1. Welch PSD over a 4-s sliding window (2000 samples) - already
        #      smooths instantaneous EEG noise via ~7 overlapping segments
        #      with 50% overlap (~8 dB white-noise SNR boost).
        #   2. EMA over feature ticks (τ=2s, ~1000-sample equivalent).
        #   3. Δ(smoothed θ/α) over 4 s - agent sees the trend explicitly.
        # The math is owned by ``ObsState`` so the runtime daemon can build
        # an *identical* observation vector at deployment.
        self._obs_state: ObsState = ObsState(tick_hz=self.config.tick_hz)
        # Restfulness model is expensive to prepare - own it here
        self._restfulness = None
        if self.config.include_restfulness:
            from app.eeg.ml_metrics import RestfulnessMetric
            self._restfulness = RestfulnessMetric()
            self._restfulness.prepare()

    def set_episode_seed_base(self, base: int) -> None:
        """Fix this env's unseeded-reset stream. Called once per vec-env
        rank so parallel collectors draw disjoint, reproducible subjects."""
        self._episode_seed_base = int(base)
        self._episode_idx = 0

    def close(self) -> None:
        if self._restfulness is not None:
            self._restfulness.release()
            self._restfulness = None

    def reset(self, *, seed: Optional[int] = None,
              options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        # SB3 auto-resets with a bare reset() at every episode boundary, so
        # an unseeded reset must not fall through to OS entropy. Derive a
        # deterministic seed from the env's own stream when the caller
        # supplies none, keeping training runs reproducible and paired.
        if seed is None:
            seed = self._episode_seed_base + self._episode_idx
        self._episode_idx += 1
        sampled = self.config.randomization.sample(seed=seed)
        self._subject = sampled
        brain = MultiChannelBrain(
            n_channels=8,
            base_params=sampled.nmm_params,
            alpha_hz=sampled.alpha_hz,
            sampling_rate=self.config.sampling_rate,
            seed=sampled.seed,
        )
        self._sim = InProcessBrainSim(
            brain=brain, photic_config=sampled.photic_config,
            session_duration_s=self.config.episode_s,
        )
        # Simulated clock: one env step advances 1 / tick_hz of *simulated*
        # time regardless of how fast the loop actually runs, so the ramp
        # limiter must be told that rather than reading time.time().
        self._supervisor = SafetySupervisor(
            tick_dt_s=1.0 / self.config.tick_hz,
        )
        self._audio = AudioController(engine=None)
        self._strobe = StrobeParams()
        n_buf = int(self.config.window_s * self.config.sampling_rate)
        n_ch = len(self.config.montage.all_active() or [0]) or 8
        # Buffer must cover at least all channels we'll touch - store full 8
        self._buf = np.zeros((8, n_buf), dtype=np.float64)
        self._buf_pos = 0
        self._buf_filled = False
        self._t = 0.0
        self._prev_features = None
        self._prev_strobe_freq_hz = None
        # Re-init temporal smoothing - each episode starts fresh so the
        # first observation doesn't leak across resets.
        self._obs_state = ObsState(tick_hz=self.config.tick_hz)
        # Warm up: ~0.5s of brain time so the NMM transient passes
        warmup_samples = int(0.5 * self.config.sampling_rate)
        for _ in range(warmup_samples):
            self._sim.step()
        obs, _ = self._observe()
        return obs, {"subject": self._subject_info()}

    def _subject_info(self) -> dict:
        """Static parameters of the sampled subject.

        Everything a per-subject stratification or an oracle policy is
        allowed to know. ``responder`` and ``cortical_gain`` are the two
        the analysis stratifies on; ``alpha_hz`` is what the oracle
        open-loop arm is permitted to see.
        """
        s = self._subject
        if s is None:
            return {}
        pc = s.photic_config
        return {
            "alpha_hz": float(s.alpha_hz),
            "responder": s.responder_type,
            "cortical_gain": float(pc.cortical_gain),
            "resonance_width_hz": float(pc.resonance_width_hz),
            "shift_per_unit_ratio": float(pc.shift_per_unit_ratio),
            "exogenous_shift": bool(pc.exogenous_shift),
            "seed": int(s.seed),
        }

    def step(self, action: np.ndarray
             ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        assert self._sim is not None and self._supervisor is not None
        stim_action = self._decode_action(action)
        # `railed_pct` is BrainFlow's ADC-full-scale detector and is
        # meaningless on synthetic NMM output in arbitrary units. There are
        # no electrodes in simulation, so report 0 and never trigger the
        # disconnect bailout here.
        safe = self._supervisor.clamp(stim_action, railed_pct=0.0)
        self._strobe = StrobeParams(
            enabled=safe.enabled,
            frequency_hz=safe.frequency_hz,
            fill=safe.fill,
            brightness=safe.brightness,
        )
        self._sim.update_strobe(self._strobe)
        self._audio.set_sync_freq(safe.frequency_hz)
        self._audio.set_strobe_depth(safe.audio_strobe_depth_pct)
        self._audio.set_master_vol(safe.audio_master_vol_pct)

        # Realised plant state entering THIS tick's simulation. Sampled
        # before the sim advances, so it is the peak actually in force
        # while these samples are generated - not the one the tick's own
        # feature extraction will install for the next tick. Report-only:
        # never enters the observation or the reward.
        resonance_hz = float(self._sim.photic_config.effective_resonance_hz)
        shift_drive = float(self._sim.photic_config.current_drowsiness)

        n_steps = int(self.config.sampling_rate / self.config.tick_hz)
        for _ in range(n_steps):
            sample = self._sim.step()  # (n_channels,)
            self._buf[:, self._buf_pos] = sample
            self._buf_pos = (self._buf_pos + 1) % self._buf.shape[1]
            if self._buf_pos == 0:
                self._buf_filled = True
            self._t += self._sim.brain.dt

        obs, features = self._observe()
        plv_val = self._plv()
        reward = compute_reward(
            prev=self._prev_features,
            curr=features,
            plv_strobe=plv_val,
            strobe_freq_hz=safe.frequency_hz,
            prev_strobe_freq_hz=self._prev_strobe_freq_hz,
            weights=self.config.reward_weights,
        )
        self._prev_features = features
        self._prev_strobe_freq_hz = safe.frequency_hz

        truncated = self._t >= self.config.episode_s
        terminated = self._supervisor.panic_active
        info = {
            "features": features,
            "plv": plv_val,
            "plv_2f": self._plv_harmonic(2),
            "stim": safe,
            "t": self._t,
            "effective_resonance_hz": resonance_hz,
            "shift_drive": shift_drive,
            "subject": self._subject_info(),
            # None unless EnvConfig.diagnostics is set. Values are the
            # UNSMOOTHED per-tick band powers; the production θ/α on
            # ``features`` is EMA-smoothed, so the analysis applies the same
            # EMA (ObsState.ema_alpha_decay) to every diagnostic channel
            # offline rather than smoothing some channels and not others.
            "diag": features.diagnostic,
        }
        return obs, reward, terminated, truncated, info

    def _decode_action(self, action: np.ndarray) -> StimAction:
        # Map [-1,1] -> physical units
        freq = float(FREQ_MIN_HZ
                     + (action[0] + 1.0) / 2.0 * (FREQ_MAX_HZ - FREQ_MIN_HZ))
        fill = float((action[1] + 1.0) / 2.0)         # 0..1, clamped by safety
        brightness = float((action[2] + 1.0) / 2.0)   # 0..1, clamped by safety
        depth_pct = float((action[3] + 1.0) / 2.0 * 100.0)  # 0..100
        vol_pct = float((action[4] + 1.0) / 2.0 * 100.0)    # 0..100
        return StimAction(
            enabled=True,
            frequency_hz=freq,
            fill=fill,
            brightness=brightness,
            color_white=200,
            audio_strobe_depth_pct=depth_pct,
            audio_binaural_vol_pct=10.0,
            audio_master_vol_pct=vol_pct,
        )

    def _observe(self) -> Tuple[np.ndarray, FeatureSnapshot]:
        # Reconstruct linear window from circular buffer
        n_buf = self._buf.shape[1]
        if self._buf_filled:
            window = np.concatenate(
                [self._buf[:, self._buf_pos:], self._buf[:, :self._buf_pos]],
                axis=1,
            )
        else:
            window = self._buf[:, :self._buf_pos] if self._buf_pos > 0 \
                     else self._buf[:, :1]
        # Notch centre is the *commanded* strobe frequency for this tick -
        # the same quantity PLV is computed against, and the same one a real
        # deployment would know without a photodiode.
        stim_notch_hz = None
        if self.config.diagnostics and self._strobe.enabled \
                and self._strobe.brightness > 0.0:
            stim_notch_hz = float(self._strobe.frequency_hz)
        features = extract_features(
            window, self.config.sampling_rate,
            self.config.montage, t_ms=int(self._t * 1000),
            stim_notch_hz=stim_notch_hz,
        )
        # There are no electrodes in simulation, so `railed_pct` (BrainFlow's
        # ADC-full-scale detector) is zeroed at the source and every consumer
        # - safety supervisor, reward, observation - sees the same value. The
        # daemon's live path is unaffected: it builds its own FeatureSnapshot
        # from real EEG.
        features.railed_pct = 0.0
        plv_val = self._plv(window=window)

        # Temporal smoothing of the noise-sensitive features
        # White-noise measurement inflates θ-band power disproportionately
        # (θ is normally small against α), corrupting θ/α. The raw features are
        # overwritten with the smoothed values so reward, observation and
        # resonance shift all read the same signal.
        self._obs_state.update(
            raw_theta_alpha=float(features.theta_alpha_ratio),
            raw_plv=float(plv_val),
        )

        # Overwrite the raw θ/α with the smoothed value - reward and
        # resonance shift both read this through the FeatureSnapshot.
        features.theta_alpha_ratio = self._obs_state.smoothed_theta_alpha

        # Feed smoothed drowsiness into the photic config so the NMM's
        # resonance peak shifts gradually, not on every noise spike.
        # This is the line that closes the loop: the agent drives θ/α, and
        # θ/α moves the plant's resonance peak. In the exogenous-drift
        # contrast condition the sim drives the peak from a fixed time
        # course instead, so we must not overwrite it here.
        if not self._sim.photic_config.exogenous_shift:
            self._sim.photic_config.current_drowsiness = float(
                self._obs_state.smoothed_theta_alpha
            )
        feat_vec = feature_vector(features)
        rest_val = 0.0
        if self._restfulness is not None and self._buf_filled:
            # RESTFULNESS expects (channels, samples) of the active EEG channels
            try:
                eeg_idxs = self.config.montage.primary_eeg_channels or [0]
                rest_val = self._restfulness.score(
                    window, eeg_idxs, self.config.sampling_rate,
                )
            except Exception:
                rest_val = 0.0

        last = [self._strobe.frequency_hz, self._strobe.fill,
                self._strobe.brightness]
        obs_parts = [feat_vec, np.array([plv_val], dtype=np.float32)]
        if self._restfulness is not None:
            obs_parts.append(np.array([rest_val], dtype=np.float32))
        obs_parts.append(np.array(last, dtype=np.float32))
        obs_parts.append(np.array(self._obs_state.temporal_context(),
                                    dtype=np.float32))
        obs = np.concatenate(obs_parts)
        return obs.astype(np.float32), features

    def _plv(self, window: Optional[np.ndarray] = None) -> float:
        if window is None:
            n_buf = self._buf.shape[1]
            if self._buf_filled:
                window = np.concatenate(
                    [self._buf[:, self._buf_pos:],
                     self._buf[:, :self._buf_pos]],
                    axis=1,
                )
            else:
                return 0.0
        if not self._strobe.enabled or self._strobe.frequency_hz <= 0:
            return 0.0
        primary = self.config.montage.primary_eeg_channels[0]
        eeg = window[primary].copy()
        # PLV is measured against the envelope the simulator actually
        # delivered over this window. A reference regenerated from a single
        # frequency would assume a constant strobe, so a ramping policy would
        # be scored against a stimulus it never received.
        env = self._delivered_env(window.shape[1])
        if env is None:
            return 0.0
        return plv_strobe_to_eeg(
            eeg, env, self.config.sampling_rate, self._strobe.frequency_hz,
        )

    def _delivered_env(self, n: int):
        """The envelope actually delivered over the last ``n`` samples.

        Returns None when no history exists (non-simulator EEG source) or the
        window contains no light at all - in which case PLV is undefined and
        callers must report 0.0 rather than substituting a reconstructed
        constant-frequency reference, which would make a policy's attainable
        PLV depend on its own ramp rate.
        """
        if self._sim is None or not hasattr(self._sim, "delivered_envelope"):
            return None
        env = self._sim.delivered_envelope(n)
        return env if np.any(env) else None

    def _plv_harmonic(self, harmonic: int = 2,
                      window: Optional[np.ndarray] = None) -> float:
        """Diagnostic PLV at a harmonic of the strobe frequency.

        Report-only: never enters the observation vector or the reward.
        """
        if window is None:
            if self._buf_filled:
                window = np.concatenate(
                    [self._buf[:, self._buf_pos:],
                     self._buf[:, :self._buf_pos]],
                    axis=1,
                )
            else:
                return 0.0
        if not self._strobe.enabled or self._strobe.frequency_hz <= 0:
            return 0.0
        primary = self.config.montage.primary_eeg_channels[0]
        eeg = window[primary].copy()
        # As in _plv: the harmonic diagnostic must be measured against the
        # delivered envelope, not a reconstructed one.
        env = self._delivered_env(window.shape[1])
        if env is None:
            return 0.0
        return plv_strobe_harmonic(
            eeg, env, self.config.sampling_rate, self._strobe.frequency_hz,
            harmonic=harmonic,
        )
