"""
BioListenerSleepConfig - every tunable value in the project, schema-validated.

Field defaults preserve current behavior exactly (values migrated from
TrainConfig, EnvConfig, RewardWeights, RandomizationRanges, IngestConfig,
AudioParams, PhoticInputConfig, and the constants in app/agent/safety.py).
Existing dataclasses keep their names so call sites don't all rewrite at
once - they construct themselves from the matching sub-section of
BioListenerSleepConfig via the ``.from_config(...)`` helpers added in app/config/facades.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class PersistenceConfig(_Section):
    mongo_url: str = Field(
        default="mongodb://localhost:27017/",
        description="MongoDB connection string for sessions/decisions/safety events.",
    )
    mongo_db: str = Field(
        default="biolistener_sleep_dev",
        description="Database name. Use one per study to keep cohorts isolated.",
    )
    studies_root: Path = Field(
        default=Path("studies"),
        description="Filesystem root for per-session folders (.fif / .parquet).",
    )
    experimental_data_root: Path = Field(
        default=Path("experimental_data"),
        description="Where BrainFlow add_streamer dumps raw EEG CSVs.",
    )


class EegConfig(_Section):
    kind: str = Field(
        default="sim",
        description="Source: 'sim' (NMM via TCP), 'biolistener' (real ESP32), or 'playback' (CSV).",
        pattern="^(sim|biolistener|playback)$",
    )
    ip_address: str = Field(default="0.0.0.0",
                             description="Local interface BoardShim binds for inbound device.")
    ip_port: int = Field(default=12345, ge=1, le=65535,
                          description="TCP port the BIOLISTENER device connects to.")
    timeout: int = Field(default=15, ge=1,
                          description="BoardShim prepare_session timeout (s).")
    sampling_rate: int = Field(default=500, ge=64, le=2000)
    pga_gain: int = Field(default=128, ge=1, le=128,
                           description="Per-channel PGA gain via config_board cmd 3.")
    active_channels: List[int] = Field(
        default_factory=lambda: list(range(8)),
        description=(
            "Which of the 8 BIOLISTENER channels are physically connected "
            "to electrodes. Channels NOT in this list are disabled at the "
            "device with {'command':2,'channel':N,'enable':0} so they don't "
            "stream noise / rail. The montage's primary/EOG/EMG channels "
            "must all be within this list - validated at session start."
        ),
    )
    montage: str = Field(default="occipital_alpha",
                          description="Channel layout: 'occipital_alpha' or 'fpz_mastoid'.")
    playback_file: Optional[Path] = Field(default=None,
                                            description="Required iff kind='playback'.")
    record_to_csv: bool = Field(default=True,
                                 description="Auto-add file streamer for raw EEG capture.")


class MaskStrobeConfig(_Section):
    """Per-session presentation params the policy does NOT control: the LED
    colour, white-channel intensity, and which of the 8 LEDs are enabled.

    The RL policy only controls frequency / fill (duty) / brightness — these
    three knobs decide *how* the strobe pulses. Colour, white channel, and
    bitmask decide *what the pulse looks like* and *which LEDs fire* — those
    are operator-set per study and should be reproducible across sessions.
    """
    color_hex: str = Field(
        default="#000000",
        pattern=r"^#?[0-9a-fA-F]{6}$",
        description=(
            "RGB strobe colour as a hex string (e.g. '#FF0000' = red, "
            "'#000000' = no RGB / use white channel only). Note: the W "
            "channel is a separate physical LED on the mask and is NOT "
            "derived from this RGB — set `white_channel` for that."
        ),
    )
    white_channel: int = Field(
        default=200, ge=0, le=255,
        description=(
            "Brightness of the dedicated white LED in each RGBW pixel "
            "(0-255). Independent from RGB. v1 real sessions strobe with "
            "white only by setting color_hex='#000000' + white_channel=200."
        ),
    )
    leds: List[int] = Field(
        default_factory=lambda: [1] * 8,
        description=(
            "Which of the 8 LED rings fire. List of 8 ints (0 = off, 1 = on), "
            "ordered LED 0 → LED 7. Examples:\n"
            "  [1,1,1,1,1,1,1,1]  - all 8 on (default)\n"
            "  [0,0,1,1,1,1,0,0]  - central 4 only\n"
            "  [1,0,1,0,1,0,1,0]  - alternating\n"
            "Legacy int form (0xFF / 255 / 0b11111111) is still accepted."
        ),
    )

    @field_validator("leds", mode="before")
    @classmethod
    def _normalize_leds(cls, v):
        """Accept three input shapes:
        (1) list[int] of length 8 with 0/1 - the new canonical form.
        (2) int 0..255 - legacy 8-bit bitmask, LSB = LED 0.
        (3) str of 8 chars '0'/'1' - convenience for YAML one-liners.
        All collapse to the canonical list[int]."""
        if isinstance(v, int):
            if not 0 <= v <= 0xFF:
                raise ValueError(f"led bitmask int {v!r} not in 0..255")
            return [(v >> i) & 1 for i in range(8)]
        if isinstance(v, str):
            if len(v) != 8 or any(c not in "01" for c in v):
                raise ValueError(
                    f"led string {v!r} must be 8 chars of '0'/'1' "
                    "(LSB first, e.g. '11111111' for all on)"
                )
            return [int(c) for c in v]
        if isinstance(v, list):
            if len(v) != 8 or any(int(x) not in (0, 1) for x in v):
                raise ValueError(
                    f"led list {v!r} must be 8 ints, each 0 or 1"
                )
            return [int(x) for x in v]
        raise TypeError(f"unsupported led spec: {type(v).__name__}")

    @property
    def led_bitmask(self) -> int:
        """Convert the canonical list form to the wire-format 8-bit
        bitmask the mask client sends. LED i contributes bit i (LSB first)."""
        return sum(bool(b) << i for i, b in enumerate(self.leds))


class MaskConfig(_Section):
    kind: str = Field(default="virtual",
                       description="'virtual' (local UDP server) or 'esp32' (real device).",
                       pattern="^(virtual|esp32)$")
    ip_address: str = Field(default="127.0.0.1",
                             description="Target IP (real ESP32 on LAN, or loopback for virtual).")
    port: int = Field(default=20000, ge=1, le=65535,
                       description="UDP/JSON port (matches ESP32 firmware default).")
    strobe: MaskStrobeConfig = Field(default_factory=MaskStrobeConfig)


class AudioSafetyConfig(_Section):
    """Hard upper bounds on the BiofeedbackEngine that cannot be exceeded
    regardless of shaper output, agent action, or operator UI input.

    Mirrors the visual ``safety.*`` block: every cap here is enforced
    inside ``BiofeedbackEngine.set_*`` methods so the engine's pyo signal
    chain physically cannot rise above the configured limits. Operator-
    set per study, independent of the perceptual shaper.
    """
    max_val_vol: float = Field(
        default=1.5, ge=0.0, le=10.0,
        description=(
            "Absolute cap on pyo's val_vol (= percent/100 * output_gain). "
            "1.5 keeps the final amplitude well under pyo's 1.0 clip ceiling "
            "for typical presets. Raise only if you've verified your preset "
            "+ hardware combo doesn't clip at the new ceiling."
        ),
    )
    max_strobe_depth_pct: float = Field(
        default=50.0, ge=0.0, le=100.0,
        description=(
            "Max isochronic AM depth the agent's action[3] can request. "
            "100%% = full-amplitude tremolo (silence between pulses), "
            "subjectively intense. 50%% gives clearly perceptible "
            "modulation without being harsh."
        ),
    )
    max_binaural_vol_pct: float = Field(
        default=20.0, ge=0.0, le=100.0,
        description=(
            "Max binaural-beat volume the engine will emit. Currently "
            "hardcoded to 10%% in the agent decoder; this cap protects "
            "against accidental overrides or future agent-controlled paths."
        ),
    )


class AudioShaperConfig(_Section):
    """Deterministic perceptual shaper - replaces the agent's master-volume
    action and the static warmth knob. Mirrors ``app.audio.shaper.AudioShaper``.
    """
    theta_alpha_baseline: float = Field(default=0.5, ge=0.0,
        description="θ/α at or below this: 'alert', no shaping")
    theta_alpha_full: float = Field(default=3.0, ge=0.1,
        description="θ/α at or above this: 'fully drowsy', shaping at endpoint")
    warmth_at_baseline_pct: float = Field(default=80.0, ge=0.0, le=100.0)
    warmth_at_full_pct: float = Field(default=25.0, ge=0.0, le=100.0)
    initial_master_vol_pct: float = Field(default=50.0, ge=0.0, le=100.0)
    drowsy_vol_taper: float = Field(default=0.4, ge=0.0, le=1.0,
        description="Fraction of initial volume removed at full drowsiness")
    end_fade_starts_at_frac: float = Field(default=0.75, ge=0.0, le=1.0)
    end_fade_min_factor: float = Field(default=0.4, ge=0.0, le=1.0,
        description="Volume multiplier at the very end of the session")


class AudioConfig(_Section):
    enabled: Optional[bool] = Field(
        default=None,
        description=(
            "True/False to force; None => auto-on for 'real' sessions, off for sim/training. "
            "Overridden by BIOLISTENER_SLEEP_ENABLE_AUDIO env var."
        ),
    )
    preset: str = Field(default="sunlight_pads",
                         description="Audio preset: 'sunlight_pads' | 'warm_blanket' | 'lunar_bowls'.")
    device_index: Optional[int] = Field(default=None,
                                          description="pyo output device index; None => auto-pick pulse/default.")
    record_to_wav: bool = Field(default=False,
                                  description="If True, dump audio.wav per session (large).")
    initial_master_vol_pct: float = Field(default=50.0, ge=0.0, le=100.0)
    initial_binaural_vol_pct: float = Field(default=10.0, ge=0.0, le=100.0)
    initial_brown_noise_pct: float = Field(default=80.0, ge=0.0, le=100.0)
    initial_warmth_pct: float = Field(default=80.0, ge=0.0, le=100.0)
    initial_reverb_pct: float = Field(default=80.0, ge=0.0, le=100.0)
    initial_strobe_depth_pct: float = Field(default=10.0, ge=0.0, le=100.0)
    output_gain: float = Field(
        default=3.0, ge=0.0, le=10.0,
        description=(
            "Linear makeup gain applied to the BiofeedbackEngine's final "
            "stereo output AFTER master volume. The pyo signal graph's "
            "natural peak amplitude is ~0.2-0.3 (preset RMS is "
            "conservative), so 100%% master volume is ~14 dB quieter than "
            "a peak-normalized .m4a played via VLC. 3.0 brings them "
            "roughly to parity; raise to 4-5 if still too quiet on your "
            "hardware, but watch for clipping above ~5."
        ),
    )
    shaper: AudioShaperConfig = Field(default_factory=AudioShaperConfig)
    safety: AudioSafetyConfig = Field(default_factory=AudioSafetyConfig)


class SafetyConfig(_Section):
    freq_min_hz: float = Field(default=3.0, ge=0.5)
    freq_max_hz: float = Field(default=14.0, ge=1.0, le=30.0)
    pse_band_lo: float = Field(default=15.0,
                                description="Lower bound of photosensitive-epilepsy high-risk band.")
    pse_band_hi: float = Field(default=25.0)
    brightness_min: float = Field(default=0.0, ge=0.0, le=1.0)
    brightness_max: float = Field(default=0.60, ge=0.0, le=1.0,
                                    description="v1 conservative cap (raise after self-testing).")
    fill_min: float = Field(default=0.0, ge=0.0, le=1.0)
    fill_max: float = Field(default=0.40, ge=0.0, le=1.0)
    session_timeout_s: float = Field(default=45 * 60, ge=10.0,
                                       description="Hard session timeout in seconds.")
    max_freq_step_hz_per_s: float = Field(default=4.0, ge=0.1)
    max_brightness_step_per_s: float = Field(default=0.20, ge=0.01,
                                                description="20% per second default.")
    railed_pct_disconnect_threshold: float = Field(default=80.0, ge=0.0, le=100.0)


class SessionConfig(_Section):
    subject_code: str = Field(default="UNSET",
                               description="Subject identifier; required for real-hw sessions.")
    audio_enabled: Optional[bool] = Field(
        default=None,
        description=(
            "Per-session audio toggle. True/False forces; None -> daemon "
            "auto-derives from eeg.kind (biolistener -> on, sim/playback -> off)."
        ),
    )
    protocol: str = Field(
        default="rl_agent",
        pattern="^(rl_agent|baseline_no_stim|open_loop_track)$",
        description=(
            "Experimental protocol = policy dimension. "
            "'rl_agent' (closed-loop SAC), 'baseline_no_stim' (no stimulation, EEG-only), "
            "or 'open_loop_track' (pre-recorded scripted strobe+audio)."
        ),
    )
    duration_s: float = Field(default=60.0, ge=1.0)
    notes: str = Field(default="")
    protocol_params: Dict[str, object] = Field(
        default_factory=dict,
        description=(
            "Free-form per-protocol parameters forwarded to the daemon's "
            "create_session payload. For open_loop_track, expected keys: "
            "strobe_config_path, library, preset, brightness."
        ),
    )


class PolicyConfig(_Section):
    name: str = Field(
        default="fixed_freq",
        pattern="^(sac|dqn|fixed_freq|linear_ramp)$",
        description=(
            "RL/baseline policy when SessionConfig.protocol == 'rl_agent'. "
            "(open_loop_track and baseline_no_stim ignore this section.)"
        ),
    )
    checkpoint_path: Optional[Path] = Field(default=None,
                                              description="Path to SAC .zip; None => use FixedFrequencyPolicy fallback.")
    # Fixed-frequency baseline defaults
    fixed_freq_hz: float = Field(default=10.0, ge=1.0, le=30.0)
    fixed_fill: float = Field(default=0.20, ge=0.0, le=1.0)
    fixed_brightness: float = Field(default=0.30, ge=0.0, le=1.0)
    # Linear ramp baseline defaults
    ramp_start_hz: float = Field(default=11.0)
    ramp_end_hz: float = Field(default=4.0)


class OpenLoopTrackConfig(_Section):
    """
    Pre-recorded open-loop strobe+audio protocol - neutrally named.

    The track data (catalog YAML, audio files, strobe-event CSVs) lives in
    an external folder supplied by the user via ``data_root``. The
    integration code in ``app/open_loop_track/`` is vendored neutrally and
    references no third-party product anywhere.
    """
    data_root: Optional[Path] = Field(
        default=Path("data/external_tracks/source"),
        description=(
            "Path to the external open-loop-track replication folder "
            "(catalog YAML + audio + strobe CSVs). The default expects a "
            "per-workstation symlink at data/external_tracks/source - see "
            "data/external_tracks/README.md for setup. Set to None to disable "
            "the open_loop_track protocol."
        ),
    )
    catalog_filename: str = Field(default="sessions.yaml",
                                    description="Catalog YAML inside data_root.")
    audio_subdir: str = Field(default="sounds",
                                description="Audio subdirectory inside data_root (folder name set by external project).")
    strobe_subdir: str = Field(default="strobe",
                                 description="Strobe-CSV subdirectory inside data_root (folder name set by external project).")
    default_activity: Optional[str] = Field(default=None,
                                              description="Activity slug to use when SessionConfig doesn't specify one.")
    default_duration_min: int = Field(default=10, ge=1, le=60)


class EnvLoopConfig(_Section):
    tick_hz: float = Field(default=4.0, ge=0.5, le=20.0,
                            description="Control-loop frequency.")
    window_s: float = Field(default=4.0, ge=1.0,
                             description="EEG rolling-window length for feature extraction.")
    episode_s: float = Field(default=60.0, ge=1.0,
                              description="Episode length during RL training.")
    sampling_rate: int = Field(default=500, ge=64, le=2000,
                                description="Synthetic-brain sampling rate (matches EegConfig).")
    include_restfulness: bool = Field(default=True,
                                        description="Include BrainFlow MLModel RESTFULNESS in observation.")


class RewardWeightsConfig(_Section):
    """Per-coefficient reward-shaping knobs.

    Either set ``variant`` to a name from
    :mod:`app.agent.reward_variants.REWARD_VARIANTS` (the recommended
    path for the reward-ablation experiment), OR override the individual
    ``k_*`` fields directly. If both are present, ``variant`` wins -
    individual fields are silently ignored to keep configs unambiguous.
    """
    variant: Optional[str] = Field(
        default=None,
        description=(
            "Named preset from app.agent.reward_variants.REWARD_VARIANTS "
            "(e.g. 'naive_alpha_max', 'naive_plv_only', "
            "'theta_alpha_delta', 'full_shaped'). If set, k_* fields below "
            "are ignored."
        ),
    )
    k_theta_alpha:       float = Field(default=1.0)
    k_theta_alpha_level: float = Field(default=0.2)
    k_hypnagogic:        float = Field(default=0.5)
    k_plv:               float = Field(default=0.4)
    k_descent:           float = Field(default=0.05)
    k_disconnect:        float = Field(default=-10.0)
    k_alpha_pwr:         float = Field(default=0.0,
        description="Raw α-power coefficient - used by 'naive_alpha_max' baseline.")
    flat_plv:            bool  = Field(default=False,
        description="Skip band-weighting on PLV - used by 'naive_plv_only' baseline.")


class TrainingConfig(_Section):
    total_steps: int = Field(default=20_000, ge=1)
    episode_s: float = Field(default=30.0, ge=1.0)
    learning_rate: float = Field(default=3e-4, gt=0)
    batch_size: int = Field(default=256, ge=1)
    buffer_size: int = Field(default=100_000, ge=100)
    gamma: float = Field(default=0.99, gt=0, le=1.0)
    tau: float = Field(default=0.005, gt=0, le=1.0)
    ent_coef: str = Field(default="auto")
    seed: int = Field(default=42)
    checkpoint_dir: Path = Field(default=Path("app/agent/checkpoints"))
    n_envs: int = Field(
        default=1, ge=1, le=64,
        description=(
            "Parallel env collectors during training. 1 -> DummyVecEnv "
            "(in-process, easy to debug). >1 -> SubprocVecEnv with one "
            "process per env on its own CPU core; clamped to os.cpu_count() "
            "at runtime. Per-env seed = training.seed + rank so subprocesses "
            "don't all sample the same NMM subject."
        ),
    )


class RandomizationRangesConfig(_Section):
    alpha_hz_range: Tuple[float, float] = Field(default=(8.5, 11.5))
    cortical_gain_range: Tuple[float, float] = Field(default=(100.0, 700.0))
    resonance_width_range: Tuple[float, float] = Field(default=(2.0, 4.5))
    p_mean_range: Tuple[float, float] = Field(default=(200.0, 240.0))
    p_std_range: Tuple[float, float] = Field(default=(15.0, 30.0))
    non_responder_prob: float = Field(default=0.10, ge=0.0, le=1.0)


class PhoticConfig(_Section):
    eyelid_transmission: float = Field(
        default=0.10,
        gt=0.0,
        le=1.0,
        description=(
            "Nominal broadband fraction of mask light reaching the retina "
            "through a closed eyelid. Degenerate with cortical_gain, which "
            "domain randomization already sweeps."
        ),
    )
    cortical_gain_default: float = Field(default=400.0)
    base_resonance_hz_default: float = Field(default=10.0)
    resonance_width_hz_default: float = Field(default=4.0)


class BrainSimConfig(_Section):
    n_channels: int = Field(default=8, ge=1, le=64)
    alpha_hz_default: float = Field(default=10.0, ge=4.0, le=20.0)
    channel_jitter: float = Field(default=0.05, ge=0.0, le=0.5)
    seed: int = Field(default=0)


class MlflowConfig(_Section):
    tracking_uri: Optional[str] = Field(default=None,
                                          description="None => MLflow default (./mlruns).")
    experiment_name: str = Field(default="sleep_onset_sac")


class DaemonConfig(_Section):
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8000, ge=1, le=65535)
    static_dir: Path = Field(default=Path("app/runtime/static"),
                              description="Vite build output served at /.")


class BioListenerSleepConfig(_Section):
    """Root configuration - every other tunable hangs off this."""

    persistence: PersistenceConfig = Field(default_factory=PersistenceConfig)
    eeg: EegConfig = Field(default_factory=EegConfig)
    mask: MaskConfig = Field(default_factory=MaskConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    open_loop_track: OpenLoopTrackConfig = Field(default_factory=OpenLoopTrackConfig)
    env: EnvLoopConfig = Field(default_factory=EnvLoopConfig)
    reward: RewardWeightsConfig = Field(default_factory=RewardWeightsConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    randomization: RandomizationRangesConfig = Field(default_factory=RandomizationRangesConfig)
    photic: PhoticConfig = Field(default_factory=PhoticConfig)
    brain_sim: BrainSimConfig = Field(default_factory=BrainSimConfig)
    mlflow: MlflowConfig = Field(default_factory=MlflowConfig)
    daemon: DaemonConfig = Field(default_factory=DaemonConfig)
