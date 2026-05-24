"""
Audio engine

    Source preset (SuperSaw + LFOs)
       │
       ├─ * isochronic AM (Sine(target_hz) -> depth)
       ▼
    Biquad lowpass (val_cutoff)
       │
       ▼
    Chorus
       │
       + Pink/Brown noise + Stereo ping-pong tester
       │
       ▼
    Freeverb (val_reverb, damp)
       │
       + Stereo binaural (carrier 150 Hz, beat = target_hz)
       │
       * val_vol
       │
       ▼
    Output

The agent does NOT touch pyo objects - it talks to AudioController
(``app.audio.controller``), which sets the SigTo values exposed here.
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from app.audio import presets as audio_presets

logger = logging.getLogger("audio.engine")


@dataclass
class AudioSeed:
    """Persisted to ``audio_seed.json`` per session for replay determinism."""
    random_seed: int
    numpy_seed: int
    pyo_seed: Optional[int] = None
    server_start_ts: float = field(default_factory=time.time)

    def apply(self) -> None:
        random.seed(self.random_seed)
        np.random.seed(self.numpy_seed)


class AudioServer:
    """Manages the pyo Server lifecycle (boot, start, stop, shutdown)."""

    def __init__(self, device_index: Optional[int] = None,
                 record_to: Optional[str] = None,
                 nchnls: int = 2, buffersize: int = 1024):
        self.device_index = device_index
        self.record_to = record_to
        self.nchnls = nchnls
        self.buffersize = buffersize
        self.server: Optional["object"] = None

    @staticmethod
    def _pick_output_device() -> int:
        from pyo import pa_get_output_devices
        names, indexes = pa_get_output_devices()
        for preferred in ("pulse", "default"):
            for name, idx in zip(names, indexes):
                if preferred in name.lower():
                    return idx
        return -1

    def start(self) -> None:
        from pyo import Server
        if self.device_index is None:
            self.device_index = self._pick_output_device()
        srv = Server(nchnls=self.nchnls, duplex=0, buffersize=self.buffersize)
        srv.setOutputDevice(self.device_index)
        srv.boot()
        if self.record_to is not None:
            srv.recordOptions(dur=-1, filename=self.record_to)
            srv.recstart()
        srv.start()
        self.server = srv
        logger.info("AudioServer running on device %s%s",
                    self.device_index,
                    f" (rec->{self.record_to})" if self.record_to else "")

    def stop(self) -> None:
        if self.server is not None:
            try:
                if self.record_to is not None:
                    self.server.recstop()
                self.server.stop()
                self.server.shutdown()
            finally:
                self.server = None
                logger.info("AudioServer stopped")


class BiofeedbackEngine:
    """
    pyo signal graph identical to biofeedback_sound_engine.BiofeedbackEngine.

    The set_* methods are kept as the agent's audio action surface, but the
    AudioController in app/audio/controller.py is the preferred façade
    because it knows the master volume range, depth ranges, etc. and clips
    values appropriately for the safety supervisor.
    """

    def __init__(self, preset_name: str = "sunlight_pads",
                 output_gain: float = 1.0,
                 max_val_vol: float = float("inf"),
                 max_strobe_depth_pct: float = 100.0,
                 max_binaural_vol_pct: float = 100.0):
        # pyo imports deferred so importing this module doesn't require
        # pyo's compiled audio backend on hosts that won't run the engine.
        from pyo import SigTo, Sine, Biquad, PinkNoise, BrownNoise, Sig, LFO
        from pyo import Pan, Chorus, Freeverb
        self.preset_name = preset_name
        # Static makeup gain applied to every val_vol assignment so 100%
        # master volume reaches the same RMS as a peak-normalized .m4a.
        # The pyo signal graph naturally peaks at ~0.2-0.3 (conservative
        # preset RMS); without this multiplier the engine is ~14 dB
        # quieter than VLC playback at the same OS volume.
        self.output_gain = float(output_gain)
        # Hard audio-safety caps mirroring the visual SafetySupervisor's
        # bounds. Operator-set in `audio.safety.*`; defaults to "no cap"
        # for backward compat with callers that don't pass them. Every
        # setter clamps against these BEFORE writing to the pyo SigTo,
        # so the engine's signal chain physically cannot exceed them.
        self.max_val_vol = float(max_val_vol)
        self.max_strobe_depth_pct = float(max_strobe_depth_pct)
        self.max_binaural_vol_pct = float(max_binaural_vol_pct)

        self.target_hz = SigTo(value=0, time=0.5, init=0)
        self.strobe_depth = SigTo(value=0, time=0.1, init=0)
        self.binaural_vol = SigTo(value=0, time=0.2, init=0)
        self.brown_vol = SigTo(value=0, time=0.5, init=0)
        self.val_cutoff = SigTo(value=500, time=0.2, init=500)
        self.val_reverb = SigTo(value=0.8, time=0.2, init=0.8)
        # Initial val_vol respects the same max_val_vol cap the setter
        # enforces - otherwise a high output_gain at boot would briefly
        # ride above the cap before the first set_master_volume_pct call.
        initial_val_vol = min(0.5 * self.output_gain, self.max_val_vol)
        self.val_vol = SigTo(value=initial_val_vol, time=0.2,
                              init=initial_val_vol)
        self.test_beep_amp = SigTo(value=0, time=0.1, init=0)

        self.raw_music = audio_presets.build(preset_name)

        noise_lfo = Sine(freq=0.05, mul=200, add=400)
        pink_L = Biquad(PinkNoise(mul=0.03), freq=noise_lfo, q=1)
        pink_R = Biquad(PinkNoise(mul=0.03), freq=noise_lfo, q=1)
        self.stereo_pink = Sig([pink_L, pink_R])
        brown_L = Biquad(BrownNoise(mul=self.brown_vol), freq=300, q=1)
        brown_R = Biquad(BrownNoise(mul=self.brown_vol), freq=300, q=1)
        self.stereo_brown = Sig([brown_L, brown_R])

        ping_pong_clk = LFO(freq=0.25, type=2).range(-1, 1)
        test_freq = (ping_pong_clk * 200) + 600
        test_beep = Sine(freq=test_freq, mul=self.test_beep_amp)
        self.test_panned = Pan(test_beep, outs=2, pan=ping_pong_clk)

        # Isochronic + binaural sync layer ("Stochastic Conductor" in the proposal).
        pulse_wave = Sine(freq=self.target_hz, mul=0.5, add=0.5)
        final_modulator = (pulse_wave * self.strobe_depth) + (1 - self.strobe_depth)
        pulsing_music = self.raw_music * final_modulator

        carrier_freq = 150
        stereo_binaural = Sig([
            Sine(freq=carrier_freq, mul=self.binaural_vol),
            Sine(freq=carrier_freq + self.target_hz, mul=self.binaural_vol),
        ])

        filtered = Biquad(pulsing_music, freq=self.val_cutoff, q=0.5)
        chorused = Chorus(filtered, depth=0.9, feedback=0.4, bal=0.5)
        texture_mix = chorused + self.stereo_pink + self.stereo_brown + self.test_panned
        music_layer = Freeverb(texture_mix, size=self.val_reverb, damp=0.8, bal=0.5)
        self.final_out = (music_layer + stereo_binaural) * self.val_vol
        self.final_out.out()

    def set_sync_frequency(self, hz: float) -> None:
        self.target_hz.value = float(hz)

    def set_strobe_depth_pct(self, percent: float) -> None:
        pct = min(float(percent), self.max_strobe_depth_pct)
        self.strobe_depth.value = max(0.0, pct) / 100.0

    def set_binaural_volume_pct(self, percent: float) -> None:
        pct = min(float(percent), self.max_binaural_vol_pct)
        self.binaural_vol.value = (max(0.0, pct) / 100.0) * 0.05

    def set_brown_noise_volume_pct(self, percent: float) -> None:
        self.brown_vol.value = (float(percent) / 100.0) * 0.1

    def set_master_volume_pct(self, percent: float) -> None:
        raw = (float(percent) / 100.0) * self.output_gain
        self.val_vol.value = max(0.0, min(raw, self.max_val_vol))

    def set_warmth_cutoff_pct(self, percent: float) -> None:
        val = float(percent)
        self.val_cutoff.value = 100 + (val * val * 0.15)

    def set_reverb_space_pct(self, percent: float) -> None:
        self.val_reverb.value = min(float(percent) / 100.0, 0.95)

    def set_test_mode(self, enabled: bool) -> None:
        self.test_beep_amp.value = 0.2 if enabled else 0.0
