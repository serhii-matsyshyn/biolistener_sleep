"""
Audio source presets - extracted verbatim from biofeedback_sound_engine.py
to keep the validated music-source design intact while exposing it as a
clean dictionary the agent can switch between.

The frequency sets, LFOs, and SuperSaw detune values are unchanged from
the original.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict

logger = logging.getLogger("audio.presets")


def _sunlight_pads() -> "object":
    """Bright SuperSaw Sunlight pads (C Major 9)."""
    # pyo imports are deferred so this module can be imported in environments
    # without an audio device (e.g. CI, training nodes).
    from pyo import Sine, SuperSaw, Sig
    freqs = [130.81, 196.00, 246.94, 293.66, 329.63]
    lfo_L = Sine(freq=[0.10, 0.13, 0.09, 0.11, 0.14], mul=0.1, add=0.15)
    lfo_R = Sine(freq=[0.11, 0.12, 0.08, 0.10, 0.15], mul=0.1, add=0.15)
    saw_L = SuperSaw(freq=[f * 0.995 for f in freqs], detune=0.4, mul=lfo_L).mix(1)
    saw_R = SuperSaw(freq=[f * 1.005 for f in freqs], detune=0.4, mul=lfo_R).mix(1)
    return Sig([saw_L, saw_R])


def _warm_blanket() -> "object":
    """Deep Comfort (G Suspended 2)."""
    from pyo import Sine, SuperSaw, Sig
    freqs = [98.00, 146.83, 196.00, 220.00, 293.66]
    lfo_L = Sine(freq=[0.05, 0.06, 0.04, 0.07, 0.05], mul=0.1, add=0.15)
    lfo_R = Sine(freq=[0.06, 0.05, 0.08, 0.04, 0.06], mul=0.1, add=0.15)
    saw_L = SuperSaw(freq=[f * 0.997 for f in freqs], detune=0.25, mul=lfo_L).mix(1)
    saw_R = SuperSaw(freq=[f * 1.003 for f in freqs], detune=0.25, mul=lfo_R).mix(1)
    return Sig([saw_L, saw_R])


def _lunar_bowls() -> "object":
    """Ethereal Spa (F Major Pentatonic)."""
    from pyo import Sine, SuperSaw, Sig
    freqs = [174.61, 196.00, 220.00, 261.63, 293.66]
    lfo_L = Sine(freq=[0.15, 0.18, 0.14, 0.20, 0.16], mul=0.08, add=0.12)
    lfo_R = Sine(freq=[0.16, 0.17, 0.15, 0.19, 0.18], mul=0.08, add=0.12)
    saw_L = SuperSaw(freq=[f * 0.999 for f in freqs], detune=0.1, mul=lfo_L).mix(1)
    saw_R = SuperSaw(freq=[f * 1.001 for f in freqs], detune=0.1, mul=lfo_R).mix(1)
    return Sig([saw_L, saw_R])


PRESETS: Dict[str, Callable[[], "object"]] = {
    "sunlight_pads": _sunlight_pads,
    "warm_blanket": _warm_blanket,
    "lunar_bowls": _lunar_bowls,
}


def build(preset_name: str) -> "object":
    if preset_name not in PRESETS:
        from pyo import Sig
        logger.warning("Unknown preset %r - defaulting to silence", preset_name)
        return Sig(0)
    return PRESETS[preset_name]()
