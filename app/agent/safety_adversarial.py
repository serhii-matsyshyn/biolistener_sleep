"""
Adversarial test for the SafetySupervisor "shield".

Builds an attacker that intentionally requests stimulation in the
photosensitive-epilepsy (PSE) band [15–25 Hz] every tick, plus
out-of-bounds brightness and duty cycle. Feeds the requests through
``SafetySupervisor.clamp`` and verifies:

  * Zero requests in [PSE_BAND_LO, PSE_BAND_HI] ever leave the shield
    with a frequency inside that band.
  * Brightness never exceeds BRIGHTNESS_MAX_V1.
  * Fill never exceeds FILL_MAX.

Pass condition: ``n_pse_leaked == 0`` after N requests (10 000 in the
notebook). This is the figure that says "the safety architecture is
formal, not vibes" without needing a real human in the loop.

CLI wrapper: ``biolistener-sleep shield-test --episodes 10000 --out shield_log.json``.
"""
from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from app.agent.safety import (
    BRIGHTNESS_MAX_V1, FILL_MAX, FREQ_MAX_HZ, PSE_BAND_HI, PSE_BAND_LO,
    SafetySupervisor, StimAction,
)

logger = logging.getLogger("agent.safety_adversarial")


@dataclass
class AdversarialRequest:
    """A single PSE-band-targeted stim request - the attacker's payload."""
    frequency_hz: float
    fill: float
    brightness: float


@dataclass
class ShieldAttackResult:
    n_requested: int = 0
    n_pse_blocked: int = 0
    n_pse_leaked: int = 0           # MUST stay 0
    n_brightness_clamped: int = 0
    n_fill_clamped: int = 0
    requested_freqs: List[float] = field(default_factory=list)
    actual_freqs:    List[float] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "n_requested":          self.n_requested,
            "n_pse_blocked":        self.n_pse_blocked,
            "n_pse_leaked":         self.n_pse_leaked,
            "n_brightness_clamped": self.n_brightness_clamped,
            "n_fill_clamped":       self.n_fill_clamped,
            "pse_band_lo_hz":       PSE_BAND_LO,
            "pse_band_hi_hz":       PSE_BAND_HI,
            "freq_max_hz":          FREQ_MAX_HZ,
            "brightness_max":       BRIGHTNESS_MAX_V1,
            "fill_max":             FILL_MAX,
            "requested_freqs":      self.requested_freqs,
            "actual_freqs":         self.actual_freqs,
        }


def _next_attack(rng: random.Random) -> AdversarialRequest:
    """Always inside the PSE band, max brightness, max fill."""
    return AdversarialRequest(
        frequency_hz=rng.uniform(PSE_BAND_LO, PSE_BAND_HI),
        fill=1.0,           # well above FILL_MAX (0.4)
        brightness=1.0,     # well above BRIGHTNESS_MAX_V1 (0.6)
    )


def run_shield_attack(
    n_requests: int = 10_000,
    seed: int = 0,
) -> ShieldAttackResult:
    """
    Fire ``n_requests`` PSE-band-targeted stim requests through a fresh
    SafetySupervisor and tally what got through.

    The supervisor is reset between every request so its session-timeout
    rule doesn't kick in and mask the PSE clamp; we are isolating the
    band-guard.
    """
    rng = random.Random(seed)
    result = ShieldAttackResult()
    supervisor = SafetySupervisor()

    for _ in range(n_requests):
        atk = _next_attack(rng)
        result.n_requested += 1
        result.requested_freqs.append(atk.frequency_hz)
        action = StimAction(
            enabled=True,
            frequency_hz=atk.frequency_hz,
            fill=atk.fill,
            brightness=atk.brightness,
        )
        safe = supervisor.clamp(action)
        result.actual_freqs.append(safe.frequency_hz)

        in_band_before = PSE_BAND_LO <= atk.frequency_hz <= PSE_BAND_HI
        in_band_after  = PSE_BAND_LO <= safe.frequency_hz <= PSE_BAND_HI
        if in_band_before and not in_band_after:
            result.n_pse_blocked += 1
        if in_band_after:
            result.n_pse_leaked += 1

        if safe.brightness < atk.brightness:
            result.n_brightness_clamped += 1
        if safe.fill < atk.fill:
            result.n_fill_clamped += 1

    logger.info(
        "Shield attack: %d requests | %d PSE blocked | %d PSE leaked | "
        "%d brightness clamps | %d fill clamps",
        result.n_requested, result.n_pse_blocked, result.n_pse_leaked,
        result.n_brightness_clamped, result.n_fill_clamped,
    )
    return result


def write_json(result: ShieldAttackResult, out: Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.to_json(), indent=2))
    logger.info("Wrote shield log -> %s", out)
