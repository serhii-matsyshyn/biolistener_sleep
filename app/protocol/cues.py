"""
Spoken cues.

The subject is motionless with eyes closed under the mask, so audio is the only
channel. The phrase set is small and fixed, so the WAVs are **rendered once and
committed** under ``data/cues/en/``: running a session needs no speech engine,
no system packages, and no network.

Playback goes through ``pygame.mixer``, the same stack as
``app.open_loop_track.audio_player`` - SDL is bundled in the wheel, so it behaves
the same on Linux, macOS and Windows. Cues use a dedicated ``Sound`` channel
rather than ``mixer.music``, which the open-loop track already owns, so the two
cannot evict each other.

To regenerate the files (only needed if a phrase changes) see
``tools/render_cues.py``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger("protocol.cues")

CUE_ROOT = Path("data/cues")
DEFAULT_LANG = "en"
LANGS = ("en",)

# Start values for the silent serial-subtraction task; one is picked per session
# so the arithmetic is not memorised across runs.
REALERT_STARTS: Tuple[int, ...] = (300, 400, 500, 700, 800)

CUES: Dict[str, str] = {
    "lie_down": (
        "Lie down and get comfortable. Adjust the mask now until it sits evenly "
        "and blocks the light. Once it is right, leave it alone - it must not "
        "move again until the session ends."),
    "settle": "Settling. Lie still with your eyes closed.",
    "eyes_open": "Open your eyes.",
    "eyes_closed": "Close your eyes.",
    "alpha_rest": ("Keep your eyes closed and rest. Let your mind wander. Do not try to think about anything."),
    "alpha_task": ("Keep your eyes closed and stay still. Starting from four hundred, subtract seven, again and again, silently."),
    "strobe_start": "Stimulation starting. Stay still.",
    "sweep_start": "Sweep starting. Stay completely still.",
    "descent_start": "Next block starting. Stay still and let yourself drift.",
    "realert_answer": "Say your final number out loud.",
    # The full Karolinska anchors the first time, so the scale means something;
    # a short reminder afterwards, since repeating ten seconds of anchors at
    # every boundary would itself keep waking the subject up.
    "kss_prompt_anchored": (
        "Rate your sleepiness now. One, extremely alert. Three, alert. "
        "Five, neither alert nor sleepy. Seven, sleepy but not fighting it. "
        "Nine, very sleepy, fighting sleep. Say your number."),
    "kss_prompt": "Rate your sleepiness. One alert, five neither, nine fighting sleep.",
    "session_end": "Session complete. You can move now.",
    "abort": "Session stopped.",
    "gate_failed": "Quality check failed. Session stopped.",
}

for _n in REALERT_STARTS:
    CUES[f"realert_start_{_n}"] = (
        f"Open your eyes but keep completely still. Starting from {_n}, "
        "subtract seven, again and again, silently.")


def realert_cue_id(seed: int) -> str:
    return f"realert_start_{REALERT_STARTS[seed % len(REALERT_STARTS)]}"


@dataclass
class CuePlayer:
    """
    Fire-and-forget playback on a dedicated mixer channel.

    Never blocks the caller and never raises: a cue that is missing or a host
    with no audio device must not stall the control loop or shift block timing.
    """

    lang: str = DEFAULT_LANG
    root: Path = CUE_ROOT
    enabled: bool = True
    volume: float = 0.9
    _channel: object = field(default=None, init=False, repr=False)
    _cache: Dict[str, object] = field(default_factory=dict, init=False, repr=False)
    _ready: Optional[bool] = field(default=None, init=False, repr=False)
    missing: set = field(default_factory=set, init=False, repr=False)

    # ---- files -----------------------------------------------------------

    def path_for(self, cue_id: str) -> Path:
        return Path(self.root) / self.lang / f"{cue_id}.wav"

    def available(self, cue_id: str) -> bool:
        return self.path_for(cue_id).exists()

    def missing_cues(self) -> Tuple[str, ...]:
        return tuple(sorted(c for c in CUES if not self.available(c)))

    # ---- mixer -----------------------------------------------------------

    def _ensure_mixer(self) -> bool:
        if self._ready is not None:
            return self._ready
        try:
            import pygame
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=22050, size=-16, channels=1,
                                  buffer=512)
            # A reserved channel keeps cues off mixer.music, which the
            # open-loop track player uses.
            pygame.mixer.set_reserved(1)
            self._channel = pygame.mixer.Channel(0)
            self._ready = True
            logger.info("cue mixer ready")
        except Exception:
            logger.warning("cue playback unavailable - session will run silent",
                           exc_info=True)
            self._ready = False
        return self._ready

    def _sound(self, cue_id: str):
        if cue_id in self._cache:
            return self._cache[cue_id]
        import pygame
        snd = pygame.mixer.Sound(str(self.path_for(cue_id)))
        snd.set_volume(self.volume)
        self._cache[cue_id] = snd
        return snd

    def preload(self) -> int:
        """Load every available cue up front, so no session tick pays for I/O."""
        if not self.enabled or not self._ensure_mixer():
            return 0
        n = 0
        for cue_id in CUES:
            if self.available(cue_id):
                try:
                    self._sound(cue_id)
                    n += 1
                except Exception:
                    logger.warning("could not load cue %r", cue_id, exc_info=True)
        return n

    # ---- playback --------------------------------------------------------

    def play(self, cue_id: str) -> bool:
        if not self.enabled:
            return False
        if not self.available(cue_id):
            if cue_id not in self.missing:
                self.missing.add(cue_id)
                logger.warning("cue %r not present at %s", cue_id,
                               self.path_for(cue_id))
            return False
        if not self._ensure_mixer():
            return False
        try:
            self._channel.play(self._sound(cue_id))
        except Exception:
            logger.warning("failed to play cue %r", cue_id, exc_info=True)
            return False
        return True

    def duration(self, cue_id: str) -> float:
        """
        Seconds the cue takes to speak, 0 if unknown.

        The caller needs this to wait for a prompt to finish before opening a
        response window - the anchored rating prompt is nearly twelve seconds,
        and a fixed wait would close the window mid-sentence.
        """
        if not self.available(cue_id) or not self._ensure_mixer():
            return 0.0
        try:
            return float(self._sound(cue_id).get_length())
        except Exception:
            logger.debug("cue length unavailable for %r", cue_id, exc_info=True)
            return 0.0

    def stop(self) -> None:
        try:
            if self._channel is not None:
                self._channel.stop()
        except Exception:
            logger.debug("cue stop failed", exc_info=True)
