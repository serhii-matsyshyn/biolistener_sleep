"""
Capturing the subject's spoken ratings.

A solo subject cannot type and cannot see a screen, so ratings are spoken. The
naive version - record the whole session on a phone and find the moments where
you spoke - makes the transcription harder than the experiment. Two things fix
that:

**Short clips, not one long file.** The runner already knows exactly when it
asked, so it records a few seconds around each prompt. Six clips of six seconds
each are reviewed in under a minute, and there is nothing to search through.

**A keypress path that skips transcription entirely.** If a keyboard is within
reach, pressing a digit records the rating directly and the clip is only a
backup. Nothing waits for either: a missed rating is logged as missing.
"""
from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("protocol.response")

CLIP_SECONDS = 6.0
SAMPLE_RATE = 16000          # speech only; keeps six clips well under a megabyte

# Seconds each candidate microphone is listened to when choosing one, and the
# peak sample value below which it is treated as carrying nothing. A jack with
# no plug in it still enumerates, still opens, and still returns frames - just
# frames of nothing - so a device has to be judged on what arrives, not on
# whether SDL will hand it over.
PROBE_S = 0.6
SILENT_PEAK = 32                       # ~0.1 % of full scale


@dataclass
class ClipRecorder:
    """
    Records one short clip per prompt into ``<session>/kss_clips/``.

    Capture goes through SDL, which pygame already bundles in its wheel and
    which the cue player already uses for playback, so recording needs no
    system package, no external binary and no separate process, and behaves
    the same on Linux, macOS and Windows.

    SDL fills the buffer on its own thread and the WAV is written when the
    clip's time is up, so the control loop never waits on the microphone. A
    host with no input device records nothing rather than failing the session.
    """

    session_dir: Path
    seconds: float = CLIP_SECONDS
    enabled: bool = True
    _threads: List[threading.Thread] = field(default_factory=list, init=False)
    _warned: bool = field(default=False, init=False)

    @property
    def clip_dir(self) -> Path:
        return Path(self.session_dir) / "kss_clips"

    _chosen_device: Optional[str] = field(default=None, init=False, repr=False)

    @staticmethod
    def _probe(name: str) -> int:
        """Peak sample value arriving from ``name`` over ``PROBE_S``."""
        import numpy as np
        import pygame
        from pygame._sdl2 import audio as sdl_audio
        buf = bytearray()
        dev = sdl_audio.AudioDevice(
            devicename=name, iscapture=True, frequency=SAMPLE_RATE,
            audioformat=pygame.AUDIO_S16, numchannels=1, chunksize=1024,
            allowed_changes=0,
            callback=lambda _d, mem: buf.extend(bytes(mem)))
        try:
            dev.pause(0)
            time.sleep(PROBE_S)
            dev.pause(1)
        finally:
            dev.close()
        a = np.frombuffer(bytes(buf), dtype=np.int16)
        return int(np.abs(a).max()) if a.size else 0

    @classmethod
    def input_device(cls) -> Optional[str]:
        """
        A capture device that is actually carrying sound, or ``None``.

        Not simply the first one SDL lists. On a laptop the headphone jack's
        microphone input enumerates ahead of the built-in one and opens
        happily with nothing plugged into it, so taking the first name records
        six files of silence and loses the spoken ratings - which is exactly
        what happened on 27 August. Each candidate is listened to briefly and
        the loudest is used; the first is kept only as a fallback when none of
        them carries anything, so a genuinely quiet room still records.
        """
        try:
            import pygame
            from pygame._sdl2 import audio as sdl_audio
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            names = [str(n) for n in sdl_audio.get_audio_device_names(True)]
        except Exception:
            return None
        if not names:
            return None
        best, best_peak = None, -1
        for n in names:
            try:
                peak = cls._probe(n)
            except Exception:
                logger.debug("could not probe %r", n, exc_info=True)
                continue
            logger.info("capture candidate %r peak %d", n, peak)
            if peak > best_peak:
                best, best_peak = n, peak
        if best is None:
            return names[0]
        if best_peak < SILENT_PEAK:
            logger.warning(
                "no microphone is carrying sound (loudest %r peaked at %d). "
                "Recording anyway, but check the input before relying on the "
                "spoken answers.", best, best_peak)
        return best

    def _warn_once(self, msg: str) -> None:
        if not self._warned:
            self._warned = True
            logger.warning("%s - ratings must be typed", msg)

    def capture(self, name: str) -> Optional[Path]:
        if not self.enabled:
            return None
        if self._chosen_device is None:
            self._chosen_device = self.input_device()
        device = self._chosen_device
        if device is None:
            self._warn_once("no capture device")
            return None

        self.clip_dir.mkdir(parents=True, exist_ok=True)
        dest = self.clip_dir / f"{name}.wav"

        def _record() -> None:
            import wave
            import pygame
            from pygame._sdl2 import audio as sdl_audio
            chunks: List[bytes] = []
            try:
                dev = sdl_audio.AudioDevice(
                    devicename=device, iscapture=True,
                    frequency=SAMPLE_RATE, audioformat=pygame.AUDIO_S16,
                    numchannels=1, chunksize=1024, allowed_changes=0,
                    callback=lambda _d, buf: chunks.append(bytes(buf)))
                dev.pause(0)
                time.sleep(self.seconds)
                dev.pause(1)
                dev.close()
                with wave.open(str(dest), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(SAMPLE_RATE)
                    w.writeframes(b"".join(chunks))
            except Exception:
                logger.warning("capture failed for %s", name, exc_info=True)

        t = threading.Thread(target=_record, name=f"clip-{name}", daemon=True)
        t.start()
        self._threads.append(t)
        return dest

    def stop(self) -> None:
        """Let a clip still running finish writing before the session closes."""
        for t in self._threads:
            t.join(timeout=self.seconds + 2.0)
        self._threads.clear()


class StdinInput:
    """
    Reads the terminal for ratings and for the stop signal.

    A bare digit is a rating; anything else stops the session. Both are one
    line, so neither needs the subject to look at anything - the digit row and
    the Enter key are findable by feel.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.stop_reason: Optional[str] = None
        self._lock = threading.Lock()
        self._pending: Optional[int] = None
        self._t = threading.Thread(target=self._watch, daemon=True) if enabled else None

    def start(self) -> "StdinInput":
        if self._t is not None:
            self._t.start()
        return self

    def _watch(self) -> None:
        try:
            for line in sys.stdin:
                text = line.strip()
                if text.isdigit() and 1 <= int(text) <= 9:
                    with self._lock:
                        self._pending = int(text)
                    continue
                self.stop_reason = "manual_stop"
                return
        except Exception:
            return

    def take_rating(self) -> Optional[int]:
        """Consume a rating typed since the last call, if any."""
        with self._lock:
            value, self._pending = self._pending, None
        return value

    def clear_rating(self) -> None:
        with self._lock:
            self._pending = None


def write_kss_skeleton(session_dir: Path, prompts: List[dict]) -> Path:
    """
    Pre-fill ``kss.csv`` with one row per prompt, values left blank.

    Transcription then means typing six numbers into rows that already exist,
    rather than reconstructing when each prompt happened.
    """
    dest = Path(session_dir) / "kss.csv"
    # Anything already on disk wins: the browser writes ratings here as they
    # are clicked, and rewriting the file blank at session end would throw
    # away every one of them.
    existing = {}
    if dest.exists():
        try:
            for row in dest.read_text(encoding="utf8").splitlines()[1:]:
                parts = row.split(",")
                if len(parts) >= 3 and parts[2].strip():
                    existing[parts[1].strip()] = parts[2].strip()
        except Exception:
            logger.debug("could not read existing kss.csv", exc_info=True)
    lines = ["t_s,block,value,clip"]
    for p in prompts:
        val = ("" if p.get("value") is None else str(p["value"])) \
            or existing.get(str(p.get("block", "")), "")
        lines.append(f"{p.get('t_s', '')},{p.get('block', '')},{val},"
                     f"{p.get('clip', '')}")
    dest.write_text("\n".join(lines) + "\n", encoding="utf8")
    return dest
