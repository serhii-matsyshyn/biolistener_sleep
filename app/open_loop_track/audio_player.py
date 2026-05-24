"""
Pre-recorded audio file playback for the open-loop-track protocol.

Uses ``pygame.mixer.music`` - same audio stack family as the source
replication program. SDL_mixer handles .mp3/.ogg/.wav/.flac natively. For
.m4a/AAC (and anything else SDL_mixer can't decode) we transcode to .ogg
via ffmpeg on first load and cache the result, so the slow path runs once.

Cache layout: ``$TMPDIR/biolistener-mixer-cache/<sha8>.ogg`` keyed by
(absolute source path, mtime). Survives daemon restarts; cleared by
deleting the cache dir.
"""
from __future__ import annotations

import hashlib
import logging
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger("open_loop_track.audio_player")

# SDL_mixer-native formats (no transcode needed).
_NATIVE_EXTS = {".mp3", ".ogg", ".wav", ".flac"}


def _cache_dir() -> Path:
    d = Path(tempfile.gettempdir()) / "biolistener-mixer-cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _transcode_to_ogg(src: Path) -> Path | None:
    """ffmpeg-decode any audio file to .ogg in the cache dir. Returns the
    cached path, or None if ffmpeg isn't available / transcode fails."""
    try:
        mtime = int(src.stat().st_mtime)
    except OSError:
        return None
    key = hashlib.sha1(f"{src.resolve()}|{mtime}".encode()).hexdigest()[:16]
    out = _cache_dir() / f"{key}.ogg"
    if out.exists() and out.stat().st_size > 0:
        return out
    logger.info("Transcoding %s -> %s (one-time)", src.name, out.name)
    try:
        # -ac 2: force stereo (mixer.init is stereo). -y: overwrite. Quiet logs.
        # -c:a libvorbis -q:a 5: ~160 kbps VBR ogg - perceptually lossless for voice.
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", str(src),
             "-ac", "2", "-c:a", "libvorbis", "-q:a", "5",
             "-loglevel", "error", str(out)],
            check=False, capture_output=True, timeout=300,
        )
        if r.returncode != 0 or not out.exists() or out.stat().st_size == 0:
            logger.warning("ffmpeg transcode failed for %s: %s",
                           src.name, r.stderr.decode(errors="ignore")[:300])
            try:
                out.unlink()
            except OSError:
                pass
            return None
        return out
    except FileNotFoundError:
        logger.warning("ffmpeg not on PATH - cannot decode %s", src.name)
        return None
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg timed out transcoding %s", src.name)
        return None


def _playable_path(src: Path) -> Path | None:
    """Return a path SDL_mixer can definitely load, transcoding if needed."""
    if not src.exists():
        return None
    if src.suffix.lower() in _NATIVE_EXTS:
        return src
    return _transcode_to_ogg(src)


class OpenLoopTrackAudioPlayer:
    """Single-file player using pygame.mixer.music."""

    _mixer_inited = False

    def __init__(self, audio_path: Path, master_vol: float = 0.5,
                 loop: bool = False):
        self.audio_path = Path(audio_path)
        self.master_vol = float(master_vol)
        self.loop = loop
        self._started = False

    @classmethod
    def _ensure_mixer(cls) -> bool:
        if cls._mixer_inited:
            return True
        try:
            import pygame
            pygame.mixer.init(frequency=44100, size=-16, channels=2,
                              buffer=512)
            cls._mixer_inited = True
            logger.info("pygame.mixer initialised")
            return True
        except Exception:
            logger.warning("pygame.mixer init failed - audio silent",
                           exc_info=True)
            return False

    def start(self) -> bool:
        """Boot the mixer (if needed) and start playback. True on success."""
        if not self.audio_path.exists():
            logger.warning("Audio file not found: %s - playback skipped",
                           self.audio_path)
            return False
        if not self._ensure_mixer():
            return False
        play_path = _playable_path(self.audio_path)
        if play_path is None:
            logger.warning("Audio file unusable (no decoder + ffmpeg failed): %s",
                           self.audio_path)
            return False
        try:
            import pygame
            pygame.mixer.music.load(str(play_path))
            pygame.mixer.music.set_volume(self.master_vol)
            pygame.mixer.music.play(loops=-1 if self.loop else 0)
            self._started = True
            logger.info("Open-loop track audio playing: %s%s",
                        self.audio_path.name,
                        "" if play_path == self.audio_path else f" (via {play_path.name})")
            return True
        except Exception:
            logger.warning(
                "OpenLoopTrackAudioPlayer.start() failed for %s",
                self.audio_path, exc_info=True,
            )
            return False

    def set_master_vol(self, vol_0_to_1: float) -> None:
        self.master_vol = max(0.0, min(1.0, float(vol_0_to_1)))
        if self._started:
            try:
                import pygame
                pygame.mixer.music.set_volume(self.master_vol)
            except Exception:
                pass

    def stop(self) -> None:
        if self._started:
            try:
                import pygame
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
            except Exception:
                pass
            self._started = False

    @property
    def is_playing(self) -> bool:
        return self._started
