"""
Session recording.

One directory per session holding everything needed to reproduce and cite it.
Writers append as the session runs, so an aborted or crashed run still leaves
usable data rather than an empty file.

Layout::

    data/sessions/<utc>_<protocol>/
        manifest.json       protocol, seed, montage, git commit, versions
        ticks.csv           per control tick
        markers.jsonl       block edges, cues, KSS prompts, aborts
        safety_events.jsonl every clamp the supervisor applied
        raw_eeg.csv         continuous stream, written by BrainFlow
        kss.csv             transcribed after the session
        notes.md            post-session note, from a fixed prompt list
"""
from __future__ import annotations

import csv
import json
import logging
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("protocol.recorder")

SESSION_ROOT = Path("data/sessions")

TICK_FIELDS = [
    "t_s", "raw_sample", "block", "phase", "probe_hz",
    "cmd_freq_hz", "cmd_fill", "cmd_brightness",
    "act_freq_hz", "act_fill", "act_brightness",
    # 1 when the light was continuous rather than flickering, so a block with
    # light in it but no strobe is not mistaken for a failed strobe command.
    "steady",
    "alpha_pwr", "theta_pwr", "hypnagogic_theta_pwr", "sigma_pwr", "beta_pwr",
    "theta_alpha_ratio", "emg_rms", "eog_energy", "railed_pct",
    # PLV and the smoothed pair complete the observation: with these the
    # policy's input can be rebuilt tick for tick after the fact, which is what
    # separates "the agent transferred badly" from "the agent was fed noise".
    "plv", "smoothed_plv", "delta_theta_alpha",
    "reward", "dose",
    # Simulation only - blank on hardware, where no such quantity exists.
    "true_resonance_hz", "true_drowsiness",
]

NOTES_PROMPTS = """# Post-session notes

Written immediately after the session, before anything is analysed.

- Start time of Session A, and of Session B.
- Hours slept the previous night; what time sleep actually came.
- Caffeine and alcohol today, and when.
- Posture used, and whether it held for the whole session.
- Did you fall asleep? Roughly when?
- Mask comfort - did it shift or need adjusting at any point?
- Was the O1 electrode painful supine? Which posture did you settle on?
- Closed-eye visuals: any geometric patterns or colour? At which frequencies?
- The re-alerting block - did it actually wake you up?
- Anything that went wrong, or anything unusual.
"""


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def _git_dirty() -> Optional[bool]:
    try:
        out = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                             text=True, timeout=5)
        return bool(out.stdout.strip())
    except Exception:
        return None


@dataclass
class SessionRecorder:
    """Owns one session directory. Use as a context manager."""

    protocol_name: str
    root: Path = SESSION_ROOT
    extra_manifest: Dict[str, Any] = field(default_factory=dict)

    dir: Path = field(init=False)
    t0: float = field(init=False, default=0.0)
    _ticks: Any = field(init=False, default=None)
    _tick_writer: Any = field(init=False, default=None)
    _markers: Any = field(init=False, default=None)
    _safety: Any = field(init=False, default=None)
    _n_ticks: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        self.dir = Path(self.root) / f"{stamp}_{self.protocol_name}"

    # ---- lifecycle -------------------------------------------------------

    def open(self) -> "SessionRecorder":
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self._ticks = (self.dir / "ticks.csv").open("w", newline="")
        self._tick_writer = csv.DictWriter(self._ticks, fieldnames=TICK_FIELDS,
                                           extrasaction="ignore")
        self._tick_writer.writeheader()
        self._markers = (self.dir / "markers.jsonl").open("w")
        self._safety = (self.dir / "safety_events.jsonl").open("w")
        (self.dir / "notes.md").write_text(NOTES_PROMPTS, encoding="utf8")
        logger.info("session dir %s", self.dir)
        return self

    def __enter__(self) -> "SessionRecorder":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close(status="error" if exc[0] else "completed")

    def close(self, status: str = "completed") -> None:
        for f in (self._ticks, self._markers, self._safety):
            try:
                if f is not None:
                    f.close()
            except Exception:
                logger.debug("close failed", exc_info=True)
        self._finalise_manifest(status)

    # ---- writers ---------------------------------------------------------

    @property
    def raw_eeg_path(self) -> Path:
        """Hand to ``IngestConfig.record_to`` before starting the stream."""
        return self.dir / "raw_eeg.csv"

    def set_time_origin(self) -> None:
        """Re-anchor marker time on the caller's clock, called at t=0.

        The directory is created before the EEG source is opened, and opening
        it can take a minute of connect retries. Markers and ticks would
        otherwise be stamped on origins that far apart, and the analysis maps
        marker time onto raw samples.
        """
        self.t0 = time.time()

    def tick(self, **row: Any) -> None:
        if self._tick_writer is None:
            raise RuntimeError("recorder is not open")
        self._tick_writer.writerow(row)
        self._n_ticks += 1
        if self._n_ticks % 200 == 0:      # bounded loss if the process dies
            self._ticks.flush()

    def marker(self, event: str, **payload: Any) -> None:
        """``event`` names the marker; payload keys are free, including ``kind``."""
        if self._markers is None:
            raise RuntimeError("recorder is not open")
        rec = {"t_s": round(time.time() - self.t0, 4), "event": event, **payload}
        self._markers.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._markers.flush()

    def safety_events(self, events: List[Any]) -> None:
        if self._safety is None or not events:
            return
        for e in events:
            rec = asdict(e) if hasattr(e, "__dataclass_fields__") else dict(e)
            self._safety.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        self._safety.flush()

    def write_manifest(self, **extra: Any) -> None:
        self.extra_manifest.update(extra)

    def _finalise_manifest(self, status: str) -> None:
        man = {
            "protocol": self.protocol_name,
            "status": status,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.t0)),
            "duration_s": round(time.time() - self.t0, 2),
            "n_ticks": self._n_ticks,
            "git_commit": _git_commit(),
            "git_dirty": _git_dirty(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            **self.extra_manifest,
        }
        (self.dir / "manifest.json").write_text(
            json.dumps(man, indent=2, ensure_ascii=False, default=str), encoding="utf8")


def load_session(path: Path) -> Dict[str, Any]:
    """Read a recorded session back. Missing optional files are ``None``."""
    import pandas as pd

    path = Path(path)
    man = json.loads((path / "manifest.json").read_text(encoding="utf8"))

    def _jsonl(name: str) -> List[dict]:
        f = path / name
        if not f.exists():
            return []
        return [json.loads(l) for l in f.read_text(encoding="utf8").splitlines() if l.strip()]

    def _csv(name: str):
        f = path / name
        return pd.read_csv(f) if f.exists() else None

    return {
        "dir": path,
        "manifest": man,
        "ticks": _csv("ticks.csv"),
        "kss": _csv("kss.csv"),
        "markers": _jsonl("markers.jsonl"),
        "safety_events": _jsonl("safety_events.jsonl"),
        "raw_eeg_path": (path / "raw_eeg.csv") if (path / "raw_eeg.csv").exists() else None,
        "notes": (path / "notes.md").read_text(encoding="utf8")
        if (path / "notes.md").exists() else None,
    }
