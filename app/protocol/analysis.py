"""
Post-session analysis.

Computation lives here rather than in the notebooks so the numbers that reach
the thesis come from code that can be tested and re-run.

Raw samples are aligned to session time through the ``raw_sample`` column in
``ticks.csv``, which records how many samples the source had produced at each
tick. Where that is unavailable - a real recording, where BrainFlow owns the
file - the mapping falls back to a constant rate anchored on the first tick.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.protocol import quality as q
from app.protocol.recorder import load_session

logger = logging.getLogger("protocol.analysis")

EDGE_DROP_S = 1.0        # discarded at each segment edge
# How "response at a probe frequency" is measured.
#
# ``plv`` - phase locking between the delivered strobe envelope and the EEG.
# ``amplitude`` - spectral power at the probe frequency.
#
# PLV is the default: spectral power at a probe also contains whatever the brain
# was already doing there, and near the subject's own alpha that endogenous
# activity dominates the driven component.
RESPONSE_METRIC = "plv"
# Resampling unit for the bootstrap. Three seconds, not one: PLV is measured
# inside a ~2 Hz band that takes about half a second to settle, so a one-second
# epoch is largely filter transient.
EPOCH_S = 3.0
BOOTSTRAP_N = 2000

# Shortest segment that still counts as one, as a fraction of the median
# segment in the same recording. Relative rather than absolute so it means the
# same thing for a full session and for a compressed rehearsal: what it is
# there to catch is a segment cut short by an abort being averaged, with equal
# weight, against segments that ran to completion.
MIN_SEGMENT_FRACTION = 0.5
MIN_SEGMENT_TICKS = 20
# Pre-registered floor on how much the command must move within a segment. An
# arm whose spread falls below this was not controlling: either pinned by the
# supervisor or emitting a constant. The descending ramp traverses 7 Hz
# linearly, giving 7/sqrt(12) ~ 2.0, so this only ever binds on the agent.
MIN_FREQ_SD_HZ = 0.15


# ---- loading -------------------------------------------------------------

def load_raw(path: Path) -> np.ndarray:
    """``(channels, samples)`` from a whitespace-delimited recording."""
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr.T if arr.shape[0] > arr.shape[1] else arr


# A packet gap - pre-flight, a pause, the stream restarting - is not a sample
# interval, so runs are cut here before the rate is measured.
_PACKET_GAP_S = 0.010
# A measured rate this far from the nominal one is not a clock difference, it
# is a damaged timestamp column; fall back rather than trust it.
_RATE_TOLERANCE = 0.25


def measure_sampling_rate(raw: np.ndarray, nominal: float) -> Optional[float]:
    """
    True sample rate from the recording's own timestamps, or ``None``.

    The device does not necessarily run at the rate it was asked for: a
    BIOLISTENER told ``data_rate 500`` rounds to a power-of-two divider and
    streams 512, while ``BoardShim.get_sampling_rate`` keeps reporting the
    nominal figure. A 2.4 % error is enough to put a commanded 6 Hz drive at
    5.86 Hz on the analysis axis and to walk block boundaries tens of seconds
    off over a session, so the rate is measured rather than declared.

    Measured over the longest gap-free run: whole-file span includes the time
    before the session clock starts and any pause, which would understate it
    badly. ``1/median(dt)`` is not usable either - timestamps are quantised to
    the millisecond, so 500 and 512 Hz both report a 2 ms median.
    """
    ts = None
    for row in raw:
        # The timestamp column is the one carrying non-decreasing unix epoch
        # values; the packet counter also rises but sits far below.
        if row.size > 2 and row[0] > 1e9 and bool(np.all(np.diff(row) >= 0)):
            ts = row
            break
    if ts is None or ts.size < 1000:
        return None
    breaks = np.flatnonzero(np.diff(ts) > _PACKET_GAP_S)
    edges = np.concatenate(([0], breaks + 1, [ts.size]))
    best, span = 0, 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a > best:
            best, span = b - a, float(ts[b - 1] - ts[a])
    if best < 1000 or span <= 0:
        return None
    rate = (best - 1) / span
    if abs(rate - nominal) > _RATE_TOLERANCE * nominal:
        logger.warning("measured sample rate %.3f Hz is implausible against a "
                       "nominal %.0f Hz; using the nominal value", rate, nominal)
        return None
    return float(rate)


@dataclass
class Session:
    dir: Path
    manifest: dict
    ticks: object
    raw: Optional[np.ndarray]
    markers: List[dict]
    kss: object
    safety_events: List[dict]
    _measured_rate: Optional[float] = None
    _timeline: object = None          # lazily built by :meth:`timeline`

    @property
    def sampling_rate(self) -> int:
        """Rate for spectral work, rounded - the PSD helpers require an int."""
        return int(round(self.sampling_rate_exact))

    @property
    def sampling_rate_exact(self) -> float:
        """
        Measured rate, falling back to what the manifest declares.

        Index arithmetic uses this rather than the rounded value: over a
        35-minute session the difference between 511.995 and 512 is still only
        a few samples, but the difference between either and a declared 500 is
        about fifty seconds.
        """
        if self._measured_rate is not None:
            return self._measured_rate
        recorded = self.manifest.get("measured_sampling_rate")
        if recorded:
            return float(recorded)
        return float(self.manifest.get("sampling_rate", 500))

    @property
    def primary_channel(self) -> int:
        """Row of ``raw`` holding the montage's primary EEG channel.

        Montage indices count EEG channels; the file may not. BrainFlow's
        streamer writes the packet counter first, so montage channel 0 is row
        1 there and row 0 in a simulated recording. ``raw_eeg_rows`` in the
        manifest says which; without it, assume the rows are already EEG.
        """
        from app.eeg.montage import ALL as MONTAGES
        m = MONTAGES[self.manifest.get("montage", "occipital_alpha")]
        idx = m.primary_eeg_channels[0]
        rows = self.manifest.get("raw_eeg_rows")
        if rows and idx < len(rows):
            return int(rows[idx])
        return idx

    def timeline(self) -> Optional[Tuple[np.ndarray, int]]:
        """
        ``(timestamps, first sample of the longest unbroken run)``.

        The recording can open with samples the board buffered before the
        session began - 22 and 44 of them in the two hardware sessions here,
        followed by an eight-minute gap - so the run is found rather than
        assumed to start at sample zero.
        """
        if self._timeline is not None:
            return self._timeline or None
        if self.raw is None:
            self._timeline = False
            return None
        for row in self.raw:
            if row.size > 2 and row[0] > 1e9 and bool(np.all(np.diff(row) >= 0)):
                breaks = np.flatnonzero(np.diff(row) > _PACKET_GAP_S)
                edges = np.concatenate(([0], breaks + 1, [row.size]))
                a, _ = max(zip(edges[:-1], edges[1:]), key=lambda e: e[1] - e[0])
                self._timeline = (row, int(a))
                return self._timeline
        self._timeline = False
        return None

    def raw_index_at(self, t_s: float) -> int:
        """Sample index in ``raw`` corresponding to session time ``t_s``."""
        tk = self.ticks
        if tk is not None and "raw_sample" in tk and tk["raw_sample"].notna().any():
            good = tk[tk["raw_sample"].notna()]
            return int(np.interp(t_s, good["t_s"].to_numpy(),
                                 good["raw_sample"].to_numpy()))
        # Hardware recordings have no per-tick sample count - BrainFlow owns
        # the file - and it starts before the session clock does. The runner
        # writes down how much ran ahead.
        offset = float(self.manifest.get("raw_offset_s", 0.0) or 0.0)
        # Looked up against the recording's own clock wherever there is one.
        # Multiplying elapsed time by a rate assumes samples arrive evenly, and
        # a stalled link that later delivers its backlog in one burst breaks
        # that assumption for every block after it. A timestamp search does not
        # care how the samples arrived, only when they were taken.
        tl = self.timeline()
        if tl is not None:
            ts, start = tl
            return int(np.searchsorted(ts, ts[start] + offset + t_s))
        return int((t_s + offset) * self.sampling_rate_exact)

    def block_bounds(self, block: str) -> Tuple[float, float]:
        starts = [m["t_s"] for m in self.markers
                  if m.get("event") == "block_start" and m.get("block") == block]
        ends = [m["t_s"] for m in self.markers
                if m.get("event") == "block_end" and m.get("block") == block]
        if not starts:
            raise KeyError(f"no block_start marker for {block!r}")
        return starts[0], (ends[0] if ends else starts[0])

    def segment(self, block: str, drop_edge_s: float = EDGE_DROP_S,
                channel: Optional[int] = None) -> np.ndarray:
        t0, t1 = self.block_bounds(block)
        i0 = self.raw_index_at(t0 + drop_edge_s)
        i1 = self.raw_index_at(t1 - drop_edge_s)
        ch = self.primary_channel if channel is None else channel
        if self.raw is None:
            raise ValueError("session has no raw recording")
        i0, i1 = max(0, i0), max(0, i1)
        if i1 - i0 < 64:
            raise ValueError(
                f"{block!r}: {i1 - i0} samples after dropping {drop_edge_s:g} s "
                f"at each edge - block too short to analyse")
        return self.raw[ch, i0:i1]


def load(path: Path) -> Session:
    d = load_session(Path(path))
    raw = None
    if d["raw_eeg_path"] is not None:
        try:
            raw = load_raw(d["raw_eeg_path"])
        except Exception:
            logger.warning("could not read %s", d["raw_eeg_path"], exc_info=True)
    nominal = float(d["manifest"].get("sampling_rate", 500))
    measured = measure_sampling_rate(raw, nominal) if raw is not None else None
    if measured is not None and abs(measured - nominal) / nominal > 0.005:
        logger.info("%s: recorded at %.3f Hz, not the declared %.0f Hz",
                    d["dir"].name, measured, nominal)
    return Session(dir=d["dir"], manifest=d["manifest"], ticks=d["ticks"],
                   raw=raw, markers=d["markers"], kss=d["kss"],
                   safety_events=d["safety_events"], _measured_rate=measured)


# ---- validation gates ----------------------------------------------------

def _as_diagnostic(g: "q.GateResult", name: str) -> "q.GateResult":
    """Same measurement, reported rather than gating."""
    from dataclasses import replace
    return replace(g, name=name, blocking=False)


def _unevaluated(name: str, label: str, err: Exception) -> q.GateResult:
    """A gate that could not be computed is a failure, not a silent absence."""
    return q.GateResult(name, label, float("nan"), float("nan"), False,
                        blocking=True, detail=f"could not evaluate: {err}")


def ladder_frequency(sess: Session) -> float:
    """
    Frequency the brightness ladder was actually driven at.

    Taken from the recording rather than from ``spec.SSVEP_CHECK_HZ``: the
    constant describes the protocol as it stands today, and a session recorded
    before it moved would be probed at a frequency it never emitted. The
    commanded value is exact - it is what the mask was told, not a measurement.
    """
    tk = sess.ticks
    if tk is not None and "act_freq_hz" in tk:
        rung = tk[tk["block"].astype(str).str.startswith("A4_bright_")]
        lit = rung[rung["act_freq_hz"] > 0]
        if len(lit):
            return float(lit["act_freq_hz"].median())
    from app.protocol.spec import SSVEP_CHECK_HZ
    return float(SSVEP_CHECK_HZ)


def _brightest_block(sess: Session) -> str:
    """Name of the highest-brightness rung of the ladder."""
    names = [m.get("block") for m in sess.markers
             if m.get("event") == "block_start"
             and str(m.get("block", "")).startswith("A4_bright_")]
    if not names:
        raise ValueError("no brightness-ladder blocks recorded")
    return max(names, key=lambda n: int(n.rsplit("_", 1)[1]))


def validation_gates(sess: Session) -> List[q.GateResult]:
    """
    Run every gate over a validation recording.

    A block that is missing or too short marks its own gate as unevaluated and
    leaves the rest to run: a session with one damaged block should still say
    what it can about the others.
    """
    fs = sess.sampling_rate
    check_hz = ladder_frequency(sess)
    out: List[q.GateResult] = []
    tk = sess.ticks

    def _try(name, label, fn):
        try:
            out.append(fn())
        except Exception as err:                       # noqa: BLE001
            logger.warning("%s not evaluated: %s", name, err)
            out.append(_unevaluated(name, label, err))

    _try("G1", "railed %", lambda: q.gate_railed(
        float(tk["railed_pct"].max()) if tk is not None and not tk.empty else 0.0))
    _try("G2", "line noise", lambda: q.gate_line_noise(
        sess.segment("A1_settle"), fs))
    def _frontal_rows():
        """Recorded channels that are not the primary EEG one."""
        from app.eeg.montage import ALL as MONTAGES
        m = MONTAGES[sess.manifest.get("montage", "occipital_alpha")]
        rows = sess.manifest.get("raw_eeg_rows") or []
        active = sess.manifest.get("active_channels")
        idx = [c for c in (active if active is not None else m.all_active())
               if c not in m.primary_eeg_channels]
        return [rows[c] for c in idx if c < len(rows)]

    _try("G3", "occipital/frontal alpha", lambda: q.gate_alpha_topography(
        [sess.segment(b) for b in ("A2b_eyes_closed", "A2d_eyes_closed")],
        [sess.segment("A2b_eyes_closed", channel=r) for r in _frontal_rows()],
        fs))
    def _unlit():
        """Blocks where the strobe was off entirely - the pickup reference."""
        out = []
        for name in ("A1_settle", "A2b_eyes_closed", "A2d_eyes_closed",
                     "A3a_alpha_rest", "A3b_alpha_task"):
            try:
                out.append(sess.segment(name))
            except Exception:                              # noqa: BLE001
                continue
        if not out:
            raise ValueError("no unstimulated block to compare against")
        return out

    def _ladder():
        segs = {}
        for m in sess.markers:
            name = m.get("block")
            if m.get("event") == "block_start" and name and name.startswith("A4_bright_"):
                segs[int(name.rsplit("_", 1)[1]) / 100.0] = sess.segment(name)
        if not segs:
            raise ValueError("no brightness-ladder blocks recorded")
        return segs

    _try("G8", "alpha peak over background", lambda: q.gate_alpha_present(
        [sess.segment(b) for b in ("A2b_eyes_closed", "A3a_alpha_rest")], fs))

    # Reported, never blocking. Both manipulations are defeated by the
    # apparatus - a mask emitting diffuse light is a Ganzfeld, and eyes-closed
    # arithmetic is the condition alpha is least sensitive to - so they are
    # kept as measurements of the subject rather than as tests of the rig.
    _try("D1", "alpha closed/open", lambda: _as_diagnostic(
        q.gate_alpha_blocking(
            [sess.segment("A2a_eyes_open"), sess.segment("A2c_eyes_open")],
            [sess.segment("A2b_eyes_closed"), sess.segment("A2d_eyes_closed")],
            fs), "D1"))
    _try("D2", "alpha rest/task", lambda: _as_diagnostic(
        q.gate_cognitive_alpha(
            [sess.segment("A3a_alpha_rest")], [sess.segment("A3b_alpha_task")],
            fs), "D2"))
    _try("G4", "SSVEP SNR", lambda: q.gate_ssvep(
        sess.segment(_brightest_block(sess)), fs, check_hz))
    _try("G5", "artifact", lambda: q.gate_artifact(
        sess.segment("A4_bright_000"), _unlit(), fs, check_hz))
    _try("G7", "dose response", lambda: q.gate_dose_response(
        _ladder(), fs, check_hz))

    def _multi():
        segs = {}
        for m in sess.markers:
            name = m.get("block")
            if m.get("event") == "block_start" and name and name.startswith("A5_multi_"):
                segs[float(name.split("_")[-1].replace("hz", ""))] = sess.segment(name)
        if not segs:
            raise ValueError("no A5 blocks recorded")
        return q.gate_frequency_specificity(segs, fs)

    _try("G6", "frequency specificity", _multi)
    return out


# ---- response curves -----------------------------------------------------

@dataclass
class ProbeResponse:
    probe_hz: float
    snr_db: float
    amplitude: float
    n_epochs: int
    epoch_amplitudes: np.ndarray


@dataclass
class ResponseCurve:
    block: str
    probes: Tuple[ProbeResponse, ...]
    # Probes that produced no usable samples, with the reason. Never silently
    # dropped: a gap in the curve changes where the peak lands.
    skipped: Tuple[Tuple[float, str], ...] = ()

    @property
    def freqs(self) -> np.ndarray:
        return np.array([p.probe_hz for p in self.probes])

    @property
    def snr(self) -> np.ndarray:
        return np.array([p.snr_db for p in self.probes])

    @property
    def amplitude(self) -> np.ndarray:
        return np.array([p.amplitude for p in self.probes])


def _dwell_for(sess: Session, block: str) -> Optional[float]:
    for b in sess.manifest.get("blocks", []):
        if b.get("name") == block:
            return b.get("dwell_s")
    return None


def response_curve(
    sess: Session, block: str, epoch_s: float = EPOCH_S,
    settle_s: Optional[float] = None, metric: str = RESPONSE_METRIC,
) -> ResponseCurve:
    """
    Entrainment strength at each probe of a sweep.

    Probe windows come from what the mask actually delivered, not from the
    planned schedule, so an overrunning transition or a clamped command removes
    those samples rather than diluting the estimate.
    """
    from app.brain_sim.photic_input import StrobeParams, strobe_envelope
    from app.eeg.plv import plv_strobe_to_eeg
    from app.protocol.sweep import settled_mask

    tk = sess.ticks
    fs = sess.sampling_rate
    blk = tk[tk["block"] == block].copy()
    if blk.empty:
        raise KeyError(f"no ticks for block {block!r}")
    blk = blk[blk["probe_hz"].notna()]

    if settle_s is None:
        # Discard entrainment build-up, but never more than a third of the
        # dwell, or a short-dwell recording would have nothing left.
        dwell = _dwell_for(sess, block)
        settle_s = 1.0 if not dwell else min(1.0, dwell / 3.0)

    keep = settled_mask(blk["t_s"].to_numpy(), blk["act_freq_hz"].to_numpy(),
                        blk["probe_hz"].to_numpy(), settle_s=settle_s)
    blk = blk[keep]

    def _score(seg: np.ndarray, f: float, fill: float, brightness: float) -> float:
        if metric == "amplitude":
            return q.peak_amplitude(q.psd_welch(seg, fs), f)
        env = strobe_envelope(
            duration_s=seg.size / fs, dt=1.0 / fs,
            strobe=StrobeParams(enabled=True, frequency_hz=f,
                                fill=fill, brightness=brightness))
        return float(plv_strobe_to_eeg(seg, env[:seg.size], fs, f))

    probes, skipped = [], []
    step = max(64, int(epoch_s * fs))
    for f, grp in blk.groupby("probe_hz"):
        i0 = sess.raw_index_at(float(grp["t_s"].min()))
        i1 = sess.raw_index_at(float(grp["t_s"].max()))
        seg = sess.raw[sess.primary_channel, i0:i1] if sess.raw is not None else np.array([])
        if seg.size < 64:
            skipped.append((float(f), f"{seg.size} settled samples"))
            continue
        fill = float(grp["act_fill"].mean()) if "act_fill" in grp else 0.4
        bright = float(grp["act_brightness"].mean()) if "act_brightness" in grp else 0.6
        # A dwell shorter than one epoch still yields a point estimate; it just
        # cannot contribute to the bootstrap, which needs at least two.
        bounds = [(k * step, (k + 1) * step) for k in range(max(1, seg.size // step))]
        vals = [_score(seg[a:b], float(f), fill, bright)
                for a, b in bounds if seg[a:b].size >= 64]
        if not vals:
            vals = [_score(seg, float(f), fill, bright)]
        psd = q.psd_welch(seg, fs)
        # Point estimate from the whole dwell, not the mean of short epochs.
        # PLV is biased upward in short windows and the bias is not uniform
        # across frequency, so averaging epoch PLVs tilts the curve toward the
        # high probes and the fit pins to the top of the grid. Epochs are kept
        # for the bootstrap interval only, where the bias is common to every
        # resample and cancels.
        probes.append(ProbeResponse(
            probe_hz=float(f), snr_db=q.peak_snr_db(psd, float(f)),
            amplitude=_score(seg, float(f), fill, bright),
            n_epochs=len(vals), epoch_amplitudes=np.array(vals)))

    if skipped:
        logger.warning("%s: %d probe(s) had no usable samples: %s", block,
                       len(skipped), ", ".join(f"{f:g} Hz ({r})" for f, r in skipped))
    probes.sort(key=lambda p: p.probe_hz)
    return ResponseCurve(block=block, probes=tuple(probes),
                         skipped=tuple(skipped))


# ---- peak estimation -----------------------------------------------------

PEAK_TOP_K = 4          # probes that contribute to the centroid
# Floor on the reported half-width: the p95 of the estimator's absolute error,
# measured against a simulated plant at known peaks. The p95 rather than the
# mean, so the quoted interval is not sized to the typical case.
#
# The bias is deliberately not subtracted. It is calibrated in simulation and a
# real head need not share it, so correcting for it would trade a wide honest
# interval for a confident wrong centre.
#
# This bounds the *absolute* peak. A shift between two sweeps of one session
# subtracts two estimates carrying much the same bias, so the shift is better
# determined than either endpoint.
PEAK_ACCURACY_HZ = 1.45


def fit_peak(freqs: Sequence[float], response: Sequence[float],
             top_k: int = PEAK_TOP_K) -> float:
    """
    Peak location as the response-weighted centroid of the strongest probes.

    A parabola through the maximum and its two neighbours uses three points and
    throws the rest away, which on a real response curve makes it noisy and
    sensitive to a single high probe. Weighting the strongest ``top_k`` probes
    by how far they stand above the curve's own floor uses more of the sweep and
    ignores the far tail, where the response is mostly the PLV floor.

    Measured against a simulated plant at known peaks, over twelve sweeps:

        parabola, 3 points   bias +0.50 Hz   |error| 0.79   sd 0.84
        centroid, all        bias +0.73 Hz   |error| 0.79   sd 0.56
        centroid, top 4      bias +0.35 Hz   |error| 0.50   sd 0.50
    """
    f = np.asarray(freqs, float)
    y = np.asarray(response, float)
    ok = np.isfinite(f) & np.isfinite(y)
    f, y = f[ok], y[ok]
    if f.size == 0:
        return float("nan")
    if f.size < 3:
        return float(f[np.argmax(y)])

    keep = min(top_k, f.size)
    idx = np.argsort(y)[-keep:]
    fk, yk = f[idx], y[idx]
    # Weight by height above the local floor: an offset common to every probe
    # carries no information about where the peak is, and leaving it in drags
    # the centroid toward the middle of the grid.
    w = np.clip(yk - yk.min(), 0.0, None)
    if w.sum() <= 0:
        return float(f[np.argmax(y)])
    return float(np.clip((fk * w).sum() / w.sum(), f.min(), f.max()))


def bootstrap_peak(
    curve: ResponseCurve, n: int = BOOTSTRAP_N, seed: int = 0,
) -> Dict[str, float]:
    """
    Peak location with a percentile interval, resampling epochs within probes.

    The interval reflects variability inside one recording, which is the only
    kind a single session can support.
    """
    rng = np.random.default_rng(seed)
    usable = [p for p in curve.probes if p.n_epochs >= 2]
    if len(usable) < 3:
        return {"peak_hz": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n_boot": 0, "spacing_hz": float("nan"),
                "resolvable": False}
    freqs = np.array([p.probe_hz for p in usable])
    # ``amplitude`` is the whole-dwell estimate; the epoch means are only for
    # the interval. Using the epoch means here would inherit their bias.
    point = fit_peak(freqs, [p.amplitude for p in usable])
    draws = np.empty(n)
    for b in range(n):
        means = [float(np.mean(rng.choice(p.epoch_amplitudes, p.n_epochs, replace=True)))
                 for p in usable]
        draws[b] = fit_peak(freqs, means)
    draws = draws[np.isfinite(draws)]
    # Recentre the interval on the point estimate. The resamples are built from
    # short epochs, whose PLV is biased upward, so their distribution sits above
    # the whole-dwell estimate - left alone it produced intervals that did not
    # even contain the point they were quoted around. The bootstrap is used for
    # the *spread*, which the bias shifts equally in every resample, and that
    # spread is hung on the unbiased centre.
    offset = point - float(np.median(draws)) if draws.size else 0.0
    lo = float(np.percentile(draws, 2.5)) + offset
    hi = float(np.percentile(draws, 97.5)) + offset
    # Each side is widened independently to the estimator's measured accuracy,
    # so an interval that sits off-centre is widened on the side that needs it.
    inflated = (point - lo) < PEAK_ACCURACY_HZ or (hi - point) < PEAK_ACCURACY_HZ
    lo = min(lo, point - PEAK_ACCURACY_HZ)
    hi = max(hi, point + PEAK_ACCURACY_HZ)
    # The bootstrap measures how repeatable the fit is, not how right it is: an
    # interval far narrower than the gap between probes is a bias signature,
    # not precision. ``resolvable`` reports whether the resamples alone were
    # informative, before the floor was applied.
    spacing = float(np.median(np.diff(freqs))) if freqs.size > 1 else float("nan")
    resolvable = not inflated
    return {"peak_hz": point, "lo": lo, "hi": hi,
            "n_boot": int(draws.size),
            "spacing_hz": spacing,
            "width_floor_applied": bool(inflated),
            "resolvable": resolvable}


def peak_table(sess: Session, blocks: Sequence[str], seed: int = 0,
               metric: str = RESPONSE_METRIC):
    """One row per sweep: peak, interval, and how much data backed it.

    ``prominence`` is the peak divided by the median across probes - how much
    of a peak there is to fit at all. Each metric has a regime where it is
    uninformative: PLV is phase-only, so a cleanly driven plant locks at every
    probe and the curve flattens; spectral amplitude carries the endogenous
    activity at the probe frequency. A fit from a curve with no prominence is
    fitting noise, whatever its interval says.
    """
    import pandas as pd
    rows = []
    for b in blocks:
        try:
            c = response_curve(sess, b, metric=metric)
            r = bootstrap_peak(c, seed=seed)
            resp = np.array([p.amplitude for p in c.probes], dtype=float)
            med = float(np.median(resp)) if resp.size else float("nan")
            rows.append({"block": b, "peak_hz": r["peak_hz"],
                         "ci_lo": r["lo"], "ci_hi": r["hi"],
                         "prominence": (float(resp.max() / med)
                                        if med and np.isfinite(med) else float("nan")),
                         "argmax_hz": (float(c.probes[int(resp.argmax())].probe_hz)
                                       if resp.size else float("nan")),
                         # Carried through so the regression can report
                         # whether the peaks it fitted were resolvable.
                         "resolvable": bool(r.get("resolvable", True)),
                         "n_probes": len(c.probes),
                         "n_epochs": sum(p.n_epochs for p in c.probes)})
        except Exception as err:
            logger.warning("sweep %s: %s", b, err)
            rows.append({"block": b, "peak_hz": float("nan"),
                         "ci_lo": float("nan"), "ci_hi": float("nan"),
                         "prominence": float("nan"), "argmax_hz": float("nan"),
                         "resolvable": False,
                         "n_probes": 0, "n_epochs": 0})
    return pd.DataFrame(rows, columns=["block", "peak_hz", "ci_lo", "ci_hi",
                                       "prominence", "argmax_hz", "resolvable",
                                       "n_probes", "n_epochs"])


def sweep_blocks(sess: Session) -> List[str]:
    seen, out = set(), []
    for b in sess.manifest.get("blocks", []):
        if b.get("kind") == "sweep" and b["name"] not in seen:
            seen.add(b["name"])
            out.append(b["name"])
    return out


# ---- primary outcome: ramp against the learned policy --------------------

ARM_OF_KIND = {"descent_ramp": "ramp", "descent_agent": "agent"}


def arm_segments(sess: Session) -> List[dict]:
    """Descent segments in run order, with their ABBA position."""
    out = []
    for b in sess.manifest.get("blocks", []):
        arm = ARM_OF_KIND.get(b.get("kind", ""))
        if arm:
            out.append({"block": b["name"], "arm": arm, "position": len(out)})
    return out


def arm_summary(sess: Session):
    """
    Per-segment reward and delivered dose.

    The agent controls brightness and the ramp does not, so dose is reported
    beside reward and the contrast is read against it.
    """
    import pandas as pd
    tk = sess.ticks
    rows = []
    segs = list(arm_segments(sess))
    counts = [len(tk[tk["block"] == s["block"]]) for s in segs]
    full = [c for c in counts if c > 0]
    # Floor of at least one tick, so a session with no usable segments reports
    # that it has no data rather than a NaN contrast.
    floor = (max(MIN_SEGMENT_TICKS, MIN_SEGMENT_FRACTION * float(np.median(full)))
             if full else 1.0)
    floor = max(floor, 1.0)
    for seg, n in zip(segs, counts):
        d = tk[tk["block"] == seg["block"]]
        # A segment cut short by an abort is averaged with the same weight as
        # a full one and contributes no variability to the bootstrap, so a
        # thirty-second remnant would quietly sit beside a five-minute block.
        if n < floor:
            if n:
                logger.warning("segment %s has %d ticks (< %.0f) - excluded",
                               seg["block"], n, floor)
            continue
        rows.append({**seg,
                     "n_ticks": len(d),
                     "mean_reward": float(d["reward"].mean()),
                     "total_reward": float(d["reward"].sum()),
                     "mean_dose": float(d["dose"].mean()),
                     "mean_freq_hz": float(d["act_freq_hz"].mean()),
                     # Noise degrades the closed-loop arm only, so recording
                     # quality has to be comparable between arms before the
                     # contrast means anything.
                     "mean_railed_pct": float(d["railed_pct"].mean())
                     if "railed_pct" in d else float("nan"),
                     # A policy driven by noise, or saturated against a rail,
                     # stops varying its command. Near-zero spread is the
                     # signature of an agent that is not really controlling.
                     "freq_sd_hz": float(d["act_freq_hz"].std())})
    return pd.DataFrame(rows)


def _block_length(series: Sequence[np.ndarray]) -> int:
    """
    Resampling block length, from how far the reward stays correlated.

    Features come from a 4 s Welch window advanced 0.25 s at a time, so
    sixteen consecutive ticks share most of their samples before any real
    dynamics are considered. A block has to be long compared with that or the
    draws are near-copies of each other and the interval comes out too narrow.
    Sized as the lag where autocorrelation first falls below 1/e, doubled.
    """
    lags = []
    for x in series:
        x = np.asarray(x, dtype=np.float64)
        if x.size < 64:
            continue
        x = x - x.mean()
        denom = float(x @ x)
        if denom <= 0:
            continue
        n = min(x.size // 2, 400)
        ac = np.array([float(x[k:] @ x[:x.size - k]) / denom
                       for k in range(1, n)])
        below = np.flatnonzero(ac < np.exp(-1.0))
        lags.append(int(below[0]) + 1 if below.size else n)
    if not lags:
        return 40
    return int(min(400, max(40, 2 * int(np.median(lags)))))


def arm_contrast(sess: Session, n_boot: int = 2000, seed: int = 0) -> dict:
    """
    Ramp minus agent, with the position balance that makes it interpretable.

    Averaging each arm over its own segments is only unbiased because the
    positions are balanced - see :func:`app.protocol.spec.check_arm_balance`.
    The interval comes from resampling *contiguous blocks* of ticks: consecutive
    ticks are strongly autocorrelated, so treating them as independent would
    produce an interval several times too narrow.
    """
    tab = arm_summary(sess)
    if tab.empty or set(tab["arm"]) != {"ramp", "agent"}:
        return {"ok": False, "reason": "both arms not present in the recording"}

    pos = {a: sorted(g["position"]) for a, g in tab.groupby("arm")}
    balanced = (len(set(len(v) for v in pos.values())) == 1
                and len(set(sum(v) for v in pos.values())) == 1)

    means = tab.groupby("arm")["mean_reward"].mean()
    point = float(means["ramp"] - means["agent"])

    rng = np.random.default_rng(seed)
    tk = sess.ticks
    per_seg = {r["block"]: tk[tk["block"] == r["block"]]["reward"].to_numpy()
               for _, r in tab.iterrows()}
    block_len = _block_length(list(per_seg.values()))
    draws = np.empty(n_boot)
    for i in range(n_boot):
        vals = {}
        for _, r in tab.iterrows():
            x = per_seg[r["block"]]
            nb = max(1, len(x) // block_len)
            starts = rng.integers(0, max(1, len(x) - block_len), size=nb)
            samp = np.concatenate([x[s:s + block_len] for s in starts]) if len(x) > block_len else x
            vals.setdefault(r["arm"], []).append(float(np.mean(samp)))
        draws[i] = np.mean(vals["ramp"]) - np.mean(vals["agent"])

    # Segments per arm, and how far apart they were. The bootstrap resamples
    # ticks inside each segment but never resamples which segments were drawn,
    # and the point estimate is a mean of segment means - so between-segment
    # variation, the dominant term at arm level, is absent from the interval.
    # With two segments per arm it cannot be estimated at all. One night is
    # therefore descriptive; the interval earns its inferential reading once
    # the ABBA and BAAB nights are pooled and each arm has four segments.
    n_per_arm = {a: int(len(g)) for a, g in tab.groupby("arm")}
    seg_spread = {a: (float(g["mean_reward"].max() - g["mean_reward"].min())
                      if len(g) > 1 else float("nan"))
                  for a, g in tab.groupby("arm")}
    enough_segments = all(n >= 3 for n in n_per_arm.values())

    dose = tab.groupby("arm")["mean_dose"].mean()
    railed = tab.groupby("arm")["mean_railed_pct"].mean()
    freq_sd = tab.groupby("arm")["freq_sd_hz"].mean()
    # Pre-registered fairness check: if the agent's segments were noisier than
    # the ramp's, it was handicapped by the recording rather than beaten, and
    # the contrast carries no directional claim.
    railed_gap = float(railed.get("agent", float("nan"))
                       - railed.get("ramp", float("nan")))
    comparable = bool(np.isfinite(railed_gap) and abs(railed_gap) <= 5.0)
    # Reported as a verdict, not a number to eyeball: a policy that stopped
    # controlling emits a near-constant command, and the contrast then
    # compares a ramp against a fixed tone rather than against a controller.
    # It does not catch the opposite failure - a policy wandering on noise
    # scores a high spread - which is what the replay diagnostic is for.
    agent_sd = float(freq_sd.get("agent", float("nan")))
    return {
        "ok": True,
        "quality_comparable": comparable,
        "mean_railed_pct": {k: float(v) for k, v in railed.items()},
        "railed_gap_agent_minus_ramp": railed_gap,
        "freq_sd_hz": {k: float(v) for k, v in freq_sd.items()},
        "agent_controlling": bool(np.isfinite(agent_sd)
                                  and agent_sd >= MIN_FREQ_SD_HZ),
        "min_freq_sd_hz": MIN_FREQ_SD_HZ,
        "n_segments_per_arm": n_per_arm,
        "segment_spread": seg_spread,
        "bootstrap_block_len": int(block_len),
        # False on a single night: the interval covers tick-level noise only.
        "interval_supports_inference": bool(enough_segments),
        "balanced": bool(balanced),
        "positions": pos,
        "ramp_minus_agent": point,
        "ci_lo": float(np.percentile(draws, 2.5)),
        "ci_hi": float(np.percentile(draws, 97.5)),
        "mean_reward": {k: float(v) for k, v in means.items()},
        "mean_dose": {k: float(v) for k, v in dose.items()},
        "dose_ratio_agent_over_ramp": (float(dose["agent"] / dose["ramp"])
                                       if dose.get("ramp", 0) else float("nan")),
        "n_boot": n_boot,
    }


def resonance_regression(sess: Session, seed: int = 0) -> dict:
    """
    Peak frequency against arousal, controlling for a monotonic drift.

    Arousal is non-monotonic within the session because of the re-alerting
    block, while electrode drift is monotonic in time. Fitting both terms is
    what separates them; a two-point contrast cannot, and charges the drift to
    arousal.
    """
    blocks = sweep_blocks(sess)
    tab = peak_table(sess, blocks, seed=seed)
    tk = sess.ticks
    rows = []
    for _, r in tab.iterrows():
        if not np.isfinite(r["peak_hz"]):
            continue
        d = tk[tk["block"] == r["block"]]
        if d.empty:
            continue
        rows.append((float(d["theta_alpha_ratio"].mean()),
                     float(d["t_s"].mean()) / 60.0, float(r["peak_hz"])))
    if len(rows) < 4:
        return {"ok": False, "reason": f"{len(rows)} usable sweeps, need >= 4",
                "n_sweeps": len(rows)}

    arousal, minutes, peaks = map(np.array, zip(*rows))
    X = np.column_stack([np.ones(len(peaks)), arousal, minutes])
    beta, *_ = np.linalg.lstsq(X, peaks, rcond=None)
    resid = peaks - X @ beta
    dof = len(peaks) - X.shape[1]
    se = (np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * (resid @ resid) / dof)
          if dof > 0 else np.full(3, np.nan))
    return {
        "ok": True,
        "hz_per_arousal": float(beta[1]), "hz_per_arousal_se": float(se[1]),
        "hz_per_minute_drift": float(beta[2]), "hz_per_minute_drift_se": float(se[2]),
        "n_sweeps": len(peaks), "dof": int(dof),
        "peaks_resolvable": bool(tab["resolvable"].all())
        if "resolvable" in tab else True,
        "arousal_range": [float(arousal.min()), float(arousal.max())],
    }


# ---- rehearsal verification (simulation only) ----------------------------

def peak_recovery(sess: Session, seed: int = 0):
    """
    Fitted peak against the value the simulator actually used, per sweep.

    Only meaningful on a simulated recording, where the plant's resonance is
    knowable; on hardware the ground-truth columns are blank and this returns an
    empty table. Running it on a full-length rehearsal is what turns "the
    pipeline executes" into "the fit recovers the right answer, with this bias
    and this interval coverage".
    """
    import pandas as pd
    # One shape whichever path is taken: a caller indexing error_hz must not
    # have to care whether there was no ground truth or no usable sweep.
    columns = ["block", "fitted_hz", "true_hz", "true_sd_hz", "error_hz",
               "ci_lo", "ci_hi", "target_stable", "covered", "resolvable",
               "n_probes", "n_skipped"]
    tk = sess.ticks
    if tk is None or "true_resonance_hz" not in tk or tk["true_resonance_hz"].isna().all():
        return pd.DataFrame(columns=columns)
    rows = []
    for b in sweep_blocks(sess):
        try:
            curve = response_curve(sess, b)
            boot = bootstrap_peak(curve, seed=seed)
        except Exception as err:
            logger.warning("sweep %s: %s", b, err)
            continue
        d = tk[tk["block"] == b]
        if d.empty or not np.isfinite(boot["peak_hz"]):
            continue
        # The peak moves within a sweep, and in simulation it moves a lot: each
        # probe drives theta/alpha, which is what the plant's resonance tracks,
        # so sweeping the probe moves the very thing being measured. Where that
        # spread is large the block mean is not a target the fit could hit, and
        # comparing against it says nothing about the estimator.
        true_hz = float(d["true_resonance_hz"].mean())
        true_sd = float(d["true_resonance_hz"].std())
        rows.append({
            "block": b,
            "fitted_hz": boot["peak_hz"],
            "true_hz": true_hz,
            "error_hz": boot["peak_hz"] - true_hz,
            "ci_lo": boot["lo"], "ci_hi": boot["hi"],
            "true_sd_hz": true_sd,
            # A target that itself wanders further than the probe spacing is not
            # a fixed quantity, and neither hitting nor missing it means much.
            "target_stable": bool(np.isfinite(true_sd)
                                  and true_sd <= 0.5 * boot.get("spacing_hz", np.inf)),
            "covered": bool(boot["lo"] <= true_hz <= boot["hi"]),
            "resolvable": boot.get("resolvable", True),
            "n_probes": len(curve.probes),
            "n_skipped": len(curve.skipped),
        })
    return pd.DataFrame(rows, columns=columns)


def recovery_summary(sess: Session, seed: int = 0) -> dict:
    """Bias and interval coverage over a rehearsal's sweeps."""
    tab = peak_recovery(sess, seed=seed)
    if tab.empty:
        # Two very different reasons produce an empty table, and saying the
        # wrong one sends the reader looking in the wrong place: a hardware
        # recording has no ground truth to compare against, while a simulated
        # one whose sweeps were too short has ground truth and nothing to
        # compare it with.
        tk = sess.ticks
        has_truth = (tk is not None and "true_resonance_hz" in tk
                     and not tk["true_resonance_hz"].isna().all())
        if not has_truth:
            return {"ok": False, "reason": "no ground truth in this recording "
                                           "(hardware sessions have none)"}
        return {"ok": False,
                "reason": f"ground truth present but no sweep yielded a usable "
                          f"peak ({len(sweep_blocks(sess))} sweeps); the sweeps "
                          f"were too short to fit"}
    return {
        "ok": True,
        "n_sweeps": int(len(tab)),
        "mean_error_hz": float(tab["error_hz"].mean()),
        "mean_abs_error_hz": float(tab["error_hz"].abs().mean()),
        "coverage": float(tab["covered"].mean()),
        "mean_ci_width_hz": float((tab["ci_hi"] - tab["ci_lo"]).mean()),
        "n_unresolvable": int((~tab["resolvable"].astype(bool)).sum()),
        "n_unstable_target": int((~tab["target_stable"].astype(bool)).sum()),
        "mean_true_sd_hz": float(tab["true_sd_hz"].mean()),
        "true_range_hz": [float(tab["true_hz"].min()), float(tab["true_hz"].max())],
    }


def drift_recovery(sess: Session, seed: int = 0) -> dict:
    """
    Does the regression separate arousal from time-drift on known ground truth?

    The simulator's peak moves with drowsiness and not with elapsed time, so a
    correctly-specified fit should load on arousal and leave the drift term near
    zero. Validated against synthetic data before; this checks it on a recording
    the pipeline actually produced.
    """
    reg = resonance_regression(sess, seed=seed)
    if not reg["ok"]:
        return {"ok": False, "reason": reg["reason"]}
    tk = sess.ticks
    if tk is None or "true_resonance_hz" not in tk or tk["true_resonance_hz"].isna().all():
        return {"ok": False, "reason": "no ground truth in this recording"}
    d = tk.dropna(subset=["true_resonance_hz", "true_drowsiness"])
    span_hz = float(d["true_resonance_hz"].max() - d["true_resonance_hz"].min())
    span_drowsy = float(d["true_drowsiness"].max() - d["true_drowsiness"].min())
    return {
        "ok": True,
        "fitted_hz_per_arousal": reg["hz_per_arousal"],
        "fitted_hz_per_minute_drift": reg["hz_per_minute_drift"],
        "true_peak_span_hz": span_hz,
        "true_drowsiness_span": span_drowsy,
        # The simulator applies no time-dependent drift, so a large fitted drift
        # term means the regression is absorbing something it should not.
        "drift_should_be_near_zero": True,
        "n_sweeps": reg["n_sweeps"],
    }


def probe_state_coupling(sess: Session, block: str) -> dict:
    """
    Does the probe move the state it is measuring?

    The plant's resonance tracks drowsiness, and each probe drives drowsiness:
    strobing near theta raises theta/alpha, strobing near alpha lowers it. Where
    that coupling is strong the sweep measures a loop rather than a response
    curve - in simulation it is strong enough that the peak crosses its whole
    range inside one sweep.

    A real head is expected to couple far more weakly, because an individual's
    alpha frequency drifts over tens of minutes rather than over the ten seconds
    a probe lasts. That is an assumption, so this measures it: the correlation
    between probe frequency and the drowsiness recorded while that probe was
    delivered. Near zero means the sweep is a passive measurement. Strongly
    negative means low probes are making the subject drowsier as they are
    measured, and the curve is partly self-inflicted.
    """
    tk = sess.ticks
    blk = tk[(tk["block"] == block) & tk["probe_hz"].notna()]
    if blk.empty:
        return {"ok": False, "reason": f"no probe ticks in {block!r}"}
    g = blk.groupby("probe_hz")["theta_alpha_ratio"].mean()
    if g.size < 3:
        return {"ok": False, "reason": f"{g.size} probes, need >= 3"}
    f = g.index.to_numpy(dtype=float)
    y = g.to_numpy(dtype=float)
    if np.std(f) == 0 or np.std(y) == 0:
        return {"ok": False, "reason": "no variation to correlate"}
    r = float(np.corrcoef(f, y)[0, 1])
    out = {"ok": True, "block": block, "r": r, "n_probes": int(g.size),
           "theta_alpha_by_probe": {float(k): float(v) for k, v in g.items()},
           "state_range": float(y.max() - y.min())}
    # A correlation this size means the probe explains a quarter of the state
    # variation across the sweep, which is enough to shape the curve.
    out["coupled"] = bool(abs(r) >= 0.5)
    return out


def sweep_coupling_report(sess: Session):
    """Probe-to-state coupling for every sweep in a session."""
    import pandas as pd
    rows = []
    for b in sweep_blocks(sess):
        c = probe_state_coupling(sess, b)
        rows.append({"block": b, "r": c.get("r", float("nan")),
                     "coupled": c.get("coupled", None),
                     "state_range": c.get("state_range", float("nan")),
                     "note": "" if c.get("ok") else c.get("reason", "")})
    return pd.DataFrame(rows)


# ---- did the policy respond to the EEG at all? ---------------------------

# Below this the agent's commands are no better explained by the EEG it saw
# than by EEG it never saw, which is what a policy driven by noise looks like.
POLICY_RESPONSE_MIN = 0.15


def policy_response(sess: Session, checkpoint: Path, seed: int = 0) -> dict:
    """
    Re-run the checkpoint on the recorded observations, and on scrambled ones.

    ``agent_controlling`` only catches a policy that stopped moving. The
    failure that matters more here is the opposite: fed features far outside
    anything it trained on, a policy wanders across the band and produces a
    perfectly healthy-looking command spread while responding to nothing. The
    two are indistinguishable from the command trace alone.

    Separating them needs a counterfactual, and a recording supplies one. The
    observation is rebuilt from ``ticks.csv`` tick by tick and pushed back
    through the policy twice: once as recorded, and once with the EEG-derived
    elements time-shuffled so they carry the same marginal distribution and no
    relationship to the moment they are paired with. If the commands come out
    the same either way, the EEG was not driving them.

    Returns ``sensitivity`` in [0, 1]: how much of the policy's own command
    range moves when the EEG is scrambled.
    """
    tk = sess.ticks
    need = ["alpha_pwr", "theta_pwr", "beta_pwr", "hypnagogic_theta_pwr",
            "sigma_pwr", "theta_alpha_ratio", "emg_rms", "eog_energy",
            "plv", "smoothed_plv", "delta_theta_alpha",
            "act_freq_hz", "act_fill", "act_brightness"]
    missing = [c for c in need if c not in tk]
    if missing:
        return {"ok": False, "reason": f"ticks.csv lacks {missing}; the full "
                                       f"observation is not present in this recording"}
    d = tk[tk["block"].astype(str).str.contains("agent")]
    if len(d) < 100:
        return {"ok": False, "reason": f"{len(d)} agent ticks, need >= 100"}

    from stable_baselines3 import SAC
    policy = SAC.load(str(checkpoint), device="cpu")

    # Element order is SleepOnsetEnv._observe(): 9 features, plv, the previous
    # action's three knobs, then the smoothed pair. railed_pct was held at zero
    # for every training step, so it is zero here too.
    eeg_cols = ["alpha_pwr", "theta_pwr", "beta_pwr", "hypnagogic_theta_pwr",
                "sigma_pwr", "theta_alpha_ratio", "emg_rms", "eog_energy"]
    feat = d[eeg_cols].to_numpy(dtype=np.float32)
    railed = np.zeros((len(d), 1), dtype=np.float32)
    plv = d[["plv"]].to_numpy(dtype=np.float32)
    last = np.roll(d[["act_freq_hz", "act_fill", "act_brightness"]]
                   .to_numpy(dtype=np.float32), 1, axis=0)
    last[0] = last[1]
    tail = d[["smoothed_plv", "delta_theta_alpha"]].to_numpy(dtype=np.float32)

    def commands(f, p, t):
        obs = np.concatenate([f[:, :8], railed, p, last, t], axis=1)
        act, _ = policy.predict(obs, deterministic=True)
        return np.asarray(act, dtype=np.float64)[:, 0]     # frequency knob

    real = commands(feat, plv, tail)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(8):
        perm = rng.permutation(len(d))
        diffs.append(float(np.mean(np.abs(
            real - commands(feat[perm], plv[perm], tail[perm])))))
    spread = float(np.percentile(real, 95) - np.percentile(real, 5))
    shift = float(np.mean(diffs))
    sensitivity = shift / spread if spread > 1e-6 else 0.0
    return {
        "ok": True,
        "n_ticks": int(len(d)),
        "command_shift_when_eeg_scrambled": round(shift, 4),
        "own_command_range": round(spread, 4),
        "sensitivity": round(sensitivity, 4),
        "threshold": POLICY_RESPONSE_MIN,
        "responded_to_eeg": bool(sensitivity >= POLICY_RESPONSE_MIN),
    }


def driver_phase_model(sess: Session, block: Optional[str] = None) -> dict:
    """
    Which clock the mask's LED driver keeps, measured rather than assumed.

    Reconstructing the delivered envelope needs to know whether the driver's
    phase carries over when the frequency is retuned (a free-running timer) or
    restarts against an absolute clock (the simulator's ``t % period``). The
    firmware is not ours, so this decides it from the recording.

    The two models are identical while the frequency holds still - the phase
    can only diverge at a retune - so this needs a block whose command moves,
    and picks the one that moves most. PLV is measured in short epochs at each
    epoch's own frequency, because a single band centred on a block-wide median
    would filter out most of a sweep.

    Whichever wins, the arm comparison stays fair: the wrong choice lowers PLV
    on both arms together. This buys sensitivity, not validity.
    """
    from app.eeg.plv import plv_strobe_to_eeg
    from app.protocol.runner import DeliveredEnvelope
    from app.agent.safety import StimAction

    tk = sess.ticks
    if tk is None or "act_freq_hz" not in tk:
        return {"ok": False, "reason": "no command history in ticks.csv"}
    if block is None:
        spread = {b: float(g["act_freq_hz"].std())
                  for b, g in tk.groupby("block")
                  if len(g) >= 40 and float(g.get("act_brightness",
                                                  g["act_freq_hz"] * 0).max()) > 0}
        spread = {b: v for b, v in spread.items() if np.isfinite(v)}
        if not spread:
            return {"ok": False, "reason": "no lit block with a moving command"}
        block = max(spread, key=spread.get)
    d = tk[tk["block"] == block]
    if len(d) < 40:
        return {"ok": False, "reason": f"{len(d)} ticks in {block!r}"}
    if float(d["act_freq_hz"].std()) < 0.1:
        return {"ok": False, "block": block,
                "reason": f"{block!r} holds one frequency, so the two models "
                          f"are identical there and the choice cannot matter"}
    try:
        eeg = sess.segment(block, drop_edge_s=0.0)
    except Exception as err:
        return {"ok": False, "reason": str(err)}

    fs = sess.sampling_rate
    tick_s = 1.0 / float(sess.manifest.get("tick_hz", 4.0))
    spt = int(round(fs * tick_s))
    n_ep = max(1, int(round(EPOCH_S / tick_s)))          # ticks per epoch
    out = {}
    for name, continuous in (("phase_continuous", True), ("restarting", False)):
        env = DeliveredEnvelope(fs, capacity_s=len(d) * tick_s + 8.0,
                                phase_continuous=continuous)
        for _, r in d.iterrows():
            env.extend(StimAction(enabled=True,
                                  frequency_hz=float(r["act_freq_hz"]),
                                  fill=float(r.get("act_fill", 0.4)),
                                  brightness=float(r.get("act_brightness", 0.6))),
                       spt)
        n = min(len(eeg), spt * len(d))
        ref = env.latest(n)
        if ref is None:
            out[name] = 0.0
            continue
        seg = np.asarray(eeg[-n:], dtype=np.float64)
        vals = []
        for k in range(0, len(d) - n_ep + 1, n_ep):
            i0, i1 = k * spt, (k + n_ep) * spt
            if i1 > n:
                break
            f_ep = float(d["act_freq_hz"].iloc[k:k + n_ep].mean())
            vals.append(plv_strobe_to_eeg(seg[i0:i1].copy(), ref[i0:i1].copy(),
                                          fs, f_ep))
        out[name] = float(np.mean(vals)) if vals else 0.0
    best = max(out, key=out.get)
    margin = abs(out["phase_continuous"] - out["restarting"])
    return {
        "ok": True, "block": block,
        "command_sd_hz": round(float(d["act_freq_hz"].std()), 3),
        "plv": {k: round(v, 4) for k, v in out.items()},
        "best": best,
        "margin": round(margin, 4),
        # Below this the recording does not separate them. That is a real
        # answer, not a failure: it means the choice is not costing anything.
        "decisive": bool(margin >= 0.05),
    }
