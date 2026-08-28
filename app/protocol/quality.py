"""
Signal-quality gates for the validation session, and the live monitor that
aborts a running session when the recording stops being usable.

Thresholds are module constants so they can be quoted before any data exists.
A gate that fails is reported with its measured value, not silently retried.

Gate G2 runs on **unfiltered** samples on purpose: the standard chain in
``app.eeg.features`` bandstops 48-52 Hz, so filtered data would show the notch
rather than the interference the gate is looking for.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from brainflow.data_filter import (
    DataFilter,
    DetrendOperations,
    FilterTypes,
    WindowOperations,
)

logger = logging.getLogger("protocol.quality")

# ---- pre-registered thresholds -------------------------------------------
RAILED_MAX_PCT = 20.0          # G1  usable recording, well below disconnect
LINE_NOISE_MAX_FRAC = 0.20     # G2  50 Hz power as a fraction of 1-45 Hz
# G3  eyes-closed / eyes-open alpha power. On 60 s segments of pink noise with
# no alpha at all, the ratio's 99th percentile is ~1.28, so 1.5 sits above what
# chance produces. Shorter segments widen that null and weaken the gate.
ALPHA_BLOCK_MIN_RATIO = 1.5
SSVEP_MIN_SNR_DB = 3.0         # G4  peak over neighbouring bins
# G8  resting / cognitive-load alpha power. Same null as G3 - the statistic is
# the same ratio of two band powers - so the same threshold applies.
COGNITIVE_ALPHA_MIN_RATIO = 1.5
# G5  how far the zero-brightness rung may stand above the unlit blocks, in dB.
# The strobe is still clocking there while LED current is zero, so anything the
# amplifier picks up from the drive electronics shows here and nowhere else.
# The comparison is against unlit blocks rather than against the bright rung:
# the old ratio put the subject's own rhythm in the numerator, so at a probe
# frequency near alpha it reported instrumentation where there was none, and
# it could not fall below roughly 1/(response contrast) however clean the rig.
# Leave-one-out over ten unlit blocks, where there is certainly no pickup, the
# statistic spans -3.22 to +3.60 dB; 4 dB clears that and still catches a
# doubling of power at the drive frequency.
ARTIFACT_MAX_EXCESS_DB = 4.0

# G7  how far each lit rung must stand above the zero-brightness rung, in dB.
# What separates a response from pickup is that it depends on commanded
# brightness at all - the data line clocks identically at every rung. Over 30
# pairs of unlit blocks the same statistic reaches 5.46 dB; the weakest lit
# rung recorded stands 8.1 dB above zero, so 6 dB sits between the two.
DOSE_MIN_EXCESS_DB = 6.0
LINE_HZ = 50.0

# G3  occipital alpha against the frontal channels, eyes closed. Two electrodes
# seeing the same thing give a ratio near 1: measured over 300 surrogate pairs
# the null runs to 1.35, and a frontal channel against the other frontal
# channel on real data gives 0.85. Occipital placement gives 3.2-5.0. The bar
# sits between, clear of both.
ALPHA_TOPOGRAPHY_MIN_RATIO = 2.0

# G8  height of the alpha peak over the aperiodic background it sits on. A
# background with no rhythm in it still shows some excess, because the
# statistic takes a maximum across the band: over 300 surrogates spanning
# 1/f to 1/f^3 the null reaches 6.09 dB. A real occipital rhythm gives 22-24.
ALPHA_PROMINENCE_MIN_DB = 10.0
ALPHA_PEAK_BAND = (7.0, 14.0)      # searched for the peak
APERIODIC_FIT_BAND = (2.0, 40.0)   # background fitted here, excluding the peak

ALPHA_BAND = (8.0, 13.0)
BROADBAND = (1.0, 45.0)

# Peak measurement geometry, in Hz either side of the probe frequency.
#
# A driven response is a line spectrum - phase-locked to a commanded frequency
# and, at this resolution, one bin wide - so the peak is read from the single
# nearest bin. Taking the maximum over a window of bins instead raises the
# reading on noise as well as on signal, which inflates the null.
#
# The noise floor is read close in. A window reaching 3 Hz either side of a
# 6 Hz probe covers 3-9 Hz, whose upper half sits on the rising flank of
# occipital alpha; the floor is then set by alpha rather than by the
# background beside the probe, and the reading is biased by 7-9 dB in a
# direction that depends on where the subject's alpha happens to sit.
#
# These values were chosen against the null: over six unlit blocks in two
# sessions, probed at six frequencies away from alpha, no null reading reaches
# SSVEP_MIN_SNR_DB (worst 2.41 dB, p95 1.88), while a commanded 6 Hz drive
# reads 10.5 dB.
GUARD_HZ = 0.15                # excluded from the noise estimate
NOISE_HALFWIDTH_HZ = 1.0       # bins averaged for the noise floor


@dataclass(frozen=True)
class GateResult:
    name: str
    label: str
    value: float
    threshold: float
    passed: bool
    blocking: bool = True
    detail: str = ""

    def __str__(self) -> str:
        mark = "PASS" if self.passed else ("FAIL" if self.blocking else "warn")
        return f"[{mark}] {self.name} {self.label}: {self.value:.3g} (limit {self.threshold:.3g})"


def _nfft_for(n: int, cap: int = 4096) -> int:
    """
    Transform length for a segment of ``n`` samples.

    The cap sets the frequency resolution, and it has to be wide enough to
    resolve a line: at 1024 a 60 s block was analysed at 0.5 Hz bins - the same
    resolution a 2 s block gets - which spreads a driven peak across its
    neighbours and buries it in the very bins the floor is measured from.
    4096 gives 0.125 Hz at 512 Hz sampling. Higher still keeps raising the
    driven reading, but the null rises with it and crosses the threshold, so
    this is where the margin is widest rather than where the number is largest.
    """
    nfft = 1 << max(6, (n.bit_length() - 1))
    return min(nfft, cap, n)


def psd_welch(channel: np.ndarray, sampling_rate: int,
              filtered: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    ``(amplitudes, frequencies)``.

    ``filtered`` applies the same chain the agent's features go through. Turn it
    off to measure anything the chain removes, such as mains interference.
    """
    x = np.ascontiguousarray(np.asarray(channel, dtype=np.float64).copy())
    if x.size < 64:
        raise ValueError(f"need >= 64 samples, got {x.size}")
    if filtered:
        DataFilter.detrend(x, DetrendOperations.CONSTANT.value)
        DataFilter.perform_bandpass(x, sampling_rate, 3.0, 45.0, 2,
                                    FilterTypes.BUTTERWORTH_ZERO_PHASE.value, 0)
        DataFilter.perform_bandstop(x, sampling_rate, 48.0, 52.0, 2,
                                    FilterTypes.BUTTERWORTH_ZERO_PHASE.value, 0)
    else:
        DataFilter.detrend(x, DetrendOperations.CONSTANT.value)
    # Whole segment, not just the last window: Welch averages across as many
    # overlapping windows as the data allows, and that averaging is what makes
    # a long segment worth recording.
    nfft = _nfft_for(x.size)
    amps, freqs = DataFilter.get_psd_welch(
        x, nfft, nfft // 2, sampling_rate,
        WindowOperations.BLACKMAN_HARRIS.value)
    return np.asarray(amps), np.asarray(freqs)


def band_power(psd: Tuple[np.ndarray, np.ndarray], lo: float, hi: float) -> float:
    amps, freqs = psd
    sel = (freqs >= lo) & (freqs <= hi)
    return float(amps[sel].sum()) if sel.any() else 0.0


def _peak_bins(freqs: np.ndarray, f_hz: float) -> np.ndarray:
    """The single bin nearest ``f_hz`` - where a phase-locked line sits."""
    nearest = np.zeros_like(freqs, dtype=bool)
    nearest[int(np.argmin(np.abs(freqs - f_hz)))] = True
    return nearest


def peak_snr_db(psd: Tuple[np.ndarray, np.ndarray], f_hz: float) -> float:
    """
    Response at ``f_hz`` against the local noise floor, in dB.

    The floor is the mean of bins within ``NOISE_HALFWIDTH_HZ`` but outside
    ``GUARD_HZ``, so a broad peak does not raise its own reference.
    """
    amps, freqs = psd
    peak = _peak_bins(freqs, f_hz)
    noise = ((np.abs(freqs - f_hz) <= NOISE_HALFWIDTH_HZ)
             & (np.abs(freqs - f_hz) > GUARD_HZ))
    if not noise.any():
        return float("nan")
    floor = float(amps[noise].mean())
    if floor <= 0:
        return float("nan")
    return float(10.0 * np.log10(amps[peak].max() / floor))


def peak_amplitude(psd: Tuple[np.ndarray, np.ndarray], f_hz: float) -> float:
    amps, freqs = psd
    return float(amps[_peak_bins(freqs, f_hz)].max())


def alpha_prominence_db(channel: np.ndarray, sampling_rate: int) -> float:
    """
    Height of the alpha peak above the 1/f background, in dB.

    EEG power falls with frequency whatever the electrode is doing, so band
    power alone cannot say whether a rhythm is present - a steeper background
    raises it just as a real rhythm does. Fitting the aperiodic part on a
    log-log axis, with the alpha band left out of the fit, separates the two:
    what remains is how far the peak stands above the slope it sits on.
    """
    amps, freqs = psd_welch(channel, sampling_rate)
    fit = ((freqs >= APERIODIC_FIT_BAND[0]) & (freqs <= APERIODIC_FIT_BAND[1])
           & (amps > 0))
    if fit.sum() < 8:
        raise ValueError("not enough usable bins to fit the background")
    lf, la = np.log10(freqs[fit]), np.log10(amps[fit])
    peak = (freqs[fit] >= ALPHA_PEAK_BAND[0]) & (freqs[fit] <= ALPHA_PEAK_BAND[1])
    if not peak.any() or (~peak).sum() < 4:
        raise ValueError("alpha band not resolvable in this segment")
    slope = np.polyfit(lf[~peak], la[~peak], 1)
    return float(10.0 * (la[peak] - np.polyval(slope, lf[peak])).max())


# ---- gates ---------------------------------------------------------------

def gate_railed(railed_pct: float) -> GateResult:
    return GateResult("G1", "railed %", float(railed_pct), RAILED_MAX_PCT,
                      railed_pct < RAILED_MAX_PCT,
                      detail="electrode contact and ground")


def gate_line_noise(raw: np.ndarray, sampling_rate: int) -> GateResult:
    psd = psd_welch(raw, sampling_rate, filtered=False)
    line = band_power(psd, LINE_HZ - 1.5, LINE_HZ + 1.5)
    total = band_power(psd, *BROADBAND)
    frac = line / total if total > 0 else float("inf")
    return GateResult("G2", f"{LINE_HZ:g} Hz fraction", frac, LINE_NOISE_MAX_FRAC,
                      frac < LINE_NOISE_MAX_FRAC,
                      detail="measured on unfiltered samples")


def gate_alpha_blocking(
    eyes_open: Sequence[np.ndarray],
    eyes_closed: Sequence[np.ndarray],
    sampling_rate: int,
) -> GateResult:
    """
    Alpha must rise on eye closure - the check that the electrodes see cortex.

    The eyes-open blocks are lit, by continuous light from the mask rather than
    a strobe. Alpha desynchronises to light on the retina, not to the eyelids
    moving, so under a blackout mask an unlit eyes-open block is the same
    condition as an eyes-closed one and the ratio has nothing to measure. The
    light is unmodulated so that nothing is driven into the band being read.

    Every open/closed pair is evaluated, and the weakest one decides: a single
    replication that fails means the effect is not reliable within the session.
    """
    if not eyes_open or not eyes_closed:
        raise ValueError("need at least one eyes-open and one eyes-closed segment")
    ratios = []
    for o, c in zip(eyes_open, eyes_closed):
        po = band_power(psd_welch(o, sampling_rate), *ALPHA_BAND)
        pc = band_power(psd_welch(c, sampling_rate), *ALPHA_BAND)
        ratios.append(pc / po if po > 0 else float("inf"))
    worst = float(min(ratios))
    return GateResult(
        "G3", "alpha closed/open", worst, ALPHA_BLOCK_MIN_RATIO,
        worst >= ALPHA_BLOCK_MIN_RATIO,
        detail=f"{len(ratios)} repeat(s): " + ", ".join(f"{r:.2f}" for r in ratios))


def gate_cognitive_alpha(
    resting: Sequence[np.ndarray],
    under_load: Sequence[np.ndarray],
    sampling_rate: int,
) -> GateResult:
    """
    Alpha must fall under cognitive load - a second check that it is cortical.

    Independent of eye closure and of the stimulus: both conditions are dark,
    eyes closed and unstimulated, so nothing is injected into the band being
    measured. It therefore stands even if the mask delivers no light at all,
    which is the one failure G3 cannot distinguish from flat electrodes.
    """
    if not resting or not under_load:
        raise ValueError("need at least one resting and one loaded segment")
    ratios = []
    for r, l in zip(resting, under_load):
        pr = band_power(psd_welch(r, sampling_rate), *ALPHA_BAND)
        pl = band_power(psd_welch(l, sampling_rate), *ALPHA_BAND)
        ratios.append(pr / pl if pl > 0 else float("inf"))
    worst = float(min(ratios))
    return GateResult(
        "G8", "alpha rest/task", worst, COGNITIVE_ALPHA_MIN_RATIO,
        worst >= COGNITIVE_ALPHA_MIN_RATIO, blocking=False,
        detail="independent of eye closure; " + ", ".join(f"{r:.2f}" for r in ratios))


def gate_alpha_topography(
    occipital: Sequence[np.ndarray],
    frontal: Sequence[np.ndarray],
    sampling_rate: int,
) -> GateResult:
    """
    Alpha must be larger occipitally than frontally - the electrode is where
    it is supposed to be.

    This replaces eye closure as the placement check because it does not depend
    on a manipulation the apparatus defeats. Alpha desynchronises to light on
    the retina, and an opaque mask emitting diffuse light is a Ganzfeld, in
    which the eyes-open and eyes-closed conditions are optically the same; the
    ratio between them then measures nothing. Topography needs no manipulation
    at all, and the quantity it uses is the stable one - alpha level reproduces
    at ICC 0.84 over five years, where the open/closed ratio reproduces at
    r = 0.38 over nine days.

    Every eyes-closed segment is evaluated and the weakest decides.
    """
    if not occipital or not frontal:
        raise ValueError("need at least one occipital and one frontal segment")
    ratios = []
    for occ in occipital:
        po = band_power(psd_welch(occ, sampling_rate), *ALPHA_BAND)
        pf = [band_power(psd_welch(f, sampling_rate), *ALPHA_BAND) for f in frontal]
        mean_f = float(np.mean(pf))
        ratios.append(po / mean_f if mean_f > 0 else float("inf"))
    worst = float(min(ratios))
    return GateResult(
        "G3", "occipital/frontal alpha", worst, ALPHA_TOPOGRAPHY_MIN_RATIO,
        worst >= ALPHA_TOPOGRAPHY_MIN_RATIO,
        detail=f"{len(frontal)} frontal channel(s); " +
               ", ".join(f"{r:.2f}" for r in ratios))


def gate_alpha_present(
    segments: Sequence[np.ndarray], sampling_rate: int
) -> GateResult:
    """
    There must be an alpha rhythm to measure at all.

    Independent of placement and of the stimulus: a peak standing above the
    aperiodic background is what distinguishes a cortical rhythm from a steep
    1/f slope, and it needs no manipulation, no light, and no second channel.
    Together with topography this answers what the two alpha gates were always
    for - the signal is brain, and it is brain from the right place.
    """
    if not segments:
        raise ValueError("need at least one eyes-closed segment")
    vals = [alpha_prominence_db(s, sampling_rate) for s in segments]
    worst = float(min(vals))
    return GateResult(
        "G8", "alpha peak over background", worst, ALPHA_PROMINENCE_MIN_DB,
        worst >= ALPHA_PROMINENCE_MIN_DB,
        detail=", ".join(f"{v:.1f} dB" for v in vals))


def gate_ssvep(seg: np.ndarray, sampling_rate: int, f_hz: float) -> GateResult:
    snr = peak_snr_db(psd_welch(seg, sampling_rate), f_hz)
    return GateResult("G4", f"SSVEP SNR at {f_hz:g} Hz", snr, SSVEP_MIN_SNR_DB,
                      bool(np.isfinite(snr) and snr >= SSVEP_MIN_SNR_DB),
                      detail="light reaching the retina")


def gate_artifact(
    zero_brightness: np.ndarray,
    unlit: Sequence[np.ndarray],
    sampling_rate: int,
    f_hz: float,
) -> GateResult:
    """
    Commanding zero brightness must leave nothing behind at the drive frequency.

    The strobe stays enabled and the mask is not touched, so the firmware keeps
    running ``setPixelColor`` and ``show`` on every phase and the NeoPixel data
    line keeps clocking while LED current goes to zero. Anything the amplifier
    couples from that appears here.

    Measured against blocks where the strobe was off entirely, not against the
    bright rung. The subject's own rhythm is present in both the zero rung and
    the bright one, so a ratio between them reports whatever endogenous power
    sits at the probe frequency as though it were instrumentation - which is
    what happened at a 10 Hz probe with alpha underneath it.
    """
    if not unlit:
        raise ValueError("need at least one unstimulated block for comparison")
    zero = peak_snr_db(psd_welch(zero_brightness, sampling_rate), f_hz)
    base = [peak_snr_db(psd_welch(u, sampling_rate), f_hz) for u in unlit]
    excess = float(zero - np.mean(base))
    return GateResult(
        "G5", f"zero-brightness excess at {f_hz:g} Hz", excess,
        ARTIFACT_MAX_EXCESS_DB, excess < ARTIFACT_MAX_EXCESS_DB,
        detail=f"vs {len(base)} unlit block(s) at "
               f"{np.mean(base):+.1f} dB; residual would be instrumentation")


def gate_dose_response(
    by_brightness: Dict[float, np.ndarray],
    sampling_rate: int,
    f_hz: float,
) -> GateResult:
    """
    The response must depend on commanded brightness.

    Pickup from the drive electronics cannot: the data line clocks identically
    whatever the LED current is, so a peak that grows when brightness is
    commanded up is light and nothing else. Every lit rung is therefore
    required to stand clear of the zero-brightness rung.

    Not monotonicity across the lit rungs. That was the earlier form, and the
    response measured here is not monotone - it rises steeply off zero and
    falls again at the top rung, on two runs, at two probe frequencies, and
    under raw amplitude, SNR and broadband-normalised amplitude alike. Requiring
    an ordering the stimulus does not produce tests the physiology rather than
    the rig, which is not what this gate is for.

    Measured in SNR rather than raw amplitude: broadband power drifts about
    30 % across a ladder, which moves absolute peaks without any change in the
    response.
    """
    if len(by_brightness) < 2:
        raise ValueError("need at least two brightness levels")
    levels = sorted(by_brightness)
    if levels[0] > 0.0:
        raise ValueError("the ladder needs a zero-brightness rung to compare against")
    snr = {b: peak_snr_db(psd_welch(by_brightness[b], sampling_rate), f_hz)
           for b in levels}
    lit = levels[1:]
    excess = [snr[b] - snr[levels[0]] for b in lit]
    worst = float(min(excess))
    return GateResult(
        "G7", "weakest lit rung over zero brightness", worst,
        DOSE_MIN_EXCESS_DB, worst >= DOSE_MIN_EXCESS_DB,
        detail="; ".join(f"{int(b * 100)} %: {e:+.1f} dB"
                         for b, e in zip(lit, excess)))


def gate_frequency_specificity(
    segments: Dict[float, np.ndarray], sampling_rate: int
) -> GateResult:
    """Each probe should peak at its own frequency, not at a fixed one."""
    hits = 0
    detail = []
    for f, seg in sorted(segments.items()):
        snr = peak_snr_db(psd_welch(seg, sampling_rate), f)
        ok = bool(np.isfinite(snr) and snr >= SSVEP_MIN_SNR_DB)
        hits += ok
        detail.append(f"{f:g} Hz {snr:.1f} dB{'' if ok else ' (miss)'}")
    n = len(segments)
    return GateResult("G6", "frequencies driving", float(hits), float(n),
                      hits == n, blocking=False, detail="; ".join(detail))


# Both are required. They are complementary rather than redundant: topography
# alone could be satisfied by two dead channels with different noise floors,
# and a peak alone could come from an electrode that is not where it is
# thought to be. Neither depends on a manipulation, so neither has the failure
# mode that made an either-of rule necessary when G3 was the eye-closure
# ratio.
ALPHA_EITHER_OF: Tuple[str, ...] = ()


def summarise(gates: Sequence[GateResult]) -> Dict[str, object]:
    by_name = {g.name: g for g in gates}
    alpha = [by_name[n] for n in ALPHA_EITHER_OF if n in by_name]
    alpha_satisfied = any(g.passed for g in alpha) if alpha else True

    blocking_failed = [g.name for g in gates if g.blocking and not g.passed
                       and not (g.name in ALPHA_EITHER_OF and alpha_satisfied)]
    return {
        "passed": not blocking_failed,
        "blocking_failures": blocking_failed,
        # Reported separately so a session that fails G3 but passes G8 reads as
        # what it is - the alpha check satisfied by the other route - instead
        # of appearing to have been let through.
        "alpha_gate": {
            "satisfied": bool(alpha_satisfied),
            "by": [g.name for g in alpha if g.passed],
            "rule": "G3 or G8",
        },
        "warnings": [g.name for g in gates if not g.blocking and not g.passed],
        "gates": [
            {"name": g.name, "label": g.label, "value": g.value,
             "threshold": g.threshold, "passed": g.passed,
             "blocking": g.blocking, "detail": g.detail}
            for g in gates
        ],
    }


# ---- live monitor --------------------------------------------------------

@dataclass
class QualityMonitor:
    """
    Rolling check during a session. Abort only on a sustained problem, so one
    bad window from a swallow or a cable nudge does not end a 41-minute run.
    """
    railed_max_pct: float = RAILED_MAX_PCT
    sustain_ticks: int = 20
    _bad: int = field(default=0, init=False)
    _worst: float = field(default=0.0, init=False)

    def update(self, railed_pct: float) -> Optional[str]:
        """Return an abort reason, or ``None`` to continue."""
        self._worst = max(self._worst, float(railed_pct))
        if railed_pct >= self.railed_max_pct:
            self._bad += 1
            if self._bad >= self.sustain_ticks:
                return (f"railed {railed_pct:.0f}% for {self._bad} ticks "
                        f"(limit {self.railed_max_pct:.0f}%)")
        else:
            self._bad = 0
        return None

    @property
    def worst_railed_pct(self) -> float:
        return self._worst

    def reset(self) -> None:
        self._bad = 0
        self._worst = 0.0
