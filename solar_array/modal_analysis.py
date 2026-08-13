"""Modal identification from event-derived displacement series.

Given the displacement time series produced by :mod:`event_vibrometer`, recover the
modal frequencies, damping ratios and operational deflection shapes of the array.

Why damping is not taken from the spectrum
------------------------------------------
The half-power method needs the -3 dB bandwidth to be resolvable. For a mode at
0.082 Hz with a damping ratio of 0.35% that bandwidth is

    df = 2 * zeta * f_n = 2 * 0.0035 * 0.082 = 5.7e-4 Hz

which would demand a Welch segment longer than half an hour. It is not obtainable
from a terminator-crossing observation. The logarithmic decrement, applied to the
free decay that follows the thermal snap, needs no frequency resolution at all --
only enough cycles to see the envelope fall. That is why damping here is estimated
in the time domain, with the half-power figure reported alongside purely as a
resolution-limited cross-check.

Everything is implemented directly on NumPy: Welch periodogram averaging, a
prominence-based peak picker, an FFT band-pass, an FFT Hilbert transform for the
analytic envelope, and a single-frequency DFT for the deflection shapes.

NumPy only -- no SDK, no scipy. Self-test::

    python modal_analysis.py
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Spectral primitives
# --------------------------------------------------------------------------- #

def welch_psd(
    x: np.ndarray,
    fs: float,
    *,
    segment_s: float = 300.0,
    overlap: float = 0.5,
    detrend: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """One-sided power spectral density by Welch's method of averaged periodograms.

    Returns ``(freqs_hz, psd)`` where ``psd`` has units of ``x**2 / Hz``.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    nperseg = min(n, max(16, int(round(segment_s * fs))))
    noverlap = int(nperseg * np.clip(overlap, 0.0, 0.95))
    step = max(1, nperseg - noverlap)

    window = np.hanning(nperseg)
    norm = fs * np.sum(window ** 2)
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)

    acc = np.zeros(freqs.size, dtype=np.float64)
    count = 0
    for start in range(0, n - nperseg + 1, step):
        seg = x[start:start + nperseg]
        if detrend:
            idx = np.arange(nperseg, dtype=np.float64)
            slope, offset = np.polyfit(idx, seg, 1)
            seg = seg - (slope * idx + offset)
        spec = np.fft.rfft(seg * window)
        acc += (np.abs(spec) ** 2) / norm
        count += 1

    if count == 0:
        return freqs, acc
    psd = acc / count
    # Fold negative frequencies into the one-sided estimate.
    if psd.size > 2:
        psd[1:-1] *= 2.0
    return freqs, psd


def find_spectral_peaks(
    freqs: np.ndarray,
    psd: np.ndarray,
    *,
    band_hz: Tuple[float, float] = (0.02, 5.0),
    min_prominence_db: float = 6.0,
    min_snr_db: float = 12.0,
    min_separation_hz: float = 0.02,
    max_peaks: int = 6,
    floor_window_bins: int = 101,
) -> List[int]:
    """Prominence-based peak picking on the log spectrum.

    Two criteria must both hold, because either alone admits false modes:

    * **Prominence** -- the drop from a peak to the highest saddle separating it
      from any taller peak. This rejects shoulders on the skirt of a strong mode.
    * **Local signal-to-noise** -- the rise above a running median of the log
      spectrum. This rejects noise peaks, which in a Welch estimate averaged over
      only a handful of segments are routinely prominent by 6-10 dB purely by
      chance. Genuine lightly damped modes stand tens of dB above the floor.
    """
    lo, hi = band_hz
    in_band = (freqs >= lo) & (freqs <= hi)
    if not in_band.any():
        return []

    db = 10.0 * np.log10(np.maximum(psd, 1e-30))
    idx_band = np.nonzero(in_band)[0]
    lo_i, hi_i = idx_band[0], idx_band[-1]

    # Running median as a robust local noise floor.
    half_w = max(1, int(floor_window_bins) // 2)
    padded = np.pad(db, half_w, mode="edge")
    floor_db = np.empty_like(db)
    for i in range(db.size):
        floor_db[i] = np.median(padded[i:i + 2 * half_w + 1])

    candidates = []
    for i in range(max(lo_i, 1), min(hi_i, db.size - 1)):
        if db[i] > db[i - 1] and db[i] >= db[i + 1]:
            candidates.append(i)
    if not candidates:
        return []

    scored = []
    for i in candidates:
        if db[i] - floor_db[i] < min_snr_db:
            continue
        left_min = db[i]
        j = i - 1
        while j >= lo_i and db[j] <= db[i]:
            left_min = min(left_min, db[j])
            j -= 1
        right_min = db[i]
        k = i + 1
        while k <= hi_i and db[k] <= db[i]:
            right_min = min(right_min, db[k])
            k += 1
        prominence = db[i] - max(left_min, right_min)
        if prominence >= min_prominence_db:
            scored.append((db[i], prominence, i))

    scored.sort(reverse=True)
    kept: List[int] = []
    for _, _, i in scored:
        if all(abs(freqs[i] - freqs[j]) >= min_separation_hz for j in kept):
            kept.append(i)
        if len(kept) >= max_peaks:
            break
    return sorted(kept)


def interpolate_peak_hz(freqs: np.ndarray, psd: np.ndarray, index: int) -> float:
    """Sub-bin peak frequency by parabolic interpolation on the log spectrum."""
    if index <= 0 or index >= psd.size - 1:
        return float(freqs[index])
    y0, y1, y2 = (10.0 * np.log10(np.maximum(psd[index - 1:index + 2], 1e-30)))
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-12:
        return float(freqs[index])
    delta = 0.5 * (y0 - y2) / denom
    delta = float(np.clip(delta, -0.5, 0.5))
    return float(freqs[index] + delta * (freqs[1] - freqs[0]))


#: Half-power width of a Hann window's main lobe, in frequency bins. A measured
#: bandwidth at or below this is the window, not the structure.
HANN_HALF_POWER_BINS = 1.44


def half_power_damping(
    freqs: np.ndarray,
    psd: np.ndarray,
    index: int,
    *,
    min_resolvable_bins: float = 4.0,
) -> Tuple[float, bool]:
    """Damping ratio from the -3 dB bandwidth.

    Returns ``(zeta, resolved)``. ``resolved`` is False when the measured bandwidth
    is not comfortably wider than the analysis window's own main lobe
    (``HANN_HALF_POWER_BINS`` wide), in which case the figure describes the
    spectrum estimator rather than the structure and must not be reported as a
    damping measurement.
    """
    peak = psd[index]
    half = peak / 2.0
    i = index
    while i > 0 and psd[i] > half:
        i -= 1
    j = index
    while j < psd.size - 1 and psd[j] > half:
        j += 1
    bin_hz = float(freqs[1] - freqs[0]) if freqs.size > 1 else 0.0
    bandwidth = float(freqs[j] - freqs[i])
    f_n = float(freqs[index])
    if f_n <= 0.0 or bin_hz <= 0.0:
        return float("nan"), False
    resolved = bandwidth > max(min_resolvable_bins, HANN_HALF_POWER_BINS) * bin_hz
    return bandwidth / (2.0 * f_n), bool(resolved)


# --------------------------------------------------------------------------- #
# Time-domain primitives
# --------------------------------------------------------------------------- #

def bandpass(x: np.ndarray, fs: float, low_hz: float, high_hz: float) -> np.ndarray:
    """Zero-phase FFT band-pass with a raised-cosine transition.

    Acceptable here because the analysis is offline over a complete record, and the
    zero phase is required for the deflection-shape phases to mean anything.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    width = max((high_hz - low_hz) * 0.25, freqs[1] if freqs.size > 1 else 1e-6)
    gain = np.zeros_like(freqs)
    gain[(freqs >= low_hz) & (freqs <= high_hz)] = 1.0
    lo_edge = (freqs < low_hz) & (freqs > low_hz - width)
    hi_edge = (freqs > high_hz) & (freqs < high_hz + width)
    gain[lo_edge] = 0.5 * (1.0 + np.cos(math.pi * (low_hz - freqs[lo_edge]) / width))
    gain[hi_edge] = 0.5 * (1.0 + np.cos(math.pi * (freqs[hi_edge] - high_hz) / width))
    return np.fft.irfft(spec * gain, n=n)


def analytic_envelope(x: np.ndarray) -> np.ndarray:
    """Envelope via the FFT Hilbert transform."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    spec = np.fft.fft(x)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1:(n + 1) // 2] = 2.0
    return np.abs(np.fft.ifft(spec * h))


def log_decrement_damping(
    t_s: np.ndarray,
    x: np.ndarray,
    f_n: float,
    *,
    bandwidth_ratio: float = 0.35,
    floor_ratio: float = 0.12,
) -> Tuple[float, float]:
    """Damping ratio from the free-decay envelope.

    The signal is band-passed around ``f_n``, its analytic envelope taken, and
    ``ln(envelope)`` regressed against time. For a single decaying mode the slope
    is ``-zeta * omega_n``.

    Samples where the envelope has fallen to ``floor_ratio`` of its peak are
    excluded, because below that the residual ambient excitation, not the decay,
    sets the level. Returns ``(zeta, r_squared)``.
    """
    t_s = np.asarray(t_s, dtype=np.float64)
    if f_n <= 0.0 or t_s.size < 8:
        return float("nan"), 0.0

    fs = 1.0 / float(np.median(np.diff(t_s)))
    half = max(f_n * bandwidth_ratio, 1e-4)
    env = analytic_envelope(bandpass(x, fs, f_n - half, f_n + half))

    peak_i = int(np.argmax(env))
    env = env[peak_i:]
    t = t_s[peak_i:] - t_s[peak_i]
    if env.size < 8 or env[0] <= 0.0:
        return float("nan"), 0.0

    keep = env > floor_ratio * env[0]
    # Use only the leading contiguous run of the decay.
    if not keep[0]:
        return float("nan"), 0.0
    end = int(np.argmin(keep)) if (~keep).any() else keep.size
    if end < 8:
        return float("nan"), 0.0

    t, y = t[:end], np.log(env[:end])
    slope, offset = np.polyfit(t, y, 1)
    pred = slope * t + offset
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    zeta = -slope / (2.0 * math.pi * f_n)
    return float(zeta), float(r2)


def single_frequency_dft(t_s: np.ndarray, x: np.ndarray, f_hz: float) -> complex:
    """Complex amplitude of ``x`` at exactly ``f_hz``, Hann-weighted."""
    t = np.asarray(t_s, dtype=np.float64)
    w = np.hanning(t.size)
    phasor = np.exp(-2j * math.pi * f_hz * (t - t[0]))
    return complex(2.0 * np.sum(x * w * phasor) / np.sum(w))


def modal_assurance_criterion(a: np.ndarray, b: np.ndarray) -> float:
    """MAC between two mode-shape vectors; 1 means collinear, 0 orthogonal."""
    a = np.asarray(a, dtype=np.complex128).ravel()
    b = np.asarray(b, dtype=np.complex128).ravel()
    num = abs(np.vdot(a, b)) ** 2
    den = float(np.real(np.vdot(a, a) * np.vdot(b, b)))
    return float(num / den) if den > 0 else 0.0


# --------------------------------------------------------------------------- #
# Identification
# --------------------------------------------------------------------------- #

@dataclass
class IdentifiedMode:
    """One identified structural mode."""

    freq_hz: float
    damping_ratio: float
    damping_r2: float
    half_power_zeta: float
    half_power_resolved: bool
    amplitude_px: float
    prominence_db: float
    shape: np.ndarray = field(default_factory=lambda: np.zeros(0))       # complex
    shape_real: np.ndarray = field(default_factory=lambda: np.zeros(0))  # normalised

    def to_record(self, gauge_names: Optional[Sequence[str]] = None) -> Dict[str, object]:
        rec: Dict[str, object] = {
            "freq_hz": round(self.freq_hz, 6),
            "damping_ratio": (None if math.isnan(self.damping_ratio)
                              else round(self.damping_ratio, 6)),
            "damping_r2": round(self.damping_r2, 4),
            "half_power_zeta": (None if math.isnan(self.half_power_zeta)
                                else round(self.half_power_zeta, 6)),
            "half_power_resolved": self.half_power_resolved,
            "amplitude_px": round(self.amplitude_px, 5),
            "prominence_db": round(self.prominence_db, 2),
            "shape_real": [round(float(v), 5) for v in self.shape_real],
        }
        if gauge_names is not None:
            rec["gauges"] = list(gauge_names)
        return rec


class ModalIdentifier:
    """Identifies modes from a multi-gauge displacement record."""

    def __init__(
        self,
        *,
        band_hz: Tuple[float, float] = (0.02, 5.0),
        segment_s: float = 300.0,
        overlap: float = 0.5,
        min_prominence_db: float = 6.0,
        min_snr_db: float = 12.0,
        min_separation_hz: float = 0.02,
        harmonic_tol_hz: float = 0.02,
        harmonic_max_gauges: int = 1,
        max_modes: int = 6,
    ) -> None:
        self.band_hz = band_hz
        self.segment_s = float(segment_s)
        self.overlap = float(overlap)
        self.min_prominence_db = float(min_prominence_db)
        self.min_snr_db = float(min_snr_db)
        self.min_separation_hz = float(min_separation_hz)
        self.harmonic_tol_hz = float(harmonic_tol_hz)
        self.harmonic_max_gauges = int(harmonic_max_gauges)
        self.max_modes = int(max_modes)

    def identify(
        self,
        t_s: np.ndarray,
        displacement: np.ndarray,
        *,
        warmup_s: float = 0.0,
        healthy: Optional[np.ndarray] = None,
        decay_start_s: Optional[float] = None,
    ) -> List[IdentifiedMode]:
        """Identify modes from ``displacement`` shaped ``(n_gauges, n_samples)``.

        ``warmup_s`` discards the drift filter's settling transient.
        ``decay_start_s`` marks the thermal snap; the damping fit uses only data
        after it, since that is the free-decay interval.
        """
        t_s = np.asarray(t_s, dtype=np.float64)
        disp = np.atleast_2d(np.asarray(displacement, dtype=np.float64))
        if t_s.size < 16 or disp.shape[1] != t_s.size:
            return []

        use = t_s >= (t_s[0] + warmup_s)
        if use.sum() < 16:
            return []
        t = t_s[use]
        d = disp[:, use]
        if healthy is not None and np.any(healthy):
            rows = np.asarray(healthy, dtype=bool)
        else:
            rows = np.ones(d.shape[0], dtype=bool)

        fs = 1.0 / float(np.median(np.diff(t)))

        # Detect peaks in each healthy gauge separately, then pool by frequency.
        # Averaging the gauge spectra is the wrong operator here: the fundamental
        # dominates every gauge by tens of dB, and in the averaged spectrum its
        # spectral leakage buries the weaker higher modes. A higher mode with an
        # antinode at even one gauge, by contrast, is easily found in that gauge's
        # own spectrum.
        gauge_indices = list(np.nonzero(rows)[0])
        per_gauge_psd: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        detections: List[Tuple[float, float, int]] = []   # (freq, prominence_db, gauge)
        freqs = np.zeros(0)
        psd_sum = None
        for i in gauge_indices:
            freqs, psd = welch_psd(d[i], fs, segment_s=self.segment_s,
                                   overlap=self.overlap)
            per_gauge_psd[i] = (freqs, psd)
            psd_sum = psd if psd_sum is None else psd_sum + psd
            db = 10.0 * np.log10(np.maximum(psd, 1e-30))
            for idx in find_spectral_peaks(
                    freqs, psd, band_hz=self.band_hz,
                    min_prominence_db=self.min_prominence_db,
                    min_snr_db=self.min_snr_db,
                    min_separation_hz=self.min_separation_hz,
                    max_peaks=self.max_modes):
                f_hz = interpolate_peak_hz(freqs, psd, idx)
                detections.append((f_hz, float(db[idx] - np.median(db)), i))

        if psd_sum is None or freqs.size == 0 or not detections:
            return []
        psd_avg = psd_sum / max(len(gauge_indices), 1)

        # Cluster detections that agree in frequency to within the peak separation,
        # keep the strongest, and require corroboration is not needed -- a single
        # confident gauge is enough for a mode with a localised antinode.
        detections.sort()
        clusters: List[List[Tuple[float, float, int]]] = []
        for det in detections:
            if clusters and det[0] - clusters[-1][-1][0] <= self.min_separation_hz:
                clusters[-1].append(det)
            else:
                clusters.append([det])

        # (freq, pooled strength, corroborating gauges) per cluster.
        cand: List[Tuple[float, float, int]] = []
        for cluster in clusters:
            weights = np.array([max(d_[1], 1e-3) for d_ in cluster])
            f_hz = float(np.average([d_[0] for d_ in cluster], weights=weights))
            cand.append((f_hz, float(weights.sum()), len({d_[2] for d_ in cluster})))

        # Reject peaks below the analysis band (residual illumination ramp).
        cand = [c for c in cand if c[0] >= self.band_hz[0]]

        # Reject 2nd/3rd-order harmonic artefacts. A true structural mode that
        # happens to lie near an integer multiple of a lower mode is corroborated
        # across several gauges (its mode shape has antinodes at multiple
        # stations); a leakage/saturation harmonic is confined to the one gauge
        # where the fundamental is strongest. So a candidate is dropped only if
        # it is (a) weaker, (b) within tolerance of 2f or 3f of a stronger peak,
        # and (c) seen at no more gauges than that stronger peak.
        cand.sort(key=lambda c: -c[1])          # strongest first
        kept: List[Tuple[float, float, int]] = []
        for c in cand:
            is_harmonic = False
            for stronger in kept:
                ratio = c[0] / stronger[0]
                n = round(ratio)
                if (2 <= n <= 3
                        and abs(c[0] - n * stronger[0]) <= self.harmonic_tol_hz
                        and c[2] <= stronger[2]
                        and c[2] <= self.harmonic_max_gauges):
                    is_harmonic = True
                    break
            if not is_harmonic:
                kept.append(c)

        kept.sort(key=lambda c: c[0])
        # Merge residual near-duplicates and cap the count.
        peak_freqs: List[float] = []
        for f_hz, _s, _g in kept:
            if peak_freqs and f_hz - peak_freqs[-1] < self.min_separation_hz:
                continue
            peak_freqs.append(f_hz)
        peak_freqs = peak_freqs[:self.max_modes]

        if decay_start_s is not None:
            decay = t >= decay_start_s
        else:
            decay = np.ones(t.size, dtype=bool)

        db = 10.0 * np.log10(np.maximum(psd_avg, 1e-30))
        modes: List[IdentifiedMode] = []
        for f_n in peak_freqs:
            idx = int(np.argmin(np.abs(freqs - f_n)))
            hp_zeta, hp_ok = half_power_damping(freqs, psd_avg, idx)

            # Damping from the free decay of the gauge with the most energy at
            # this frequency, so a localised mode is measured where it is strong.
            band_energy = []
            for i in range(d.shape[0]):
                if not rows[i]:
                    band_energy.append(-1.0)
                    continue
                amp = abs(single_frequency_dft(t, d[i], f_n))
                band_energy.append(amp)
            best = int(np.argmax(band_energy))
            zeta, r2 = log_decrement_damping(t[decay], d[best][decay], f_n)

            shape = np.array([single_frequency_dft(t, d[i], f_n)
                              for i in range(d.shape[0])], dtype=np.complex128)
            peak_i = int(np.argmax(np.abs(shape)))
            reference = shape[peak_i]
            normalised = shape / reference if abs(reference) > 0 else shape
            shape_real = np.real(normalised)

            modes.append(IdentifiedMode(
                freq_hz=f_n,
                damping_ratio=zeta,
                damping_r2=r2,
                half_power_zeta=hp_zeta,
                half_power_resolved=hp_ok,
                amplitude_px=float(np.abs(shape[peak_i])),
                prominence_db=float(db[idx] - np.median(db)),
                shape=shape,
                shape_real=shape_real,
            ))

        modes.sort(key=lambda m: m.freq_hz)
        return modes

    @classmethod
    def from_config(cls, cfg: dict) -> "ModalIdentifier":
        m = cfg.get("modal_analysis", {})
        band = m.get("band_hz", [0.02, 5.0])
        return cls(
            band_hz=(float(band[0]), float(band[1])),
            segment_s=float(m.get("welch_window_s", 300.0)),
            overlap=float(m.get("welch_overlap", 0.5)),
            min_prominence_db=float(m.get("peak_prominence_db", 6.0)),
            min_snr_db=float(m.get("peak_snr_db", 12.0)),
            min_separation_hz=float(m.get("min_peak_separation_hz", 0.02)),
            harmonic_tol_hz=float(m.get("harmonic_tol_hz", 0.02)),
            harmonic_max_gauges=int(m.get("harmonic_max_gauges", 1)),
            max_modes=int(m.get("max_modes", 6)),
        )


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #

def _self_test() -> int:
    ok = True

    def check(label: str, condition: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= bool(condition)
        print(f"  [{'PASS' if condition else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))

    fs = 200.0
    t = np.arange(0.0, 900.0, 1.0 / fs)
    rng = np.random.default_rng(5)

    print("welch_psd / find_spectral_peaks")
    truth = [(0.082, 1.00), (0.310, 0.35), (0.515, 0.22), (1.440, 0.06)]
    sig = sum(a * np.sin(2 * math.pi * f * t + 0.3 * k)
              for k, (f, a) in enumerate(truth))
    sig = sig + 0.02 * rng.standard_normal(t.size)

    freqs, psd = welch_psd(sig, fs, segment_s=300.0)
    print(f"    resolution {freqs[1] - freqs[0]:.5f} Hz over {t[-1]:.0f} s")
    peaks = find_spectral_peaks(freqs, psd, band_hz=(0.02, 5.0),
                                min_prominence_db=6.0, max_peaks=6)
    found = sorted(interpolate_peak_hz(freqs, psd, i) for i in peaks)
    check("exactly four modes found, no noise peaks", len(found) == 4,
          f"{[f'{f:.4f}' for f in found]}")
    if len(found) == 4:
        errs = [abs(f - tf) / tf * 100 for f, (tf, _) in zip(found, truth)]
        print(f"    frequency error: {[f'{e:.2f}%' for e in errs]}")
        check("frequency error below 2%", max(errs) < 2.0)

    print("half_power_damping resolution limit")
    fundamental = peaks[int(np.argmin([abs(freqs[i] - 0.082) for i in peaks]))]
    hp, resolved = half_power_damping(freqs, psd, fundamental)
    needed = 2 * 0.0035 * 0.082
    print(f"    -3 dB bandwidth needed {needed:.2e} Hz vs bin {freqs[1] - freqs[0]:.2e} Hz "
          f"(would require a {1.0 / needed / 60:.0f} min segment)")
    check("half-power correctly reports unresolved", not resolved,
          f"zeta_hp = {hp:.4f} is the window main lobe, not a measurement")

    print("log_decrement_damping")
    for f_n, zeta in ((0.082, 0.0035), (0.515, 0.0060)):
        decay = np.exp(-zeta * 2 * math.pi * f_n * t) * np.sin(2 * math.pi * f_n * t)
        decay = decay + 0.002 * rng.standard_normal(t.size)
        est, r2 = log_decrement_damping(t, decay, f_n)
        err = abs(est - zeta) / zeta * 100
        print(f"    f={f_n} Hz: true zeta {zeta:.4f} -> estimated {est:.4f} "
              f"({err:.1f}% error, R^2 {r2:.4f})")
        check(f"damping recovered at {f_n} Hz", err < 15.0 and r2 > 0.95)

    print("bandpass / analytic_envelope")
    mixed = (np.sin(2 * math.pi * 0.082 * t) + np.sin(2 * math.pi * 1.44 * t))
    isolated = bandpass(mixed, fs, 0.05, 0.15)
    env = analytic_envelope(isolated)
    mid = slice(t.size // 4, 3 * t.size // 4)
    check("band-pass isolates the fundamental",
          abs(float(np.mean(env[mid])) - 1.0) < 0.1,
          f"envelope {float(np.mean(env[mid])):.3f} of unit amplitude")

    print("ModalIdentifier - multi-gauge with anti-phase shape")
    # Two modes: the first in phase across gauges, the second anti-phase across
    # the mid-span node, which is what a genuine higher bending mode looks like.
    xi = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    shape1 = xi ** 2
    shape2 = np.sin(2 * math.pi * xi)
    disp = np.stack([
        shape1[i] * np.sin(2 * math.pi * 0.082 * t)
        + 0.4 * shape2[i] * np.sin(2 * math.pi * 0.515 * t + 0.7)
        + 0.01 * rng.standard_normal(t.size)
        for i in range(xi.size)
    ])

    ident = ModalIdentifier(band_hz=(0.02, 3.0), segment_s=300.0, max_modes=4)
    modes = ident.identify(t, disp, warmup_s=0.0)
    check("two modes identified", len(modes) == 2, f"{[f'{m.freq_hz:.4f}' for m in modes]}")
    if len(modes) == 2:
        check("fundamental accurate", abs(modes[0].freq_hz - 0.082) / 0.082 < 0.02,
              f"{modes[0].freq_hz:.4f} Hz")
        mac1 = modal_assurance_criterion(modes[0].shape_real, shape1)
        mac2 = modal_assurance_criterion(modes[1].shape_real, shape2)
        print(f"    MAC against true shapes: mode 1 {mac1:.4f}, mode 2 {mac2:.4f}")
        check("mode shapes recovered", mac1 > 0.98 and mac2 > 0.98)
        signs = np.sign(modes[1].shape_real)
        check("anti-phase preserved across the node",
              len(set(signs.tolist())) == 2, f"signs {signs.tolist()}")
        rec = modes[0].to_record(["G1", "G2", "G3", "G4", "G5"])
        check("record serialisable", "freq_hz" in rec and len(rec["shape_real"]) == 5)

    print("\nSELF-TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
