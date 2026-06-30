"""
Audio-visual drone detection: fuses event camera propeller detection with
ReSpeaker 4-mic array acoustic detection using Bayesian sensor fusion.

System components:
  - Prophesee EVK4-HD (1280x720, IMX636) with 220-degree fisheye lens
  - ReSpeaker 4-Mic Array (USB or Pi HAT) via PyAudio

Detection pipeline:
    1. Visual: per-pixel frequency map + connected-component clustering + tracking
  2. Acoustic: mic array FFT for blade-pass frequency + improved multi-pair GCC-PHAT
               for DOA (frequency-weighted, 4x TDOA resolution, works >15cm)
  3. Fusion: sequential Bayesian updating with frequency cross-validation

Usage:
    python detect_drone_fused.py -i <event_file>                  # visual only
    python detect_drone_fused.py -i <event_file> --enable-audio   # fused mode
    python detect_drone_fused.py --enable-audio                   # live camera + mic

Latency budget (typical DJI Mavic-class drone at 189 Hz BPF):
    Visual first detect  :  ~40-60 ms  (7 periods @ 189 Hz + overhead)
    Acoustic first detect:  ~50-120 ms (FFT window fill + USB buffer)
    Fused high confidence: ~100-200 ms (both modalities agree)
"""

import argparse
import json
import math
import os
import threading
import time
import numpy as np
from collections import deque

import cv2


# ---------------------------------------------------------------------------
# Frequency cross-validation
# ---------------------------------------------------------------------------

def cross_validate_frequency(visual_freq_hz, acoustic_freq_hz,
                             tolerance_ratio=0.05, max_harmonic=4):
    """Check if visual and acoustic frequencies are consistent.

    Accounts for one sensor possibly detecting a harmonic of the fundamental.

    Returns:
        (is_consistent, confidence, (harmonic_v, harmonic_a))
    """
    for n_v in range(1, max_harmonic + 1):
        for n_a in range(1, max_harmonic + 1):
            fundamental_v = visual_freq_hz / n_v
            fundamental_a = acoustic_freq_hz / n_a
            ratio = abs(fundamental_v - fundamental_a) / max(fundamental_a, 1e-6)
            if ratio < tolerance_ratio:
                confidence = 1.0 / (n_v * n_a)
                return True, confidence, (n_v, n_a)
    return False, 0.0, (0, 0)


def load_azimuth_calibration(model_path):
    """Load visual->acoustic azimuth linear calibration model from JSON.

    Expected keys:
      - gain
      - offset_deg
    Optional keys:
      - camera_hfov_deg
      - metadata
    """
    if not model_path:
        return None
    if not os.path.exists(model_path):
        print(f"[Calib] File not found: {model_path}")
        return None
    try:
        with open(model_path, "r", encoding="utf-8") as f:
            model = json.load(f)
        gain = float(model.get("gain", 1.0))
        offset_deg = float(model.get("offset_deg", 0.0))
        return {
            "gain": gain,
            "offset_deg": offset_deg,
            "camera_hfov_deg": float(model.get("camera_hfov_deg", 220.0)),
            "metadata": model.get("metadata", {}),
        }
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"[Calib] Failed to load model '{model_path}': {e}")
        return None


def pixel_x_to_visual_azimuth_deg(center_x_px, width, camera_hfov_deg):
    """Map image x-coordinate to camera azimuth (deg).

    Uses a rectilinear (pinhole) projection, which is correct for a standard
    CCTV/C-mount lens. The focal length in pixels is derived from the supplied
    horizontal FOV:  f_px = (width/2) / tan(HFOV/2).  Azimuth is then the
    arctangent of the horizontal pixel offset from the optical centre.
    """
    half_hfov_rad = math.radians(camera_hfov_deg) / 2.0
    # Guard against degenerate / fisheye-style FOVs (>=180 deg) where the
    # rectilinear model is undefined; fall back to a linear map.
    if camera_hfov_deg >= 179.0 or half_hfov_rad <= 0.0:
        norm_x = (center_x_px / max(width - 1, 1)) * 2.0 - 1.0
        return float(norm_x * (camera_hfov_deg / 2.0))
    f_px = (width / 2.0) / math.tan(half_hfov_rad)
    dx = center_x_px - (width - 1) / 2.0
    return float(math.degrees(math.atan2(dx, f_px)))


def apply_visual_azimuth_calibration(raw_visual_azimuth_deg, calib_model):
    """Apply linear calibration: az_cal = gain * az_raw + offset."""
    if calib_model is None:
        return raw_visual_azimuth_deg
    return float(calib_model["gain"] * raw_visual_azimuth_deg + calib_model["offset_deg"])


class AzimuthCalibrationCollector:
    """Realtime cross-modal spatial calibration.

    As a co-located source (phone tone taped to the fan) is swept LATERALLY
    across the field of view, the event camera reports the fan's visual azimuth
    and the mic array reports the sound's DOA. This collector pairs the two and
    fits a linear visual->acoustic model:

        acoustic_doa = gain * visual_azimuth + offset

    Samples are accumulated into visual-azimuth bins so a stationary dwell
    cannot dominate the least-squares fit; a clean left-right sweep then yields
    even coverage across the FOV. The resulting JSON is directly consumable by
    --azimuth-calibration.
    """

    def __init__(self, visual_bin_deg=2.0, min_bin_samples=5,
                 min_acoustic_span_deg=8.0, solid_r2=0.6, max_bin_samples=400):
        self.visual_bin_deg = max(0.5, float(visual_bin_deg))
        self.min_bin_samples = max(1, int(min_bin_samples))
        self.min_acoustic_span_deg = float(min_acoustic_span_deg)
        self.solid_r2 = float(solid_r2)
        self.max_bin_samples = max(10, int(max_bin_samples))
        # bin_index -> {"vis": [..], "doa": [..]}  (raw samples for robust medians)
        self.bins = {}
        self.total = 0

    def add(self, visual_az_deg, doa_deg):
        b = int(round(visual_az_deg / self.visual_bin_deg))
        rec = self.bins.get(b)
        if rec is None:
            rec = {"vis": [], "doa": []}
            self.bins[b] = rec
        rec["vis"].append(float(visual_az_deg))
        rec["doa"].append(float(doa_deg))
        # Bound memory on long runs: keep the most recent samples per bin.
        if len(rec["doa"]) > self.max_bin_samples:
            rec["vis"] = rec["vis"][-self.max_bin_samples:]
            rec["doa"] = rec["doa"][-self.max_bin_samples:]
        self.total += 1

    def clear(self):
        self.bins.clear()
        self.total = 0

    def _bin_stats(self):
        """Robust per-bin summary: median visual, median DOA, sample count.

        Only bins with enough samples are returned, so noisy under-sampled
        bins (a quick pass of the sweep) cannot corrupt the fit. Medians make
        each bin robust to the GCC-PHAT DOA's occasional wild outliers.
        """
        xs, ys, ws = [], [], []
        for rec in self.bins.values():
            n = len(rec["doa"])
            if n < self.min_bin_samples:
                continue
            xs.append(float(np.median(rec["vis"])))
            ys.append(float(np.median(rec["doa"])))
            ws.append(n)
        return (np.asarray(xs, dtype=float),
                np.asarray(ys, dtype=float),
                np.asarray(ws, dtype=float))

    @staticmethod
    def _weighted_lsq(x, y, w):
        sw = np.sqrt(w)
        A = np.vstack([x * sw, sw]).T
        sol, *_ = np.linalg.lstsq(A, y * sw, rcond=None)
        return float(sol[0]), float(sol[1])

    def fit(self):
        x, y, w = self._bin_stats()
        if len(x) < 3:
            return None
        # Need genuine lateral spread, otherwise the slope is undetermined.
        if float(np.ptp(x)) < 5.0:
            return None

        # Weighted least squares with iterative outlier rejection. Bins whose
        # median DOA sits far from the trend (reverberation / bad windows) are
        # dropped, then the line is refit on the consistent inliers.
        keep = np.ones(len(x), dtype=bool)
        gain = offset = 0.0
        for _ in range(3):
            xs, ys, ws = x[keep], y[keep], w[keep]
            if len(xs) < 3 or float(np.ptp(xs)) < 5.0:
                break
            gain, offset = self._weighted_lsq(xs, ys, ws)
            resid = y - (gain * x + offset)
            mad = float(np.median(np.abs(resid - np.median(resid)))) or 1.0
            new_keep = np.abs(resid) <= max(3.0 * 1.4826 * mad, 4.0)
            if new_keep.sum() < 3 or np.array_equal(new_keep, keep):
                keep = new_keep if new_keep.sum() >= 3 else keep
                break
            keep = new_keep

        xi, yi, wi = x[keep], y[keep], w[keep]
        if len(xi) < 3:
            return None
        gain, offset = self._weighted_lsq(xi, yi, wi)
        resid = yi - (gain * xi + offset)
        rmse = float(np.sqrt(np.average(resid ** 2, weights=wi)))

        # R^2 (coefficient of determination) = how much of the acoustic-DOA
        # variation the visual angle explains. This is the SOLIDITY score.
        ybar = float(np.average(yi, weights=wi))
        ss_tot = float(np.sum(wi * (yi - ybar) ** 2))
        ss_res = float(np.sum(wi * resid ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-9 else 0.0

        acoustic_span = float(yi.max() - yi.min())
        visual_span = float(xi.max() - xi.min())
        solid = bool(r2 >= self.solid_r2
                     and acoustic_span >= self.min_acoustic_span_deg
                     and len(xi) >= 4)

        return {
            "gain": float(gain),
            "offset_deg": float(offset),
            "rmse_deg": rmse,
            "r2": float(r2),
            "solid": solid,
            "n_bins": int(len(xi)),
            "n_bins_total": int(len(x)),
            "n_samples": self.total,
            "visual_span_deg": visual_span,
            "acoustic_span_deg": acoustic_span,
            "visual_range_deg": [float(xi.min()), float(xi.max())],
            "acoustic_range_deg": [float(yi.min()), float(yi.max())],
        }

    def save(self, path, camera_hfov_deg, extra_metadata=None):
        fit = self.fit()
        if fit is None:
            return None
        model = {
            "gain": fit["gain"],
            "offset_deg": fit["offset_deg"],
            "camera_hfov_deg": float(camera_hfov_deg),
            "metadata": {
                "kind": "visual_to_acoustic_linear",
                "rmse_deg": fit["rmse_deg"],
                "r2": fit["r2"],
                "solid": fit["solid"],
                "n_bins": fit["n_bins"],
                "n_samples": fit["n_samples"],
                "visual_range_deg": fit["visual_range_deg"],
                "acoustic_range_deg": fit["acoustic_range_deg"],
                "acoustic_span_deg": fit["acoustic_span_deg"],
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        }
        if extra_metadata:
            model["metadata"].update(extra_metadata)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(model, f, indent=2)
        return model


def visual_azimuth_deg_to_pixel_x(azimuth_deg, width, camera_hfov_deg):
    """Inverse of pixel_x_to_visual_azimuth_deg: map an azimuth back to image x.

    Used to draw the acoustic DOA as a marker on the frequency map. Returns a
    float pixel column, clamped to the image bounds.
    """
    half_hfov_rad = math.radians(camera_hfov_deg) / 2.0
    if camera_hfov_deg >= 179.0 or half_hfov_rad <= 0.0:
        norm_x = azimuth_deg / (camera_hfov_deg / 2.0)
        x = (norm_x + 1.0) / 2.0 * max(width - 1, 1)
    else:
        f_px = (width / 2.0) / math.tan(half_hfov_rad)
        dx = f_px * math.tan(math.radians(azimuth_deg))
        x = dx + (width - 1) / 2.0
    return float(min(max(x, 0.0), width - 1))


# ---------------------------------------------------------------------------
# Sequential Bayesian detector
# ---------------------------------------------------------------------------

class BayesianDroneDetector:
    """Sequential Bayesian fusion for real-time drone detection.

    Updates P(drone) as new visual and acoustic evidence arrives.
    """

    def __init__(self, prior=0.001, decay_rate=0.995,
                 min_freq=50, max_freq=500, detection_threshold=0.8):
        self.p_drone = prior
        self.base_prior = prior
        self.decay_rate = decay_rate
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.detection_threshold = detection_threshold

    def _freq_in_range(self, freq_hz):
        return self.min_freq <= freq_hz <= self.max_freq

    def _visual_likelihood(self, snr, freq_hz):
        """P(V|D) and P(V|~D)."""
        if not self._freq_in_range(freq_hz):
            return 0.1, 0.3
        p_v_d = 1.0 / (1.0 + np.exp(-(snr - 5.0)))
        p_v_nd = 0.01 * np.exp(-0.5 * snr)
        return p_v_d, max(p_v_nd, 1e-9)

    def _acoustic_likelihood(self, snr, freq_hz, elevation_deg=None):
        """P(A|D) and P(A|~D)."""
        if not self._freq_in_range(freq_hz):
            return 0.05, 0.2
        p_a_d = 1.0 / (1.0 + np.exp(-(snr - 4.0)))
        if elevation_deg is not None and elevation_deg > 10:
            p_a_nd = 0.005  # few elevated non-drone sources
        else:
            p_a_nd = 0.05
        return p_a_d, max(p_a_nd, 1e-9)

    def update(self, visual_evidence=None, acoustic_evidence=None):
        """Bayesian update with new evidence.

        visual_evidence:  dict(snr, freq_hz)  or None
        acoustic_evidence: dict(snr, freq_hz, doa_azimuth, doa_elevation) or None

        Returns updated P(drone).
        """
        p_d = self.p_drone
        p_nd = 1.0 - p_d

        # Visual update
        if visual_evidence is not None:
            pv_d, pv_nd = self._visual_likelihood(
                visual_evidence["snr"], visual_evidence["freq_hz"])
            denom = pv_d * p_d + pv_nd * p_nd
            p_d = pv_d * p_d / denom
            p_nd = 1.0 - p_d

        # Acoustic update
        if acoustic_evidence is not None:
            pa_d, pa_nd = self._acoustic_likelihood(
                acoustic_evidence["snr"], acoustic_evidence["freq_hz"],
                acoustic_evidence.get("doa_elevation"))
            denom = pa_d * p_d + pa_nd * p_nd
            p_d = pa_d * p_d / denom
            p_nd = 1.0 - p_d

        # Frequency cross-validation bonus
        if visual_evidence and acoustic_evidence:
            matched, conf, _ = cross_validate_frequency(
                visual_evidence["freq_hz"], acoustic_evidence["freq_hz"])
            if matched:
                pf_d = 0.99 * conf
                pf_nd = 0.001
                denom = pf_d * p_d + pf_nd * p_nd
                p_d = pf_d * p_d / denom

        self.p_drone = np.clip(p_d, 1e-9, 1.0 - 1e-9)
        return self.p_drone

    def decay(self):
        """Without new evidence, belief drifts toward prior."""
        self.p_drone = (self.decay_rate * self.p_drone
                        + (1 - self.decay_rate) * self.base_prior)

    @property
    def detected(self):
        return self.p_drone > self.detection_threshold

    @property
    def confidence(self):
        return self.p_drone


# ---------------------------------------------------------------------------
# Acoustic processing thread (ReSpeaker 4-mic array)
# ---------------------------------------------------------------------------

class AcousticProcessor:
    """Processes audio from the ReSpeaker 4-mic array in a background thread.

    Computes:
      - FFT on each mic channel to find blade-pass frequency peaks
      - Simple GCC-PHAT DOA estimation between mic pairs

    ReSpeaker USB Mic Array v2.0 geometry (circular, 32 mm radius):
        Mic 0: (-32, 0, 0) mm
        Mic 1: ( 0, -32, 0) mm
        Mic 2: (+32, 0, 0) mm
        Mic 3: ( 0, +32, 0) mm
    """

    # Mic positions in meters (USB Mic Array v2.0)
    MIC_POSITIONS = np.array([
        [-0.032, 0.0, 0.0],
        [0.0, -0.032, 0.0],
        [0.032, 0.0, 0.0],
        [0.0, 0.032, 0.0],
    ])
    SPEED_OF_SOUND = 343.0  # m/s

    def __init__(self, sample_rate=16000, chunk_size=1024, n_channels=4,
                 fft_window_sec=0.2, min_freq=50, max_freq=500, min_snr=3.0,
                 device_index=None, doa_min_confidence=1.5,
                 doa_smoothing_window=7):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.n_channels = n_channels
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.min_snr = min_snr
        self.doa_min_confidence = doa_min_confidence
        self.device_index = device_index

        fft_window_samples = int(fft_window_sec * sample_rate)
        self.buffer_maxlen = fft_window_samples
        # Per-channel rolling audio buffers
        self.buffers = [deque(maxlen=fft_window_samples) for _ in range(n_channels)]

        # DOA temporal smoothing: keep recent valid bearings and report a
        # median-filtered value so the on-screen dot stops jittering.
        self.doa_history = deque(maxlen=max(1, doa_smoothing_window))
        self.smoothed_doa = None

        # Latest results (thread-safe via GIL for simple reads)
        self.latest_result = None  # dict or None
        self._running = False
        self._thread = None

    def start(self):
        """Start the audio capture + analysis thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self):
        """Background thread: capture audio and run analysis."""
        try:
            import pyaudio
        except ImportError:
            print("[Audio] PyAudio not available. Acoustic detection disabled.")
            return

        pa = pyaudio.PyAudio()
        try:
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=self.n_channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                input_device_index=self.device_index,
            )
        except Exception as e:
            print(f"[Audio] Failed to open mic stream: {e}")
            pa.terminate()
            return

        print(f"[Audio] Streaming from ReSpeaker ({self.sample_rate} Hz, "
              f"{self.n_channels} ch, chunk={self.chunk_size})")

        try:
            while self._running:
                try:
                    raw = stream.read(self.chunk_size, exception_on_overflow=False)
                except Exception:
                    continue

                # Deinterleave into per-channel samples
                samples = np.frombuffer(raw, dtype=np.int16).reshape(-1, self.n_channels)
                for ch in range(self.n_channels):
                    self.buffers[ch].extend(samples[:, ch].astype(np.float64))

                # Analyze when buffer is full
                if len(self.buffers[0]) >= self.buffer_maxlen:
                    analysis_end_mono = time.monotonic()
                    result = self._analyze()
                    if result is not None:
                        window_dur = self.buffer_maxlen / float(self.sample_rate)
                        result["window_start_mono_s"] = analysis_end_mono - window_dur
                        result["window_end_mono_s"] = analysis_end_mono
                        result["analysis_mono_s"] = analysis_end_mono
                    self.latest_result = result
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

    def _analyze(self):
        """Run FFT + DOA on buffered audio."""
        n = len(self.buffers[0])
        if n < 64:
            return None

        # Average spectrum across all channels for frequency detection
        window = np.hanning(n)
        all_spectra = []
        channel_data = []
        for ch in range(self.n_channels):
            sig = np.array(self.buffers[ch])
            channel_data.append(sig)
            sig_dc = sig - np.mean(sig)
            spectrum = np.fft.rfft(sig_dc * window)
            all_spectra.append(np.abs(spectrum))

        avg_spectrum = np.mean(all_spectra, axis=0) * 2.0 / n
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sample_rate)

        # Find peak in target range
        valid = (freqs >= self.min_freq) & (freqs <= self.max_freq)
        if not np.any(valid):
            return None

        valid_mags = avg_spectrum[valid]
        valid_freqs = freqs[valid]
        peak_idx = np.argmax(valid_mags)
        peak_freq = float(valid_freqs[peak_idx])
        peak_mag = float(valid_mags[peak_idx])

        noise_floor = float(np.median(valid_mags))
        snr = peak_mag / noise_floor if noise_floor > 0 else peak_mag

        if snr < self.min_snr:
            return None

        # GCC-PHAT DOA estimation: use multi-pair fusion for robustness.
        # Returns None when no mic pair was confident this window.
        raw_doa = self._multi_pair_doa_fusion(channel_data)

        # Temporal smoothing: median-filter recent valid bearings so the
        # reported direction (and the on-screen dot) stops dancing. We also
        # expose a spread metric used to gate noisy audio-only fusion updates.
        if raw_doa is not None:
            self.doa_history.append(raw_doa)
        if len(self.doa_history) > 0:
            hist = sorted(self.doa_history)
            self.smoothed_doa = float(hist[len(hist) // 2])  # median
            doa_spread = float(max(self.doa_history) - min(self.doa_history))
        else:
            doa_spread = None

        # Report the smoothed bearing (fall back to last known / 0.0).
        reported_doa = (self.smoothed_doa
                        if self.smoothed_doa is not None else 0.0)

        return {
            "freq_hz": peak_freq,
            "snr": snr,
            "magnitude": peak_mag,
            "doa_azimuth": reported_doa,
            "doa_azimuth_raw": raw_doa,
            "doa_spread_deg": doa_spread,
            "doa_valid": raw_doa is not None,
            "doa_elevation": None,  # planar array cannot resolve elevation
        }

    def _multi_pair_doa_fusion(self, channel_data):
        """Estimate DOA using all available mic pairs with weighted fusion.
        
        ReSpeaker 4-mic geometry:
          Mic 0: left (-32mm)
          Mic 1: front (-32mm, rotated 90°)
          Mic 2: right (+32mm)
          Mic 3: back (+32mm, rotated 90°)
        
        Compute DOA from multiple pairs:
          - Pair 0-2: x-axis (horizontal left-right), 64 mm baseline
          - Pair 1-3: y-axis (horizontal front-back), 64 mm baseline
        
        Fuse by weighting confidence of each pair.
        """
        # Pair 0-2 (x-axis): mics are 64mm apart horizontally
        doa_x = self._gcc_phat_doa(channel_data[0], channel_data[2], mic_dist=0.064)
        
        # Pair 1-3 (y-axis): mics are 64mm apart vertically (front-back)
        doa_y = self._gcc_phat_doa(channel_data[1], channel_data[3], mic_dist=0.064)
        
        # Fuse confident pairs. None means that pair was not confident this
        # window; return None only when neither pair produced an estimate.
        if doa_x is not None and doa_y is not None:
            fused_doa = (doa_x + doa_y) / 2.0
        elif doa_x is not None:
            fused_doa = doa_x
        elif doa_y is not None:
            fused_doa = doa_y
        else:
            fused_doa = None
        
        return fused_doa

    def _gcc_phat_doa(self, sig1, sig2, mic_dist):
        """Estimate DOA angle using frequency-weighted GCC-PHAT with high-resolution TDOA.

        Key improvements for far-field (>15cm) detection:
          1. Frequency-band weighting: emphasize target frequency (122-203 Hz)
          2. High-resolution cross-correlation: use 2x zero-padding for TDOA precision
          3. Phase unwrapping: track continuous phase for larger delays
          4. Confidence metric: reject low-confidence estimates

        Returns azimuth in degrees (0 = broadside, +-90 = endfire) or 0 if confidence low.
        """
        n = len(sig1)
        
        # Step 1: Compute cross-correlation with frequency weighting
        S1 = np.fft.rfft(sig1)
        S2 = np.fft.rfft(sig2)
        cross = S1 * np.conj(S2)
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sample_rate)
        
        # Weight by energy in target frequency band (self.min_freq to self.max_freq)
        freq_band = (freqs >= self.min_freq) & (freqs <= self.max_freq)
        weight = np.zeros_like(freqs)
        weight[freq_band] = np.abs(S1[freq_band]) + np.abs(S2[freq_band])
        weight = weight / (np.max(weight) + 1e-10)
        
        # Apply frequency weighting: emphasize target band
        cross_weighted = cross * (1.0 + 4.0 * weight)
        
        # Step 2: Normalize for GCC-PHAT
        magnitude = np.abs(cross_weighted)
        magnitude[magnitude < 1e-10] = 1e-10
        gcc_phat = cross_weighted / magnitude
        
        # Step 3: High-resolution TDOA via inverse FFT with zero-padding
        # Use 4x padding for sub-sample delay estimation
        n_padded = n * 4
        gcc = np.fft.irfft(gcc_phat, n=n_padded)
        
        # Step 4: Search for peak delay with confidence metric
        max_delay_samples = int(mic_dist / self.SPEED_OF_SOUND * self.sample_rate) + 1
        search_range = min(max_delay_samples + 2, n_padded // 2)
        gcc_shifted = np.concatenate([gcc[-search_range:], gcc[:search_range + 1]])
        
        # Find top 2 peaks for confidence estimation
        peak_idx = np.argmax(np.abs(gcc_shifted))
        peak_val = np.abs(gcc_shifted[peak_idx])
        
        # Remove peak and find second-highest
        gcc_no_peak = np.abs(gcc_shifted.copy())
        search_width = max(5, search_range // 20)
        gcc_no_peak[max(0, peak_idx - search_width):min(len(gcc_no_peak), peak_idx + search_width)] = 0
        second_peak = np.max(gcc_no_peak)
        
        # Confidence = ratio of main peak to second peak (higher is better)
        confidence = peak_val / (second_peak + 1e-6)
        
        # Step 5: Estimate delay with sub-sample precision
        # Use parabolic interpolation around peak for accuracy
        if 0 < peak_idx < len(gcc_shifted) - 1:
            y0 = np.abs(gcc_shifted[peak_idx - 1])
            y1 = np.abs(gcc_shifted[peak_idx])
            y2 = np.abs(gcc_shifted[peak_idx + 1])
            # Parabolic peak interpolation
            denom = 2 * (y0 - 2 * y1 + y2)
            if abs(denom) > 1e-10:
                peak_frac = (y2 - y0) / denom
                delay_samples = peak_idx - search_range + peak_frac
            else:
                delay_samples = peak_idx - search_range
        else:
            delay_samples = peak_idx - search_range
        
        # Account for padding in delay estimation
        delay_samples = delay_samples / 4.0
        
        delay_sec = delay_samples / self.sample_rate
        max_delay = mic_dist / self.SPEED_OF_SOUND
        delay_sec = np.clip(delay_sec, -max_delay, max_delay)
        
        # Step 6: Convert delay to angle
        sin_theta = delay_sec * self.SPEED_OF_SOUND / mic_dist
        sin_theta = np.clip(sin_theta, -1.0, 1.0)
        
        # Only return high-confidence estimates (tunable confidence ratio)
        if confidence < self.doa_min_confidence:
            # Low confidence: no usable estimate this window.
            return None
        
        return float(np.degrees(np.arcsin(sin_theta)))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Audio-visual drone detection with Bayesian sensor fusion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Input
    parser.add_argument("-i", "--input-event-file", dest="event_file_path", default="",
                        help="Path to event file (RAW/DAT/HDF5). Empty = live camera.")

    # Frequency range
    parser.add_argument("--min-freq", dest="min_freq", type=float, default=140)
    parser.add_argument("--max-freq", dest="max_freq", type=float, default=200)
    parser.add_argument("--num-blades", dest="num_blades", type=int, default=2)

    # Visual frequency map range (decoupled from the acoustic band). The event
    # camera reports per-pixel flicker that can be a harmonic of the acoustic
    # blade-pass tone, so keep this wide to avoid the propeller dropping out of
    # the heatmap while it is still spinning.
    parser.add_argument("--visual-min-freq", dest="visual_min_freq", type=float,
                        default=50)
    parser.add_argument("--visual-max-freq", dest="visual_max_freq", type=float,
                        default=500)

    # Visual tuning
    parser.add_argument("--filter-length", dest="filter_length", type=int, default=4)
    parser.add_argument("--max-period-diff", dest="max_period_diff", type=int, default=1500)
    parser.add_argument("--freq-precision", dest="freq_precision", type=float, default=5.0)
    parser.add_argument("--min-cluster-pixels", dest="min_cluster_pixels", type=int, default=20,
                        help="Minimum vibrating pixels in a connected cluster. Raised from 2 "
                             "to suppress tiny 2-4 px ghost tracks that otherwise reach high "
                             "confidence. Real propellers span hundreds-thousands of pixels.")
    parser.add_argument("--dilate-radius", dest="dilate_radius", type=int, default=5,
                        help="Morphological dilation radius for connecting nearby pixels.")
    parser.add_argument("--max-freq-cv", dest="max_freq_cv", type=float, default=0.3,
                        help="Maximum within-cluster frequency coefficient of variation.")
    parser.add_argument("--min-hits", dest="min_hits", type=int, default=5,
                        help="Minimum track hits before a propeller is confirmed.")
    parser.add_argument("--max-age", dest="max_age", type=int, default=8,
                        help="Frames without detection before a track is dropped.")
    parser.add_argument("--track-distance", dest="track_distance", type=float, default=100)
    parser.add_argument("--freq-tolerance", dest="freq_tolerance", type=float, default=0.3)
    parser.add_argument("--confidence-threshold", dest="confidence_threshold", type=float,
                        default=0.95,
                        help="Bayesian confidence threshold for visual propeller confirmation.")
    parser.add_argument("--delta-t", dest="delta_t", type=int, default=500,
                        help="Event slice (us). 500 -> 2 kHz sampling -> 1 kHz Nyquist.")

    # Audio
    parser.add_argument("--enable-audio", dest="enable_audio", action="store_true",
                        help="Enable ReSpeaker 4-mic array acoustic detection.")
    parser.add_argument("--audio-rate", dest="audio_rate", type=int, default=16000,
                        help="Audio sampling rate (Hz). ReSpeaker supports 16000/48000.")
    parser.add_argument("--audio-chunk", dest="audio_chunk", type=int, default=1024,
                        help="Audio chunk size in samples.")
    parser.add_argument("--audio-channels", dest="audio_channels", type=int, default=4)
    parser.add_argument("--audio-device", dest="audio_device", type=int, default=None,
                        help="PyAudio device index for ReSpeaker.")
    parser.add_argument("--audio-fft-window", dest="audio_fft_window", type=float, default=0.2,
                        help="Audio FFT window duration (seconds).")
    parser.add_argument("--audio-min-snr", dest="audio_min_snr", type=float, default=3.0)
    parser.add_argument("--doa-min-confidence", dest="doa_min_confidence", type=float,
                        default=1.5,
                        help="GCC-PHAT peak-to-second-peak ratio required to assign a "
                             "direction. Lower = more directions assigned (more range, "
                             "noisier); higher = stricter (fewer, cleaner).")
    parser.add_argument("--doa-smoothing-window", dest="doa_smoothing_window", type=int,
                        default=7,
                        help="Number of recent valid DOA estimates median-filtered to "
                             "produce a stable bearing (reduces on-screen dot jitter). "
                             "1 disables smoothing.")
    parser.add_argument("--audio-doa-max-spread", dest="audio_doa_max_spread_deg",
                        type=float, default=30.0,
                        help="Max spread (deg) across the recent DOA window allowed for an "
                             "audio-only Bayesian update. Rejects unstable, dancing "
                             "directions that indicate noise rather than a real source.")

    # ReSpeaker LED ring (physical direction indicator)
    parser.add_argument("--enable-led-ring", dest="enable_led_ring", action="store_true",
                        help="Drive the ReSpeaker v2.0 12-LED ring to point at the detector's "
                             "azimuth (requires WinUSB bound to Interface 3 via Zadig).")
    parser.add_argument("--led-brightness", dest="led_brightness", type=int, default=12,
                        help="LED ring brightness 0-31.")
    parser.add_argument("--led0-offset", dest="led0_offset_deg", type=float, default=0.0,
                        help="Physical mounting offset (deg) of LED index 0 for ring alignment.")
    parser.add_argument("--led-counter-clockwise", dest="led_counter_clockwise",
                        action="store_true",
                        help="Reverse ring direction if the lit LED moves the wrong way.")

    # Fusion
    parser.add_argument("--prior", dest="prior", type=float, default=0.001,
                        help="Bayesian prior P(drone).")
    parser.add_argument("--detection-threshold", dest="detection_threshold", type=float,
                        default=0.8, help="P(drone) threshold for confirmed detection.")
    parser.add_argument("--visual-min-pixels", dest="visual_min_pixels", type=int, default=30,
                        help="Minimum visual cluster pixels to use visual evidence in fusion.")
    parser.add_argument("--visual-min-track-confidence", dest="visual_min_track_confidence",
                        type=float, default=0.98,
                        help="Minimum visual track confidence to use visual evidence in fusion.")
    parser.add_argument("--acoustic-min-hits", dest="acoustic_min_hits", type=int, default=3,
                        help="Required consecutive stable acoustic hits before fusion update.")
    parser.add_argument("--acoustic-max-freq-jitter", dest="acoustic_max_freq_jitter", type=float,
                        default=25.0,
                        help="Max frame-to-frame acoustic peak frequency jump (Hz) to count as stable.")
    parser.add_argument("--audio-confirm-snr", dest="audio_confirm_snr", type=float, default=6.0,
                        help="Minimum acoustic SNR for audio-only fusion updates.")
    parser.add_argument("--max-angle-residual", dest="max_angle_residual_deg", type=float,
                        default=35.0,
                        help="Max allowed visual-vs-audio azimuth mismatch (deg) for cross-modal fusion.")

    # Timing
    parser.add_argument("--update-freq", dest="update_freq", type=float, default=20)
    parser.add_argument("--print-interval", dest="print_interval_sec", type=float, default=0.5)
    parser.add_argument("--rt-max-lag-sec", dest="rt_max_lag_sec", type=float, default=0.15,
                        help="When the visual pipeline falls more than this many "
                             "seconds behind real time, skip the expensive "
                             "clustering for that frame so the backlog drains "
                             "(keeps the GUI realtime).")
    parser.add_argument("--no-rt-drop", dest="no_rt_drop", action="store_true",
                        help="Disable realtime catch-up frame dropping (process "
                             "every frame even if lag accumulates).")
    parser.add_argument("-f", "--replay-factor", dest="replay_factor", type=float, default=1)
    parser.add_argument("--max-association-lag", dest="max_association_lag_sec", type=float,
                        default=0.20,
                        help="Max allowed |visual_ts-audio_ts| (seconds) for fused update.")
    parser.add_argument("--session-log", dest="session_log_path", type=str, default="",
                        help="Optional JSONL path for periodic evidence/fusion logs.")
    parser.add_argument("--camera-hfov-deg", dest="camera_hfov_deg", type=float, default=63.8,
                        help="Camera horizontal field of view in degrees for visual azimuth. "
                             "Default 63.8 deg = 5 mm rectilinear lens on the IMX636 "
                             "(6.22 mm sensor width).")
    parser.add_argument("--azimuth-calibration", dest="azimuth_calibration_path", type=str,
                        default="",
                        help="Optional JSON calibration model mapping visual azimuth to acoustic frame.")
    parser.add_argument("--calibrate-azimuth", dest="calibrate_azimuth_path", type=str,
                        default="",
                        help="Realtime calibration mode: sweep a co-located source "
                             "laterally; pairs of (visual_azimuth, acoustic_DOA) are "
                             "collected and a linear visual->acoustic model is written "
                             "to this JSON on exit (or on 's'). Use with --enable-audio.")
    parser.add_argument("--calib-visual-bin-deg", dest="calib_visual_bin_deg", type=float,
                        default=2.0,
                        help="Visual-azimuth bin width (deg) for even calibration coverage.")
    parser.add_argument("--calib-max-doa-spread-deg", dest="calib_max_doa_spread_deg",
                        type=float, default=20.0,
                        help="Reject calibration samples whose smoothed DOA spread "
                             "exceeds this (deg) so only steady bearings are used.")

    args = parser.parse_args()

    nyquist = 1e6 / args.delta_t / 2
    if args.visual_max_freq > nyquist:
        parser.error(
            f"--visual-max-freq ({args.visual_max_freq} Hz) > Nyquist ({nyquist:.0f} Hz). "
            f"Decrease --delta-t to at least {int(1e6 / (2 * args.visual_max_freq))} us.")
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    try:
        from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator, is_live_camera
        from metavision_sdk_analytics import FrequencyMapAsyncAlgorithm, DominantValueMapAlgorithm, \
            HeatMapFrameGeneratorAlgorithm
        from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
        from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIKeyEvent, UIAction
        from detect_propeller import FrequencyMapAnalyzer, PropellerTracker, draw_detections_on_frame
    except Exception as e:
        print("[Init] Failed to import Metavision SDK modules.")
        print("[Init] Ensure your environment is activated and SDK paths are configured.")
        print(f"[Init] Import error: {e}")
        return

    delta_t = args.delta_t
    fs = 1e6 / delta_t
    print_interval_samples = max(1, int(args.print_interval_sec * fs))

    # --- Event camera setup ---
    mv_iterator = EventsIterator(input_path=args.event_file_path, delta_t=delta_t)
    height, width = mv_iterator.get_size()
    if not is_live_camera(args.event_file_path):
        mv_iterator = LiveReplayEventsIterator(mv_iterator, replay_factor=args.replay_factor)

    # SDK frequency map
    freq_algo = FrequencyMapAsyncAlgorithm(
        width=width, height=height,
        filter_length=args.filter_length,
        min_freq=args.visual_min_freq, max_freq=args.visual_max_freq,
        diff_thresh_us=args.max_period_diff)
    freq_algo.update_frequency = args.update_freq

    dominant_algo = DominantValueMapAlgorithm(
        args.visual_min_freq, args.visual_max_freq, args.freq_precision,
        args.min_cluster_pixels)

    heat_gen = HeatMapFrameGeneratorAlgorithm(
        args.visual_min_freq, args.visual_max_freq, args.freq_precision,
        width, height, "Hz")
    freq_img = heat_gen.get_output_image()
    freq_full_height = heat_gen.full_height

    # Production-optimized visual analyzer and tracker
    analyzer = FrequencyMapAnalyzer(
        width=width, height=height,
        min_freq=args.visual_min_freq,
        max_freq=args.visual_max_freq,
        num_blades=args.num_blades,
        min_pixels=args.min_cluster_pixels,
        dilate_radius=args.dilate_radius,
        max_freq_cv=args.max_freq_cv,
    )
    tracker = PropellerTracker(
        max_distance=args.track_distance,
        freq_tolerance=args.freq_tolerance,
        min_hits=args.min_hits,
        max_age=args.max_age,
        confidence_threshold=args.confidence_threshold,
    )

    # --- Audio setup ---
    audio_proc = None
    if args.enable_audio:
        audio_proc = AcousticProcessor(
            sample_rate=args.audio_rate,
            chunk_size=args.audio_chunk,
            n_channels=args.audio_channels,
            fft_window_sec=args.audio_fft_window,
            min_freq=args.min_freq, max_freq=args.max_freq,
            min_snr=args.audio_min_snr,
            doa_min_confidence=args.doa_min_confidence,
            doa_smoothing_window=args.doa_smoothing_window,
            device_index=args.audio_device)
        audio_proc.start()

    # --- Bayesian detector ---
    bayesian = BayesianDroneDetector(
        prior=args.prior, min_freq=args.min_freq, max_freq=args.max_freq,
        detection_threshold=args.detection_threshold)

    # --- Optional ReSpeaker LED ring (physical azimuth indicator) ---
    led_ring = None
    if args.enable_led_ring:
        try:
            from respeaker_led_control import find_device, PixelRing
            led_ring = PixelRing(find_device())
            led_ring.set_brightness(args.led_brightness)
            led_ring.off()
            print("[LED] Ring acquired; will indicate detector azimuth.")
        except Exception as e:
            led_ring = None
            print(f"[LED] Ring unavailable ({e}).")
            print("[LED] Bind WinUSB to ReSpeaker Interface 3 with Zadig, then retry.")

    led_cw = not args.led_counter_clockwise
    led_state = {"last": None}  # remembers last drawn mode to limit USB traffic

    def update_led_ring(p_drone, detected, fused_az, acoustic_az):
        """Drive the ring: red flash when a drone is confirmed, cyan toward the
        acoustic bearing while scanning, off when silent."""
        if led_ring is None:
            return
        try:
            if detected and fused_az is not None:
                key = ("DET", int(fused_az / 10))
                if key != led_state["last"]:
                    led_ring.point_at(fused_az, rgb=(255, 0, 0),
                                      led0_offset_deg=args.led0_offset_deg,
                                      clockwise=led_cw, beam_width=1)
                    led_state["last"] = key
            elif acoustic_az is not None:
                key = ("DOA", int(acoustic_az / 10))
                if key != led_state["last"]:
                    led_ring.point_at(acoustic_az, rgb=(0, 255, 255),
                                      led0_offset_deg=args.led0_offset_deg,
                                      clockwise=led_cw, beam_width=0)
                    led_state["last"] = key
            else:
                if led_state["last"] != "OFF":
                    led_ring.off()
                    led_state["last"] = "OFF"
        except Exception:
            pass    # --- Optional visual->acoustic azimuth calibration model ---
    azimuth_calibration = load_azimuth_calibration(args.azimuth_calibration_path)
    if azimuth_calibration is not None:
        print("[Calib] Loaded azimuth model: "
              f"gain={azimuth_calibration['gain']:.4f}, "
              f"offset={azimuth_calibration['offset_deg']:.2f} deg")

    # --- Optional realtime cross-modal calibration collector ---
    calib_collector = None
    if args.calibrate_azimuth_path:
        calib_collector = AzimuthCalibrationCollector(
            visual_bin_deg=args.calib_visual_bin_deg)
        print("[Calib] CALIBRATION MODE ON -> "
              f"{args.calibrate_azimuth_path}")
        print("[Calib] Sweep the co-located source LEFT<->RIGHT across the FOV.")
        print("[Calib] Keys in a window: 's' save model now, 'c' clear samples.")
        print("[Calib] Model is also saved automatically on exit (Q/ESC).")

    # --- Optional session logger (JSONL) ---
    session_log_fp = None
    if args.session_log_path:
        try:
            session_log_fp = open(args.session_log_path, "w", encoding="utf-8")
            print(f"[Log] Writing session trace to: {args.session_log_path}")
        except OSError as e:
            print(f"[Log] Could not open session log '{args.session_log_path}': {e}")
            session_log_fp = None

    # --- Windows ---
    ev_window = MTWindow(title="Events - Fused Drone Detector",
                         width=width, height=height,
                         mode=BaseWindow.RenderMode.BGR, open_directly=True)
    freq_window = MTWindow(title="Frequency Map",
                           width=width, height=freq_full_height,
                           mode=BaseWindow.RenderMode.BGR, open_directly=True)

    def keyboard_cb(key, scancode, action, mods):
        if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
            ev_window.set_close_flag()
            freq_window.set_close_flag()
        elif calib_collector is not None and action == UIAction.RELEASE:
            if key == UIKeyEvent.KEY_S:
                saved = calib_collector.save(
                    args.calibrate_azimuth_path, args.camera_hfov_deg)
                if saved is not None:
                    md = saved["metadata"]
                    tag = "SOLID" if md.get("solid") else "WEAK"
                    print(f"\n[Calib] Saved ({tag}) -> {args.calibrate_azimuth_path}  "
                          f"gain={saved['gain']:.4f}  "
                          f"offset={saved['offset_deg']:.2f} deg  "
                          f"R2={md['r2']:.2f}  "
                          f"aSpan={md['acoustic_span_deg']:.1f} deg  "
                          f"rmse={md['rmse_deg']:.2f} deg  "
                          f"(bins={md['n_bins']})")
                    if not md.get("solid"):
                        print("[Calib]   -> WEAK fit. Sweep the phone wider and "
                              "slower; the acoustic DOA must visibly swing.")
                else:
                    print("\n[Calib] Not enough data yet "
                          "(>=3 bins, >=5 samples/bin, >5 deg sweep). Keep sweeping.")
            elif key == UIKeyEvent.KEY_C:
                calib_collector.clear()
                print("\n[Calib] Cleared collected samples.")

    ev_window.set_keyboard_callback(keyboard_cb)
    freq_window.set_keyboard_callback(keyboard_cb)

    event_frame_gen = PeriodicFrameGenerationAlgorithm(
        sensor_width=width, sensor_height=height, fps=25, palette=ColorPalette.Dark)

    confirmed_propellers = []
    latest_visual_evidence = [None]

    def on_cd_frame_cb(ts, cd_frame):
        if confirmed_propellers:
            draw_detections_on_frame(cd_frame, confirmed_propellers, args.num_blades)
        ev_window.show_async(cd_frame)

    event_frame_gen.set_output_callback(on_cd_frame_cb)

    # Realtime guard: compare sensor time (ts, microseconds) to wall clock so we
    # can DROP stale visual work when the pipeline falls behind, instead of
    # letting the live-camera backlog (and the on-screen lag) accumulate.
    rt_state = {"ts0": None, "wall0": None, "skips": 0, "frames": 0}

    def on_freq_map(ts, freq_map):
        nonlocal confirmed_propellers

        # --- Lag estimate (sensor-time elapsed vs wall-time elapsed) ---
        now = time.monotonic()
        if rt_state["ts0"] is None:
            rt_state["ts0"] = ts
            rt_state["wall0"] = now
            lag_s = 0.0
        else:
            sensor_elapsed = (ts - rt_state["ts0"]) / 1e6
            wall_elapsed = now - rt_state["wall0"]
            lag_s = wall_elapsed - sensor_elapsed
        rt_state["frames"] += 1

        # When we are more than ~150 ms behind real time, skip the expensive
        # clustering/heatmap for this frame and reuse the last result. This
        # bounds per-frame cost so the backlog drains and the GUI stays live.
        if lag_s > args.rt_max_lag_sec and not args.no_rt_drop:
            rt_state["skips"] += 1
            heat_gen.generate_bgr_heat_map(freq_map, freq_img)
            cv2.putText(freq_img, f"REALTIME CATCH-UP (lag {lag_s*1000:.0f} ms)",
                        (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 165, 255), 1, cv2.LINE_AA)
            freq_window.show_async(freq_img)
            return

        candidates = analyzer.analyze(freq_map)
        confirmed_propellers = tracker.update(candidates)
        latest_visual_evidence[0] = None

        heat_gen.generate_bgr_heat_map(freq_map, freq_img)

        if confirmed_propellers:
            best = confirmed_propellers[0]
            center_x_px = float(best["x"])
            raw_visual_azimuth_deg = pixel_x_to_visual_azimuth_deg(
                center_x_px=center_x_px, width=width,
                camera_hfov_deg=args.camera_hfov_deg)
            calibrated_visual_azimuth_deg = apply_visual_azimuth_calibration(
                raw_visual_azimuth_deg, azimuth_calibration)
            latest_visual_evidence[0] = {
                "freq_hz": best["freq_hz"],
                "snr": float(best.get("confidence", 0.0) * 10.0),
                "evidence_mono_s": time.monotonic(),
                "x": center_x_px,
                "y": float(best["y"]),
                "pixels": int(best["pixels"]),
                "bbox": best["bbox"],
                "visual_azimuth_raw_deg": raw_visual_azimuth_deg,
                "visual_azimuth_deg": calibrated_visual_azimuth_deg,
                "track_id": int(best["id"]),
                "track_confidence": float(best.get("confidence", 0.0)),
                "analysis_time_ms": float(analyzer.analysis_time_ms),
            }

            x, y, w, h = best["bbox"]
            margin = max(8, int(max(w, h) * 0.2))
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(width, x + w + margin)
            y2 = min(height, y + h + margin)
            cv2.rectangle(freq_img, (x1, y1), (x2, y2), (255, 255, 255), 2)
            label = f"Visual: {best['freq_hz']:.0f} Hz ({best['rpm']:.0f} RPM)"
            cv2.putText(freq_img, label, (x1, max(y1 - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
        else:
            success, dom_freq = dominant_algo.compute_dominant_value(freq_map)
            if success:
                rpm = (dom_freq / args.num_blades) * 60
                label = f"Visual: {dom_freq:.0f} Hz ({rpm:.0f} RPM)"
                cv2.putText(freq_img, label, (10, height - 30),
                            cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)

        # Show Bayesian confidence
        status = "DRONE DETECTED" if bayesian.detected else "scanning"
        color = (0, 0, 255) if bayesian.detected else (0, 255, 0)
        cv2.putText(freq_img, f"P(drone)={bayesian.confidence:.3f}  [{status}]",
                    (10, height - 10), cv2.FONT_HERSHEY_PLAIN, 1.0, color, 1)

        status_line = (f"Cand: {len(candidates)}  Tracks: {len(tracker.tracks)}  "
                       f"Confirmed: {len(confirmed_propellers)}  "
                       f"analyze={analyzer.analysis_time_ms:.1f}ms")
        cv2.putText(freq_img, status_line, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)

        # --- Acoustic DOA indicator: dot + vertical line at the sound azimuth ---
        if audio_proc is not None and audio_proc.latest_result is not None:
            ar = audio_proc.latest_result
            doa_az = ar.get("doa_azimuth")
            if doa_az is not None:
                spread = ar.get("doa_spread_deg")
                stable = (spread is not None
                          and spread <= args.audio_doa_max_spread_deg)
                # Solid cyan when the smoothed bearing is stable; dim/hollow
                # when it is still dancing so the operator can trust it or not.
                color = (255, 255, 0) if stable else (120, 120, 80)
                doa_x = int(visual_azimuth_deg_to_pixel_x(
                    doa_az, width, args.camera_hfov_deg))
                cv2.line(freq_img, (doa_x, 26), (doa_x, height - 16),
                         color, 1, cv2.LINE_AA)
                if stable:
                    cv2.circle(freq_img, (doa_x, height // 2), 8, color, -1,
                               cv2.LINE_AA)
                    cv2.circle(freq_img, (doa_x, height // 2), 8, (0, 0, 0), 1,
                               cv2.LINE_AA)
                else:
                    cv2.circle(freq_img, (doa_x, height // 2), 8, color, 1,
                               cv2.LINE_AA)
                spread_txt = f", +/-{spread:.0f}deg" if spread is not None else ""
                doa_label = (f"Sound {doa_az:+.0f} deg ({ar.get('freq_hz', 0):.0f} Hz"
                             f"{spread_txt})")
                lx = min(max(doa_x - 60, 5), width - 200)
                cv2.putText(freq_img, doa_label, (lx, height // 2 - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            color, 1, cv2.LINE_AA)

        # --- Realtime calibration sample collection + overlay ---
        if calib_collector is not None:
            ve = latest_visual_evidence[0]
            collected = False
            if (ve is not None and audio_proc is not None
                    and audio_proc.latest_result is not None):
                ar2 = audio_proc.latest_result
                doa_raw = ar2.get("doa_azimuth")
                spread2 = ar2.get("doa_spread_deg")
                # Gate: confident visual track + steady, valid acoustic bearing.
                good_visual = (ve.get("pixels", 0) >= args.visual_min_pixels
                               and ve.get("track_confidence", 0.0)
                               >= args.visual_min_track_confidence)
                good_doa = (ar2.get("doa_valid")
                            and doa_raw is not None
                            and (spread2 is None
                                 or spread2 <= args.calib_max_doa_spread_deg))
                if good_visual and good_doa:
                    calib_collector.add(ve["visual_azimuth_raw_deg"], doa_raw)
                    collected = True
            # Recompute the least-squares fit at most a few times/sec (it is
            # only for the on-screen readout) instead of every 20 Hz frame.
            if (collected or now - rt_state.get("last_fit_t", 0.0) > 0.25):
                rt_state["last_fit"] = calib_collector.fit()
                rt_state["last_fit_t"] = now
            fit = rt_state.get("last_fit")
            if fit is not None:
                ctxt = (f"CALIB n={calib_collector.total} bins={fit['n_bins']}  "
                        f"gain={fit['gain']:.3f} off={fit['offset_deg']:+.1f}  "
                        f"R2={fit['r2']:.2f} aSpan={fit['acoustic_span_deg']:.0f}deg "
                        f"rmse={fit['rmse_deg']:.1f}deg")
                if fit["solid"]:
                    ctxt2 = "SOLID - press s to save"
                    c2 = (0, 255, 0)
                else:
                    ctxt2 = "NOT solid: sweep wider/slower (need R2>=0.6, aSpan>=8deg)"
                    c2 = (0, 165, 255)
            else:
                ctxt = (f"CALIB n={calib_collector.total}  "
                        f"sweep LEFT<->RIGHT (need >=3 bins, >=5 samp/bin)")
                ctxt2 = "s=save  c=clear"
                c2 = (180, 180, 180)
            cv2.putText(freq_img, ctxt, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255) if collected else (0, 160, 160), 1,
                        cv2.LINE_AA)
            cv2.putText(freq_img, ctxt2, (10, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, c2, 1,
                        cv2.LINE_AA)

        freq_window.show_async(freq_img)

    freq_algo.set_output_callback(on_freq_map)

    # --- Print header ---
    nyquist = fs / 2
    print("=" * 78)
    print("  Audio-Visual Drone Detector (Bayesian Fusion)")
    print(f"  Sensor          : {width}x{height}  (lens HFOV {args.camera_hfov_deg:.1f} deg, rectilinear)")
    print(f"  Visual sampling : {fs:.0f} Hz  (delta_t={delta_t} us, Nyquist={nyquist:.0f} Hz)")
    print(f"  Freq range      : {args.min_freq}-{args.max_freq} Hz  ({args.num_blades} blades)")
    print(f"  Visual freq map : {args.visual_min_freq}-{args.visual_max_freq} Hz  "
          f"(decoupled from acoustic band)")
    print(f"  Visual pipeline : FrequencyMapAnalyzer + PropellerTracker (v3 realtime path)")
    print(f"  Min cluster px  : {args.min_cluster_pixels}")
    print(f"  Dilate radius   : {args.dilate_radius}")
    print(f"  Max freq CV     : {args.max_freq_cv}")
    print(f"  Tracking        : min_hits={args.min_hits}, max_age={args.max_age}, "
          f"dist={args.track_distance}, freq_tol={args.freq_tolerance}")
    print(f"  Confidence      : visual threshold={args.confidence_threshold:.0%}")
    print(f"  Audio           : {'ENABLED' if args.enable_audio else 'DISABLED'}")
    if args.enable_audio:
        print(f"    Sample rate   : {args.audio_rate} Hz")
        print(f"    FFT window    : {args.audio_fft_window}s")
        print(f"    DOA method    : GCC-PHAT (4-mic, ~32 mm radius)")
        print(f"    DOA min conf  : {args.doa_min_confidence:.2f} "
              f"(peak/2nd-peak ratio to assign direction)")
        print(f"    DOA smoothing : median over {args.doa_smoothing_window} windows")
    print(f"  Fusion          : Sequential Bayesian (prior={args.prior})")
    print(f"  Detection thresh: P(drone) > {args.detection_threshold}")
    print(f"  Visual gate     : pixels>={args.visual_min_pixels}, "
          f"track_conf>={args.visual_min_track_confidence:.2f}")
    print(f"  Audio gate      : hits>={args.acoustic_min_hits}, "
          f"jitter<={args.acoustic_max_freq_jitter:.1f} Hz, "
          f"audio-only SNR>={args.audio_confirm_snr:.1f}")
    print(f"  Audio DOA gate  : audio-only spread <= {args.audio_doa_max_spread_deg:.0f} deg")
    print(f"  Angle gate      : |v-a| <= {args.max_angle_residual_deg:.1f} deg")
    print(f"  Assoc gate      : |dt| <= {args.max_association_lag_sec * 1000:.0f} ms")
    print(f"  Session log     : {args.session_log_path if args.session_log_path else 'DISABLED'}")
    print(f"  Camera HFOV     : {args.camera_hfov_deg:.1f} deg")
    print("  Azimuth calib   : "
          f"{args.azimuth_calibration_path if args.azimuth_calibration_path else 'DISABLED'}")
    print("  Calibrate mode  : "
          f"{args.calibrate_azimuth_path if args.calibrate_azimuth_path else 'DISABLED'}")
    print(f"  LED ring        : "
          f"{'ENABLED' if (args.enable_led_ring and led_ring is not None) else 'DISABLED'}")
    print("=" * 78)
    print()

    sample_count = 0
    t_start = time.monotonic()
    acoustic_state = {"hits": 0, "last_freq_hz": None}

    # --- Realtime drop-to-live state ---
    # The main loop must keep pace with the live event stream. When the wall
    # clock runs ahead of the sensor timeline, the camera buffer is backing up,
    # so we shed the OPTIONAL work (CD visualization) and, if badly behind,
    # drop whole batches to drain the buffer and stay at the live edge.
    loop_ts0 = [None]
    loop_wall0 = [None]
    drop_max_lag = args.rt_max_lag_sec
    drop_hard_lag = max(args.rt_max_lag_sec * 3.0, 0.4)

    for evs in mv_iterator:
        EventLoop.poll_and_dispatch()
        sample_count += 1

        # Estimate how far behind real time we are (cheap: last event ts).
        loop_lag = 0.0
        if len(evs):
            ev_ts = int(evs["t"][-1])
            now_w = time.monotonic()
            if loop_ts0[0] is None:
                loop_ts0[0] = ev_ts
                loop_wall0[0] = now_w
            else:
                sensor_el = (ev_ts - loop_ts0[0]) / 1e6
                wall_el = now_w - loop_wall0[0]
                loop_lag = wall_el - sensor_el

        behind = (loop_lag > drop_max_lag) and not args.no_rt_drop
        hard_behind = (loop_lag > drop_hard_lag) and not args.no_rt_drop

        if hard_behind:
            # Severely behind: drain this batch without any heavy processing so
            # the buffer empties and we snap back to the live edge.
            if ev_window.should_close() or freq_window.should_close():
                break
            continue

        # CD visualization is cosmetic; skip feeding it while we are behind.
        if not behind:
            event_frame_gen.process_events(evs)
        freq_algo.process_events(evs)

        if ev_window.should_close() or freq_window.should_close():
            break

        # --- Periodic fusion + printout ---
        if sample_count % print_interval_samples != 0:
            continue

        elapsed = time.monotonic() - t_start
        wall_time_ms = elapsed * 1000

        # Visual evidence comes from the optimized frequency-map callback path.
        visual_evidence = latest_visual_evidence[0]

        # Acoustic evidence
        acoustic_evidence = None
        if audio_proc and audio_proc.latest_result:
            r = audio_proc.latest_result
            acoustic_evidence = {
                "freq_hz": r["freq_hz"],
                "snr": r["snr"],
                "doa_azimuth": r["doa_azimuth"],
                "doa_elevation": r.get("doa_elevation"),
                "doa_spread_deg": r.get("doa_spread_deg"),
                "doa_valid": r.get("doa_valid"),
                "evidence_mono_s": r.get("analysis_mono_s"),
                "window_start_mono_s": r.get("window_start_mono_s"),
                "window_end_mono_s": r.get("window_end_mono_s"),
            }

            prev_f = acoustic_state["last_freq_hz"]
            if prev_f is None:
                acoustic_state["hits"] = 1
            elif abs(acoustic_evidence["freq_hz"] - prev_f) <= args.acoustic_max_freq_jitter:
                acoustic_state["hits"] += 1
            else:
                acoustic_state["hits"] = 1
            acoustic_state["last_freq_hz"] = acoustic_evidence["freq_hz"]
            acoustic_evidence["persistence_hits"] = acoustic_state["hits"]
        else:
            acoustic_state["hits"] = 0
            acoustic_state["last_freq_hz"] = None

        # Time association gate: keep modality-specific evidence, but only fuse both
        # if the timestamps are close enough.
        association_lag_s = None
        fuse_visual = visual_evidence
        fuse_acoustic = acoustic_evidence
        association_ok = True

        # Visual quality gate
        if fuse_visual is not None:
            if (fuse_visual.get("pixels", 0) < args.visual_min_pixels
                    or fuse_visual.get("track_confidence", 0.0) < args.visual_min_track_confidence):
                fuse_visual = None

        # Acoustic stability gate
        if fuse_acoustic is not None:
            if fuse_acoustic.get("persistence_hits", 0) < args.acoustic_min_hits:
                fuse_acoustic = None

        if visual_evidence and acoustic_evidence:
            v_ts = visual_evidence.get("evidence_mono_s")
            a_ts = acoustic_evidence.get("evidence_mono_s")
            if v_ts is not None and a_ts is not None:
                association_lag_s = abs(v_ts - a_ts)
                association_ok = association_lag_s <= args.max_association_lag_sec
                if not association_ok:
                    # Avoid overconfident fusion from stale cross-modal evidence.
                    fuse_acoustic = None

        # Angle consistency gate
        gated_angle_residual = None
        if fuse_visual and fuse_acoustic:
            gated_angle_residual = abs(fuse_visual["visual_azimuth_deg"] - fuse_acoustic["doa_azimuth"])
            if gated_angle_residual > 180.0:
                gated_angle_residual = 360.0 - gated_angle_residual
            if gated_angle_residual > args.max_angle_residual_deg:
                fuse_acoustic = None

        # Audio-only updates need stronger SNR.
        if fuse_visual is None and fuse_acoustic is not None:
            if fuse_acoustic["snr"] < args.audio_confirm_snr:
                fuse_acoustic = None

        # Audio-only updates also need a STABLE direction. A bearing that dances
        # across a wide arc between windows is noise/reverberation, not a real
        # source, so reject it before it can latch P(drone) high.
        if fuse_visual is None and fuse_acoustic is not None:
            spread = fuse_acoustic.get("doa_spread_deg")
            if spread is not None and spread > args.audio_doa_max_spread_deg:
                fuse_acoustic = None

        # Bayesian update
        if fuse_visual or fuse_acoustic:
            p = bayesian.update(fuse_visual, fuse_acoustic)
        else:
            bayesian.decay()
            p = bayesian.confidence

        # Print status
        sim_time = sample_count * delta_t / 1e6
        v_str = (f"freq={visual_evidence['freq_hz']:6.1f} Hz, "
                 f"SNR={visual_evidence['snr']:.1f}"
                 if visual_evidence else "---")
        a_str = (f"freq={acoustic_evidence['freq_hz']:6.1f} Hz, "
                 f"SNR={acoustic_evidence['snr']:.1f}, "
                 f"DOA={acoustic_evidence['doa_azimuth']:.0f} deg"
                 if acoustic_evidence else "---")

        # Frequency cross-validation
        xval_str = ""
        if visual_evidence and acoustic_evidence:
            matched, conf, harmonics = cross_validate_frequency(
                visual_evidence["freq_hz"], acoustic_evidence["freq_hz"])
            if matched:
                xval_str = f"  FREQ MATCH (h={harmonics}, conf={conf:.2f})"

        assoc_str = ""
        if association_lag_s is not None:
            assoc_state = "ok" if association_ok else "stale"
            assoc_str = f"  ASSOC dt={association_lag_s*1000:.0f} ms [{assoc_state}]"

        angle_str = ""
        if visual_evidence and acoustic_evidence:
            angle_residual = abs(visual_evidence["visual_azimuth_deg"]
                                 - acoustic_evidence["doa_azimuth"])
            if angle_residual > 180.0:
                angle_residual = 360.0 - angle_residual
            angle_str = (f"  ANG v={visual_evidence['visual_azimuth_deg']:.1f} deg"
                         f" a={acoustic_evidence['doa_azimuth']:.1f} deg"
                         f" err={angle_residual:.1f} deg")

        status = "*** DRONE ***" if bayesian.detected else ""
        print(f"[{sim_time:7.2f}s] P={p:.4f} "
              f" V:[{v_str}]  A:[{a_str}]{assoc_str}{xval_str}{angle_str}  {status}")

        # Drive the physical LED ring toward the detector's azimuth.
        fused_az_for_led = None
        if visual_evidence is not None:
            fused_az_for_led = visual_evidence.get("visual_azimuth_deg")
        if fused_az_for_led is None and acoustic_evidence is not None:
            fused_az_for_led = acoustic_evidence.get("doa_azimuth")
        acoustic_az_for_led = (acoustic_evidence.get("doa_azimuth")
                               if acoustic_evidence is not None else None)
        update_led_ring(p, bayesian.detected, fused_az_for_led, acoustic_az_for_led)

        if confirmed_propellers:
            for i, track in enumerate(confirmed_propellers[:3]):
                print(f"           Propeller {i+1}: "
                      f"ID{track['id']} {track['freq_hz']:.1f} Hz, {track['rpm']:.0f} RPM, "
                      f"{track['pixels']} px, conf={track.get('confidence', 0.0):.1%}")

        if session_log_fp:
            log_rec = {
                "sim_time_s": sim_time,
                "wall_time_ms": wall_time_ms,
                "p_drone": p,
                "detected": bool(bayesian.detected),
                "association": {
                    "ok": association_ok,
                    "lag_s": association_lag_s,
                    "max_lag_s": args.max_association_lag_sec,
                },
                "visual": visual_evidence,
                "acoustic": acoustic_evidence,
                "fused_visual_present": fuse_visual is not None,
                "fused_acoustic_present": fuse_acoustic is not None,
                "fusion_gates": {
                    "visual_min_pixels": args.visual_min_pixels,
                    "visual_min_track_confidence": args.visual_min_track_confidence,
                    "acoustic_min_hits": args.acoustic_min_hits,
                    "acoustic_max_freq_jitter": args.acoustic_max_freq_jitter,
                    "audio_confirm_snr": args.audio_confirm_snr,
                    "max_angle_residual_deg": args.max_angle_residual_deg,
                    "gated_angle_residual_deg": gated_angle_residual,
                },
            }
            session_log_fp.write(json.dumps(log_rec) + "\n")
            session_log_fp.flush()

    # Cleanup
    if audio_proc:
        audio_proc.stop()
    if calib_collector is not None:
        saved = calib_collector.save(args.calibrate_azimuth_path, args.camera_hfov_deg)
        if saved is not None:
            md = saved["metadata"]
            tag = "SOLID" if md.get("solid") else "WEAK - re-run wider/slower"
            print("\n[Calib] FINAL MODEL saved -> "
                  f"{args.calibrate_azimuth_path}  [{tag}]")
            print(f"[Calib]   acoustic_doa = {saved['gain']:.4f} * visual_az "
                  f"{saved['offset_deg']:+.2f} deg")
            print(f"[Calib]   R2={md['r2']:.2f}  "
                  f"acoustic_span={md['acoustic_span_deg']:.1f} deg  "
                  f"rmse={md['rmse_deg']:.2f} deg")
            print(f"[Calib]   bins={md['n_bins']}  "
                  f"samples={md['n_samples']}  "
                  f"visual span={md['visual_range_deg']}")
            print("[Calib] Use it:  --azimuth-calibration "
                  f"{args.calibrate_azimuth_path}")
        else:
            print("\n[Calib] No model saved: needs >=3 visual bins (>=5 samples "
                  "each) over a >5 deg sweep. Re-run and sweep the source wider.")
    if led_ring is not None:
        try:
            led_ring.trace()  # hand the ring back to firmware auto-DOA
        except Exception:
            pass
    if session_log_fp:
        session_log_fp.close()
    ev_window.destroy()
    freq_window.destroy()
    print("\nDone.")


if __name__ == "__main__":
    main()
