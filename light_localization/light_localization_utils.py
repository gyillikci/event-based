"""
Shared utilities for the light localization system.

Contains:
  - LightROIExtractor: segments individual light fixtures from frequency maps
  - FrequencyFeatureExtractor: FFT-based harmonic feature extraction from event streams
  - frequency_distance: metric for comparing light fingerprints
  - weighted_centroid: position estimation from matched lights
  - harmonic_ratios: extract harmonic ratio vector from spectrum
"""

import numpy as np
import cv2
import time
from collections import deque


# ---------------------------------------------------------------------------
#  Spatial segmentation: extract individual light fixture ROIs from freq map
# ---------------------------------------------------------------------------

class LightROIExtractor:
    """Segments individual light fixtures from the SDK's per-pixel frequency map.

    Pipeline:
      1. Threshold frequency map → binary mask of pixels with detected flicker.
      2. Downscale → morphological close (fill gaps in fixture shapes).
      3. Connected-component labelling → individual light ROIs.
      4. Filter by size, aspect ratio, and frequency consistency.

    Follows the pattern of FrequencyMapAnalyzer in detect_propeller.py but
    optimized for ceiling-mounted light fixtures (larger ROIs, rectangular shapes,
    lower per-pixel frequency variance).
    """

    DOWNSCALE = 4  # 1280×720 → 320×180

    def __init__(self, width, height, min_freq=50.0, max_freq=70000.0,
                 min_pixels=20, max_pixels=50000, dilate_radius=8,
                 max_freq_cv=0.15, min_aspect=0.1, max_aspect=10.0):
        """
        Args:
            width, height: Sensor resolution.
            min_freq, max_freq: Frequency band to consider (Hz).
            min_pixels: Minimum active pixels for a valid light fixture.
            max_pixels: Maximum active pixels (filter out full-frame noise).
            dilate_radius: Morphological dilation to bridge gaps within a fixture.
            max_freq_cv: Maximum coefficient of variation within a fixture.
            min_aspect, max_aspect: Aspect ratio constraints for fixture shape.
        """
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.max_freq_cv = max_freq_cv
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect

        DS = self.DOWNSCALE
        self.h_full = height
        self.w_full = width
        self.h_ds = height // DS
        self.w_ds = width // DS

        # Morphological kernels
        ds_radius = max(1, dilate_radius // DS)
        k = 2 * ds_radius + 1
        self.dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        close_k = max(3, k + 2)
        self.close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k))

        # Pre-allocated buffers
        self._mask_full = np.empty((height, width), dtype=np.uint8)
        self._mask_small = np.empty((self.h_ds, self.w_ds), dtype=np.uint8)
        self._mask_processed = np.empty((self.h_ds, self.w_ds), dtype=np.uint8)

        # Timing
        self.analysis_time_ms = 0.0

    def extract(self, freq_map):
        """Extract light fixture ROIs from the per-pixel frequency map.

        Args:
            freq_map: 2D float array (H×W), value = detected frequency in Hz,
                      0 = no vibration detected at that pixel.

        Returns:
            List of dicts, each representing a detected light fixture:
              - roi_id (int): Sequential ID for this extraction
              - centroid_x, centroid_y (float): Pixel centroid in full resolution
              - bbox (tuple): (x, y, w, h) bounding box in full resolution
              - median_freq (float): Median frequency in Hz
              - mean_freq (float): Mean frequency in Hz
              - freq_std (float): Standard deviation of frequency within ROI
              - freq_cv (float): Coefficient of variation
              - pixel_count (int): Number of active pixels
              - mask (np.ndarray): Boolean mask of active pixels within bbox
        """
        t0 = time.perf_counter()
        DS = self.DOWNSCALE

        # Step 1: Binary mask of pixels in target frequency band
        valid = (freq_map >= self.min_freq) & (freq_map <= self.max_freq)
        active_count = int(np.count_nonzero(valid))
        if active_count < self.min_pixels:
            self._update_timing(t0)
            return []

        # Step 2: Downscale for fast morphology
        np.multiply(valid, 255, out=self._mask_full, casting='unsafe')
        cv2.resize(self._mask_full, (self.w_ds, self.h_ds),
                   dst=self._mask_small, interpolation=cv2.INTER_NEAREST)

        # Step 3: Morphological close (fill internal gaps) + dilate (bridge edges)
        cv2.morphologyEx(self._mask_small, cv2.MORPH_CLOSE, self.close_kernel,
                         dst=self._mask_processed)
        cv2.dilate(self._mask_processed, self.dilate_kernel,
                   dst=self._mask_processed, iterations=1)

        # Step 4: Connected components
        num_labels, labels_small, stats, centroids = \
            cv2.connectedComponentsWithStats(self._mask_processed, connectivity=8)

        rois = []
        roi_id = 0
        for i in range(1, num_labels):  # skip background
            # Map bounding box back to full resolution
            x_ds = int(stats[i, cv2.CC_STAT_LEFT])
            y_ds = int(stats[i, cv2.CC_STAT_TOP])
            w_box = int(stats[i, cv2.CC_STAT_WIDTH]) * DS
            h_box = int(stats[i, cv2.CC_STAT_HEIGHT]) * DS
            x_full = x_ds * DS
            y_full = y_ds * DS

            # Clamp to image bounds
            x2 = min(x_full + w_box, self.w_full)
            y2 = min(y_full + h_box, self.h_full)
            w_clamped = x2 - x_full
            h_clamped = y2 - y_full

            if w_clamped < 2 or h_clamped < 2:
                continue

            # Aspect ratio filter
            aspect = w_clamped / max(1, h_clamped)
            if aspect < self.min_aspect or aspect > self.max_aspect:
                continue

            # ROI analysis on full-resolution frequency map
            roi_valid = valid[y_full:y2, x_full:x2]
            pixel_count = int(np.count_nonzero(roi_valid))

            if pixel_count < self.min_pixels or pixel_count > self.max_pixels:
                continue

            roi_freqs = freq_map[y_full:y2, x_full:x2]
            cluster_freqs = roi_freqs[roi_valid]
            if len(cluster_freqs) == 0:
                continue

            median_freq = float(np.median(cluster_freqs))
            mean_freq = float(np.mean(cluster_freqs))
            freq_std = float(np.std(cluster_freqs)) if len(cluster_freqs) > 1 else 0.0
            freq_cv = freq_std / mean_freq if mean_freq > 0 else 0.0

            if freq_cv > self.max_freq_cv:
                continue

            cx = float(centroids[i][0]) * DS
            cy = float(centroids[i][1]) * DS

            rois.append({
                "roi_id": roi_id,
                "centroid_x": cx,
                "centroid_y": cy,
                "bbox": (x_full, y_full, w_clamped, h_clamped),
                "median_freq": median_freq,
                "mean_freq": mean_freq,
                "freq_std": freq_std,
                "freq_cv": freq_cv,
                "pixel_count": pixel_count,
                "mask": roi_valid.copy(),
            })
            roi_id += 1

        # Sort by pixel count descending (largest fixture first)
        rois.sort(key=lambda r: r["pixel_count"], reverse=True)
        self._update_timing(t0)
        return rois

    def _update_timing(self, t0):
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if self.analysis_time_ms == 0:
            self.analysis_time_ms = elapsed_ms
        else:
            self.analysis_time_ms = 0.2 * elapsed_ms + 0.8 * self.analysis_time_ms


# ---------------------------------------------------------------------------
#  FFT-based frequency feature extraction from event time series
# ---------------------------------------------------------------------------

class FrequencyFeatureExtractor:
    """Extracts frequency-domain features from per-ROI event accumulations.

    Divides the sensor area corresponding to a light fixture ROI into a single
    accumulation zone, counts events per time slice, and computes FFT to extract:
      - Fundamental frequency f₀
      - Harmonic amplitudes a₁..aₙ relative to fundamental
      - Spectral bandwidth (quality factor Q)
      - Peak SNR

    Follows the FFT pattern from PropellerGridAnalyzer in propeller_utils.py.
    """

    def __init__(self, fft_window_samples=512, num_harmonics=5, min_snr=3.0):
        """
        Args:
            fft_window_samples: Number of time slices in the FFT window.
            num_harmonics: Number of harmonics to extract (including fundamental).
            min_snr: Minimum signal-to-noise ratio for a valid detection.
        """
        self.fft_window_samples = fft_window_samples
        self.num_harmonics = num_harmonics
        self.min_snr = min_snr

        # Event accumulation buffer per ROI (keyed by roi_id)
        self._buffers = {}

    def reset(self):
        """Clear all accumulation buffers."""
        self._buffers.clear()

    def ensure_buffer(self, roi_id):
        """Create an accumulation buffer for a new ROI if it doesn't exist."""
        if roi_id not in self._buffers:
            self._buffers[roi_id] = deque(maxlen=self.fft_window_samples)

    def accumulate_events(self, roi_id, events, bbox):
        """Count events within the ROI's bounding box for this time slice.

        Args:
            roi_id: Identifier for the light fixture.
            events: Structured numpy array with fields x, y, t, p.
            bbox: (x, y, w, h) bounding box of the fixture.
        """
        self.ensure_buffer(roi_id)

        if events.size == 0:
            self._buffers[roi_id].append(0)
            return

        bx, by, bw, bh = bbox
        # Filter events within bounding box
        in_roi = ((events["x"] >= bx) & (events["x"] < bx + bw) &
                  (events["y"] >= by) & (events["y"] < by + bh))
        self._buffers[roi_id].append(int(np.count_nonzero(in_roi)))

    def extract_features(self, roi_id, sample_rate_hz, min_freq=0.0, max_freq=np.inf):
        """Compute frequency-domain features from the accumulated event time series.

        Args:
            roi_id: Identifier for the light fixture.
            sample_rate_hz: Sampling rate = 1e6 / delta_t.
            min_freq, max_freq: Frequency band to search for the fundamental.

        Returns:
            Dict with keys:
              - fundamental_freq (float): Detected fundamental frequency in Hz
              - harmonic_ratios (list[float]): Ratios a₁/a₀, a₂/a₀, ... aₙ/a₀
              - peak_snr (float): SNR of the fundamental peak
              - spectral_bandwidth (float): 3dB bandwidth of fundamental peak in Hz
              - spectrum (np.ndarray): Full magnitude spectrum
              - freqs (np.ndarray): Corresponding frequency axis
            Returns None if insufficient data or no significant peak found.
        """
        if roi_id not in self._buffers:
            return None

        buf = self._buffers[roi_id]
        n = len(buf)
        if n < 32:
            return None

        signal = np.array(buf, dtype=np.float64)
        if np.mean(signal) < 0.5:
            return None

        # DC removal + windowing
        sig = signal - np.mean(signal)
        window = np.hanning(n)
        sig_windowed = sig * window

        # FFT
        spectrum = np.fft.rfft(sig_windowed)
        freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)
        magnitudes = np.abs(spectrum) * 2.0 / n

        # Restrict to target band
        valid_mask = (freqs >= min_freq) & (freqs <= max_freq)
        if not np.any(valid_mask):
            return None

        valid_mags = magnitudes[valid_mask]
        valid_freqs = freqs[valid_mask]

        # Find fundamental peak
        peak_idx = np.argmax(valid_mags)
        peak_mag = valid_mags[peak_idx]
        fundamental_freq = valid_freqs[peak_idx]

        # SNR computation
        noise_mask = np.ones(len(valid_mags), dtype=bool)
        # Exclude bins near the peak
        peak_half_width = max(1, int(n * 0.01))
        low = max(0, peak_idx - peak_half_width)
        high = min(len(valid_mags), peak_idx + peak_half_width + 1)
        noise_mask[low:high] = False
        noise_floor = np.median(valid_mags[noise_mask]) if np.any(noise_mask) else 0.0
        snr = peak_mag / noise_floor if noise_floor > 0 else peak_mag

        if snr < self.min_snr:
            return None

        # Extract harmonic amplitudes
        harmonic_ratios = []
        freq_resolution = sample_rate_hz / n
        for k in range(1, self.num_harmonics + 1):
            harmonic_freq = fundamental_freq * (k + 1)
            if harmonic_freq > freqs[-1]:
                harmonic_ratios.append(0.0)
                continue
            # Find the bin closest to the harmonic frequency
            h_idx = np.argmin(np.abs(freqs - harmonic_freq))
            h_amplitude = magnitudes[h_idx]
            ratio = h_amplitude / peak_mag if peak_mag > 0 else 0.0
            harmonic_ratios.append(float(ratio))

        # Spectral bandwidth (3dB width of fundamental peak)
        half_power = peak_mag / np.sqrt(2)
        peak_global_idx = np.argmin(np.abs(freqs - fundamental_freq))
        bw_low = peak_global_idx
        bw_high = peak_global_idx
        while bw_low > 0 and magnitudes[bw_low] >= half_power:
            bw_low -= 1
        while bw_high < len(magnitudes) - 1 and magnitudes[bw_high] >= half_power:
            bw_high += 1
        spectral_bandwidth = float(freqs[bw_high] - freqs[bw_low])

        return {
            "fundamental_freq": float(fundamental_freq),
            "harmonic_ratios": harmonic_ratios,
            "peak_snr": float(snr),
            "spectral_bandwidth": spectral_bandwidth,
            "peak_magnitude": float(peak_mag),
            "spectrum": magnitudes.copy(),
            "freqs": freqs.copy(),
        }


# ---------------------------------------------------------------------------
#  Fingerprint distance metric
# ---------------------------------------------------------------------------

def frequency_distance(fp1, fp2, freq_weight=0.5, harmonic_weight=0.3,
                       bandwidth_weight=0.2):
    """Compute distance between two light fingerprints.

    Args:
        fp1, fp2: Fingerprint dicts with keys:
            - fundamental_freq (float)
            - harmonic_ratios (list[float])
            - spectral_bandwidth (float)
        freq_weight: Weight for fundamental frequency difference.
        harmonic_weight: Weight for harmonic ratio cosine distance.
        bandwidth_weight: Weight for bandwidth difference.

    Returns:
        Float distance [0, ∞). Lower = more similar.
    """
    # Fundamental frequency: relative difference
    f1, f2 = fp1["fundamental_freq"], fp2["fundamental_freq"]
    if f1 <= 0 or f2 <= 0:
        return float('inf')
    freq_dist = abs(f1 - f2) / max(f1, f2)

    # Harmonic ratios: cosine distance
    h1 = np.array(fp1.get("harmonic_ratios", [0.0]), dtype=np.float64)
    h2 = np.array(fp2.get("harmonic_ratios", [0.0]), dtype=np.float64)
    # Pad to same length
    max_len = max(len(h1), len(h2))
    h1_padded = np.zeros(max_len)
    h2_padded = np.zeros(max_len)
    h1_padded[:len(h1)] = h1
    h2_padded[:len(h2)] = h2

    dot = np.dot(h1_padded, h2_padded)
    norm1 = np.linalg.norm(h1_padded)
    norm2 = np.linalg.norm(h2_padded)
    if norm1 > 0 and norm2 > 0:
        cosine_sim = dot / (norm1 * norm2)
        harmonic_dist = 1.0 - cosine_sim
    else:
        harmonic_dist = 1.0

    # Bandwidth: relative difference
    bw1 = fp1.get("spectral_bandwidth", 0.0)
    bw2 = fp2.get("spectral_bandwidth", 0.0)
    max_bw = max(bw1, bw2)
    bw_dist = abs(bw1 - bw2) / max_bw if max_bw > 0 else 0.0

    total = (freq_weight * freq_dist +
             harmonic_weight * harmonic_dist +
             bandwidth_weight * bw_dist)
    return float(total)


# ---------------------------------------------------------------------------
#  Position estimation utilities
# ---------------------------------------------------------------------------

def weighted_centroid(positions, weights):
    """Estimate position as weighted centroid of known light positions.

    Args:
        positions: Array of shape (N, 2) or (N, 3) with light coordinates.
        weights: Array of shape (N,) with positive weights (e.g., similarity scores).

    Returns:
        Estimated position as 1D array of shape (2,) or (3,).
    """
    positions = np.asarray(positions, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    if positions.ndim != 2 or weights.ndim != 1:
        raise ValueError("positions must be (N, D) and weights must be (N,)")
    if len(positions) != len(weights):
        raise ValueError("positions and weights must have same length")
    if len(positions) == 0:
        raise ValueError("Need at least one position for centroid estimation")

    # Normalize weights
    w_sum = np.sum(weights)
    if w_sum <= 0:
        # Fallback to uniform weighting
        return np.mean(positions, axis=0)

    w_norm = weights / w_sum
    return np.sum(positions * w_norm[:, np.newaxis], axis=0)


def triangulate_from_bearings(light_positions_3d, bearing_vectors_2d,
                              camera_height=1.2):
    """Estimate 2D position from angular bearings to known ceiling lights.

    Uses least-squares intersection of bearing lines projected onto the
    horizontal plane.

    Args:
        light_positions_3d: Array (N, 3) of light positions [x, y, z] in meters.
        bearing_vectors_2d: Array (N, 2) of unit bearing vectors [dx, dy] from
                            camera to each light, projected onto the horizontal plane.
        camera_height: Height of the camera above the floor (meters).

    Returns:
        Estimated 2D position (x, y) as 1D array, or None if under-determined.
    """
    N = len(light_positions_3d)
    if N < 2:
        return None

    light_pos = np.asarray(light_positions_3d, dtype=np.float64)
    bearings = np.asarray(bearing_vectors_2d, dtype=np.float64)

    # Set up the linear system: for each light i,
    # the camera position p satisfies: (p - light_xy_i) × bearing_i = 0
    # which gives: bearing_iy * (px - lx_i) - bearing_ix * (py - ly_i) = 0
    A = np.zeros((N, 2))
    b = np.zeros(N)
    for i in range(N):
        bx, by = bearings[i]
        lx, ly = light_pos[i, 0], light_pos[i, 1]
        A[i, 0] = by
        A[i, 1] = -bx
        b[i] = by * lx - bx * ly

    # Least-squares solution
    result, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    if rank < 2:
        return None
    return result


def harmonic_ratios(spectrum, freqs, fundamental_freq, num_harmonics=5):
    """Extract harmonic amplitude ratios from a magnitude spectrum.

    Args:
        spectrum: 1D magnitude spectrum (from np.fft.rfft).
        freqs: Corresponding frequency axis (from np.fft.rfftfreq).
        fundamental_freq: Detected fundamental frequency in Hz.
        num_harmonics: Number of harmonics to extract (2nd, 3rd, ...).

    Returns:
        List of float ratios [a₂/a₁, a₃/a₁, ..., aₙ/a₁].
    """
    if fundamental_freq <= 0 or len(spectrum) == 0:
        return [0.0] * num_harmonics

    # Find fundamental amplitude
    f0_idx = np.argmin(np.abs(freqs - fundamental_freq))
    a0 = spectrum[f0_idx]
    if a0 <= 0:
        return [0.0] * num_harmonics

    ratios = []
    for k in range(2, num_harmonics + 2):
        hk_freq = fundamental_freq * k
        if hk_freq > freqs[-1]:
            ratios.append(0.0)
            continue
        hk_idx = np.argmin(np.abs(freqs - hk_freq))
        ratios.append(float(spectrum[hk_idx] / a0))

    return ratios


def pixel_to_bearing(cx, cy, camera_matrix, dist_coeffs=None):
    """Convert pixel coordinates to a 3D unit bearing vector.

    Args:
        cx, cy: Pixel coordinates of the light centroid.
        camera_matrix: 3×3 intrinsic camera matrix.
        dist_coeffs: Distortion coefficients (optional).

    Returns:
        Unit bearing vector (3,) in camera frame: [right, down, forward].
    """
    point = np.array([[[cx, cy]]], dtype=np.float64)
    if dist_coeffs is not None:
        point = cv2.undistortPoints(point, camera_matrix, dist_coeffs, P=camera_matrix)

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx0 = camera_matrix[0, 2]
    cy0 = camera_matrix[1, 2]

    px = point[0, 0, 0]
    py = point[0, 0, 1]

    bearing = np.array([
        (px - cx0) / fx,
        (py - cy0) / fy,
        1.0,
    ], dtype=np.float64)

    # Normalize
    norm = np.linalg.norm(bearing)
    if norm > 0:
        bearing /= norm
    return bearing
