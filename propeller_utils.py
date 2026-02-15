"""
Shared propeller detection utilities used by detect_propeller.py and
detect_drone_fused.py.

Contains:
  - PropellerGridAnalyzer: grid-based event-rate FFT for periodic signal detection
  - cluster_detections: groups adjacent grid cells with similar frequencies
"""

import numpy as np
from collections import deque


class PropellerGridAnalyzer:
    """Divides the sensor into a grid and runs FFT on event rates per cell.

    Each cell accumulates event counts per time slice. FFT on this time series
    reveals periodic activity at propeller blade-pass frequencies.
    """

    def __init__(self, width, height, grid_cells, fft_window_samples,
                 min_freq, max_freq, num_blades, min_snr):
        self.width = width
        self.height = height
        self.grid_cells = grid_cells
        self.cell_w = width / grid_cells
        self.cell_h = height / grid_cells
        self.num_blades = num_blades
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.min_snr = min_snr

        # Per-cell event count buffers (rolling window)
        n_cells = grid_cells * grid_cells
        self.buffers = [deque(maxlen=fft_window_samples) for _ in range(n_cells)]

    def accumulate(self, events):
        """Count events falling into each grid cell for this time slice."""
        if events.size == 0:
            for buf in self.buffers:
                buf.append(0)
            return

        cols = np.clip((events["x"] / self.cell_w).astype(int), 0, self.grid_cells - 1)
        rows = np.clip((events["y"] / self.cell_h).astype(int), 0, self.grid_cells - 1)
        indices = rows * self.grid_cells + cols

        counts = np.bincount(indices, minlength=self.grid_cells * self.grid_cells)
        for i, buf in enumerate(self.buffers):
            buf.append(int(counts[i]))

    def analyze(self, fs):
        """Run FFT on each cell and return detected propeller regions.

        Args:
            fs: Sampling rate in Hz.

        Returns:
            List of dicts with keys: row, col, freq_hz, rpm, snr, magnitude
        """
        detections = []
        for r in range(self.grid_cells):
            for c in range(self.grid_cells):
                idx = r * self.grid_cells + c
                buf = self.buffers[idx]
                n = len(buf)
                if n < 16:
                    continue

                signal = np.array(buf, dtype=np.float64)
                # Skip cells with almost no events
                if np.mean(signal) < 1.0:
                    continue

                sig = signal - np.mean(signal)
                window = np.hanning(n)
                sig_windowed = sig * window

                spectrum = np.fft.rfft(sig_windowed)
                freqs = np.fft.rfftfreq(n, d=1.0 / fs)
                magnitudes = np.abs(spectrum) * 2.0 / n

                # Only consider target frequency range
                valid = (freqs >= self.min_freq) & (freqs <= self.max_freq)
                if not np.any(valid):
                    continue

                valid_mags = magnitudes[valid]
                valid_freqs = freqs[valid]

                peak_idx = np.argmax(valid_mags)
                peak_mag = valid_mags[peak_idx]
                peak_freq = valid_freqs[peak_idx]

                # SNR: peak magnitude vs median of the rest
                noise_floor = np.median(valid_mags)
                if noise_floor > 0:
                    snr = peak_mag / noise_floor
                else:
                    snr = peak_mag  # if no noise, any signal is good

                if snr >= self.min_snr:
                    rpm = (peak_freq / self.num_blades) * 60
                    detections.append({
                        "row": r, "col": c,
                        "freq_hz": float(peak_freq),
                        "rpm": float(rpm),
                        "snr": float(snr),
                        "magnitude": float(peak_mag),
                    })
        return detections


def cluster_detections(detections):
    """Group adjacent grid cells with similar frequencies into propeller clusters.

    Returns list of dicts: center_row, center_col, freq_hz, rpm, n_cells, avg_snr
    """
    if not detections:
        return []

    # Simple connected-component clustering on grid adjacency + frequency similarity
    used = [False] * len(detections)
    clusters = []

    for i, det in enumerate(detections):
        if used[i]:
            continue
        # Start a new cluster
        cluster = [det]
        used[i] = True
        queue = [i]

        while queue:
            ci = queue.pop(0)
            cd = detections[ci]
            for j, other in enumerate(detections):
                if used[j]:
                    continue
                # Adjacent in grid?
                dr = abs(cd["row"] - other["row"])
                dc = abs(cd["col"] - other["col"])
                if dr <= 1 and dc <= 1:
                    # Similar frequency? (within 20%)
                    freq_ratio = max(cd["freq_hz"], other["freq_hz"]) / max(min(cd["freq_hz"], other["freq_hz"]), 1e-6)
                    if freq_ratio <= 1.2:
                        cluster.append(other)
                        used[j] = True
                        queue.append(j)

        # Summarize cluster
        freqs = [d["freq_hz"] for d in cluster]
        rpms = [d["rpm"] for d in cluster]
        snrs = [d["snr"] for d in cluster]
        rows = [d["row"] for d in cluster]
        cols = [d["col"] for d in cluster]
        clusters.append({
            "center_row": float(np.mean(rows)),
            "center_col": float(np.mean(cols)),
            "freq_hz": float(np.median(freqs)),
            "rpm": float(np.median(rpms)),
            "n_cells": len(cluster),
            "avg_snr": float(np.mean(snrs)),
        })

    # Sort by number of cells (largest cluster = most confident)
    clusters.sort(key=lambda c: c["n_cells"], reverse=True)
    return clusters
