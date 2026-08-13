"""Reusable Metavision Pro (SDK) pipeline stages — verified working.

This module wraps the Metavision SDK Pro algorithms that (a) have Python
bindings and (b) were verified end-to-end against synthetic events in the
project's Python 3.9 / SDK environment. It gives the detectors
(detect_propeller.py, detect_active_markers.py, light_localization/...) a
single, correct place to consume these algorithms instead of hand-rolled
equivalents.

Verified stages
---------------
Filters (EventCD -> EventCD):
  * ActivityNoiseFilterAlgorithm    - drops isolated noise events
  * SpatioTemporalContrastAlgorithm - STC contrast/redundancy filter
  * TrailFilterAlgorithm            - trailing-event filter
  * AntiFlickerAlgorithm            - removes mains/ambient flicker band
  * RoiMaskAlgorithm                - spatial region-of-interest gating
Frequency detection:
  * FrequencyAlgorithm + FrequencyClusteringAlgorithm  - per-event flicker
    frequency estimation grouped into spatial clusters (SDK replacement for
    the custom connected-component clustering in propeller_utils.py).
Tracking:
  * TrackingAlgorithm (+ TrackingConfig) - generic motion tracking.
Maps / viz:
  * FrequencyMapAsyncAlgorithm      - per-pixel frequency map
  * DominantValueMapAlgorithm       - dominant value of a map
  * HeatMapFrameGeneratorAlgorithm  - BGR heat-map rendering

Environment (IMPORTANT)
-----------------------
Run only with the SDK-compatible interpreter:
    C:\\Users\\z003n5uc\\AppData\\Local\\Programs\\Python\\Python39\\python.exe
with the native DLL folder on PATH:
    $env:PATH = "C:\\Users\\z003n5uc\\Desktop\\event-based\\bin;" + $env:PATH

Self-test (no camera needed):
    python sdk_star_stages.py
"""
from __future__ import annotations

import numpy as np

from metavision_sdk_analytics import (
    DominantValueMapAlgorithm,
    FrequencyMapAsyncAlgorithm,
    HeatMapFrameGeneratorAlgorithm,
    TrackingAlgorithm,
    TrackingConfig,
)
from metavision_sdk_cv import (
    ActivityNoiseFilterAlgorithm,
    AntiFlickerAlgorithm,
    FrequencyAlgorithm,
    FrequencyClusteringAlgorithm,
    RoiMaskAlgorithm,
    SpatioTemporalContrastAlgorithm,
    TrailFilterAlgorithm,
)


class EventPreprocessor:
    """Chain of noise / flicker / ROI filters applied to EventCD buffers.

    Each enabled stage runs in order. ``process`` returns a NumPy EventCD
    array so it can be fed straight into the frequency or tracking stages.
    """

    def __init__(
        self,
        width: int,
        height: int,
        *,
        noise_filter: str | None = "stc",       # "stc" | "activity" | "trail" | None
        noise_threshold_us: int = 10_000,
        stc_cut_trail: bool = True,
        anti_flicker: bool = False,
        flicker_band_hz: tuple[float, float] = (45.0, 65.0),
        flicker_filter_length: int = 7,
        roi_rect: tuple[int, int, int, int] | None = None,  # (x0, y0, x1, y1)
    ) -> None:
        self.width = width
        self.height = height
        self._stages = []

        if noise_filter == "stc":
            f = SpatioTemporalContrastAlgorithm(width, height, noise_threshold_us,
                                                stc_cut_trail)
        elif noise_filter == "activity":
            f = ActivityNoiseFilterAlgorithm(width, height, noise_threshold_us)
        elif noise_filter == "trail":
            f = TrailFilterAlgorithm(width, height, noise_threshold_us)
        elif noise_filter is None:
            f = None
        else:
            raise ValueError(f"unknown noise_filter: {noise_filter!r}")
        if f is not None:
            self._stages.append(f)

        if anti_flicker:
            self._stages.append(
                AntiFlickerAlgorithm(width, height,
                                     filter_length=flicker_filter_length,
                                     min_freq=flicker_band_hz[0],
                                     max_freq=flicker_band_hz[1])
            )

        if roi_rect is not None:
            mask = np.zeros((height, width), dtype=np.float64)
            x0, y0, x1, y1 = roi_rect
            mask[y0:y1, x0:x1] = 1.0
            self._stages.append(RoiMaskAlgorithm(mask))

        self._buffers = [s.get_empty_output_buffer() for s in self._stages]

    def process(self, events: np.ndarray) -> np.ndarray:
        """Run every enabled filter in sequence; return filtered EventCD array."""
        data = events
        for stage, buf in zip(self._stages, self._buffers):
            stage.process_events(data, buf)
            data = buf.numpy()
        return data


class FrequencyClusterDetector:
    """Per-event frequency estimation + spatial clustering.

    SDK-native replacement for the connected-component clustering in
    ``propeller_utils.py``. ``process`` returns the clustering output buffer's
    NumPy view with fields:
    ``(x, y, t, frequency, id, n_pixels, n_events)``.
    """

    def __init__(
        self,
        width: int,
        height: int,
        *,
        min_freq: float = 10.0,
        max_freq: float = 300.0,
        filter_length: int = 4,
        diff_thresh_us: int = 1500,
        min_cluster_size: int = 3,
        max_frequency_diff: float = 10.0,
        max_time_diff_us: int = 100_000,
        filter_alpha: float = 0.1,
    ) -> None:
        self.freq = FrequencyAlgorithm(width, height, filter_length=filter_length,
                                       min_freq=min_freq, max_freq=max_freq,
                                       diff_thresh_us=diff_thresh_us)
        self.clustering = FrequencyClusteringAlgorithm(
            width, height, min_cluster_size=min_cluster_size,
            max_frequency_diff=max_frequency_diff, max_time_diff=max_time_diff_us,
            filter_alpha=filter_alpha,
        )
        self._freq_buf = self.freq.get_empty_output_buffer()
        self._clus_buf = self.clustering.get_empty_output_buffer()

    def process(self, events: np.ndarray) -> np.ndarray:
        self.freq.process_events(events, self._freq_buf)
        self.clustering.process_events(self._freq_buf, self._clus_buf)
        return self._clus_buf.numpy()


class MotionTracker:
    """Thin wrapper over the SDK generic ``TrackingAlgorithm``.

    Returns tracking events with fields ``(x, y, t, x_, y_, width_, height_,
    object_id_, event_id_)``.
    """

    def __init__(self, width: int, height: int, config: TrackingConfig | None = None):
        self.algo = TrackingAlgorithm(width, height, config or TrackingConfig())
        self._buf = self.algo.get_empty_output_buffer()

    def process(self, events: np.ndarray) -> np.ndarray:
        self.algo.process_events(events, self._buf)
        return self._buf.numpy()


class FrequencyMapViz:
    """Per-pixel frequency map, dominant value and BGR heat-map rendering.

    ``update`` pushes events and refreshes the latest float32 frequency map.
    ``dominant`` returns ``(is_valid, value_hz)``. ``render`` returns a BGR
    image (height includes an SDK-drawn value-scale legend bar).
    """

    def __init__(
        self,
        width: int,
        height: int,
        *,
        min_freq: float = 10.0,
        max_freq: float = 300.0,
        filter_length: int = 4,
        diff_thresh_us: int = 1500,
        freq_precision: float = 5.0,
        min_pixel_count: int = 5,
    ) -> None:
        self.width = width
        self.height = height
        self._map: np.ndarray | None = None
        self.freq_map = FrequencyMapAsyncAlgorithm(
            width, height, filter_length=filter_length, min_freq=min_freq,
            max_freq=max_freq, diff_thresh_us=diff_thresh_us,
        )
        self.freq_map.set_output_callback(self._on_map)
        self.dominant_algo = DominantValueMapAlgorithm(
            min_freq, max_freq, freq_precision, min_pixel_count)
        self.heat = HeatMapFrameGeneratorAlgorithm(
            min_freq, max_freq, freq_precision, width, height, "Hz")
        # HeatMap renders into a frame taller than the sensor (legend bar).
        self._heat_img = np.asarray(self.heat.get_output_image())

    def _on_map(self, ts, freq_map) -> None:
        self._map = np.array(freq_map, copy=True)

    def update(self, events: np.ndarray) -> np.ndarray | None:
        self.freq_map.process_events(events)
        return self._map

    def dominant(self) -> tuple[bool, float]:
        if self._map is None:
            return (False, 0.0)
        return self.dominant_algo.compute_dominant_value(self._map)

    def render(self) -> np.ndarray | None:
        if self._map is None:
            return None
        self.heat.generate_bgr_heat_map(self._map.astype(np.float32), self._heat_img)
        return np.asarray(self.heat.get_output_image())


# --------------------------------------------------------------------------- #
#  Self-test: synthetic events, no camera required.                           #
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    from metavision_sdk_base import EventCD

    W, H, FREQ, DUR = 320, 240, 100.0, 400_000
    half = int(1e6 / (2 * FREQ))
    ev = []
    for i, t in enumerate(range(0, DUR, half)):
        pol = i % 2
        for x in range(150, 166):
            for y in range(110, 126):
                ev.append((x, y, pol, t))
    rng = np.random.default_rng(0)
    for _ in range(4000):
        ev.append((int(rng.integers(0, W)), int(rng.integers(0, H)),
                   int(rng.integers(0, 2)), int(rng.integers(0, DUR))))
    events = np.array(ev, dtype=EventCD)
    events.sort(order="t")

    ok = True

    pre = EventPreprocessor(W, H, noise_filter="activity", anti_flicker=True,
                            flicker_band_hz=(45, 65), roi_rect=(120, 90, 200, 160))
    filtered = pre.process(events)
    print(f"[preproc]  {len(events)} -> {len(filtered)} events")
    ok &= 0 < len(filtered) <= len(events)

    det = FrequencyClusterDetector(W, H, min_freq=50, max_freq=200,
                                   min_cluster_size=3, max_frequency_diff=10)
    clusters = det.process(events)
    freqs = clusters["frequency"] if len(clusters) else np.array([])
    print(f"[freqclus] {len(clusters)} cluster(s); "
          f"freq~{np.median(freqs):.1f}Hz" if len(clusters) else "[freqclus] 0 clusters")
    ok &= len(clusters) >= 1 and abs(np.median(freqs) - FREQ) / FREQ < 0.2

    viz = FrequencyMapViz(W, H, min_freq=50, max_freq=200)
    viz.update(events)
    valid, val = viz.dominant()
    img = viz.render()
    print(f"[map]      dominant valid={valid} value={val:.1f}Hz; "
          f"heatmap={None if img is None else img.shape}")
    ok &= valid and abs(val - FREQ) / FREQ < 0.2 and img is not None and img.sum() > 0

    # moving blob for tracker
    mv = []
    steps = DUR // 1000
    for k, t in enumerate(range(0, DUR, 1000)):
        cx = 40 + (k * 240) // steps
        for dx in range(-10, 11):
            for dy in range(-10, 11):
                mv.append((min(max(cx + dx, 0), W - 1),
                           min(max(120 + dy, 0), H - 1),
                           int(rng.integers(0, 2)), t))
    moving = np.array(mv, dtype=EventCD)
    moving.sort(order="t")
    trk = MotionTracker(W, H)
    tracks = trk.process(moving)
    print(f"[tracker]  {len(tracks)} tracking event(s)")
    ok &= len(tracks) >= 1

    print("=" * 50)
    print("SELF-TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import os
    import sys

    _bin = r"C:\Users\z003n5uc\Desktop\event-based\bin"
    if hasattr(os, "add_dll_directory") and os.path.isdir(_bin):
        os.add_dll_directory(_bin)
    sys.exit(_selftest())
