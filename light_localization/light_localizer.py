"""
Light Localizer — Runtime positioning using ceiling light fingerprints.

Identifies visible ceiling lights by matching their frequency signatures against
a calibrated fingerprint database, then estimates the camera's indoor position
using weighted centroid or triangulation.

Detection pipeline:
  1. Per-pixel frequency map (FrequencyMapAsyncAlgorithm from Metavision SDK)
  2. Spatial segmentation into individual light fixture ROIs (LightROIExtractor)
  3. Frequency feature extraction per ROI (FrequencyFeatureExtractor / SDK dominant freq)
  4. Fingerprint matching against calibrated database (FingerprintDatabase.find_nearest)
  5. Position estimation (weighted centroid / AOA triangulation)

Usage:
    localizer = LightLocalizer(database_path="calibration_data/room1.json")
    # In the event processing loop:
    position = localizer.process_frequency_map(freq_map)
"""

import time
import numpy as np
from typing import Optional

from light_localization_utils import (
    LightROIExtractor,
    FrequencyFeatureExtractor,
    weighted_centroid,
    triangulate_from_bearings,
    pixel_to_bearing,
)
from light_fingerprint import FingerprintDatabase, LightFingerprint


# ---------------------------------------------------------------------------
#  Light Matcher — links observed ROIs to known fingerprints
# ---------------------------------------------------------------------------

class LightMatcher:
    """Matches observed light ROIs to known fingerprints in the database.

    Uses a two-stage matching strategy:
      1. Coarse: filter candidates by frequency band and approximate frequency
      2. Fine: rank candidates by full feature distance (frequency + harmonics)

    Maintains temporal consistency: if a light was matched in the previous frame,
    prefer the same match unless a significantly better candidate appears.
    """

    def __init__(self, database, max_distance=0.3, freq_tolerance_hz=2000.0):
        """
        Args:
            database: FingerprintDatabase instance.
            max_distance: Maximum feature distance to accept a match.
            freq_tolerance_hz: Coarse frequency tolerance for candidate filtering.
        """
        self.database = database
        self.max_distance = max_distance
        self.freq_tolerance_hz = freq_tolerance_hz

        # Temporal consistency: previous frame's matches
        self._prev_matches = {}  # roi_id → (light_id, distance)

    def match(self, rois, features_per_roi, room_id=None):
        """Match observed light ROIs to known fingerprints.

        Args:
            rois: List of dicts from LightROIExtractor.extract().
            features_per_roi: Dict mapping roi_id → feature dict from
                              FrequencyFeatureExtractor, or None for SDK-only features.
            room_id: Restrict matching to a specific room.

        Returns:
            List of match dicts:
              - roi: The original ROI dict
              - fingerprint: Matched LightFingerprint (or None)
              - distance: Feature distance (lower = better)
              - confidence: Match confidence [0, 1]
        """
        matches = []
        used_light_ids = set()

        for roi in rois:
            rid = roi["roi_id"]

            # Build query feature vector
            if features_per_roi and rid in features_per_roi and features_per_roi[rid] is not None:
                query = features_per_roi[rid]
            else:
                # Fallback: use SDK frequency map features only
                query = {
                    "fundamental_freq": roi["median_freq"],
                    "harmonic_ratios": [],
                    "spectral_bandwidth": roi.get("freq_std", 0.0) * 2,
                }

            # Stage 1: coarse filter by frequency
            candidates = self.database.find_by_frequency(
                query["fundamental_freq"],
                tolerance_hz=self.freq_tolerance_hz,
                room_id=room_id,
            )

            if not candidates:
                # Fall back to nearest-neighbor search
                nearest = self.database.find_nearest(
                    query, top_k=3, room_id=room_id
                )
                candidates_with_dist = nearest
            else:
                # Stage 2: fine ranking by full feature distance
                candidates_with_dist = []
                for fp in candidates:
                    if fp.light_id in used_light_ids:
                        continue
                    from light_localization_utils import frequency_distance
                    dist = frequency_distance(query, fp.feature_vector())
                    candidates_with_dist.append((fp, dist))
                candidates_with_dist.sort(key=lambda x: x[1])

            # Select best match
            best_fp = None
            best_dist = float('inf')
            for fp, dist in candidates_with_dist:
                if fp.light_id not in used_light_ids and dist <= self.max_distance:
                    best_fp = fp
                    best_dist = dist
                    break

            # Temporal consistency bonus: prefer previous match if close
            if rid in self._prev_matches:
                prev_id, prev_dist = self._prev_matches[rid]
                if best_fp and best_fp.light_id != prev_id:
                    # Only switch if new match is significantly better
                    prev_fp = self.database.get(prev_id)
                    if prev_fp and prev_id not in used_light_ids:
                        from light_localization_utils import frequency_distance
                        prev_dist_now = frequency_distance(query, prev_fp.feature_vector())
                        if prev_dist_now < best_dist * 1.3:  # 30% hysteresis
                            best_fp = prev_fp
                            best_dist = prev_dist_now

            confidence = max(0.0, 1.0 - best_dist / self.max_distance) if best_fp else 0.0

            match_result = {
                "roi": roi,
                "fingerprint": best_fp,
                "distance": best_dist,
                "confidence": confidence,
            }
            matches.append(match_result)

            if best_fp:
                used_light_ids.add(best_fp.light_id)
                self._prev_matches[rid] = (best_fp.light_id, best_dist)

        return matches


# ---------------------------------------------------------------------------
#  Position Estimator — converts matched lights to a position estimate
# ---------------------------------------------------------------------------

class PositionEstimator:
    """Estimates camera position from matched ceiling lights.

    Supports two methods:
      1. Weighted centroid: uses match confidence × signal strength as weights
      2. AOA triangulation: uses pixel bearing vectors to known 3D light positions

    Includes temporal smoothing via exponential moving average.
    """

    def __init__(self, method="centroid", smoothing_alpha=0.3,
                 camera_matrix=None, dist_coeffs=None, camera_height=1.2):
        """
        Args:
            method: 'centroid' or 'triangulation'.
            smoothing_alpha: EMA smoothing factor (0 = no update, 1 = no smoothing).
            camera_matrix: 3×3 intrinsic matrix (required for triangulation).
            dist_coeffs: Distortion coefficients (optional).
            camera_height: Camera height above floor in meters.
        """
        self.method = method
        self.smoothing_alpha = smoothing_alpha
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.camera_height = camera_height

        self._position_ema = None  # Smoothed position estimate
        self._last_timestamp = 0

    def estimate(self, matches, timestamp=None):
        """Estimate position from a list of light matches.

        Args:
            matches: List of match dicts from LightMatcher.match().
            timestamp: Current timestamp in microseconds (for temporal tracking).

        Returns:
            Dict with keys:
              - position_2d (np.ndarray): Estimated (x, y) in meters, or None
              - position_raw (np.ndarray): Unsmoothed estimate
              - num_matches (int): Number of valid matches used
              - matched_lights (list[str]): IDs of matched lights
              - confidence (float): Overall confidence [0, 1]
              - method (str): Method used for this estimate
        """
        # Filter to valid matches only
        valid = [m for m in matches
                 if m["fingerprint"] is not None and m["confidence"] > 0.1]

        if not valid:
            return {
                "position_2d": self._position_ema,
                "position_raw": None,
                "num_matches": 0,
                "matched_lights": [],
                "confidence": 0.0,
                "method": self.method,
            }

        if self.method == "triangulation" and self.camera_matrix is not None and len(valid) >= 2:
            raw_pos = self._triangulate(valid)
        else:
            raw_pos = self._centroid(valid)

        # Temporal smoothing
        if raw_pos is not None:
            if self._position_ema is None:
                self._position_ema = raw_pos.copy()
            else:
                self._position_ema = (self.smoothing_alpha * raw_pos +
                                      (1 - self.smoothing_alpha) * self._position_ema)

        # Overall confidence: weighted average of match confidences
        confidences = [m["confidence"] for m in valid]
        overall_conf = float(np.mean(confidences)) if confidences else 0.0

        return {
            "position_2d": self._position_ema.copy() if self._position_ema is not None else None,
            "position_raw": raw_pos,
            "num_matches": len(valid),
            "matched_lights": [m["fingerprint"].light_id for m in valid],
            "confidence": overall_conf,
            "method": self.method,
        }

    def _centroid(self, valid_matches):
        """Weighted centroid position estimation."""
        positions = []
        weights = []
        for m in valid_matches:
            fp = m["fingerprint"]
            pos = fp.position_3d
            positions.append([pos[0], pos[1]])  # Use x, y only
            # Weight: confidence × pixel count (proxy for proximity/signal strength)
            w = m["confidence"] * max(1, fp.pixel_count / 100.0)
            weights.append(w)

        return weighted_centroid(np.array(positions), np.array(weights))

    def _triangulate(self, valid_matches):
        """AOA triangulation position estimation."""
        light_positions = []
        bearings = []

        for m in valid_matches:
            fp = m["fingerprint"]
            roi = m["roi"]

            light_positions.append(list(fp.position_3d))
            bearing_3d = pixel_to_bearing(
                roi["centroid_x"], roi["centroid_y"],
                self.camera_matrix, self.dist_coeffs
            )
            # Project to horizontal plane (take x, y components)
            bearing_2d = bearing_3d[:2]
            norm = np.linalg.norm(bearing_2d)
            if norm > 0:
                bearing_2d = bearing_2d / norm
            bearings.append(bearing_2d)

        result = triangulate_from_bearings(
            np.array(light_positions),
            np.array(bearings),
            camera_height=self.camera_height,
        )
        return result

    def reset(self):
        """Reset the temporal smoothing state."""
        self._position_ema = None


# ---------------------------------------------------------------------------
#  LightLocalizer — main facade combining all components
# ---------------------------------------------------------------------------

class LightLocalizer:
    """Main localization class combining detection, matching, and positioning.

    This is the primary interface for the localization system. It wraps:
      - LightROIExtractor (spatial segmentation)
      - FrequencyFeatureExtractor (FFT harmonic analysis)
      - LightMatcher (fingerprint matching)
      - PositionEstimator (position computation)

    Usage in the event processing loop:
        localizer = LightLocalizer(database_path="calibration_data/room1.json",
                                   width=1280, height=720)

        # Option A: Process a frequency map from FrequencyMapAsyncAlgorithm
        result = localizer.process_frequency_map(freq_map)

        # Option B: Also feed raw events for FFT analysis (higher accuracy)
        localizer.accumulate_events(events)
        result = localizer.process_frequency_map(freq_map, use_fft=True,
                                                  sample_rate_hz=2000)
    """

    def __init__(self, database_path=None, database=None,
                 width=1280, height=720, room_id=None,
                 min_freq=50.0, max_freq=70000.0,
                 estimation_method="centroid",
                 camera_matrix=None, camera_height=1.2,
                 smoothing_alpha=0.3,
                 max_match_distance=0.3,
                 fft_window_samples=512):
        """
        Args:
            database_path: Path to a FingerprintDatabase JSON file.
            database: Pre-loaded FingerprintDatabase (alternative to path).
            width, height: Sensor resolution.
            room_id: Restrict matching to a specific room.
            min_freq, max_freq: Frequency range for light detection.
            estimation_method: 'centroid' or 'triangulation'.
            camera_matrix: 3×3 intrinsics (for triangulation).
            camera_height: Camera height in meters.
            smoothing_alpha: Temporal smoothing factor [0, 1].
            max_match_distance: Maximum distance to accept a fingerprint match.
            fft_window_samples: FFT window size for harmonic analysis.
        """
        # Load database
        if database is not None:
            self.database = database
        elif database_path is not None:
            self.database = FingerprintDatabase.load(database_path)
        else:
            self.database = FingerprintDatabase()

        self.room_id = room_id
        self.width = width
        self.height = height

        # Components
        self.roi_extractor = LightROIExtractor(
            width, height,
            min_freq=min_freq, max_freq=max_freq,
            min_pixels=20, max_pixels=50000,
            dilate_radius=8, max_freq_cv=0.15,
        )

        self.feature_extractor = FrequencyFeatureExtractor(
            fft_window_samples=fft_window_samples,
            num_harmonics=5, min_snr=3.0,
        )

        self.matcher = LightMatcher(
            self.database,
            max_distance=max_match_distance,
        )

        self.estimator = PositionEstimator(
            method=estimation_method,
            smoothing_alpha=smoothing_alpha,
            camera_matrix=camera_matrix,
            camera_height=camera_height,
        )

        # State
        self._current_rois = []
        self._frame_count = 0
        self._last_result = None

        # Timing
        self.total_time_ms = 0.0

    def accumulate_events(self, events):
        """Feed raw events for FFT-based harmonic feature extraction.

        Should be called every time slice (delta_t) before process_frequency_map().

        Args:
            events: Structured numpy array with fields x, y, t, p.
        """
        for roi in self._current_rois:
            self.feature_extractor.accumulate_events(
                roi["roi_id"], events, roi["bbox"]
            )

    def process_frequency_map(self, freq_map, use_fft=False, sample_rate_hz=2000.0,
                              timestamp=None):
        """Process a frequency map to estimate the camera's position.

        Args:
            freq_map: 2D float array from FrequencyMapAsyncAlgorithm.
            use_fft: If True, use FFT features from accumulate_events() for matching.
            sample_rate_hz: Effective sample rate = 1e6 / delta_t.
            timestamp: Current timestamp in microseconds.

        Returns:
            Dict with keys:
              - position_2d: Estimated (x, y) in meters (or None)
              - num_lights: Number of detected light fixtures
              - num_matches: Number of successfully matched lights
              - matched_lights: List of matched light IDs
              - confidence: Overall confidence [0, 1]
              - rois: Detected ROI details
              - timing_ms: Processing time in milliseconds
        """
        t0 = time.perf_counter()

        # Step 1: Extract light fixture ROIs
        rois = self.roi_extractor.extract(freq_map)
        self._current_rois = rois

        # Step 2: Extract FFT features if requested
        features_per_roi = {}
        if use_fft:
            for roi in rois:
                features = self.feature_extractor.extract_features(
                    roi["roi_id"], sample_rate_hz,
                    min_freq=self.roi_extractor.min_freq,
                    max_freq=self.roi_extractor.max_freq,
                )
                if features is not None:
                    features_per_roi[roi["roi_id"]] = features

        # Step 3: Match ROIs to known fingerprints
        matches = self.matcher.match(rois, features_per_roi, room_id=self.room_id)

        # Step 4: Estimate position
        pos_result = self.estimator.estimate(matches, timestamp=timestamp)

        # Update timing
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.total_time_ms = 0.2 * elapsed_ms + 0.8 * self.total_time_ms
        self._frame_count += 1

        result = {
            "position_2d": pos_result["position_2d"],
            "position_raw": pos_result["position_raw"],
            "num_lights": len(rois),
            "num_matches": pos_result["num_matches"],
            "matched_lights": pos_result["matched_lights"],
            "confidence": pos_result["confidence"],
            "method": pos_result["method"],
            "rois": rois,
            "matches": matches,
            "timing_ms": elapsed_ms,
            "frame": self._frame_count,
        }
        self._last_result = result
        return result

    def get_status(self):
        """Return a human-readable status string for display overlay."""
        if self._last_result is None:
            return "Light Localizer: no data"

        r = self._last_result
        pos = r["position_2d"]
        pos_str = f"({pos[0]:.2f}, {pos[1]:.2f})" if pos is not None else "unknown"
        return (
            f"Lights: {r['num_lights']} detected, {r['num_matches']} matched | "
            f"Pos: {pos_str} | Conf: {r['confidence']:.0%} | "
            f"{self.total_time_ms:.1f}ms"
        )

    def reset(self):
        """Reset all state (for re-initialization)."""
        self.feature_extractor.reset()
        self.estimator.reset()
        self._current_rois = []
        self._frame_count = 0
        self._last_result = None
