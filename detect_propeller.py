"""
Drone propeller rotation detector for event-based cameras.

Detects spinning propellers in the field of view and estimates their
rotational frequency (Hz) and RPM in real-time.

Detection pipeline:
  1. Per-pixel frequency map  (FrequencyMapAsyncAlgorithm from Metavision SDK)
     - Each pixel independently detects periodic brightness changes.
  2. Connected-component spatial clustering
     - Groups adjacent vibrating pixels with similar frequencies.
     - Even a few pixels (3+) at a consistent frequency = candidate propeller.
  3. Temporal tracking
     - Propeller candidates must persist across multiple frames to be confirmed.
     - Smooths position and frequency estimates over time.
     - Eliminates transient noise spikes.

Typical drone propeller frequencies:
    Heavy-lift   :  67 - 167 Hz  (4,000 - 10,000 RPM)
    Camera drones:  67 - 250 Hz  (4,000 - 15,000 RPM)
    Racing/FPV   : 500 - 833 Hz  (30,000 - 50,000 RPM)
  Blade-pass frequency = rotation_freq * num_blades (usually x2 for 2-blade)

Usage:
    python detect_propeller.py -i <path_to_event_file>
    python detect_propeller.py                            # live camera
    python detect_propeller.py -i recording.raw --max-freq 800 --num-blades 3
"""

import argparse
import time
import numpy as np

from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator, is_live_camera
from metavision_sdk_analytics import FrequencyMapAsyncAlgorithm, DominantValueMapAlgorithm, \
    HeatMapFrameGeneratorAlgorithm
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction, UIKeyEvent
import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect drone propeller rotation frequency and RPM from event camera data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Input
    parser.add_argument(
        "-i", "--input-event-file", dest="event_file_path", default="",
        help="Path to input event file (RAW, DAT or HDF5). If not specified, the live camera is used.",
    )

    # Frequency range
    parser.add_argument(
        "--min-freq", dest="min_freq", type=float, default=10,
        help="Minimum propeller frequency to detect (Hz).",
    )
    parser.add_argument(
        "--max-freq", dest="max_freq", type=float, default=300,
        help="Maximum propeller frequency to detect (Hz). "
             "Set higher for racing drones (e.g. 1000).",
    )
    parser.add_argument(
        "--num-blades", dest="num_blades", type=int, default=3,
        help="Number of propeller blades. RPM = (detected_freq / num_blades) * 60.",
    )

    # SDK frequency algorithm tuning
    parser.add_argument(
        "--filter-length", dest="filter_length", type=int, default=4,
        help="Number of successive periods to confirm a vibration. "
             "Lower = more sensitive to weak/distant signals.",
    )
    parser.add_argument(
        "--max-period-diff", dest="max_period_diff", type=int, default=1500,
        help="Max difference between two periods to be considered the same (us). "
             "Higher = more tolerant of noisy/distant signals.",
    )
    parser.add_argument(
        "--freq-precision", dest="freq_precision", type=float, default=5.0,
        help="Width of frequency bins in Hz for histogram display.",
    )

    # Spatial clustering
    parser.add_argument(
        "--min-cluster-pixels", dest="min_cluster_pixels", type=int, default=2,
        help="Minimum vibrating pixels in a connected cluster to be a candidate. "
             "Set to 1 only for extreme range (very noisy — pair with higher min-hits).",
    )
    parser.add_argument(
        "--dilate-radius", dest="dilate_radius", type=int, default=5,
        help="Morphological dilation radius to connect nearby vibrating pixels. "
             "Larger = bridges bigger gaps (good for sparse distant signals).",
    )
    parser.add_argument(
        "--max-freq-cv", dest="max_freq_cv", type=float, default=0.3,
        help="Max coefficient of variation of frequencies within a cluster. "
             "Lower = stricter frequency consistency required.",
    )

    # Temporal tracking
    parser.add_argument(
        "--min-hits", dest="min_hits", type=int, default=5,
        help="Minimum consecutive detections before a propeller is confirmed. "
             "Higher = fewer false positives but slower response.",
    )
    parser.add_argument(
        "--max-age", dest="max_age", type=int, default=8,
        help="Frames without detection before a track is dropped.",
    )
    parser.add_argument(
        "--track-distance", dest="track_distance", type=float, default=100,
        help="Max pixel distance to match a new detection to an existing track.",
    )
    parser.add_argument(
        "--freq-tolerance", dest="freq_tolerance", type=float, default=0.3,
        help="Max relative frequency difference to match detection to track.",
    )

    # Timing
    parser.add_argument(
        "--delta-t", dest="delta_t", type=int, default=500,
        help="Event slice duration (us). Determines sampling rate. "
             "Default 500 us -> 2000 Hz sampling -> 1000 Hz Nyquist limit.",
    )
    parser.add_argument(
        "--update-freq", dest="update_freq", type=float, default=10,
        help="How often (Hz) the frequency heatmap and detections update. "
             "Lower = less CPU load = less latency.",
    )

    # Replay
    parser.add_argument(
        "-f", "--replay-factor", dest="replay_factor", type=float, default=1,
        help="Replay speed factor (>1 = slow-motion, <1 = speed-up).",
    )

    args = parser.parse_args()

    # Validate Nyquist
    nyquist = 1e6 / args.delta_t / 2
    if args.max_freq > nyquist:
        parser.error(
            f"--max-freq ({args.max_freq} Hz) exceeds Nyquist limit ({nyquist:.0f} Hz) "
            f"for --delta-t {args.delta_t} us. Decrease --delta-t to at least "
            f"{int(1e6 / (2 * args.max_freq))} us."
        )
    return args


# ---------------------------------------------------------------------------
#  Spatial clustering: connected components on the per-pixel frequency map
# ---------------------------------------------------------------------------

class FrequencyMapAnalyzer:
    """Finds propeller candidate regions in the SDK's per-pixel frequency map.

    Pipeline:
      1. Threshold the frequency map to get a binary mask of vibrating pixels.
      2. Dilate the mask to bridge small gaps (propeller pixels may be sparse).
      3. Connected-component labelling to find spatial clusters.
      4. For each cluster: check size, frequency consistency, compute stats.
    """

    # Downscale factor for the binary mask before morphology + CC analysis.
    # 4x downscale: 1280x720 -> 320x180  (~16x fewer pixels to process).
    DOWNSCALE = 4

    def __init__(self, min_freq, max_freq, num_blades,
                 min_pixels=3, dilate_radius=5, max_freq_cv=0.3):
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.num_blades = num_blades
        self.min_pixels = min_pixels
        self.max_freq_cv = max_freq_cv
        # Structuring element for dilation (adjusted for downscaled image)
        ds_radius = max(1, dilate_radius // self.DOWNSCALE)
        k = 2 * ds_radius + 1
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    def analyze(self, freq_map):
        """Extract propeller candidate regions from the frequency map.

        Args:
            freq_map: 2D float array where each pixel = detected frequency (Hz),
                      0 means no vibration detected.

        Returns:
            List of dicts with keys:
              x, y      - centroid
              freq_hz   - median frequency
              rpm       - estimated RPM
              pixels    - number of active pixels
              bbox      - (x, y, w, h)
              freq_cv   - coefficient of variation of frequency within cluster
        """
        DS = self.DOWNSCALE
        h_full, w_full = freq_map.shape

        # Step 1: binary mask of pixels vibrating in our target range
        valid = (freq_map >= self.min_freq) & (freq_map <= self.max_freq)

        # Early exit: count active pixels cheaply
        active_count = int(np.count_nonzero(valid))
        if active_count < self.min_pixels:
            return []

        # Step 2: downscale mask for fast morphology + connected components
        mask_full = valid.astype(np.uint8) * 255
        h_ds, w_ds = h_full // DS, w_full // DS
        mask_small = cv2.resize(mask_full, (w_ds, h_ds),
                                interpolation=cv2.INTER_NEAREST)

        # Step 3: dilate to bridge nearby pixels (on small image = very fast)
        mask_dilated = cv2.dilate(mask_small, self.kernel, iterations=1)

        # Step 4: connected components on the small image
        num_labels, labels_small, stats, centroids = \
            cv2.connectedComponentsWithStats(mask_dilated, connectivity=8)

        detections = []
        for i in range(1, num_labels):  # skip label 0 = background
            area_dilated = stats[i, cv2.CC_STAT_AREA]
            # Minimum area check (scaled for downsampled image)
            if area_dilated < max(1, self.min_pixels // (DS * DS)):
                continue

            # Map bounding box back to full resolution
            x_ds = int(stats[i, cv2.CC_STAT_LEFT])
            y_ds = int(stats[i, cv2.CC_STAT_TOP])
            w_box = int(stats[i, cv2.CC_STAT_WIDTH]) * DS
            h_box = int(stats[i, cv2.CC_STAT_HEIGHT]) * DS
            x_full = x_ds * DS
            y_full = y_ds * DS

            # Clamp to image bounds
            x2 = min(x_full + w_box, w_full)
            y2 = min(y_full + h_box, h_full)

            # Get actual vibrating pixels from the full-res valid mask (ROI only)
            roi_valid = valid[y_full:y2, x_full:x2]
            actual_pixels = int(np.count_nonzero(roi_valid))
            if actual_pixels < self.min_pixels:
                continue

            # Step 5: frequency statistics (on full-res ROI)
            roi_freqs = freq_map[y_full:y2, x_full:x2]
            cluster_freqs = roi_freqs[roi_valid]
            if len(cluster_freqs) == 0:
                continue

            median_freq = float(np.median(cluster_freqs))
            if median_freq <= 0:
                continue

            # Coefficient of variation: std / mean (0 = perfect consistency)
            if len(cluster_freqs) > 1:
                freq_cv = float(np.std(cluster_freqs) / median_freq)
            else:
                freq_cv = 0.0

            if freq_cv > self.max_freq_cv:
                continue

            rpm = (median_freq / self.num_blades) * 60
            # Centroid in full-res coordinates
            cx = float(centroids[i][0]) * DS
            cy = float(centroids[i][1]) * DS

            detections.append({
                "x": cx,
                "y": cy,
                "freq_hz": median_freq,
                "rpm": rpm,
                "pixels": actual_pixels,
                "bbox": (x_full, y_full, x2 - x_full, y2 - y_full),
                "freq_cv": freq_cv,
            })

        # Sort by pixel count (largest = most confident)
        detections.sort(key=lambda d: d["pixels"], reverse=True)
        return detections


# ---------------------------------------------------------------------------
#  Temporal tracking: only report propellers that persist across frames
# ---------------------------------------------------------------------------

class PropellerTracker:
    """Tracks propeller detections across frames for temporal consistency.

    A detection must be seen in at least `min_hits` out of the last
    `max_age` frames to be confirmed as a real propeller. This eliminates
    transient noise spikes.
    """

    def __init__(self, max_distance=100, freq_tolerance=0.3,
                 min_hits=3, max_age=8):
        self.max_distance = max_distance
        self.freq_tolerance = freq_tolerance
        self.min_hits = min_hits
        self.max_age = max_age
        self.tracks = []
        self.next_id = 1

    def update(self, detections):
        """Match new detections to existing tracks, create/remove tracks.

        Args:
            detections: List of dicts from FrequencyMapAnalyzer.analyze()

        Returns:
            List of confirmed track dicts (hits >= min_hits), sorted by pixel count.
        """
        matched_tracks = set()
        matched_dets = set()

        # Greedy matching: for each detection, find the best matching track
        for di, det in enumerate(detections):
            best_track_idx = None
            best_dist = self.max_distance

            for ti, track in enumerate(self.tracks):
                if ti in matched_tracks:
                    continue

                # Spatial distance
                dist = np.sqrt((det["x"] - track["x"])**2 +
                               (det["y"] - track["y"])**2)
                if dist > self.max_distance:
                    continue

                # Frequency similarity
                f1, f2 = det["freq_hz"], track["freq_hz"]
                freq_ratio = max(f1, f2) / max(min(f1, f2), 1e-6)
                if freq_ratio > 1.0 + self.freq_tolerance:
                    continue

                if dist < best_dist:
                    best_dist = dist
                    best_track_idx = ti

            if best_track_idx is not None:
                # Update existing track with exponential smoothing
                track = self.tracks[best_track_idx]
                alpha = 0.3
                track["x"] = alpha * det["x"] + (1 - alpha) * track["x"]
                track["y"] = alpha * det["y"] + (1 - alpha) * track["y"]
                track["freq_hz"] = alpha * det["freq_hz"] + (1 - alpha) * track["freq_hz"]
                track["rpm"] = alpha * det["rpm"] + (1 - alpha) * track["rpm"]
                track["pixels"] = det["pixels"]
                track["bbox"] = det["bbox"]
                track["freq_cv"] = det["freq_cv"]
                track["hits"] += 1
                track["age"] = 0
                matched_tracks.add(best_track_idx)
                matched_dets.add(di)

        # Create new tracks for unmatched detections
        for di, det in enumerate(detections):
            if di in matched_dets:
                continue
            self.tracks.append({
                "id": self.next_id,
                "x": det["x"],
                "y": det["y"],
                "freq_hz": det["freq_hz"],
                "rpm": det["rpm"],
                "pixels": det["pixels"],
                "bbox": det["bbox"],
                "freq_cv": det["freq_cv"],
                "hits": 1,
                "age": 0,
            })
            self.next_id += 1

        # Age unmatched tracks and prune dead ones
        for ti, track in enumerate(self.tracks):
            if ti not in matched_tracks:
                track["age"] += 1

        self.tracks = [t for t in self.tracks if t["age"] <= self.max_age]

        # Return only confirmed tracks
        confirmed = [t for t in self.tracks if t["hits"] >= self.min_hits]
        confirmed.sort(key=lambda t: t["pixels"], reverse=True)
        return confirmed


# ---------------------------------------------------------------------------
#  Visualization helpers
# ---------------------------------------------------------------------------

def draw_detections_on_frame(frame, confirmed_tracks, num_blades):
    """Draw bounding boxes and labels for confirmed propellers directly on frame.

    Draws in-place to avoid an expensive full-frame copy (~2.7 MB at 1280x720).
    The frame is overwritten by PeriodicFrameGenerationAlgorithm on the next
    callback anyway, so in-place modification is safe.
    """
    for track in confirmed_tracks:
        x, y, w, h = track["bbox"]
        margin = max(10, int(max(w, h) * 0.3))
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)

        # Color: green, brighter for more hits (cap at 10 to avoid overflow)
        brightness = min(255, 100 + min(track["hits"], 10) * 15)
        color = (0, brightness, 0)
        thickness = 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        freq = track["freq_hz"]
        rpm = track["rpm"]
        px = track["pixels"]
        label1 = f"ID{track['id']}: {freq:.0f}Hz {rpm:.0f}RPM"
        label2 = f"{px}px hits={track['hits']}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_thick = 1
        (tw1, th1), _ = cv2.getTextSize(label1, font, font_scale, text_thick)
        (tw2, th2), _ = cv2.getTextSize(label2, font, font_scale, text_thick)
        tw = max(tw1, tw2)

        ty = y1 - 5
        if ty - th1 - th2 - 8 < 0:
            ty = y2 + th1 + 5

        cv2.rectangle(frame, (x1, ty - th1 - th2 - 8),
                      (x1 + tw + 6, ty + 4), (0, 0, 0), -1)
        cv2.putText(frame, label1, (x1 + 3, ty - th2 - 4),
                    font, font_scale, color, text_thick, cv2.LINE_AA)
        cv2.putText(frame, label2, (x1 + 3, ty),
                    font, font_scale, color, text_thick, cv2.LINE_AA)

        cx, cy_pt = int(track["x"]), int(track["y"])
        cv2.drawMarker(frame, (cx, cy_pt), color,
                       cv2.MARKER_CROSS, 12, thickness)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    delta_t = args.delta_t

    # Events iterator
    mv_iterator = EventsIterator(input_path=args.event_file_path, delta_t=delta_t)
    height, width = mv_iterator.get_size()

    if not is_live_camera(args.event_file_path):
        mv_iterator = LiveReplayEventsIterator(mv_iterator, replay_factor=args.replay_factor)

    # --- SDK frequency map algorithm (per-pixel frequency detection) ---
    freq_algo = FrequencyMapAsyncAlgorithm(
        width=width, height=height,
        filter_length=args.filter_length,
        min_freq=args.min_freq,
        max_freq=args.max_freq,
        diff_thresh_us=args.max_period_diff,
    )
    freq_algo.update_frequency = args.update_freq

    # Dominant value extractor (for the heatmap legend)
    dominant_algo = DominantValueMapAlgorithm(
        args.min_freq, args.max_freq, args.freq_precision, args.min_cluster_pixels)

    # Heatmap frame generator
    heat_gen = HeatMapFrameGeneratorAlgorithm(
        args.min_freq, args.max_freq, args.freq_precision, width, height, "Hz")
    freq_img = heat_gen.get_output_image()
    freq_full_height = heat_gen.full_height

    # --- Spatial analyzer + Temporal tracker ---
    analyzer = FrequencyMapAnalyzer(
        min_freq=args.min_freq,
        max_freq=args.max_freq,
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
    )

    # --- Windows ---
    ev_window = MTWindow(title="Propeller Detector - Events + Detections",
                         width=width, height=height,
                         mode=BaseWindow.RenderMode.BGR, open_directly=True)
    freq_window = MTWindow(title="Frequency Map",
                           width=width, height=freq_full_height,
                           mode=BaseWindow.RenderMode.BGR, open_directly=True)

    def keyboard_cb(key, scancode, action, mods):
        if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
            ev_window.set_close_flag()
            freq_window.set_close_flag()

    ev_window.set_keyboard_callback(keyboard_cb)
    freq_window.set_keyboard_callback(keyboard_cb)

    # Event frame generator — we intercept the frame to draw detections on it
    event_frame_gen = PeriodicFrameGenerationAlgorithm(
        sensor_width=width, sensor_height=height, fps=25, palette=ColorPalette.Dark)

    # Shared state between callbacks
    confirmed_propellers = []      # latest confirmed tracks
    last_print_ts = [0]            # timestamp of last console printout
    prev_confirmed_ids = [set()]   # for detecting changes

    def on_cd_frame_cb(ts, cd_frame):
        """Called ~25 fps: overlay detection bounding boxes on the event frame."""
        if confirmed_propellers:
            draw_detections_on_frame(cd_frame, confirmed_propellers,
                                    args.num_blades)
        ev_window.show_async(cd_frame)

    event_frame_gen.set_output_callback(on_cd_frame_cb)

    def on_freq_map(ts, freq_map):
        """Called at update_freq Hz: analyze freq map, track, visualize."""
        nonlocal confirmed_propellers

        # --- Spatial clustering on the frequency map ---
        candidates = analyzer.analyze(freq_map)

        # --- Temporal tracking ---
        confirmed_propellers = tracker.update(candidates)

        # --- Generate heatmap + overlay detections ---
        heat_gen.generate_bgr_heat_map(freq_map, freq_img)

        # Draw detection boxes on the heatmap too
        if confirmed_propellers:
            for track in confirmed_propellers:
                x, y, w, h = track["bbox"]
                margin = max(8, int(max(w, h) * 0.2))
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(width, x + w + margin)
                y2 = min(height, y + h + margin)
                cv2.rectangle(freq_img, (x1, y1), (x2, y2), (255, 255, 255), 2)
                label = f"{track['freq_hz']:.0f}Hz {track['rpm']:.0f}RPM"
                cv2.putText(freq_img, label, (x1, max(y1 - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 1, cv2.LINE_AA)

        # Dominant frequency overlay
        success, dominant_freq = dominant_algo.compute_dominant_value(freq_map)
        if success:
            rpm = (dominant_freq / args.num_blades) * 60
            text = f"Dominant: {dominant_freq:.0f} Hz  ({rpm:.0f} RPM, {args.num_blades} blades)"
            cv2.putText(freq_img, text, (10, height - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)

        # Status line
        n_candidates = len(candidates)
        n_confirmed = len(confirmed_propellers)
        n_tracks = len(tracker.tracks)
        status = f"Candidates: {n_candidates}  Tracks: {n_tracks}  Confirmed: {n_confirmed}"
        cv2.putText(freq_img, status, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        freq_window.show_async(freq_img)

        # --- Console output: only on changes or periodically ---
        ts_sec = ts / 1e6
        current_ids = set(t["id"] for t in confirmed_propellers)
        changed = (current_ids != prev_confirmed_ids[0])
        periodic = (ts_sec - last_print_ts[0] >= 2.0)  # every 2s max

        if changed or (periodic and confirmed_propellers):
            prev_confirmed_ids[0] = current_ids
            last_print_ts[0] = ts_sec

            if confirmed_propellers:
                print(f"[{ts_sec:7.2f}s] === {n_confirmed} CONFIRMED PROPELLER(S) ===  "
                      f"({n_candidates} candidates, {n_tracks} tracks)")
                for track in confirmed_propellers:
                    print(f"    >> ID{track['id']:3d}: "
                          f"freq={track['freq_hz']:6.1f} Hz, "
                          f"RPM={track['rpm']:7.0f}, "
                          f"pixels={track['pixels']:4d}, "
                          f"hits={track['hits']:3d}, "
                          f"pos=({track['x']:.0f},{track['y']:.0f})")

    freq_algo.set_output_callback(on_freq_map)

    # Print header
    nyquist = 1e6 / delta_t / 2
    print("=" * 75)
    print("  Drone Propeller Rotation Detector  (v2 - spatial + temporal)")
    print(f"  Sensor          : {width}x{height}")
    print(f"  Frequency range : {args.min_freq} - {args.max_freq} Hz")
    print(f"  Blades assumed  : {args.num_blades}")
    print(f"  Min cluster px  : {args.min_cluster_pixels}")
    print(f"  Dilate radius   : {args.dilate_radius}")
    print(f"  Max freq CV     : {args.max_freq_cv}")
    print(f"  Tracking: min_hits={args.min_hits}, max_age={args.max_age}, "
          f"dist={args.track_distance}, freq_tol={args.freq_tolerance}")
    print(f"  Update rate     : {args.update_freq} Hz")
    print(f"  delta_t         : {delta_t} us  (Nyquist={nyquist:.0f} Hz)")
    print("=" * 75)
    print()

    for evs in mv_iterator:
        EventLoop.poll_and_dispatch()
        event_frame_gen.process_events(evs)
        freq_algo.process_events(evs)

        if ev_window.should_close() or freq_window.should_close():
            break

    ev_window.destroy()
    freq_window.destroy()
    print("\nDone.")


if __name__ == "__main__":
    main()
