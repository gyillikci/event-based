"""
Active LED marker detector for event-based cameras.

Detects blinking LEDs (active markers) in the field of view and identifies each
one by its blink frequency. Designed for the Arduino Nano 33 BLE firmware in
arduino/active_markers/active_markers.ino, whose built-in LEDs blink at the
fixed frequencies listed in active_markers.json.

This reuses the proven frequency pipeline from detect_propeller.py:
  1. Per-pixel frequency map      (FrequencyMapAsyncAlgorithm, Metavision SDK)
  2. Connected-component clustering (FrequencyMapAnalyzer)
  3. Temporal tracking + Bayesian confidence (PropellerTracker)
On top of that it adds marker identification: each confirmed track's frequency
is matched to the nearest known LED frequency from active_markers.json.

Key differences vs. propeller detection:
  * High frequency band (500-2000 Hz) -> smaller delta_t (Nyquist headroom).
  * num_blades = 1 -> the detected frequency IS the marker frequency (no RPM).
  * Tight frequency tolerance so distinct markers never merge.

Usage:
    python detect_active_markers.py                       # live camera
    python detect_active_markers.py -i recording.raw      # from a file
    python detect_active_markers.py --config active_markers.json
"""

import argparse
import json
import os

import numpy as np

from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator, is_live_camera
from metavision_sdk_analytics import FrequencyMapAsyncAlgorithm, DominantValueMapAlgorithm, \
    HeatMapFrameGeneratorAlgorithm
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIKeyEvent
import cv2

# Reuse the spatial clustering and temporal tracking from the propeller detector.
from detect_propeller import FrequencyMapAnalyzer, PropellerTracker


DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "active_markers.json")


def load_marker_config(path):
    """Load the LED marker frequency table.

    Returns:
        (markers, band, match_tol_hz) where markers is a list of dicts with keys
        id, name, freq_hz, color_bgr; band is (min_freq, max_freq); match_tol_hz
        is the max |detected - nominal| frequency difference for identification.
    """
    with open(path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)

    markers = []
    for m in cfg["markers"]:
        markers.append({
            "id": int(m["id"]),
            "name": str(m["name"]),
            "freq_hz": float(m["freq_hz"]),
            "color_bgr": tuple(int(c) for c in m.get("color_bgr", [0, 255, 0])),
        })

    band = cfg.get("band", {})
    min_freq = float(band.get("min_freq_hz", 500))
    max_freq = float(band.get("max_freq_hz", 2000))
    match_tol_hz = float(cfg.get("match_tolerance_hz", 120))
    return markers, (min_freq, max_freq), match_tol_hz


def assign_marker(freq_hz, markers, tol_hz):
    """Match a detected frequency to the nearest known marker within tolerance.

    Returns the marker dict, or None if no marker is within tol_hz.
    """
    best = None
    best_diff = tol_hz
    for m in markers:
        diff = abs(freq_hz - m["freq_hz"])
        if diff <= best_diff:
            best_diff = diff
            best = m
    return best


def parse_args(markers_default, band_default, tol_default):
    parser = argparse.ArgumentParser(
        description="Detect and identify blinking LED markers from event camera data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Input
    parser.add_argument(
        "-i", "--input-event-file", dest="event_file_path", default="",
        help="Path to input event file (RAW, DAT or HDF5). If not specified, the live camera is used.",
    )
    parser.add_argument(
        "--config", dest="config_path", default=DEFAULT_CONFIG,
        help="Path to the marker frequency table (JSON).",
    )

    # Frequency range (defaults come from the marker config band)
    parser.add_argument(
        "--min-freq", dest="min_freq", type=float, default=band_default[0],
        help="Minimum marker frequency to detect (Hz).",
    )
    parser.add_argument(
        "--max-freq", dest="max_freq", type=float, default=band_default[1],
        help="Maximum marker frequency to detect (Hz).",
    )
    parser.add_argument(
        "--match-tolerance", dest="match_tolerance", type=float, default=tol_default,
        help="Max |detected - nominal| frequency difference (Hz) to identify a marker.",
    )

    # SDK frequency algorithm tuning
    parser.add_argument(
        "--filter-length", dest="filter_length", type=int, default=4,
        help="Number of successive periods to confirm a blinking pixel.",
    )
    parser.add_argument(
        "--max-period-diff", dest="max_period_diff", type=int, default=100,
        help="Max difference between two periods to be considered the same (us). "
             "Small for the high band so distinct markers stay separable.",
    )
    parser.add_argument(
        "--freq-precision", dest="freq_precision", type=float, default=10.0,
        help="Width of frequency bins in Hz for histogram display.",
    )

    # Spatial clustering — LEDs are small point sources.
    parser.add_argument(
        "--min-cluster-pixels", dest="min_cluster_pixels", type=int, default=2,
        help="Minimum blinking pixels in a connected cluster to be a candidate.",
    )
    parser.add_argument(
        "--dilate-radius", dest="dilate_radius", type=int, default=5,
        help="Morphological dilation radius to connect nearby blinking pixels.",
    )
    parser.add_argument(
        "--max-freq-cv", dest="max_freq_cv", type=float, default=0.3,
        help="Max coefficient of variation of frequencies within a cluster.",
    )

    # Temporal tracking
    parser.add_argument(
        "--min-hits", dest="min_hits", type=int, default=3,
        help="Minimum consecutive detections before a marker is confirmed.",
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
        "--freq-tolerance", dest="freq_tolerance", type=float, default=0.1,
        help="Max relative frequency difference to match detection to track. "
             "Tight (0.1) so distinct markers do not merge.",
    )
    parser.add_argument(
        "--confidence-threshold", dest="confidence_threshold", type=float, default=0.9,
        help="Bayesian confidence threshold to confirm a marker (0-1).",
    )

    # Timing
    parser.add_argument(
        "--delta-t", dest="delta_t", type=int, default=1000,
        help="Event batch size (us) handed to the SDK per loop iteration. Larger "
             "= fewer Python iterations/sec = lower live latency. Detection is "
             "period-based, so this does NOT limit the detectable frequency.",
    )
    parser.add_argument(
        "--update-freq", dest="update_freq", type=float, default=30,
        help="How often (Hz) the frequency map and detections update.",
    )

    # Replay
    parser.add_argument(
        "-f", "--replay-factor", dest="replay_factor", type=float, default=1,
        help="Replay speed factor (>1 = slow-motion, <1 = speed-up).",
    )

    args = parser.parse_args()

    # delta_t only controls how often event batches are handed to the SDK. The
    # frequency algorithms measure per-pixel blink periods from event timestamps
    # (1 us resolution), so the detectable frequency is NOT limited by
    # 1/(2*delta_t): a larger delta_t just means fewer Python iterations per
    # second, which lowers live-camera latency.
    if args.max_freq <= args.min_freq:
        parser.error(f"--max-freq ({args.max_freq}) must exceed --min-freq ({args.min_freq}).")
    return args


# ---------------------------------------------------------------------------
#  Visualization
# ---------------------------------------------------------------------------

def draw_markers_on_frame(frame, tracks, markers, tol_hz):
    """Overlay identified markers on a frame (in-place).

    Matched markers are drawn in their configured color with an "M<id> <name>"
    label; unidentified blinking sources are drawn in gray with a "?" label.
    """
    for track in tracks:
        marker = assign_marker(track["freq_hz"], markers, tol_hz)

        x, y, w, h = track["bbox"]
        margin = max(10, int(max(w, h) * 0.3))
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)

        if marker is not None:
            color = marker["color_bgr"]
            label = f"M{marker['id']} {marker['name']}: {track['freq_hz']:.0f}Hz"
        else:
            color = (150, 150, 150)
            label = f"?: {track['freq_hz']:.0f}Hz"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_thick = 1
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, text_thick)
        ty = y1 - 5
        if ty - th - 4 < 0:
            ty = y2 + th + 5
        cv2.rectangle(frame, (x1, ty - th - 6), (x1 + tw + 6, ty + 4), (0, 0, 0), -1)
        cv2.putText(frame, label, (x1 + 3, ty), font, font_scale, color,
                    text_thick, cv2.LINE_AA)

        cx, cy = int(track["x"]), int(track["y"])
        cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 12, 2)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    # Pre-parse --config only, so the marker table can seed the full parser's
    # frequency-band and tolerance defaults.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", dest="config_path", default=DEFAULT_CONFIG)
    pre_args, _ = pre.parse_known_args()

    markers, band, match_tol = load_marker_config(pre_args.config_path)

    args = parse_args(markers, band, match_tol)
    match_tol = args.match_tolerance

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

    dominant_algo = DominantValueMapAlgorithm(
        args.min_freq, args.max_freq, args.freq_precision, args.min_cluster_pixels)

    heat_gen = HeatMapFrameGeneratorAlgorithm(
        args.min_freq, args.max_freq, args.freq_precision, width, height, "Hz")
    freq_img = heat_gen.get_output_image()
    freq_full_height = heat_gen.full_height

    # --- Spatial analyzer + Temporal tracker (num_blades=1 -> freq == marker freq) ---
    analyzer = FrequencyMapAnalyzer(
        width=width, height=height,
        min_freq=args.min_freq,
        max_freq=args.max_freq,
        num_blades=1,
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

    # --- Windows ---
    ev_window = MTWindow(title="Active Marker Detector - Events + Markers",
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

    event_frame_gen = PeriodicFrameGenerationAlgorithm(
        sensor_width=width, sensor_height=height, fps=25, palette=ColorPalette.Dark)

    confirmed_markers = []
    last_print_ts = [0]
    prev_confirmed_ids = [set()]

    def on_cd_frame_cb(ts, cd_frame):
        if confirmed_markers:
            draw_markers_on_frame(cd_frame, confirmed_markers, markers, match_tol)
        ev_window.show_async(cd_frame)

    event_frame_gen.set_output_callback(on_cd_frame_cb)

    def on_freq_map(ts, freq_map):
        nonlocal confirmed_markers

        candidates = analyzer.analyze(freq_map)
        confirmed_markers = tracker.update(candidates)

        heat_gen.generate_bgr_heat_map(freq_map, freq_img)

        if confirmed_markers:
            for track in confirmed_markers:
                marker = assign_marker(track["freq_hz"], markers, match_tol)
                x, y, w, h = track["bbox"]
                margin = max(8, int(max(w, h) * 0.2))
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(width, x + w + margin)
                y2 = min(height, y + h + margin)
                if marker is not None:
                    label = f"M{marker['id']} {marker['name']} {track['freq_hz']:.0f}Hz"
                else:
                    label = f"? {track['freq_hz']:.0f}Hz"
                cv2.rectangle(freq_img, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(freq_img, label, (x1, max(y1 - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 1, cv2.LINE_AA)

        success, dominant_freq = dominant_algo.compute_dominant_value(freq_map)
        if success:
            cv2.putText(freq_img, f"Dominant: {dominant_freq:.0f} Hz",
                        (10, height - 10), cv2.FONT_HERSHEY_PLAIN, 1.0,
                        (255, 255, 255), 1)

        n_candidates = len(candidates)
        n_confirmed = len(confirmed_markers)
        n_tracks = len(tracker.tracks)
        status = f"Candidates: {n_candidates}  Tracks: {n_tracks}  Markers: {n_confirmed}"
        cv2.putText(freq_img, status, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        freq_window.show_async(freq_img)

        # Console output on changes or periodically.
        ts_sec = ts / 1e6
        current_ids = set(t["id"] for t in confirmed_markers)
        changed = (current_ids != prev_confirmed_ids[0])
        periodic = (ts_sec - last_print_ts[0] >= 2.0)

        if changed or (periodic and confirmed_markers):
            prev_confirmed_ids[0] = current_ids
            last_print_ts[0] = ts_sec
            if confirmed_markers:
                timing = f"analyze={analyzer.analysis_time_ms:.1f}ms"
                print(f"[{ts_sec:7.2f}s] === {n_confirmed} MARKER(S) ===  "
                      f"({n_candidates} cand, {n_tracks} trk, {timing})")
                for track in confirmed_markers:
                    marker = assign_marker(track["freq_hz"], markers, match_tol)
                    ident = (f"M{marker['id']} {marker['name']}"
                             if marker is not None else "unidentified")
                    conf = track.get("confidence", 0.0)
                    print(f"    >> track{track['id']:3d}: "
                          f"freq={track['freq_hz']:6.1f} Hz  "
                          f"[{ident}]  "
                          f"pixels={track['pixels']:4d}, "
                          f"conf={conf:.1%}, hits={track['hits']:3d}, "
                          f"pos=({track['x']:.0f},{track['y']:.0f})")

    freq_algo.set_output_callback(on_freq_map)

    # Header
    cycle_ms = 1000.0 / args.update_freq
    print("=" * 78)
    print("  Active LED Marker Detector  (reuses propeller frequency pipeline)")
    print(f"  Sensor          : {width}x{height}")
    print(f"  Frequency range : {args.min_freq} - {args.max_freq} Hz")
    print(f"  Match tolerance : +/- {match_tol:.0f} Hz")
    print(f"  Markers ({len(markers)})     : " +
          ", ".join(f"M{m['id']} {m['name']}={m['freq_hz']:.0f}Hz" for m in markers))
    print(f"  Min cluster px  : {args.min_cluster_pixels}")
    print(f"  Tracking        : min_hits={args.min_hits}, max_age={args.max_age}, "
          f"dist={args.track_distance}, freq_tol={args.freq_tolerance}")
    print(f"  Confidence      : Bayesian threshold={args.confidence_threshold:.0%}")
    print(f"  Update rate     : {args.update_freq} Hz  (cycle={cycle_ms:.0f} ms)")
    print(f"  delta_t         : {delta_t} us  (event batch size; not a freq limit)")
    print("=" * 78)
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
