"""
Capture Flicker Map — Record per-pixel frequency maps of ceiling lights.

Points the event camera at the ceiling and captures the frequency map produced
by the Metavision SDK's FrequencyMapAsyncAlgorithm. Segments individual light
fixtures and saves their frequency fingerprints to a JSON database.

This is the primary calibration tool for the light localization system.

Usage:
    # Live camera — point at ceiling and capture
    python capture_flicker_map.py --room-id office1

    # From a recording
    python capture_flicker_map.py -i ceiling_recording.raw --room-id office1

    # Electronic ballast mode (high frequency)
    python capture_flicker_map.py --min-freq 15000 --max-freq 70000 --delta-t 5

    # Magnetic ballast mode (low frequency)
    python capture_flicker_map.py --min-freq 50 --max-freq 500 --delta-t 500
"""

import argparse
import sys
import os
import time
import json
import numpy as np
import cv2

# Add parent directory to path for SDK imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator, is_live_camera
from metavision_sdk_analytics import FrequencyMapAsyncAlgorithm, DominantValueMapAlgorithm, \
    HeatMapFrameGeneratorAlgorithm
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction, UIKeyEvent

from light_localization_utils import LightROIExtractor, FrequencyFeatureExtractor
from light_fingerprint import LightFingerprint, FingerprintDatabase


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture ceiling light frequency fingerprints using event camera.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Input
    parser.add_argument(
        "-i", "--input-event-file", dest="event_file_path", default="",
        help="Path to input event file (RAW, DAT or HDF5). "
             "If not specified, the live camera is used.",
    )

    # Frequency range
    parser.add_argument(
        "--min-freq", dest="min_freq", type=float, default=50,
        help="Minimum light flicker frequency to detect (Hz).",
    )
    parser.add_argument(
        "--max-freq", dest="max_freq", type=float, default=500,
        help="Maximum light flicker frequency to detect (Hz). "
             "Use 70000 for electronic ballast fluorescent lights.",
    )

    # SDK tuning
    parser.add_argument(
        "--filter-length", dest="filter_length", type=int, default=7,
        help="Number of successive periods to confirm a vibration. "
             "Higher = more robust, slower convergence.",
    )
    parser.add_argument(
        "--max-period-diff", dest="max_period_diff", type=int, default=5000,
        help="Max difference between two periods to be the same (us). "
             "Use 5 for high-frequency electronic ballasts.",
    )
    parser.add_argument(
        "--update-freq", dest="update_freq", type=float, default=10,
        help="Frequency map update rate (Hz).",
    )
    parser.add_argument(
        "--freq-precision", dest="freq_precision", type=float, default=2.0,
        help="Width of frequency bins in Hz for histogram display.",
    )

    # Spatial segmentation
    parser.add_argument(
        "--min-pixels", dest="min_pixels", type=int, default=20,
        help="Minimum active pixels for a valid light fixture.",
    )
    parser.add_argument(
        "--dilate-radius", dest="dilate_radius", type=int, default=8,
        help="Morphological dilation radius to fill gaps within fixtures.",
    )
    parser.add_argument(
        "--max-freq-cv", dest="max_freq_cv", type=float, default=0.15,
        help="Max coefficient of variation of frequency within a fixture.",
    )

    # FFT analysis
    parser.add_argument(
        "--fft-window", dest="fft_window", type=int, default=512,
        help="FFT window size (samples). Larger = better frequency resolution.",
    )
    parser.add_argument(
        "--num-harmonics", dest="num_harmonics", type=int, default=5,
        help="Number of harmonics to extract for each fingerprint.",
    )

    # Timing
    parser.add_argument(
        "--delta-t", dest="delta_t", type=int, default=500,
        help="Event slice duration (us). Determines effective sampling rate. "
             "Use 5 us for electronic ballasts (max_freq > 10 kHz).",
    )
    parser.add_argument(
        "-f", "--replay-factor", dest="replay_factor", type=float, default=1,
        help="Replay speed factor (>1 = slow-motion, <1 = speed-up).",
    )

    # Calibration parameters
    parser.add_argument(
        "--room-id", dest="room_id", type=str, default="room1",
        help="Room identifier for the fingerprint database.",
    )
    parser.add_argument(
        "--capture-duration", dest="capture_duration", type=float, default=0,
        help="Auto-stop after N seconds (0 = manual stop with Q/ESC).",
    )
    parser.add_argument(
        "--auto-save", dest="auto_save", action="store_true",
        help="Automatically save fingerprints on exit.",
    )

    # Output
    parser.add_argument(
        "-o", "--output-dir", dest="output_dir", type=str,
        default="calibration_data",
        help="Directory to save fingerprint database and visualizations.",
    )

    args = parser.parse_args()

    # Nyquist validation
    nyquist = 1e6 / args.delta_t / 2
    if args.max_freq > nyquist:
        parser.error(
            f"--max-freq ({args.max_freq} Hz) exceeds Nyquist limit ({nyquist:.0f} Hz) "
            f"for --delta-t {args.delta_t} us. Decrease --delta-t to at least "
            f"{int(1e6 / (2 * args.max_freq))} us."
        )
    return args


def main():
    args = parse_args()
    delta_t = args.delta_t
    sample_rate_hz = 1e6 / delta_t

    # --- Event source ---
    mv_iterator = EventsIterator(input_path=args.event_file_path, delta_t=delta_t)
    height, width = mv_iterator.get_size()

    if not is_live_camera(args.event_file_path):
        mv_iterator = LiveReplayEventsIterator(mv_iterator, replay_factor=args.replay_factor)

    # --- SDK frequency map algorithm ---
    freq_algo = FrequencyMapAsyncAlgorithm(
        width=width, height=height,
        filter_length=args.filter_length,
        min_freq=args.min_freq,
        max_freq=args.max_freq,
        diff_thresh_us=args.max_period_diff,
    )
    freq_algo.update_frequency = args.update_freq

    # Dominant value extractor
    dominant_algo = DominantValueMapAlgorithm(
        args.min_freq, args.max_freq, args.freq_precision, args.min_pixels)

    # Heatmap generator
    heat_gen = HeatMapFrameGeneratorAlgorithm(
        args.min_freq, args.max_freq, args.freq_precision, width, height, "Hz")
    freq_img = heat_gen.get_output_image()
    freq_full_height = heat_gen.full_height

    # --- Light ROI extractor ---
    roi_extractor = LightROIExtractor(
        width=width, height=height,
        min_freq=args.min_freq, max_freq=args.max_freq,
        min_pixels=args.min_pixels, max_pixels=50000,
        dilate_radius=args.dilate_radius,
        max_freq_cv=args.max_freq_cv,
    )

    # --- FFT feature extractor ---
    fft_extractor = FrequencyFeatureExtractor(
        fft_window_samples=args.fft_window,
        num_harmonics=args.num_harmonics,
        min_snr=2.0,
    )

    # --- Fingerprint database ---
    db = FingerprintDatabase()
    db.metadata["description"] = f"Calibration for room '{args.room_id}' captured on {time.strftime('%Y-%m-%d %H:%M:%S')}"

    # --- Windows ---
    ev_window = MTWindow(title="Light Localizer - Events",
                         width=width, height=height,
                         mode=BaseWindow.RenderMode.BGR, open_directly=True)
    freq_window = MTWindow(title="Light Localizer - Frequency Map",
                           width=width, height=freq_full_height,
                           mode=BaseWindow.RenderMode.BGR, open_directly=True)

    save_requested = [False]

    def keyboard_cb(key, scancode, action, mods):
        if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
            ev_window.set_close_flag()
            freq_window.set_close_flag()
        elif key == UIKeyEvent.KEY_S and action == UIAction.RELEASE:
            save_requested[0] = True
            print("[SAVE] Fingerprint save requested — will save on next freq map update.")

    ev_window.set_keyboard_callback(keyboard_cb)
    freq_window.set_keyboard_callback(keyboard_cb)

    # Event visualization
    event_frame_gen = PeriodicFrameGenerationAlgorithm(
        sensor_width=width, sensor_height=height, fps=25, palette=ColorPalette.Dark)

    # Shared state
    current_rois = []
    light_colors = {}  # roi_id → color for consistent visualization
    color_palette = [
        (0, 255, 0), (255, 100, 0), (0, 200, 255), (255, 0, 255),
        (255, 255, 0), (0, 255, 255), (128, 255, 128), (255, 128, 0),
    ]
    start_time = [time.time()]

    def on_cd_frame_cb(ts, cd_frame):
        """Draw detected light fixtures on the event frame."""
        for roi in current_rois:
            rid = roi["roi_id"]
            if rid not in light_colors:
                light_colors[rid] = color_palette[rid % len(color_palette)]
            color = light_colors[rid]
            bx, by, bw, bh = roi["bbox"]
            cv2.rectangle(cd_frame, (bx, by), (bx + bw, by + bh), color, 2)
            label = f"L{rid}: {roi['median_freq']:.0f}Hz"
            cv2.putText(cd_frame, label, (bx, max(by - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        ev_window.show_async(cd_frame)

    event_frame_gen.set_output_callback(on_cd_frame_cb)

    def on_freq_map(ts, freq_map):
        """Analyze frequency map: segment lights, extract features, visualize."""
        nonlocal current_rois

        # Segment light fixtures
        rois = roi_extractor.extract(freq_map)
        current_rois = rois

        # Extract FFT features for each ROI
        features = {}
        for roi in rois:
            f = fft_extractor.extract_features(
                roi["roi_id"], sample_rate_hz,
                min_freq=args.min_freq, max_freq=args.max_freq,
            )
            if f is not None:
                features[roi["roi_id"]] = f

        # Build/update fingerprints
        if save_requested[0]:
            db.fingerprints.clear()
            for roi in rois:
                light_id = f"{args.room_id}_light_{roi['roi_id']}"
                fp = LightFingerprint(
                    light_id=light_id,
                    position_3d=(0.0, 0.0, 0.0),  # To be filled by operator
                    room_id=args.room_id,
                )
                fp.set_from_roi(roi)
                if roi["roi_id"] in features:
                    fp.set_frequency_features(features[roi["roi_id"]])
                else:
                    # Use SDK frequency map data as fallback
                    fp.fundamental_freq = roi["median_freq"]
                    fp.peak_snr = 1.0 / max(0.01, roi["freq_cv"])
                    fp.calibration_timestamp = time.time()
                db.add(fp)

            # Save to file
            output_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), args.output_dir
            )
            os.makedirs(output_dir, exist_ok=True)
            db_path = os.path.join(output_dir, f"{args.room_id}_fingerprints.json")
            db.save(db_path)
            print(f"\n[SAVED] {len(db)} fingerprints → {db_path}")
            print(db.summary())
            print()

            # Also save the frequency map visualization
            vis_path = os.path.join(output_dir, f"{args.room_id}_freq_map.png")
            heat_gen.generate_bgr_heat_map(freq_map, freq_img)
            cv2.imwrite(vis_path, freq_img[:height, :])
            print(f"[SAVED] Frequency map → {vis_path}")

            save_requested[0] = False

        # --- Visualization ---
        heat_gen.generate_bgr_heat_map(freq_map, freq_img)

        # Draw ROI boxes on heatmap
        for roi in rois:
            rid = roi["roi_id"]
            if rid not in light_colors:
                light_colors[rid] = color_palette[rid % len(color_palette)]
            color = light_colors[rid]
            bx, by, bw, bh = roi["bbox"]
            cv2.rectangle(freq_img, (bx, by), (bx + bw, by + bh), color, 2)

            # Label with frequency and feature details
            label = f"L{rid}: {roi['median_freq']:.1f}Hz"
            cv2.putText(freq_img, label, (bx, max(by - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            # Show additional features if available
            if rid in features:
                f = features[rid]
                detail = f"SNR:{f['peak_snr']:.1f} BW:{f['spectral_bandwidth']:.1f}Hz"
                cv2.putText(freq_img, detail, (bx, by + bh + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
                if f["harmonic_ratios"]:
                    hr_str = " ".join(f"h{i+2}:{r:.2f}" for i, r in enumerate(f["harmonic_ratios"][:3]))
                    cv2.putText(freq_img, hr_str, (bx, by + bh + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)

        # Status bar
        n_rois = len(rois)
        n_features = len(features)
        elapsed = time.time() - start_time[0]
        status = (f"Lights: {n_rois} | FFT features: {n_features} | "
                  f"Room: {args.room_id} | {elapsed:.0f}s | "
                  f"Press S to save, Q to quit")
        cv2.putText(freq_img, status, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # Dominant frequency
        success, dom_freq = dominant_algo.compute_dominant_value(freq_map)
        if success:
            cv2.putText(freq_img, f"Dominant: {dom_freq:.1f} Hz", (10, height - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)

        freq_window.show_async(freq_img)

        # Console periodic update
        if int(elapsed) % 5 == 0 and int(elapsed) > 0:
            if rois:
                print(f"[{elapsed:5.0f}s] {n_rois} light(s) detected:")
                for roi in rois:
                    rid = roi["roi_id"]
                    feat_str = ""
                    if rid in features:
                        f = features[rid]
                        feat_str = f" SNR={f['peak_snr']:.1f}, harmonics={len(f['harmonic_ratios'])}"
                    print(f"    L{rid}: {roi['median_freq']:.1f}Hz, "
                          f"{roi['pixel_count']}px, CV={roi['freq_cv']:.3f}{feat_str}")

    freq_algo.set_output_callback(on_freq_map)

    # --- Print header ---
    nyquist = 1e6 / delta_t / 2
    print("=" * 78)
    print("  Light Flicker Fingerprint Capture Tool")
    print(f"  Sensor          : {width}x{height}")
    print(f"  Frequency range : {args.min_freq} – {args.max_freq} Hz")
    print(f"  Sample rate     : {sample_rate_hz:.0f} Hz (delta_t={delta_t} us)")
    print(f"  Nyquist limit   : {nyquist:.0f} Hz")
    print(f"  FFT window      : {args.fft_window} samples ({args.fft_window/sample_rate_hz*1000:.0f} ms)")
    print(f"  Room ID         : {args.room_id}")
    print(f"  Output dir      : {args.output_dir}")
    print()
    print("  Controls:")
    print("    S — Save fingerprints to database")
    print("    Q/ESC — Quit")
    print("=" * 78)
    print()

    # --- Main event loop ---
    for evs in mv_iterator:
        EventLoop.poll_and_dispatch()
        event_frame_gen.process_events(evs)
        freq_algo.process_events(evs)

        # Accumulate events for FFT analysis
        for roi in current_rois:
            fft_extractor.accumulate_events(roi["roi_id"], evs, roi["bbox"])

        # Auto-stop
        if args.capture_duration > 0:
            elapsed = time.time() - start_time[0]
            if elapsed >= args.capture_duration:
                save_requested[0] = True
                print(f"\n[AUTO] Capture duration reached ({args.capture_duration}s). Saving...")
                # Process one more to trigger save
                break

        if ev_window.should_close() or freq_window.should_close():
            break

    # Auto-save on exit if requested
    if args.auto_save and len(db) == 0 and current_rois:
        save_requested[0] = True
        # Manual save since we're outside the callback
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), args.output_dir
        )
        os.makedirs(output_dir, exist_ok=True)
        for roi in current_rois:
            light_id = f"{args.room_id}_light_{roi['roi_id']}"
            fp = LightFingerprint(
                light_id=light_id,
                position_3d=(0.0, 0.0, 0.0),
                room_id=args.room_id,
            )
            fp.set_from_roi(roi)
            fp.fundamental_freq = roi["median_freq"]
            fp.calibration_timestamp = time.time()
            db.add(fp)
        db_path = os.path.join(output_dir, f"{args.room_id}_fingerprints.json")
        db.save(db_path)
        print(f"\n[AUTO-SAVED] {len(db)} fingerprints → {db_path}")

    ev_window.destroy()
    freq_window.destroy()
    print("\nCapture complete.")
    if len(db) > 0:
        print(db.summary())


if __name__ == "__main__":
    main()
