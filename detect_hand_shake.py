"""
Hand-shake detector for event-based cameras.

Tracks the centroid of events over time and uses FFT to estimate:
  - Shake frequency (Hz)
  - Shake magnitude (pixels peak-to-peak)

The centroid of all events in each time slice reflects global camera motion.
When the camera (or your hand holding it) shakes, the centroid oscillates,
and FFT on that signal reveals the frequency and amplitude of the shake.

Usage:
    python detect_hand_shake.py -i <path_to_event_file>
    python detect_hand_shake.py                          # live camera
"""

import argparse
import numpy as np
from collections import deque

from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator, is_live_camera
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction, UIKeyEvent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect hand-shake frequency and magnitude from event camera data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input-event-file", dest="event_file_path", default="",
        help="Path to input event file (RAW, DAT or HDF5). If not specified, the live camera is used.",
    )
    parser.add_argument(
        "--analysis-window", dest="analysis_window_sec", type=float, default=2.0,
        help="Rolling window duration in seconds used for FFT analysis.",
    )
    parser.add_argument(
        "--print-interval", dest="print_interval_sec", type=float, default=0.5,
        help="How often (in seconds) to print shake stats to the console.",
    )
    parser.add_argument(
        "--delta-t", dest="delta_t", type=int, default=5000,
        help="Duration of each event slice in microseconds. "
             "Determines the sampling rate for centroid tracking (1/delta_t Hz).",
    )
    return parser.parse_args()


def analyze_shake(timestamps_us, centroids_x, centroids_y):
    """Run FFT on centroid displacement to find dominant shake frequency and magnitude.

    Args:
        timestamps_us: 1-D array of timestamps in microseconds.
        centroids_x: 1-D array of centroid x positions (pixels).
        centroids_y: 1-D array of centroid y positions (pixels).

    Returns:
        dict with keys:
            freq_x, freq_y  - dominant shake frequency per axis (Hz)
            mag_x, mag_y    - peak-to-peak amplitude per axis (pixels)
            freq_combined    - dominant frequency of combined displacement (Hz)
            mag_combined     - peak-to-peak amplitude of combined displacement (pixels)
        or None if insufficient data.
    """
    n = len(timestamps_us)
    if n < 8:
        return None

    # Sampling rate from actual timestamps
    duration_sec = (timestamps_us[-1] - timestamps_us[0]) / 1e6
    if duration_sec <= 0:
        return None
    fs = (n - 1) / duration_sec  # effective sampling rate (Hz)

    results = {}
    for axis_name, signal in [("x", centroids_x), ("y", centroids_y)]:
        # Remove DC (mean position) so we look at oscillation only
        sig = signal - np.mean(signal)

        # Apply Hann window to reduce spectral leakage
        window = np.hanning(n)
        sig_windowed = sig * window

        # FFT
        spectrum = np.fft.rfft(sig_windowed)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        magnitudes = np.abs(spectrum) * 2.0 / n  # normalize

        # Ignore DC bin (index 0) and very low frequencies (< 1 Hz)
        min_freq_hz = 1.0
        valid = freqs >= min_freq_hz
        if not np.any(valid):
            results[f"freq_{axis_name}"] = 0.0
            results[f"mag_{axis_name}"] = 0.0
            continue

        valid_freqs = freqs[valid]
        valid_mags = magnitudes[valid]

        peak_idx = np.argmax(valid_mags)
        results[f"freq_{axis_name}"] = float(valid_freqs[peak_idx])
        # Peak-to-peak = 2 * amplitude (amplitude is the FFT magnitude for that bin)
        results[f"mag_{axis_name}"] = float(valid_mags[peak_idx] * 2.0)

    # Combined displacement: sqrt(dx^2 + dy^2)
    dx = centroids_x - np.mean(centroids_x)
    dy = centroids_y - np.mean(centroids_y)
    combined = np.sqrt(dx ** 2 + dy ** 2)
    sig_c = combined - np.mean(combined)
    window = np.hanning(n)
    sig_c_windowed = sig_c * window
    spectrum_c = np.fft.rfft(sig_c_windowed)
    freqs_c = np.fft.rfftfreq(n, d=1.0 / fs)
    mags_c = np.abs(spectrum_c) * 2.0 / n

    valid_c = freqs_c >= 1.0
    if np.any(valid_c):
        peak_c = np.argmax(mags_c[valid_c])
        results["freq_combined"] = float(freqs_c[valid_c][peak_c])
        results["mag_combined"] = float(mags_c[valid_c][peak_c] * 2.0)
    else:
        results["freq_combined"] = 0.0
        results["mag_combined"] = 0.0

    return results


def main():
    args = parse_args()

    delta_t = args.delta_t  # microseconds per slice
    analysis_window_samples = int(args.analysis_window_sec * 1e6 / delta_t)
    print_interval_samples = int(args.print_interval_sec * 1e6 / delta_t)

    # Events iterator
    mv_iterator = EventsIterator(input_path=args.event_file_path, delta_t=delta_t)
    height, width = mv_iterator.get_size()

    if not is_live_camera(args.event_file_path):
        mv_iterator = LiveReplayEventsIterator(mv_iterator)

    # Rolling buffers for centroid history
    ts_buf = deque(maxlen=analysis_window_samples)
    cx_buf = deque(maxlen=analysis_window_samples)
    cy_buf = deque(maxlen=analysis_window_samples)

    sample_count = 0

    # Window for event visualization
    with MTWindow(title="Hand Shake Detector", width=width, height=height,
                  mode=BaseWindow.RenderMode.BGR) as window:

        def keyboard_cb(key, scancode, action, mods):
            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                window.set_close_flag()

        window.set_keyboard_callback(keyboard_cb)

        # Frame generator for visualization
        event_frame_gen = PeriodicFrameGenerationAlgorithm(
            sensor_width=width, sensor_height=height, fps=25,
            palette=ColorPalette.Dark,
        )

        def on_cd_frame_cb(ts, cd_frame):
            window.show_async(cd_frame)

        event_frame_gen.set_output_callback(on_cd_frame_cb)

        print("=" * 65)
        print("  Hand-Shake Detector")
        print(f"  Sensor: {width}x{height}")
        print(f"  Sampling rate: {1e6/delta_t:.0f} Hz  (delta_t={delta_t} us)")
        print(f"  FFT window: {args.analysis_window_sec}s "
              f"({analysis_window_samples} samples)")
        print("=" * 65)
        print()

        for evs in mv_iterator:
            EventLoop.poll_and_dispatch()
            event_frame_gen.process_events(evs)

            if window.should_close():
                break

            # Compute centroid of this event slice
            if evs.size == 0:
                continue

            cx = float(np.mean(evs["x"]))
            cy = float(np.mean(evs["y"]))
            ts = float(evs["t"][-1])

            ts_buf.append(ts)
            cx_buf.append(cx)
            cy_buf.append(cy)

            sample_count += 1

            # Print shake stats periodically
            if sample_count % print_interval_samples == 0 and len(ts_buf) >= 8:
                result = analyze_shake(
                    np.array(ts_buf), np.array(cx_buf), np.array(cy_buf)
                )
                if result is not None:
                    print(
                        f"Shake  |  "
                        f"X: freq={result['freq_x']:6.1f} Hz, mag={result['mag_x']:5.1f} px  |  "
                        f"Y: freq={result['freq_y']:6.1f} Hz, mag={result['mag_y']:5.1f} px  |  "
                        f"Combined: freq={result['freq_combined']:6.1f} Hz, "
                        f"mag={result['mag_combined']:5.1f} px"
                    )

    print("\nDone.")


if __name__ == "__main__":
    main()
