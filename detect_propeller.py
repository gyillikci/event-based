"""
Drone propeller rotation detector for event-based cameras.

Detects spinning propellers in the field of view and estimates their
rotational frequency (Hz) and RPM in real-time using two approaches:

  1. Per-pixel frequency map  (FrequencyMapAsyncAlgorithm from Metavision SDK)
     - Shows a heatmap of which pixels are oscillating and at what frequency.
     - Great for localizing propellers spatially.

  2. ROI-based event-rate FFT
     - Counts events per time slice in a grid of cells.
     - Applies FFT to each cell's event-rate time series.
     - Detects dominant periodic signal per cell -> frequency & RPM.
     - Clusters nearby active cells to report per-propeller stats.

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
import numpy as np

from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator, is_live_camera
from metavision_sdk_analytics import FrequencyMapAsyncAlgorithm, DominantValueMapAlgorithm, \
    HeatMapFrameGeneratorAlgorithm
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction, UIKeyEvent
import cv2

from propeller_utils import PropellerGridAnalyzer, cluster_detections


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
        "--min-freq", dest="min_freq", type=float, default=50,
        help="Minimum propeller frequency to detect (Hz).",
    )
    parser.add_argument(
        "--max-freq", dest="max_freq", type=float, default=500,
        help="Maximum propeller frequency to detect (Hz). "
             "Set higher for racing drones (e.g. 1000).",
    )
    parser.add_argument(
        "--num-blades", dest="num_blades", type=int, default=2,
        help="Number of propeller blades. RPM = (detected_freq / num_blades) * 60.",
    )

    # SDK frequency algorithm tuning
    parser.add_argument(
        "--filter-length", dest="filter_length", type=int, default=7,
        help="Number of successive periods needed to confirm a vibration.",
    )
    parser.add_argument(
        "--max-period-diff", dest="max_period_diff", type=int, default=500,
        help="Max difference between two periods to be considered the same (us). "
             "Lower = stricter periodicity requirement.",
    )
    parser.add_argument(
        "--freq-precision", dest="freq_precision", type=float, default=5.0,
        help="Width of frequency bins in Hz for histogram display.",
    )
    parser.add_argument(
        "--min-pixel-count", dest="min_pixel_count", type=int, default=25,
        help="Minimum vibrating pixels to consider a frequency real (not noise).",
    )

    # FFT grid analysis
    parser.add_argument(
        "--grid-cells", dest="grid_cells", type=int, default=16,
        help="Number of grid cells per axis for ROI-based FFT analysis.",
    )
    parser.add_argument(
        "--fft-window", dest="fft_window_sec", type=float, default=0.5,
        help="Rolling FFT window duration in seconds.",
    )
    parser.add_argument(
        "--min-snr", dest="min_snr", type=float, default=3.0,
        help="Minimum signal-to-noise ratio for a cell's FFT peak to be reported.",
    )

    # Timing
    parser.add_argument(
        "--delta-t", dest="delta_t", type=int, default=500,
        help="Event slice duration (us). Determines sampling rate. "
             "Default 500 us -> 2000 Hz sampling -> 1000 Hz Nyquist limit.",
    )
    parser.add_argument(
        "--update-freq", dest="update_freq", type=float, default=25,
        help="How often (Hz) the frequency heatmap updates.",
    )
    parser.add_argument(
        "--print-interval", dest="print_interval_sec", type=float, default=0.5,
        help="How often (seconds) to print detected propeller stats.",
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


def main():
    args = parse_args()

    delta_t = args.delta_t
    fs = 1e6 / delta_t  # sampling rate in Hz
    fft_window_samples = int(args.fft_window_sec * fs)
    print_interval_samples = max(1, int(args.print_interval_sec * fs))

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

    # Dominant value extractor
    dominant_algo = DominantValueMapAlgorithm(
        args.min_freq, args.max_freq, args.freq_precision, args.min_pixel_count)

    # Heatmap frame generator
    heat_gen = HeatMapFrameGeneratorAlgorithm(
        args.min_freq, args.max_freq, args.freq_precision, width, height, "Hz")
    freq_img = heat_gen.get_output_image()
    freq_full_height = heat_gen.full_height

    # --- Grid-based FFT analyzer ---
    grid_analyzer = PropellerGridAnalyzer(
        width, height,
        grid_cells=args.grid_cells,
        fft_window_samples=fft_window_samples,
        min_freq=args.min_freq,
        max_freq=args.max_freq,
        num_blades=args.num_blades,
        min_snr=args.min_snr,
    )

    # --- Windows ---
    # Event viewer
    ev_window = MTWindow(title="Events - Propeller Detector", width=width, height=height,
                         mode=BaseWindow.RenderMode.BGR, open_directly=True)
    # Frequency heatmap
    freq_window = MTWindow(title="Frequency Map", width=width, height=freq_full_height,
                           mode=BaseWindow.RenderMode.BGR, open_directly=True)

    def keyboard_cb(key, scancode, action, mods):
        if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
            ev_window.set_close_flag()
            freq_window.set_close_flag()

    ev_window.set_keyboard_callback(keyboard_cb)
    freq_window.set_keyboard_callback(keyboard_cb)

    # Event frame generator
    event_frame_gen = PeriodicFrameGenerationAlgorithm(
        sensor_width=width, sensor_height=height, fps=25, palette=ColorPalette.Dark)

    def on_cd_frame_cb(ts, cd_frame):
        ev_window.show_async(cd_frame)

    event_frame_gen.set_output_callback(on_cd_frame_cb)

    # Frequency map callback
    latest_freq_map = [None]  # mutable container for closure

    def on_freq_map(ts, freq_map):
        latest_freq_map[0] = freq_map.copy()

        # Generate and display heatmap
        heat_gen.generate_bgr_heat_map(freq_map, freq_img)

        # Overlay dominant frequency text
        success, dominant_freq = dominant_algo.compute_dominant_value(freq_map)
        if success:
            rpm = (dominant_freq / args.num_blades) * 60
            text = f"Dominant: {dominant_freq:.0f} Hz  ({rpm:.0f} RPM, {args.num_blades} blades)"
            cv2.putText(freq_img, text, (10, height - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)

        freq_window.show_async(freq_img)

    freq_algo.set_output_callback(on_freq_map)

    # Print header
    nyquist = fs / 2
    print("=" * 75)
    print("  Drone Propeller Rotation Detector")
    print(f"  Sensor         : {width}x{height}")
    print(f"  Sampling rate  : {fs:.0f} Hz  (delta_t={delta_t} us)")
    print(f"  Nyquist limit  : {nyquist:.0f} Hz")
    print(f"  Frequency range: {args.min_freq} - {args.max_freq} Hz")
    print(f"  Blades assumed : {args.num_blades}")
    print(f"  FFT grid       : {args.grid_cells}x{args.grid_cells} cells")
    print(f"  FFT window     : {args.fft_window_sec}s ({fft_window_samples} samples)")
    print(f"  Min SNR        : {args.min_snr}")
    print("=" * 75)
    print()

    sample_count = 0

    for evs in mv_iterator:
        EventLoop.poll_and_dispatch()
        event_frame_gen.process_events(evs)
        freq_algo.process_events(evs)

        if ev_window.should_close() or freq_window.should_close():
            break

        # Feed events into grid analyzer
        grid_analyzer.accumulate(evs)
        sample_count += 1

        # Periodic printout from grid FFT
        if sample_count % print_interval_samples == 0:
            detections = grid_analyzer.analyze(fs)
            clusters = cluster_detections(detections)

            if clusters:
                print(f"[{sample_count * delta_t / 1e6:7.2f}s] "
                      f"Detected {len(clusters)} propeller region(s):")
                for i, cl in enumerate(clusters):
                    print(f"    Propeller {i+1}: "
                          f"freq={cl['freq_hz']:6.1f} Hz, "
                          f"RPM={cl['rpm']:7.0f}, "
                          f"cells={cl['n_cells']}, "
                          f"SNR={cl['avg_snr']:.1f}")
            else:
                print(f"[{sample_count * delta_t / 1e6:7.2f}s] "
                      f"No propeller detected")

    ev_window.destroy()
    freq_window.destroy()
    print("\nDone.")


if __name__ == "__main__":
    main()
