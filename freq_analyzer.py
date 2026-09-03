"""Live frequency analyser with hardware ROI / ERC and a lag readout.

Same algorithm as the Metavision vibration-estimation sample, plus the three knobs
that actually decide whether the stream keeps up with the camera:

  --roi     hardware window: the sensor only emits events inside it. Biggest lever
            by far -- a bright kHz LED lighting the whole array can produce hundreds
            of Mev/s, which no host-side consumer can drain.
  --erc     hardware event-rate cap (events/s).
  --delta-t batch size; larger batches = fewer Python iterations per second.

The periodic stats line reports the input rate and, more importantly, the lag:
how far the displayed data has fallen behind wall-clock time. A lag that grows
monotonically means the consumer is slower than the source, not that latency is
merely high.
"""

import argparse
import os
import sys
import time

_SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples", "analytics",
                           "python_samples", "metavision_vibration_estimation")
sys.path.insert(0, _SAMPLE_DIR)

from metavision_core.event_io import EventsIterator
from metavision_core.event_io.raw_reader import initiate_device
from metavision_hal import I_ROI
from metavision_sdk_analytics import FrequencyMapAsyncAlgorithm
from metavision_sdk_core import ColorPalette, PeriodicFrameGenerationAlgorithm
from metavision_sdk_ui import BaseWindow, EventLoop, MTWindow, UIKeyEvent

from vibration_gui import VibrationGUI


class RawEventViewer:
    """Plain event display: no frequency analysis, just accumulated events."""

    def __init__(self, width, height, fps, accumulation_time_us):
        self._window = MTWindow(title="Raw events", width=width, height=height,
                                mode=BaseWindow.RenderMode.BGR, open_directly=True)

        def keyboard_cb(key, scancode, action, mods):
            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                self._window.set_close_flag()

        self._window.set_keyboard_callback(keyboard_cb)

        self._frame_gen = PeriodicFrameGenerationAlgorithm(
            sensor_width=width, sensor_height=height, fps=fps,
            accumulation_time_us=accumulation_time_us, palette=ColorPalette.Dark)
        self._frame_gen.set_output_callback(lambda ts, frame: self._window.show_async(frame))

    def process(self, evs):
        self._frame_gen.process_events(evs)

    def should_close(self):
        return self._window.should_close()

    def destroy(self):
        self._window.destroy()


class FrequencyConsumer:
    """Frequency map pipeline wrapped to match RawEventViewer's interface."""

    def __init__(self, width, height, args):
        self._gui = VibrationGUI(width=width, height=height,
                                 min_freq=args.min_freq,
                                 max_freq=args.max_freq,
                                 freq_precision=args.freq_precision_hz,
                                 min_pixel_count=args.min_pixel_count,
                                 out_video=args.out_video)
        self._algo = FrequencyMapAsyncAlgorithm(width=width,
                                                height=height,
                                                filter_length=args.filter_length,
                                                min_freq=args.min_freq,
                                                max_freq=args.max_freq,
                                                diff_thresh_us=args.max_period_diff)
        self._algo.update_frequency = args.update_freq_hz
        self._algo.set_output_callback(lambda ts, freq_map: self._gui.show(freq_map))

    def process(self, evs):
        self._algo.process_events(evs)

    def should_close(self):
        return self._gui.should_close()

    def destroy(self):
        self._gui.destroy_window()


def parse_args():
    parser = argparse.ArgumentParser(description="Live frequency analyser (ROI/ERC tuned).",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input-event-file', dest='event_file_path', default="",
                        help="RAW/HDF5 file to replay. Default: live camera.")

    hw = parser.add_argument_group('Camera throughput options (live camera only)')
    hw.add_argument('--roi', type=int, nargs=4, metavar=('X', 'Y', 'W', 'H'), default=None,
                    help="Hardware ROI window. Events outside it are never generated.")
    hw.add_argument('--erc', type=int, default=None,
                    help="Hardware event-rate cap in events/s (e.g. 20000000).")
    hw.add_argument('--delta-t', dest='delta_t', type=int, default=5000,
                    help="Event batch size in us.")
    hw.add_argument('--bias', action='append', default=[], metavar='NAME=VALUE',
                    help="Pixel bias to set, repeatable. Raising bias_diff_on/bias_diff_off "
                         "desensitises the pixels in bright scenes; raising bias_hpf rejects "
                         "slow illumination changes while keeping a kHz modulation.")

    est = parser.add_argument_group('Estimation options')
    est.add_argument('--raw', action='store_true',
                     help="Skip the frequency analysis and just display the events.")
    est.add_argument('--fps', type=int, default=25,
                     help="Display rate of the --raw view.")
    est.add_argument('--accumulation-time-us', dest='accumulation_time_us', type=int, default=10000,
                     help="Event accumulation time per frame in the --raw view.")
    est.add_argument('--min-freq', dest='min_freq', type=float, default=200)
    est.add_argument('--max-freq', dest='max_freq', type=float, default=1500)
    est.add_argument('--filter-length', dest='filter_length', type=int, default=4,
                     help="Number of successive periods needed to declare a frequency.")
    est.add_argument('--max-period-diff', dest='max_period_diff', type=int, default=200,
                     help="Max difference (us) between two periods considered the same.")
    est.add_argument('--update-freq', dest='update_freq_hz', type=float, default=25,
                     help="Algorithm/display update rate (Hz).")
    est.add_argument('--freq-precision', dest='freq_precision_hz', type=float, default=5,
                     help="Width of the frequency histogram bins (Hz).")
    est.add_argument('--min-pixel-count', dest='min_pixel_count', type=int, default=10,
                     help="Minimum pixel count for a frequency to count as real.")

    parser.add_argument('-o', '--out-video', dest='out_video', type=str, default="")
    parser.add_argument('--stats-period', dest='stats_period', type=float, default=1.0,
                        help="Seconds between throughput/lag reports. 0 disables them.")

    args = parser.parse_args()
    if args.min_freq >= args.max_freq:
        parser.error("--min-freq must be below --max-freq")

    biases = {}
    for item in args.bias:
        name, sep, value = item.partition('=')
        if not sep:
            parser.error("--bias expects NAME=VALUE, got {!r}".format(item))
        try:
            biases[name.strip()] = int(value)
        except ValueError:
            parser.error("--bias value for {!r} is not an integer".format(name.strip()))
    args.bias = biases

    return args


def build_iterator(args):
    if args.event_file_path:
        return EventsIterator(input_path=args.event_file_path, delta_t=args.delta_t)

    device = initiate_device(path="")

    if args.roi is not None:
        i_roi = device.get_i_roi()
        if i_roi is None:
            print("WARNING: camera exposes no ROI facility, --roi ignored")
        else:
            i_roi.set_window(I_ROI.Window(*args.roi))
            i_roi.enable(True)
            print("Hardware ROI: x={} y={} w={} h={}".format(*args.roi))

    if args.erc is not None:
        erc = device.get_i_erc_module()
        if erc is None:
            print("WARNING: camera exposes no ERC facility, --erc ignored")
        else:
            erc.set_cd_event_rate(args.erc)
            erc.enable(True)
            print("Hardware ERC cap: {} ev/s".format(args.erc))

    if args.bias:
        ll_biases = device.get_i_ll_biases()
        if ll_biases is None:
            print("WARNING: camera exposes no bias facility, --bias ignored")
        else:
            for name, value in args.bias.items():
                ll_biases.set(name, value)
                print("Bias {} = {}".format(name, value))

    return EventsIterator.from_device(device=device, delta_t=args.delta_t)


def main():
    args = parse_args()

    mv_iterator = build_iterator(args)
    height, width = mv_iterator.get_size()

    if args.raw:
        consumer = RawEventViewer(width, height, args.fps, args.accumulation_time_us)
    else:
        consumer = FrequencyConsumer(width, height, args)

    wall_start = None
    stream_start_us = None
    next_report = None
    events_since_report = 0

    for evs in mv_iterator:
        EventLoop.poll_and_dispatch()
        consumer.process(evs)

        if args.stats_period > 0 and evs.size:
            now = time.perf_counter()
            if wall_start is None:
                wall_start, stream_start_us, next_report = now, int(evs['t'][0]), now + args.stats_period
            events_since_report += evs.size
            if now >= next_report:
                elapsed = now - wall_start
                stream_elapsed = (int(evs['t'][-1]) - stream_start_us) / 1e6
                print("in {:6.2f} Mev/s   lag {:7.0f} ms".format(
                    events_since_report / (now - next_report + args.stats_period) / 1e6,
                    (elapsed - stream_elapsed) * 1e3))
                events_since_report = 0
                next_report = now + args.stats_period

        if consumer.should_close():
            break

    consumer.destroy()


if __name__ == "__main__":
    main()
