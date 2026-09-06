"""Record a fixed-duration RAW from the live camera.

EventsIterator can read but not write, so this drops to HAL, which is the layer
that actually owns the raw stream. Used to capture the LED frequency sweep, where
an exact, unattended duration matters more than a live view.

Usage:
    python record_raw.py out.raw --duration-s 30
    python record_raw.py out.raw --duration-s 30 --biases samples/recording.bias
"""

import argparse
import sys
import time

import metavision_hal as mh


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output", help="Path to the RAW file to write.")
    p.add_argument("--duration-s", type=float, default=30.0)
    p.add_argument("--serial", default="", help="Camera serial; empty = first available.")
    p.add_argument("--biases", default="", help="Optional bias file to load before recording.")
    args = p.parse_args()

    device = mh.DeviceDiscovery.open(args.serial)
    if device is None:
        print("error: no camera found", file=sys.stderr)
        return 1

    if args.biases:
        biases = device.get_i_ll_biases()
        if biases is None:
            print("warning: camera exposes no bias facility, ignoring --biases", file=sys.stderr)
        else:
            with open(args.biases, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.split("%")[0].strip()
                    if not line:
                        continue
                    value, name = line.split()[0], line.split()[-1]
                    try:
                        biases.set(name, int(value))
                    except Exception as exc:               # a bias the sensor does not expose
                        print(f"  skipped bias {name}: {exc}", file=sys.stderr)
            print(f"loaded biases from {args.biases}")

    geom = device.get_i_geometry()
    print(f"camera {geom.get_width()}x{geom.get_height()}, recording {args.duration_s:.1f} s "
          f"-> {args.output}")

    es = device.get_i_events_stream()
    es.log_raw_data(args.output)
    es.start()

    t0 = time.time()
    last_report = 0.0
    try:
        while True:
            elapsed = time.time() - t0
            if elapsed >= args.duration_s:
                break
            if es.wait_next_buffer() > 0:
                es.get_latest_raw_data()          # drain, the logger writes it to disk
            if elapsed - last_report >= 5.0:
                last_report = elapsed
                print(f"  {elapsed:5.1f}s / {args.duration_s:.0f}s")
    finally:
        es.stop_log_raw_data()
        es.stop()

    print(f"done, {time.time() - t0:.1f} s recorded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
