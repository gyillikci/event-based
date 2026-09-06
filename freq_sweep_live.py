"""Interactive, real-time LED blink-frequency probe.

Drives arduino/freq_sweep/freq_sweep.ino over serial and measures, live, whether
the event camera still resolves the blink. Change the frequency with the arrow
keys and watch the answer update in the same second.

WHAT IT MEASURES, AND WHY NOT WITH THE SDK. The obvious approach - run
FrequencyMapAsyncAlgorithm and see when it stops reporting - cannot separate two
very different failures: the sensor no longer producing clean events, and the
algorithm's period matching being mistuned for that frequency. So this works on
raw per-pixel event timestamps instead: for a blinking LED, the intervals between
successive same-polarity events on one pixel ARE the blink period. That is the
sensor's own answer, with no algorithm parameter in the way.

Two derived numbers say which stage is binding:

    ev/period   ~2.0 means the pixel emits both the ON and the OFF edge of every
                blink. Falling towards 1.0 and below means it is skipping edges,
                so the SENSOR is out of bandwidth or in refractory.
    error %     how far the measured period is from the one actually emitted.
                Large while ev/period stays near 2 points at timestamp
                resolution, not at the pixel.

The reference LED is measured alongside as a control: if the swept LED washes out
while the reference stays clean, the loss is a frequency effect and not the board
drifting out of frame.

Keys (letters work too, for terminals that swallow the arrows):
    up    / w     next rung of the ladder
    down  / s     previous rung
    right / d     fine adjust up, x1.1
    left  / x     fine adjust down, /1.1
    a             run the board's unattended ladder
    space         blackout the swept LED (0 Hz)
    r             re-lock onto the brightest blobs
    q / ESC       quit

Usage:
    python freq_sweep_live.py                    # auto-detect the serial port
    python freq_sweep_live.py --port COM9
    python freq_sweep_live.py --no-serial        # measure only, set frequency yourself
    python freq_sweep_live.py --csv sweep.csv
"""

import argparse
import sys
import time
from collections import deque

import cv2
import numpy as np

LADDER = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]


# --------------------------------------------------------------------------------------
# Board link
# --------------------------------------------------------------------------------------

class Board:
    """Serial link to the sketch. Degrades to a no-op if there is no board.

    Speaks to either firmware revision:

      interactive   the current freq_sweep.ino, which answers commands and
                    reports "sweep_actual=<hz>". Frequency is settable from here.
      listen-only   the earlier auto-sweep revision, which takes no commands but
                    broadcasts "step <n>: <hz> Hz" as it walks its ladder. The
                    frequency cannot be set, but it IS known, which is all the
                    measurement needs.

    Detected at startup by asking "?" and seeing whether anything comes back.
    """

    def __init__(self, port="", baud=115200):
        self.ser = None
        self.last_reply = ""
        self.sweep_actual_hz = 0.0
        self.interactive = False
        if port == "none":
            return
        try:
            import serial
            from serial.tools import list_ports
        except ImportError:
            print("pyserial not installed; running measure-only", file=sys.stderr)
            return
        if not port:
            candidates = [p.device for p in list_ports.comports()]
            if not candidates:
                print("no serial port found; running measure-only", file=sys.stderr)
                return
            port = candidates[0]
        try:
            self.ser = serial.Serial(port, baud, timeout=0.6)
            time.sleep(2.0)          # the nRF52840 USB CDC needs a moment after open
            self.ser.reset_input_buffer()
        except Exception as exc:
            print(f"could not open {port}: {exc}; running measure-only", file=sys.stderr)
            self.ser = None
            return

        # Probe which firmware is on the board.
        self.ser.write(b"?\n")
        time.sleep(0.6)
        reply = self.ser.read(self.ser.in_waiting).decode(errors="replace")
        self.interactive = "sweep_actual=" in reply
        if reply.strip():
            self._parse(reply.strip().splitlines()[-1])
        if self.interactive:
            print(f"board on {port}: interactive firmware, frequency is settable")
        else:
            print(f"board on {port}: auto-sweep firmware, LISTEN-ONLY - it broadcasts its "
                  f"frequency but takes no commands. Flash the current freq_sweep.ino for "
                  f"interactive control.")

    def _parse(self, line):
        """Read the commanded frequency out of either firmware's output."""
        self.last_reply = line
        for tok in line.split():
            if tok.startswith("sweep_actual="):
                try:
                    self.sweep_actual_hz = float(tok.split("=")[1])
                except ValueError:
                    pass
                return
        # Auto-sweep firmware: "step 3: 5000 Hz"
        if line.startswith("step ") and line.endswith(" Hz"):
            try:
                self.sweep_actual_hz = float(line.split(":")[1].strip().split()[0])
            except (IndexError, ValueError):
                pass

    def send(self, cmd):
        if self.ser is None or not self.interactive:
            return ""
        try:
            self.ser.write((cmd + "\n").encode())
            reply = self.ser.readline().decode(errors="replace").strip()
            if reply:
                self._parse(reply)
            return reply
        except Exception as exc:
            return f"ERR {exc}"

    def poll(self):
        """Pick up unsolicited lines: both firmwares announce each rung."""
        if self.ser is None:
            return
        try:
            while self.ser.in_waiting:
                line = self.ser.readline().decode(errors="replace").strip()
                if line:
                    self._parse(line)
        except Exception:
            pass

    def close(self):
        if self.ser is not None:
            self.ser.close()


# --------------------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------------------

class BlobMeter:
    """Blink statistics for one LED, from raw per-pixel event intervals."""

    def __init__(self, cx, cy, half=14, history=12):
        self.set_centre(cx, cy, half)
        self.periods = deque(maxlen=history)
        self.ev_per_period = deque(maxlen=history)
        self.coherences = deque(maxlen=history)
        self.active_px = 0
        self.ev_per_s = 0.0

    def set_centre(self, cx, cy, half=14):
        self.cx, self.cy, self.half = int(cx), int(cy), int(half)

    @property
    def roi(self):
        return (self.cx - self.half, self.cx + self.half + 1,
                self.cy - self.half, self.cy + self.half + 1)

    def update(self, evs, dt_s, width):
        x0, x1, y0, y1 = self.roi
        m = ((evs["x"] >= x0) & (evs["x"] < x1) & (evs["y"] >= y0) & (evs["y"] < y1))
        sub = evs[m]
        if sub.size < 8:
            self.ev_per_s = 0.0
            return
        self.ev_per_s = sub.size / dt_s if dt_s > 0 else 0.0

        on = sub[sub["p"] == 1]
        if on.size < 8:
            return
        idx = on["y"].astype(np.int64) * width + on["x"]
        t = on["t"].astype(np.int64)
        # Consecutive same-pixel intervals: sort by pixel, then time, and diff
        # only where the pixel index does not change.
        order = np.lexsort((t, idx))
        idx_s, t_s = idx[order], t[order]
        same = idx_s[1:] == idx_s[:-1]
        d = np.diff(t_s)[same]
        d = d[d > 0]
        if d.size < 8:
            return

        # The blink period is the dominant short interval; longer ones are the
        # pixel having missed an edge.
        med = float(np.median(d))
        core = d[(d > 0.75 * med) & (d < 1.25 * med)]
        period_us = float(np.median(core)) if core.size else med
        self.periods.append(period_us)

        # COHERENCE is what separates a blinking LED from any other bright thing:
        # a square wave puts nearly every interval within a few percent of the
        # median, whereas scene texture or a flickering lamp scatters them. This
        # is the score the blob lock is chosen on - brightness alone picked the
        # wrong target.
        self.coherences.append(core.size / d.size)

        self.active_px = int(np.unique(idx).size)
        if period_us > 0 and dt_s > 0 and self.active_px:
            n_periods = dt_s * 1e6 / period_us
            self.ev_per_period.append((sub.size / self.active_px) / n_periods)

    @property
    def hz(self):
        if not self.periods:
            return 0.0
        p = float(np.median(self.periods))
        return 1e6 / p if p > 0 else 0.0

    @property
    def evp(self):
        return float(np.median(self.ev_per_period)) if self.ev_per_period else 0.0

    @property
    def coherence(self):
        return float(np.median(self.coherences)) if self.coherences else 0.0


def find_blobs(counts, n=2, box=14, min_events=200):
    work = counts.copy()
    out = []
    for _ in range(n):
        i = int(np.argmax(work))
        y, x = divmod(i, work.shape[1])
        if work[y, x] < min_events:
            break
        out.append((x, y))
        y0, y1 = max(0, y - 3 * box), min(work.shape[0], y + 3 * box)
        x0, x1 = max(0, x - 3 * box), min(work.shape[1], x + 3 * box)
        work[y0:y1, x0:x1] = 0
    return out


def verdict_for(evp, err_pct, hz):
    if hz <= 0:
        return "no blink", (0, 0, 255)
    if evp < 0.8:
        return "SENSOR LIMIT: pixel skipping edges", (0, 80, 255)
    if abs(err_pct) > 10:
        return "period mismeasured", (0, 200, 255)
    if evp < 1.5:
        return "marginal: some edges lost", (0, 220, 255)
    return "ok", (80, 255, 80)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", default="", help="Serial port; empty auto-detects, 'none' disables.")
    p.add_argument("--no-serial", action="store_true", help="Measure only, do not drive the board.")
    p.add_argument("--delta-t", type=int, default=20000, help="Event slice, us.")
    p.add_argument("--csv", default="", help="Log every measurement here.")
    p.add_argument("--roi-half", type=int, default=14, help="Half-size of the LED ROI, px.")
    p.add_argument("--lock-delay-s", type=float, default=1.5,
                   help="Accumulate activity this long before choosing candidates.")
    p.add_argument("--lock-eval-s", type=float, default=0.8,
                   help="Judge candidate blobs for this long before committing.")
    p.add_argument("--lock-candidates", type=int, default=6,
                   help="Bright blobs evaluated; the most periodic ones win.")
    p.add_argument("--track-tol", type=float, default=0.25,
                   help="Relative tolerance for a blob to count as following the commanded "
                        "frequency. This is what tells the marker apart from ambient flicker.")
    p.add_argument("--mismatch-slices", type=int, default=120,
                   help="Slices a locked blob may ignore the commanded frequency before the "
                        "lock is dropped (120 x 20ms = 2.4s, longer than a step transition).")
    p.add_argument("--min-coherence", type=float, default=0.55,
                   help="Fraction of intervals within +/-25%% of the median needed to "
                        "call a blob a blinking LED rather than scene clutter.")
    args = p.parse_args()

    from metavision_core.event_io import EventsIterator
    from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette

    board = Board("none" if args.no_serial else args.port)
    ladder_i = 1
    if board.ser is not None:
        board.send(f"f {LADDER[ladder_i]}")

    it = EventsIterator(input_path="", delta_t=args.delta_t)
    height, width = it.get_size()
    print(f"camera {width}x{height}")

    frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=width, sensor_height=height,
                                                 fps=25, palette=ColorPalette.Dark)
    latest = {"img": None}
    frame_gen.set_output_callback(lambda ts, f: latest.__setitem__("img", f.copy()))

    counts = np.zeros(height * width, np.float64)
    meters, candidates = [], []
    lock_state, lock_since = "accumulate", time.time()
    mismatch_slices = 0
    csv = None
    if args.csv:
        csv = open(args.csv, "w", encoding="utf-8")
        csv.write("t_s,commanded_hz,actual_hz,sweep_measured_hz,sweep_err_pct,sweep_ev_per_period,"
                  "sweep_active_px,sweep_ev_per_s,sweep_coherence,"
                  "ref_measured_hz,ref_ev_per_period\n")

    win = "LED frequency probe - up/down ladder, left/right fine, a sweep, space off, q quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 760)

    t_start = time.time()
    frames = 0
    try:
        for evs in it:
            frames += 1
            board.poll()

            if evs.size:
                idx = evs["y"].astype(np.int64) * width + evs["x"]
                counts *= 0.90                      # decay, so the lock follows the LEDs
                counts += np.bincount(idx, minlength=height * width)

            # LOCK-ON, in three stages. Picking the brightest blob on the first
            # slice is what produced a confident measurement of a flickering lamp,
            # so candidates are now judged on how periodic they are, not how bright.
            dt_s = args.delta_t / 1e6
            now = time.time()
            if lock_state == "accumulate":
                if now - lock_since >= args.lock_delay_s and counts.max() > 400:
                    candidates = [BlobMeter(bx, by, half=args.roi_half)
                                  for bx, by in find_blobs(counts.reshape(height, width),
                                                           n=args.lock_candidates)]
                    if candidates:
                        lock_state, lock_since = "evaluate", now
            elif lock_state == "evaluate":
                if now - lock_since >= args.lock_eval_s:
                    # Coherence alone is not enough: a mains-driven lamp or an LED
                    # bulb's PWM is perfectly periodic too, and one at ~5 kHz was
                    # picked up and reported as the marker for a whole run. When the
                    # commanded frequency is known, the real LED is the blob that
                    # FOLLOWS it, so require a match before committing.
                    want = board.sweep_actual_hz
                    scored = []
                    for c in candidates:
                        if c.hz <= 0:
                            continue
                        tracks = (want <= 0) or (abs(c.hz - want) / want <= args.track_tol)
                        scored.append((c.coherence + (1.0 if tracks else 0.0), tracks, c))
                    scored.sort(key=lambda s: -s[0])
                    good = [c for score, tracks, c in scored
                            if c.coherence >= args.min_coherence and (tracks or want <= 0)]
                    if good:
                        meters = good[:2]
                        lock_state = "locked"
                        print(f"locked {len(meters)} blob(s): " + ", ".join(
                            f"({m.cx},{m.cy}) {m.hz:.0f}Hz coh={m.coherence:.2f}" for m in meters))
                    else:
                        if scored:
                            b = scored[0][2]
                            why = (f"best candidate ({b.cx},{b.cy}) {b.hz:.0f}Hz "
                                   f"coh={b.coherence:.2f}" +
                                   (f", does not follow the commanded {want:.0f}Hz"
                                    if want > 0 and not scored[0][1] else ""))
                        else:
                            why = "no periodic candidate at all"
                        print(f"no marker locked: {why}; still looking")
                        candidates, counts[:] = [], 0
                        lock_state, lock_since = "accumulate", now

            if evs.size:
                for m in (meters if lock_state == "locked" else candidates):
                    m.update(evs, dt_s, width)
                frame_gen.process_events(evs)

            # Drop the lock if the target stops looking periodic, or stops following
            # the commanded frequency - the latter is what exposes an ambient
            # flicker that was locked onto by accident.
            if lock_state == "locked" and meters:
                want = board.sweep_actual_hz
                incoherent = meters[0].coherence < args.min_coherence * 0.6
                off_freq = want > 0 and abs(meters[0].hz - want) / want > args.track_tol
                if off_freq:
                    mismatch_slices += 1
                else:
                    mismatch_slices = 0
                if incoherent or mismatch_slices > args.mismatch_slices:
                    why = "not periodic" if incoherent else "stopped following the commanded frequency"
                    print(f"lock lost ({why}), re-acquiring")
                    meters, candidates, counts[:] = [], [], 0
                    mismatch_slices = 0
                    lock_state, lock_since = "accumulate", time.time()

            img = latest["img"]
            if img is None:
                continue
            vis = img.copy()

            commanded = board.sweep_actual_hz if board.ser is not None else 0.0

            # The swept LED is whichever blob is closest to the commanded frequency;
            # the other is the reference control.
            sweep_m, ref_m = None, None
            if meters:
                if commanded > 0 and len(meters) > 1:
                    k = int(np.argmin([abs(m.hz - commanded) for m in meters]))
                    sweep_m = meters[k]
                    ref_m = meters[1 - k] if len(meters) > 1 else None
                else:
                    sweep_m = meters[0]
                    ref_m = meters[1] if len(meters) > 1 else None

            for m, label in ((sweep_m, "SWEEP"), (ref_m, "REF")):
                if m is None:
                    continue
                x0, x1, y0, y1 = m.roi
                col = (80, 255, 80) if label == "SWEEP" else (255, 200, 80)
                cv2.rectangle(vis, (x0, y0), (x1, y1), col, 2)
                cv2.putText(vis, f"{label} {m.hz:.0f}Hz", (x0, max(12, y0 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

            panel = np.zeros((150, vis.shape[1], 3), np.uint8)
            meas = sweep_m.hz if sweep_m else 0.0
            evp = sweep_m.evp if sweep_m else 0.0
            err = 100.0 * (meas - commanded) / commanded if commanded > 0 else float("nan")
            vtext, vcol = verdict_for(evp, err, meas)

            lines = [
                (f"commanded {commanded:9.0f} Hz    measured {meas:9.0f} Hz    "
                 f"err {err:+6.1f} %", (255, 255, 255)),
                (f"ev/period {evp:5.2f}   coherence {sweep_m.coherence if sweep_m else 0:4.2f}   "
                 f"px {sweep_m.active_px if sweep_m else 0:4d}   "
                 f"ev/s {sweep_m.ev_per_s if sweep_m else 0:9.0f}", (200, 200, 200)),
                (f"reference {ref_m.hz if ref_m else 0:7.0f} Hz  ev/period "
                 f"{ref_m.evp if ref_m else 0:4.2f}   (control)", (255, 200, 80)),
                (vtext, vcol),
            ]
            if lock_state != "locked":
                lines[-1] = (f"[{lock_state}] looking for a coherent blink...", (0, 200, 255))
            for i, (text, col) in enumerate(lines):
                cv2.putText(panel, text, (10, 26 + 32 * i), cv2.FONT_HERSHEY_SIMPLEX,
                            0.62, col, 1, cv2.LINE_AA)
            status = board.last_reply[:90]
            if board.ser is not None and not board.interactive:
                status = f"LISTEN-ONLY (auto-sweep firmware, keys inactive)   {status}"
            if status:
                cv2.putText(panel, status, (10, 144),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1, cv2.LINE_AA)

            cv2.imshow(win, np.vstack([vis, panel]))

            if csv and sweep_m and commanded > 0 and lock_state == "locked":
                csv.write(f"{time.time()-t_start:.3f},{commanded:.1f},{board.sweep_actual_hz:.2f},"
                          f"{meas:.1f},{err:.2f},{evp:.3f},{sweep_m.active_px},"
                          f"{sweep_m.ev_per_s:.0f},{sweep_m.coherence:.3f},"
                          f"{ref_m.hz if ref_m else 0:.1f},"
                          f"{ref_m.evp if ref_m else 0:.3f}\n")

            # waitKeyEx, not waitKey(..) & 0xFF: masking to a byte destroys the
            # arrow-key codes, which differ between Windows and GTK builds.
            key = cv2.waitKeyEx(1)
            if key == -1:
                continue
            ascii_key = key & 0xFF if 0 <= key < 0x110000 and (key & ~0xFF) == 0 else -1
            UP = key in (2490368, 82) or ascii_key == ord("w")
            DOWN = key in (2621440, 84) or ascii_key == ord("s")
            RIGHT = key in (2555904, 83) or ascii_key == ord("d")
            LEFT = key in (2424832, 81) or ascii_key == ord("x")

            if ascii_key in (27, ord("q")):
                break
            elif UP:
                ladder_i = min(len(LADDER) - 1, ladder_i + 1)
                board.send(f"f {LADDER[ladder_i]}")
            elif DOWN:
                ladder_i = max(0, ladder_i - 1)
                board.send(f"f {LADDER[ladder_i]}")
            elif RIGHT:
                board.send(f"f {int(max(1, board.sweep_actual_hz * 1.1))}")
            elif LEFT:
                board.send(f"f {int(max(1, board.sweep_actual_hz / 1.1))}")
            elif ascii_key == ord("a"):
                board.send("sweep")
            elif ascii_key == ord(" "):
                board.send("f 0")
            elif ascii_key == ord("r"):
                meters, candidates, counts[:] = [], [], 0
                lock_state, lock_since = "accumulate", time.time()
    finally:
        cv2.destroyAllWindows()
        if csv:
            csv.close()
            print(f"wrote {args.csv}")
        board.close()
    print(f"{frames} slices in {time.time()-t_start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
