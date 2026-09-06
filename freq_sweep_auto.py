"""Automatic LED frequency characterisation. No window, no interaction.

Observes the board's ladder for a few cycles and prints where the blink stops
being resolvable. Everything is decided from the data; nothing to click.

HOW THE LEDs ARE FOUND. Earlier attempts picked the brightest blob, then the most
periodic blob, and both locked onto room lighting - an LED bulb's PWM driver is
bright AND perfectly periodic. This version uses two facts that only the board
satisfies:

  LOCATE    for each rung, every pixel in the frame is tested for intervals
            matching the commanded period, and a pixel scores a point if it
            matches. The swept LED matches at MANY rungs because it changes with
            the ladder; a room light sits at one fixed frequency and can match at
            most one rung. Counting rungs matched, not brightness, is what
            finally separates them - activity-based locating kept finding a
            380 kev/s lamp instead of a 2 m-distant LED.
  SEPARATE  the located pixels are split into two clusters, because the board has
            two LEDs (D13 and the RGB package) a known 26.47 mm apart.
  IDENTIFY  the board broadcasts the frequency it is emitting. The SWEPT LED is
            the cluster whose measured frequency FOLLOWS that; the other cluster
            sits at a fixed frequency and is the reference. Ambient flicker
            follows nothing, so it cannot be mistaken for either.

WHAT THE NUMBERS MEAN.
  ev/period  ~2.0: the pixel emits both the ON and the OFF edge of every blink.
             Towards 1.0 it is skipping edges; near 0 the blink is gone. This is
             the sensor's own limit, with no SDK algorithm in the way.
  reference  the other LED, held at a fixed frequency for the whole run. If the
             swept LED degrades while this stays clean, the cause is frequency
             and not the board moving or the biases being wrong.

Usage:
    python freq_sweep_auto.py                       # ~2 ladder cycles
    python freq_sweep_auto.py --cycles 3 --csv sweep.csv
"""

import argparse
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

SETTLE_S = 0.9        # discarded after a rung change: PWM restart plus our own latency


class IntervalStream:
    """Streaming per-pixel inter-event intervals for same-polarity events."""

    def __init__(self, npix, width):
        self.last = np.full(npix, -1, np.int64)
        self.width = width

    def __call__(self, evs):
        on = evs[evs["p"] == 1]
        if on.size < 4:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        idx = on["y"].astype(np.int64) * self.width + on["x"]
        t = on["t"].astype(np.int64)
        # Sort by pixel then time so consecutive same-pixel events are adjacent,
        # and stitch to the previous slice through self.last.
        order = np.lexsort((t, idx))
        idx_s, t_s = idx[order], t[order]
        same = np.empty(idx_s.size, bool)
        same[0] = False
        same[1:] = idx_s[1:] == idx_s[:-1]
        prev = self.last[idx_s]
        d = np.where(same, np.concatenate([[0], np.diff(t_s)]), t_s - prev)
        valid = (d > 0) & (d < 500000) & (same | (prev >= 0))
        self.last[idx_s] = t_s
        return d[valid], idx_s[valid]


def split_clusters(mask_idx, width, min_sep_px=4.0):
    """Split located pixels into the board's two LEDs, if they are resolvable."""
    ys, xs = np.divmod(mask_idx, width)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    if pts.shape[0] < 6:
        return [mask_idx]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centres = cv2.kmeans(pts, 2, None, crit, 5, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    if np.linalg.norm(centres[0] - centres[1]) < min_sep_px or \
            min((labels == 0).sum(), (labels == 1).sum()) < 3:
        return [mask_idx]
    return [mask_idx[labels == 0], mask_idx[labels == 1]]


def period_of(iv):
    """Dominant blink period from a bag of intervals."""
    if iv.size < 20:
        return 0.0
    med = float(np.median(iv))
    core = iv[(iv > 0.75 * med) & (iv < 1.25 * med)]
    return float(np.median(core)) if core.size else med


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", default="COM9", help="Serial port of the board.")
    p.add_argument("--cycles", type=float, default=2.0, help="Ladder cycles to observe.")
    p.add_argument("--delta-t", type=int, default=20000, help="Event slice, us.")
    p.add_argument("--min-intervals", type=int, default=20,
                   help="Matching intervals a pixel needs within one rung to score that rung.")
    p.add_argument("--min-rungs", type=int, default=3,
                   help="Rungs a pixel must follow before it is accepted as the swept LED. "
                        "Ambient flicker sits at one frequency, so it can match at most one.")
    p.add_argument("--tol", type=float, default=0.20,
                   help="Relative tolerance when matching an interval to the commanded period.")
    p.add_argument("--csv", default="", help="Write the per-rung summary here.")
    args = p.parse_args()

    from metavision_core.event_io import EventsIterator
    sys.path.insert(0, ".")
    from freq_sweep_live import Board

    board = Board("none" if args.port == "none" else args.port)
    if board.ser is None:
        print("error: no board; the commanded frequency is required", file=sys.stderr)
        return 1

    it = EventsIterator(input_path="", delta_t=args.delta_t)
    height, width = it.get_size()
    npix = height * width
    print(f"camera {width}x{height}")

    intervals = IntervalStream(npix, width)
    rung_hits = np.zeros(npix, np.int16)      # rungs whose rate a pixel followed
    match_counts = np.zeros(npix, np.int32)   # interval count within the current rung
    sum_iv = np.zeros(npix, np.float64)       # interval sum within the current rung

    groups = None
    group_lut = np.full(npix, -1, np.int8)
    # per (group, rung): intervals, event count, duration
    stats = defaultdict(lambda: {"iv": [], "ev": 0, "dur": 0.0})
    cur_hz, cur_since = 0.0, 0.0
    rungs_seen, stream_t = 0, 0.0

    print("locating: scoring pixels by how many rungs their blink rate follows...")
    for evs in it:
        board.poll()
        if evs.size == 0:
            continue
        stream_t = evs["t"][-1] / 1e6
        d, idx = intervals(evs)

        if board.sweep_actual_hz != cur_hz:
            if groups is None and cur_hz > 0:
                # Close the rung out. A pixel scores this rung only if its MEAN
                # interval matches the commanded period. Counting individually
                # matching intervals is not enough: a pixel that misses edges
                # emits intervals at multiples of its own period, so a fixed
                # 3.8 kHz lamp produces 1827 and 2088 us gaps that sit inside the
                # 2000 us window and let it fake several rungs. A mean cannot be
                # faked that way - it stays near the source's own period.
                target = 1e6 / cur_hz
                n = match_counts                       # interval count this rung
                mean_iv = np.where(n > 0, sum_iv / np.maximum(n, 1), 0.0)
                follows = (n >= args.min_intervals) & \
                          (np.abs(mean_iv - target) <= args.tol * target)
                rung_hits += follows.astype(np.int16)
                match_counts[:] = 0
                sum_iv[:] = 0.0
                best = int(rung_hits.max())
                if rungs_seen >= 8 and best >= args.min_rungs:
                    mask_idx = np.flatnonzero(rung_hits >= max(args.min_rungs, best - 1))
                    groups = split_clusters(mask_idx, width)
                    for g, gi in enumerate(groups):
                        group_lut[gi] = g
                        ys, xs = np.divmod(gi, width)
                        print(f"  LED cluster {g}: {gi.size:3d} px at "
                              f"({int(xs.mean())},{int(ys.mean())}), followed up to {best} rungs")
                    if len(groups) == 1:
                        print("  one cluster only - at this distance the two LEDs are not "
                              "separable, so the reference control is unavailable")
            cur_hz, cur_since = board.sweep_actual_hz, stream_t
            rungs_seen += 1
            print(f"  [{stream_t:6.1f}s] rung {cur_hz:9.0f} Hz"
                  f"{'  (locating)' if groups is None else ''}")

        if cur_hz <= 0 or stream_t - cur_since < SETTLE_S:
            continue

        if groups is None:
            # Accumulate per-pixel interval count and sum, so the rung can be
            # scored on the mean when it closes.
            if d.size:
                match_counts += np.bincount(idx, minlength=npix).astype(np.int32)
                sum_iv += np.bincount(idx, weights=d.astype(np.float64), minlength=npix)
            if rungs_seen > 3 * 8:
                print(f"\nNO SWEPT LED FOUND: across three full cycles, no pixel followed the "
                      f"commanded frequency at {args.min_rungs} or more rungs.\n"
                      "The board's D13 LED is not visible to the camera. The reference LED "
                      "alone cannot stand in for it - it never changes frequency.")
                return 2
            continue

        for g in range(len(groups)):
            sel = group_lut[idx] == g
            s = stats[(g, cur_hz)]
            if sel.any():
                s["iv"].append(d[sel])
            s["ev"] += int((group_lut[evs["y"].astype(np.int64) * width + evs["x"]] == g).sum())
            s["dur"] += args.delta_t / 1e6

        if rungs_seen > (args.cycles + 1) * 8 + 1:
            break

    board.close()

    rung_list = sorted({hz for _, hz in stats if hz > 0})
    if not rung_list:
        print("no rungs observed")
        return 2

    # The swept LED is the cluster whose measured frequency actually moves.
    spread = {}
    for g in range(len(groups)):
        f = [1e6 / period_of(np.concatenate(stats[(g, hz)]["iv"]))
             for hz in rung_list
             if stats[(g, hz)]["iv"] and period_of(np.concatenate(stats[(g, hz)]["iv"])) > 0]
        spread[g] = (max(f) / min(f)) if len(f) > 1 and min(f) > 0 else 1.0
    sweep_g = max(spread, key=spread.get)
    ref_g = next((g for g in range(len(groups)) if g != sweep_g), None)
    print(f"\nswept LED = cluster {sweep_g} (frequency range x{spread[sweep_g]:.1f}); "
          f"reference = {'cluster %d' % ref_g if ref_g is not None else 'not resolved'}")

    print(f"\n{'commanded':>10} {'measured':>10} {'err':>8} {'ev/period':>10} {'px':>4} "
          f"{'ev/s':>9} {'ref Hz':>8}  verdict")
    print("-" * 92)
    rows = []
    for hz in rung_list:
        s = stats[(sweep_g, hz)]
        iv = np.concatenate(s["iv"]) if s["iv"] else np.empty(0)
        npx = max(1, groups[sweep_g].size)
        ev_s = s["ev"] / s["dur"] if s["dur"] else 0.0
        ref_hz = 0.0
        if ref_g is not None and stats[(ref_g, hz)]["iv"]:
            rp = period_of(np.concatenate(stats[(ref_g, hz)]["iv"]))
            ref_hz = 1e6 / rp if rp > 0 else 0.0
        period = period_of(iv)
        if period <= 0:
            print(f"{hz:>9.0f}Hz {'-':>10} {'-':>8} {'-':>10} {npx:>4} {ev_s:>9.0f} "
                  f"{ref_hz:>8.0f}  BLINK GONE")
            rows.append((hz, 0.0, 0.0, ev_s, ref_hz))
            continue
        meas = 1e6 / period
        evper = (s["ev"] / npx) / (s["dur"] * 1e6 / period) if s["dur"] else 0.0
        err = 100.0 * (meas - hz) / hz
        v = ("ok" if abs(err) < 10 and evper >= 1.5 else
             "marginal: some edges lost" if abs(err) < 10 and evper >= 0.8 else
             "SENSOR LIMIT: skipping edges" if evper < 0.8 else "period mismeasured")
        print(f"{hz:>9.0f}Hz {meas:>9.0f}Hz {err:>+7.1f}% {evper:>10.2f} {npx:>4} {ev_s:>9.0f} "
              f"{ref_hz:>8.0f}  {v}")
        rows.append((hz, meas, evper, ev_s, ref_hz))

    if args.csv:
        with open(args.csv, "w", encoding="utf-8") as fh:
            fh.write("commanded_hz,measured_hz,ev_per_period,ev_per_s,reference_hz\n")
            for r in rows:
                fh.write(",".join(f"{v:.3f}" for v in r) + "\n")
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
