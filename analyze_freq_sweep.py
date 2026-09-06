"""Find where an LED blink stops being resolvable, and which stage gives out first.

Reads a recording of arduino/freq_sweep/freq_sweep.ino and reports, per rung of
the frequency ladder, whether the blink survived and how accurately it was
measured.

WHY IT MEASURES THE PIXEL DIRECTLY. The obvious approach - run the SDK's
FrequencyMapAsyncAlgorithm and see when it stops reporting - cannot separate two
very different failures: the sensor no longer producing clean events, and the
algorithm's period matching being mistuned for that frequency. So the primary
measurement here bypasses the SDK entirely and works on raw per-pixel event
timestamps:

    for each LED pixel, the intervals between successive same-polarity events
    ARE the blink period, and their spread is the timing jitter

That is the sensor's own answer, free of any algorithm parameter. Two derived
numbers say which stage is binding:

    events_per_period  ~2.0 means the pixel is tracking every ON and OFF edge.
                       Falling towards 1.0 and below means the pixel is skipping
                       edges - it is out of bandwidth or in refractory, and the
                       SENSOR is the limit.
    period_error       how far the measured period is from the commanded one.
                       Large while events_per_period stays near 2 points at
                       timestamp resolution, not at the pixel.

Usage:
    python analyze_freq_sweep.py sweep.raw
    python analyze_freq_sweep.py sweep.raw --plot sweep.png
"""

import argparse
import sys

import numpy as np

from metavision_core.event_io import EventsIterator

# Must match the ladder in arduino/freq_sweep/freq_sweep.ino.
SWEEP_HZ = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
GAP_MS = 600
DWELL_MS = 3000


def accumulate_activity(path, delta_t=50000):
    """Pass 1: per-pixel event counts, and the event rate over time."""
    it = EventsIterator(input_path=path, delta_t=delta_t)
    height, width = it.get_size()
    flat = np.zeros(height * width, np.int64)
    rate_t, rate_n = [], []
    for evs in it:
        if evs.size:
            # bincount on flattened indices, not np.add.at: the latter takes an
            # unbuffered slow path and a sweep recording runs to 100M+ events.
            idx = evs["y"].astype(np.int64) * width + evs["x"]
            flat += np.bincount(idx, minlength=height * width)
            rate_t.append(evs["t"][-1] / 1e6)
            rate_n.append(evs.size)
    if not rate_t:
        return None, np.array([]), np.array([])
    return flat.reshape(height, width), np.array(rate_t), np.array(rate_n)


def find_blobs(counts, n_blobs=2, box=12):
    """Locate the brightest activity centres, as (x, y, total_events) ROIs."""
    work = counts.copy()
    blobs = []
    for _ in range(n_blobs):
        idx = int(np.argmax(work))
        y, x = divmod(idx, work.shape[1])
        if work[y, x] == 0:
            break
        y0, y1 = max(0, y - box), min(work.shape[0], y + box + 1)
        x0, x1 = max(0, x - box), min(work.shape[1], x + box + 1)
        blobs.append({"x": x, "y": y, "roi": (x0, x1, y0, y1),
                      "events": int(counts[y0:y1, x0:x1].sum())})
        work[y0:y1, x0:x1] = 0
    return blobs


def collect_roi_events(path, rois, delta_t=50000):
    """Pass 2: keep only the events inside the LED ROIs."""
    kept = [[] for _ in rois]
    for evs in EventsIterator(input_path=path, delta_t=delta_t):
        if not evs.size:
            continue
        for i, (x0, x1, y0, y1) in enumerate(rois):
            m = ((evs["x"] >= x0) & (evs["x"] < x1) &
                 (evs["y"] >= y0) & (evs["y"] < y1))
            if m.any():
                kept[i].append(evs[m].copy())
    return [np.concatenate(k) if k else np.empty(0, dtype=[("x", "<u2"), ("y", "<u2"),
                                                           ("p", "<i2"), ("t", "<i8")])
            for k in kept]


def segment_by_gaps(ev, min_gap_ms=300, min_burst_ms=800):
    """Split a burst train into the sweep's steps, using the blackouts between them."""
    if ev.size == 0:
        return []
    t = ev["t"].astype(np.int64)
    # Bin at 10 ms and call a bin quiet if it holds almost nothing.
    bin_us = 10000
    first, last = t[0], t[-1]
    nbins = int((last - first) // bin_us) + 1
    hist = np.bincount(((t - first) // bin_us).astype(np.int64), minlength=nbins)
    active = hist > max(3, 0.05 * np.median(hist[hist > 0]) if (hist > 0).any() else 3)

    segments, start = [], None
    for i, a in enumerate(active):
        if a and start is None:
            start = i
        elif not a and start is not None:
            if (i - start) * bin_us / 1000.0 >= min_burst_ms:
                segments.append((first + start * bin_us, first + i * bin_us))
            start = None
    if start is not None and (len(active) - start) * bin_us / 1000.0 >= min_burst_ms:
        segments.append((first + start * bin_us, first + len(active) * bin_us))
    return segments


def measure_segment(ev, t0, t1, trim_us=200000):
    """Blink statistics for one step, from the raw per-pixel event timestamps."""
    # Trim the edges: the PWM restart and the first events after a blackout are
    # not representative of steady state.
    m = (ev["t"] >= t0 + trim_us) & (ev["t"] < t1 - trim_us)
    sub = ev[m]
    if sub.size < 50:
        return None

    dur_s = (sub["t"][-1] - sub["t"][0]) / 1e6
    pix = sub["x"].astype(np.int64) * 100000 + sub["y"]
    active_px = len(np.unique(pix))

    # Per-pixel ON-to-ON intervals: this is the blink period as the sensor saw it.
    intervals = []
    on = sub[sub["p"] == 1]
    for key in np.unique(on["x"].astype(np.int64) * 100000 + on["y"])[:60]:
        px_t = on["t"][(on["x"].astype(np.int64) * 100000 + on["y"]) == key]
        if px_t.size >= 4:
            d = np.diff(np.sort(px_t.astype(np.int64)))
            intervals.append(d[d > 0])
    if not intervals:
        return None
    iv = np.concatenate(intervals)
    if iv.size < 20:
        return None

    # The blink period is the dominant short interval; long ones are dropouts.
    med = float(np.median(iv))
    core = iv[(iv > 0.5 * med) & (iv < 1.5 * med)]
    period_us = float(np.median(core)) if core.size else med

    return {
        "duration_s": dur_s,
        "events": int(sub.size),
        "active_px": active_px,
        "ev_per_s": sub.size / dur_s if dur_s > 0 else 0.0,
        "period_us": period_us,
        "measured_hz": 1e6 / period_us if period_us > 0 else 0.0,
        "period_jitter_us": float(np.std(core)) if core.size > 1 else float("nan"),
        # Per pixel, per period: 2.0 means every ON and OFF edge is being emitted.
        "ev_per_period": (sub.size / active_px) / (dur_s * 1e6 / period_us)
        if active_px and dur_s > 0 else 0.0,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="RAW/HDF5 recording of the sweep.")
    p.add_argument("--plot", default="", help="Write a summary plot here.")
    p.add_argument("--ladder", default="", help="Comma-separated Hz, if you changed the sketch.")
    args = p.parse_args()

    ladder = [float(v) for v in args.ladder.split(",")] if args.ladder else SWEEP_HZ

    print(f"pass 1: activity map over {args.input}")
    counts, rate_t, rate_n = accumulate_activity(args.input)
    if counts is None:
        print("error: no events in the recording", file=sys.stderr)
        return 1
    total = counts.sum()
    print(f"  {total/1e6:.1f} Mev, {int((counts > 0).sum())} pixels active")

    blobs = find_blobs(counts, n_blobs=2)
    if not blobs:
        print("error: no LED found", file=sys.stderr)
        return 1
    for i, b in enumerate(blobs):
        print(f"  blob {i}: ({b['x']},{b['y']}) {b['events']/1e6:.2f} Mev")

    print("pass 2: collecting LED events")
    roi_ev = collect_roi_events(args.input, [b["roi"] for b in blobs])

    # The swept LED is the one that goes dark between steps; the reference does not.
    seg_sets = [segment_by_gaps(ev) for ev in roi_ev]
    sweep_i = int(np.argmax([len(s) for s in seg_sets]))
    segments = seg_sets[sweep_i]
    ev = roi_ev[sweep_i]
    print(f"  swept LED = blob {sweep_i} at ({blobs[sweep_i]['x']},{blobs[sweep_i]['y']}), "
          f"{len(segments)} bursts found (ladder has {len(ladder)})")

    if len(segments) != len(ladder):
        print(f"  NOTE: burst count does not match the ladder. The recording may not start at a "
              f"step boundary, or high rungs may have produced no detectable blink at all - "
              f"pairing is by order, so read the table with that in mind.")

    print(f"\n{'commanded':>10} {'measured':>10} {'error':>8} {'jitter':>8} {'ev/period':>10} "
          f"{'px':>5} {'ev/s':>10}  verdict")
    print("-" * 88)
    rows = []
    for i, (t0, t1) in enumerate(segments):
        want = ladder[i] if i < len(ladder) else float("nan")
        st = measure_segment(ev, t0, t1)
        if st is None:
            print(f"{want:>9.0f}Hz {'-':>10} {'-':>8} {'-':>8} {'-':>10} {'-':>5} {'-':>10}  "
                  f"NO BLINK RESOLVED")
            rows.append((want, np.nan, np.nan))
            continue
        err_pct = 100.0 * (st["measured_hz"] - want) / want if want else float("nan")
        if st["ev_per_period"] < 0.8:
            verdict = "SENSOR: skipping edges"
        elif abs(err_pct) > 10:
            verdict = "period mismeasured"
        else:
            verdict = "ok"
        print(f"{want:>9.0f}Hz {st['measured_hz']:>9.0f}Hz {err_pct:>+7.1f}% "
              f"{st['period_jitter_us']:>7.2f}us {st['ev_per_period']:>10.2f} "
              f"{st['active_px']:>5d} {st['ev_per_s']:>10.0f}  {verdict}")
        rows.append((want, st["measured_hz"], st["ev_per_period"]))

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        arr = np.array([r for r in rows if not np.isnan(r[1])])
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        if arr.size:
            ax[0].loglog(arr[:, 0], arr[:, 1], "o-", color="#2b6cb0", label="measured")
            lim = [min(arr[:, 0]) * 0.8, max(arr[:, 0]) * 1.2]
            ax[0].loglog(lim, lim, "--", color="#999", label="ideal")
            ax[0].set_xlabel("commanded (Hz)"); ax[0].set_ylabel("measured (Hz)")
            ax[0].legend(); ax[0].grid(alpha=0.3, which="both")
            ax[1].semilogx(arr[:, 0], arr[:, 2], "o-", color="#c05621")
            ax[1].axhline(2.0, ls="--", color="#999")
            ax[1].axhline(0.8, ls=":", color="#c53030")
            ax[1].set_xlabel("commanded (Hz)")
            ax[1].set_ylabel("events per period per pixel")
            ax[1].grid(alpha=0.3, which="both")
        fig.suptitle("LED frequency sweep: where the blink stops resolving")
        plt.tight_layout(); plt.savefig(args.plot, dpi=110)
        print(f"\nwrote {args.plot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
