#!/usr/bin/env python
"""Live event-based vibration monitor for flexible solar arrays.

IAC 2026 -- Event-Based Vibration Monitoring of Flexible Solar Arrays Through
Orbital Sunrise.

This is the only module in :mod:`solar_array` that imports the Metavision SDK.
Every physics and signal-processing stage it drives (``glare_guard``,
``event_vibrometer``, ``modal_analysis``) is deliberately SDK-free and
numpy-only, so the science is unit-testable off-hardware and this file is a thin
real-time shell around it.

It runs one pipeline over an event stream from any of:

  * a live Prophesee camera (EVK4)         -- ``monitor_solar_array.py``
  * a recorded RAW / HDF5 file             -- ``monitor_solar_array.py rec.raw``
  * the orbital-sunrise simulator          -- ``monitor_solar_array.py --synthetic``

The stream is sliced at a fixed cadence and each slice flows through:

    EventRateGovernor (hardware ERC on a live camera, else software decimation)
        -> GlareGuard  (common-mode rejection + solar-disk masking)
        -> VibrometerArray  (per-gauge sub-pixel edge displacement)
        -> ModalIdentifier  (frequencies, damping, mode shapes; run periodically)

with a cv2 overlay (gauges, mask, common-mode index, live modal table) and an
optional JSONL session log.

Environment (IMPORTANT)
-----------------------
Run only with the SDK-compatible interpreter and the native DLL folder on PATH::

    $env:PATH = "C:\\Users\\z003n5uc\\Desktop\\event-based\\bin;" + $env:PATH
    & "C:\\Users\\z003n5uc\\AppData\\Local\\Programs\\Python\\Python39\\python.exe" \\
        monitor_solar_array.py --synthetic --duration 60

A live camera is not needed for ``--synthetic``; that path imports no SDK code.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from glare_guard import GlareGuard, GuardedSlice
from event_vibrometer import VibrometerArray
from modal_analysis import IdentifiedMode, ModalIdentifier

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "solar_array_config.json"


# --------------------------------------------------------------------------- #
# Event stream abstraction
# --------------------------------------------------------------------------- #

@dataclass
class StreamSlice:
    """One fixed-cadence slice of events, source-agnostic."""

    ts_us: int
    dt_us: int
    events: np.ndarray                 # EventCD: fields x, y, p, t
    truth: Optional[Dict[str, object]] = None


class EventStream:
    """Base class: a sliced event source with a known sensor geometry."""

    width: int
    height: int
    is_live: bool = False
    erc_module: object = None
    biases: object = None

    def slices(self) -> Iterator[StreamSlice]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class SyntheticStream(EventStream):
    """Wraps the orbital-sunrise simulator. Imports no SDK code."""

    def __init__(self, cfg: dict, *, scale: int, duration_s: float,
                 slice_dt_us: int, seed: Optional[int] = None) -> None:
        from orbital_sunrise_sim import OrbitalSunriseSimulator

        self._sim = OrbitalSunriseSimulator.from_config(cfg, scale=scale, seed=seed)
        self._sim.slice_dt_us = int(slice_dt_us)
        self.width = self._sim.width
        self.height = self._sim.height
        self.duration_s = float(duration_s)
        self.is_live = False

    def slices(self) -> Iterator[StreamSlice]:
        for sl in self._sim.iterate(duration_s=self.duration_s):
            yield StreamSlice(ts_us=int(sl.ts_us), dt_us=int(sl.dt_us),
                              events=sl.events, truth=sl.truth)


class MetavisionStream(EventStream):
    """Wraps a live Prophesee camera or a recorded RAW/HDF5 file.

    All Metavision imports are confined to this class so the ``--synthetic`` path
    and the numpy-only science modules never touch the SDK.
    """

    def __init__(self, input_path: str, *, slice_dt_us: int,
                 erc_target_evs: Optional[float] = None) -> None:
        from metavision_core.event_io import EventsIterator, is_live_camera

        self._input_path = input_path
        self._slice_dt_us = int(slice_dt_us)
        self.is_live = is_live_camera(input_path)

        # delta_t sets the slice cadence directly at the reader.
        self._iter = EventsIterator(input_path=input_path, delta_t=self._slice_dt_us)
        self.height, self.width = self._iter.get_size()

        # Reach through to the hardware for the event-rate controller and biases.
        reader = getattr(self._iter, "reader", None)
        device = getattr(reader, "device", None) if reader is not None else None
        if device is not None:
            erc = device.get_i_erc_module()
            if erc is not None and erc_target_evs is not None:
                erc.set_cd_event_rate(int(erc_target_evs))
                erc.enable(True)
                self.erc_module = erc
            self.biases = device.get_i_ll_biases()

    def slices(self) -> Iterator[StreamSlice]:
        for evs in self._iter:
            if evs.size == 0:
                continue
            # EventsIterator yields the raw slice; t is absolute in us.
            ts_us = int(evs["t"][0])
            yield StreamSlice(ts_us=ts_us, dt_us=self._slice_dt_us, events=evs)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

class OverlayRenderer:
    """Renders an event slice with the guard/vibrometer/modal overlays.

    Pre-allocates its frame buffer once, following the composable-stage pattern
    in ``sdk_star_stages.py``: uniform ``render()`` signature, no per-call
    allocation on the hot path.
    """

    BG = 32
    ON = (60, 220, 60)
    OFF = (200, 90, 60)

    def __init__(self, width: int, height: int, gauges, cell_size_px: int,
                 *, max_display_w: int = 1280) -> None:
        self.width = int(width)
        self.height = int(height)
        self.gauges = gauges
        self.cell_size_px = int(cell_size_px)
        self._frame = np.empty((self.height, self.width, 3), dtype=np.uint8)
        self._scale = min(1.0, max_display_w / float(max(self.width, 1)))

    def render(self, sl: StreamSlice, guarded: GuardedSlice,
               vib: VibrometerArray, modes: List[IdentifiedMode],
               hud: Dict[str, object]) -> np.ndarray:
        import cv2

        f = self._frame
        f[:] = self.BG

        # Raw events: OFF first so ON paints over it where they coincide.
        ev = sl.events
        if ev.size:
            x = np.clip(ev["x"].astype(np.int64), 0, self.width - 1)
            y = np.clip(ev["y"].astype(np.int64), 0, self.height - 1)
            off = ev["p"] <= 0
            f[y[off], x[off]] = self.OFF
            f[y[~off], x[~off]] = self.ON

        # Masked cells (solar-disk glare guard) as a translucent red wash.
        mask = guarded.mask
        if mask is not None and mask.any():
            cs = self.cell_size_px
            wash = f.copy()
            for r, c in zip(*np.nonzero(mask)):
                y0, x0 = r * cs, c * cs
                wash[y0:y0 + cs, x0:x0 + cs] = (0, 0, 160)
            cv2.addWeighted(wash, 0.35, f, 0.65, 0.0, dst=f)

        # Gauges: green if healthy, amber if starved/dropping out.
        healthy = vib.healthy()
        for i, g in enumerate(self.gauges):
            x0, x1, y0, y1 = g.bounds(self.width, self.height)
            colour = (0, 210, 0) if (i < healthy.size and healthy[i]) else (0, 165, 255)
            cv2.rectangle(f, (x0, y0), (x1 - 1, y1 - 1), colour, 1)

        self._draw_hud(f, cv2, guarded, modes, hud)

        if self._scale < 1.0:
            f = cv2.resize(f, None, fx=self._scale, fy=self._scale,
                           interpolation=cv2.INTER_NEAREST)
        return f

    def _draw_hud(self, f, cv2, guarded: GuardedSlice,
                  modes: List[IdentifiedMode], hud: Dict[str, object]) -> None:
        cm = guarded.common_mode
        lines = [
            "SOLAR-ARRAY VIBRATION MONITOR",
            f"t={float(hud.get('t_s', 0.0)):8.1f}s  profile={guarded.profile:<10}",
            f"rate={float(guarded.stats.get('raw_rate_evs', 0.0)) / 1e6:6.2f} Mev/s"
            f"  decim x{float(guarded.stats.get('decimation_factor', 1.0)):.1f}",
            f"common-mode idx={cm.index:5.2f}  bal={cm.balance:+5.2f}"
            f"  cov={cm.coverage:4.2f}",
            f"gauges healthy {int(hud.get('n_healthy', 0))}/{int(hud.get('n_gauges', 0))}"
            f"  mask {float(guarded.stats.get('masked_cell_fraction', 0.0)) * 100:4.1f}%",
        ]
        y = 18
        for i, txt in enumerate(lines):
            colour = (0, 255, 255) if i == 0 else (235, 235, 235)
            cv2.putText(f, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1,
                        cv2.LINE_AA)
            y += 18

        y += 6
        cv2.putText(f, "IDENTIFIED MODES", (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 255), 1, cv2.LINE_AA)
        y += 18
        if not modes:
            cv2.putText(f, "  (accumulating -- needs warm-up)", (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1,
                        cv2.LINE_AA)
        else:
            cv2.putText(f, "   f[Hz]   zeta%   Q     prom", (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1,
                        cv2.LINE_AA)
            y += 16
            for m in modes[:6]:
                zeta = "  n/a" if np.isnan(m.damping_ratio) else f"{m.damping_ratio * 100:5.2f}"
                q = "  n/a" if (np.isnan(m.damping_ratio) or m.damping_ratio <= 0) \
                    else f"{1.0 / (2.0 * m.damping_ratio):5.0f}"
                txt = f"  {m.freq_hz:6.3f}  {zeta}  {q}  {m.prominence_db:5.1f}dB"
                cv2.putText(f, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                            (120, 255, 120), 1, cv2.LINE_AA)
                y += 16


# --------------------------------------------------------------------------- #
# Monitor
# --------------------------------------------------------------------------- #

class SolarArrayMonitor:
    """Orchestrates the guard -> vibrometer -> modal pipeline over a stream."""

    def __init__(self, cfg: dict, stream: EventStream, *, scale: int,
                 slice_dt_us: int, modal_interval_s: float,
                 enable_bias_scheduling: bool, session_log_fp=None) -> None:
        self.cfg = cfg
        self.stream = stream
        self.slice_dt_s = slice_dt_us * 1e-6
        self.modal_interval_s = float(modal_interval_s)
        self.session_log_fp = session_log_fp

        gg = cfg.get("glare_guard", {})
        # Only hand the scheduler real biases when explicitly enabled: the bias
        # profiles in the config are marked CALIBRATE-PER-UNIT / CITE-REQUIRED and
        # must not be pushed to hardware unvalidated.
        biases = stream.biases if enable_bias_scheduling else None
        self.guard = GlareGuard.from_config(
            cfg, stream.width, stream.height, scale=scale,
            erc_module=stream.erc_module, biases=biases,
        )
        self.vibrometer = VibrometerArray.from_config(
            cfg, stream.width, stream.height,
            slice_dt_s=self.slice_dt_s, scale=scale,
            contrast_sign=float(cfg.get("scene", {}).get("contrast_sign", 1.0)),
        )
        self.identifier = ModalIdentifier.from_config(cfg)
        self.cell_size_px = int(gg.get("cell_size_px", 16))

        self.modes: List[IdentifiedMode] = []
        self._decay_start_s: Optional[float] = None
        self._last_modal_t = -1e18
        self._t0_us: Optional[int] = None

    def _relative_t_s(self, ts_us: int) -> float:
        if self._t0_us is None:
            self._t0_us = ts_us
        return (ts_us - self._t0_us) * 1e-6

    def _maybe_identify(self, t_s: float) -> None:
        warm = self.vibrometer.warmup_s
        if t_s < warm + self.identifier.segment_s:
            return
        if t_s - self._last_modal_t < self.modal_interval_s:
            return
        self._last_modal_t = t_s
        ts, disp = self.vibrometer.series()
        if ts.size < 16:
            return
        self.modes = self.identifier.identify(
            ts, disp, warmup_s=warm, healthy=self.vibrometer.healthy(),
            decay_start_s=self._decay_start_s,
        )

    def _log(self, t_s: float, sl: StreamSlice, guarded: GuardedSlice) -> None:
        if self.session_log_fp is None:
            return
        cm = guarded.common_mode
        rec = {
            "t_s": round(t_s, 4),
            "ts_us": int(sl.ts_us),
            "profile": guarded.profile,
            "stream": {
                "raw_events": int(sl.events.size),
                "kept_events": int(guarded.events.size),
                "raw_rate_evs": guarded.stats.get("raw_rate_evs"),
                "decimation_factor": guarded.stats.get("decimation_factor"),
            },
            "glare_guard": {
                "common_mode_index": round(cm.index, 4),
                "polarity_balance": round(cm.balance, 4),
                "coverage": round(cm.coverage, 4),
                "masked_cell_fraction": guarded.stats.get("masked_cell_fraction"),
            },
            "vibrometer": {
                "n_healthy": int(self.vibrometer.healthy().sum()),
                "n_gauges": len(self.vibrometer.gauges),
                "mean_dropout": round(float(self.vibrometer.dropout_fraction().mean()), 4),
            },
            "modes": [m.to_record() for m in self.modes],
        }
        if sl.truth is not None:
            rec["truth"] = {
                "irradiance_w_m2": sl.truth.get("irradiance_w_m2"),
                "tip_displacement_m": sl.truth.get("tip_displacement_m"),
            }
        self.session_log_fp.write(json.dumps(rec) + "\n")
        self.session_log_fp.flush()

    def run(self, *, renderer: Optional[OverlayRenderer], max_slices: int = 0,
            quiet: bool = False) -> Dict[str, object]:
        import_cv2 = renderer is not None
        cv2 = None
        if import_cv2:
            import cv2  # noqa: F401  (kept local so headless runs need no GUI libs)
            win = "solar-array vibration monitor"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        n = 0
        t_start = time.perf_counter()
        last_report = t_start
        peak_rate = 0.0

        try:
            for sl in self.stream.slices():
                t_s = self._relative_t_s(sl.ts_us)
                guarded = self.guard.process(sl.ts_us, sl.dt_us, sl.events)
                self.vibrometer.update(sl.ts_us, guarded.events, guarded)

                # First strong common-mode transient marks the thermal snap:
                # anchor the free-decay damping fit just after it.
                if (self._decay_start_s is None and t_s > self.vibrometer.warmup_s
                        and guarded.common_mode.index < 0.2
                        and guarded.stats.get("raw_rate_evs", 0.0)
                        < float(self.cfg.get("glare_guard", {}).get("terminator_exit_rate_evs", 2e6))):
                    self._decay_start_s = t_s

                self._maybe_identify(t_s)
                self._log(t_s, sl, guarded)

                peak_rate = max(peak_rate, float(guarded.stats.get("raw_rate_evs", 0.0)))

                if renderer is not None:
                    hud = {
                        "t_s": t_s,
                        "n_healthy": int(self.vibrometer.healthy().sum()),
                        "n_gauges": len(self.vibrometer.gauges),
                    }
                    frame = renderer.render(sl, guarded, self.vibrometer,
                                            self.modes, hud)
                    cv2.imshow(win, frame)
                    if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                        break

                n += 1
                if not quiet and (time.perf_counter() - last_report) > 2.0:
                    last_report = time.perf_counter()
                    print(f"  t={t_s:8.1f}s  slices={n}  "
                          f"rate={float(guarded.stats.get('raw_rate_evs', 0.0)) / 1e6:6.2f} Mev/s"
                          f"  cm_idx={guarded.common_mode.index:4.2f}"
                          f"  modes={len(self.modes)}"
                          f"  healthy={int(self.vibrometer.healthy().sum())}"
                          f"/{len(self.vibrometer.gauges)}")
                if max_slices and n >= max_slices:
                    break
        finally:
            if import_cv2 and cv2 is not None:
                cv2.destroyAllWindows()
            self.stream.close()

        # Final identification over the whole record.
        ts, disp = self.vibrometer.series()
        if ts.size >= 16:
            self.modes = self.identifier.identify(
                ts, disp, warmup_s=self.vibrometer.warmup_s,
                healthy=self.vibrometer.healthy(),
                decay_start_s=self._decay_start_s,
            )

        wall_s = time.perf_counter() - t_start
        return {
            "slices": n,
            "wall_s": round(wall_s, 2),
            "peak_rate_evs": peak_rate,
            "decay_start_s": self._decay_start_s,
            "modes": [m.to_record(self.vibrometer.names) for m in self.modes],
        }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_stream(args, cfg: dict) -> EventStream:
    if args.synthetic:
        return SyntheticStream(
            cfg, scale=args.scale, duration_s=args.duration,
            slice_dt_us=args.slice_dt_us, seed=args.seed,
        )
    erc_target = None if args.no_erc else float(
        cfg.get("glare_guard", {}).get("erc_target_evs", 20e6))
    return MetavisionStream(
        args.input_path, slice_dt_us=args.slice_dt_us, erc_target_evs=erc_target,
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Live event-based vibration monitor for flexible solar arrays.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("input_path", nargs="?", default="",
                    help="RAW/HDF5 file, or empty for a live camera")
    ap.add_argument("--synthetic", action="store_true",
                    help="drive the orbital-sunrise simulator instead of a camera")
    ap.add_argument("--config", dest="config_path", default=str(DEFAULT_CONFIG),
                    help="path to solar_array_config.json")
    ap.add_argument("--scale", type=int, default=1,
                    help="synthetic-only sensor downscale (1 = full 1280x720)")
    ap.add_argument("--duration", dest="duration", type=float, default=1400.0,
                    help="synthetic-only run length in seconds")
    ap.add_argument("--slice-dt-us", dest="slice_dt_us", type=int, default=20000,
                    help="event slice cadence in microseconds")
    ap.add_argument("--modal-interval", dest="modal_interval_s", type=float,
                    default=30.0, help="seconds between modal re-identifications")
    ap.add_argument("--no-erc", action="store_true",
                    help="do not enable the hardware event-rate controller")
    ap.add_argument("--enable-bias-scheduling", action="store_true",
                    help="push per-profile biases to hardware (CALIBRATE PER UNIT first)")
    ap.add_argument("--no-display", dest="no_display", action="store_true",
                    help="headless: no cv2 window")
    ap.add_argument("--max-slices", dest="max_slices", type=int, default=0,
                    help="stop after this many slices (0 = unlimited; smoke test)")
    ap.add_argument("--session-log", dest="session_log_path", default="",
                    help="write a JSONL trace to this path")
    ap.add_argument("--seed", type=int, default=None, help="synthetic RNG seed")
    ap.add_argument("--quiet", action="store_true", help="suppress periodic console lines")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config_path).read_text(encoding="utf-8"))

    print("IAC 2026 - live solar-array vibration monitor")
    print(f"config : {args.config_path}")
    if args.synthetic:
        print(f"source : synthetic  scale 1/{args.scale}  duration {args.duration:.0f}s")
    else:
        print(f"source : {args.input_path or 'LIVE CAMERA'}")

    stream = build_stream(args, cfg)
    print(f"sensor : {stream.width}x{stream.height}"
          f"  {'LIVE' if stream.is_live else 'replay'}"
          f"  erc={'on' if stream.erc_module else 'off'}"
          f"  biases={'wired' if stream.biases else 'none'}")

    session_log_fp = None
    if args.session_log_path:
        session_log_fp = open(args.session_log_path, "w", encoding="utf-8")
        print(f"log    : {args.session_log_path}")

    monitor = SolarArrayMonitor(
        cfg, stream, scale=args.scale, slice_dt_us=args.slice_dt_us,
        modal_interval_s=args.modal_interval_s,
        enable_bias_scheduling=args.enable_bias_scheduling,
        session_log_fp=session_log_fp,
    )

    renderer = None
    if not args.no_display:
        renderer = OverlayRenderer(stream.width, stream.height,
                                   monitor.vibrometer.gauges, monitor.cell_size_px)

    print("running - press Q or Esc in the window to stop\n")
    try:
        summary = monitor.run(renderer=renderer, max_slices=args.max_slices,
                              quiet=args.quiet)
    finally:
        if session_log_fp is not None:
            session_log_fp.close()

    print("\n=== summary ===")
    print(f"  slices        : {summary['slices']}  in {summary['wall_s']}s wall")
    print(f"  peak raw rate : {summary['peak_rate_evs'] / 1e6:.1f} Mev/s")
    print(f"  thermal snap  : "
          + ("t=%.1fs" % summary["decay_start_s"] if summary["decay_start_s"]
             is not None else "not detected"))
    print(f"  modes found   : {len(summary['modes'])}")
    for m in summary["modes"]:
        zeta = m["damping_ratio"]
        zeta_txt = "n/a" if zeta is None else f"{zeta * 100:.2f}%"
        print(f"    f = {m['freq_hz']:7.3f} Hz  zeta = {zeta_txt:>7}  "
              f"prom = {m['prominence_db']:.1f} dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
