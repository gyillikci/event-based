"""Sub-pixel displacement gauges driven by signed event integration.

For a logarithmic pixel the event equation is

    d(log I) = -grad(log I) . dx

so the *signed* event count accumulated over a region of interest is proportional
to the displacement of the contrast pattern crossing it. Integrating the signed
count therefore yields displacement directly, with no intensity image and no
feature tracking. Each region of interest behaves as a virtual strain gauge --
an optical accelerometer bolted to a fixed spot in the image while the structure
moves through it.

Two properties make this work through an orbital sunrise:

* The estimator is **differential**. A globally correlated illumination transient
  adds the same signed count to every pixel, so it appears as a bias proportional
  to the gauge area, which :mod:`glare_guard` estimates and removes.
* The signed sum **self-cancels symmetric texture**. Periodic solar-cell striping
  inside a gauge contributes matched ON and OFF populations that annihilate, while
  the asymmetric panel boundary contributes in full. The gauge is therefore
  selective for the structural edge it is placed on.

Sign convention
---------------
Gauge signs are fixed geometrically, never inferred from the data. Torsion and
higher bending modes put parts of the structure genuinely in anti-phase, so
resolving signs by correlating against an array consensus would erase exactly the
mode-shape information the analysis is trying to recover. With a panel brighter
than its background, the top edge moving down converts panel to background (OFF)
and the bottom edge moving down converts background to panel (ON); hence
``sign = contrast_sign * (-1 top, +1 bottom)``.

Absolute scale
--------------
The gain in events per pixel of displacement depends on the local log-contrast and
the pixel threshold. Modal *frequencies*, *damping ratios* and *normalised* mode
shapes are all invariant to it; only absolute amplitude is not. The gain therefore
defaults to a physics-based estimate and can be refined with :meth:`calibrate_gain`
against a reference instrument.

NumPy only -- no SDK, no scipy. Self-test::

    python event_vibrometer.py
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Gauge definition
# --------------------------------------------------------------------------- #

@dataclass
class GaugeSpec:
    """A fixed rectangular region of interest straddling a structural edge."""

    name: str
    x_px: int
    y_px: int
    half_width_px: int
    half_height_px: int
    xi: float                  # normalised span coordinate, for mode shapes
    edge: str                  # "top" or "bottom"
    sign: float = 1.0

    def bounds(self, width: int, height: int) -> Tuple[int, int, int, int]:
        x0 = int(max(0, self.x_px - self.half_width_px))
        x1 = int(min(width, self.x_px + self.half_width_px + 1))
        y0 = int(max(0, self.y_px - self.half_height_px))
        y1 = int(min(height, self.y_px + self.half_height_px + 1))
        return x0, x1, y0, y1

    def area_px(self, width: int, height: int) -> int:
        x0, x1, y0, y1 = self.bounds(width, height)
        return max(0, (x1 - x0)) * max(0, (y1 - y0))


def place_gauges(
    xi_values: Sequence[float],
    *,
    root_x_px: float,
    tip_x_px: float,
    centre_y_px: float,
    chord_px: float,
    half_width_px: int = 40,
    half_height_px: int = 18,
    edges: Sequence[str] = ("top", "bottom"),
    contrast_sign: float = 1.0,
) -> List[GaugeSpec]:
    """Place gauges on the undeformed panel boundary at the given span stations.

    ``contrast_sign`` is +1 when the panel is brighter than its background and -1
    otherwise. It is a static property of the scene, established once at
    commissioning, not a per-run free parameter.
    """
    span = tip_x_px - root_x_px
    gauges: List[GaugeSpec] = []
    for xi in xi_values:
        x = root_x_px + xi * span
        for edge in edges:
            if edge not in ("top", "bottom"):
                raise ValueError(f"unknown edge: {edge!r}")
            offset = -0.5 * chord_px if edge == "top" else 0.5 * chord_px
            sign = contrast_sign * (-1.0 if edge == "top" else 1.0)
            gauges.append(GaugeSpec(
                name=f"{edge[0].upper()}{xi:.2f}",
                x_px=int(round(x)),
                y_px=int(round(centre_y_px + offset)),
                half_width_px=int(half_width_px),
                half_height_px=int(half_height_px),
                xi=float(xi),
                edge=edge,
                sign=sign,
            ))
    return gauges


def estimate_gain_events_per_px(
    roi_width_px: int,
    log_contrast: float,
    contrast_threshold: float,
) -> float:
    """Physics-based gain: signed events produced per pixel of edge displacement.

    A step edge of log-contrast ``log_contrast`` sweeping one pixel across a region
    ``roi_width_px`` wide reassigns that many pixels, each emitting
    ``log_contrast / contrast_threshold`` events.
    """
    return max(roi_width_px * log_contrast / max(contrast_threshold, 1e-9), 1e-9)


# --------------------------------------------------------------------------- #
# Drift control
# --------------------------------------------------------------------------- #

class HighPassFilter:
    """Cascaded first-order high-pass, applied per sample to remove integration drift.

    Residual common-mode bias, ON/OFF threshold asymmetry and leak do not integrate
    into a constant offset -- they integrate into a linear *ramp*. A single pole
    cannot reject a ramp: its steady-state output converges to ``slope * RC``, which
    for a 0.01 Hz corner is 16 seconds' worth of drift. Two cascaded poles reject a
    ramp completely, because the first stage turns it into a step and the second
    stage removes the step.

    The cost in the modal band is small: at 0.082 Hz a 0.01 Hz second-order corner
    costs 1.5% in amplitude and about 14 degrees in phase.
    """

    def __init__(self, corner_hz: float, dt_s: float, order: int = 2) -> None:
        self.corner_hz = float(corner_hz)
        self.dt_s = float(dt_s)
        self.order = max(1, int(order))
        if corner_hz <= 0.0:
            self.alpha = 1.0
        else:
            rc = 1.0 / (2.0 * math.pi * corner_hz)
            self.alpha = rc / (rc + self.dt_s)
        self._y = [0.0] * self.order
        self._x = [0.0] * self.order
        self._primed = False

    def __call__(self, x: float) -> float:
        if not self._primed:
            self._x = [x] * self.order
            self._y = [0.0] * self.order
            self._primed = True
            return 0.0
        value = x
        for k in range(self.order):
            self._y[k] = self.alpha * (self._y[k] + value - self._x[k])
            self._x[k] = value
            value = self._y[k]
        return value

    def reset(self) -> None:
        self._y = [0.0] * self.order
        self._x = [0.0] * self.order
        self._primed = False

    @property
    def warmup_s(self) -> float:
        """Settling time before the output is trustworthy.

        A 0.01 Hz corner has a 15.9 s time constant, and each cascaded pole adds
        its own transient. Modal estimates must discard this leading interval --
        which is affordable here only because the measurement continues for many
        minutes after the terminator crossing.
        """
        if self.corner_hz <= 0.0:
            return 0.0
        rc = 1.0 / (2.0 * math.pi * self.corner_hz)
        return 5.0 * rc * self.order


# --------------------------------------------------------------------------- #
# Gauge array
# --------------------------------------------------------------------------- #

@dataclass
class GaugeSample:
    """One gauge reading for one slice."""

    increment_px: float
    displacement_px: float
    signed_events: float
    common_mode_events: float
    activity: int
    dropout: bool


class VibrometerArray:
    """Runs a set of :class:`GaugeSpec` regions over guarded event slices."""

    def __init__(
        self,
        gauges: Sequence[GaugeSpec],
        width: int,
        height: int,
        *,
        slice_dt_s: float,
        gain_events_per_px: float = 1.0,
        highpass_hz: float = 0.01,
        highpass_order: int = 2,
        min_mean_activity: float = 0.5,
        gate_on_common_mode: bool = True,
    ) -> None:
        if not gauges:
            raise ValueError("at least one gauge is required")
        self.gauges = list(gauges)
        self.width = int(width)
        self.height = int(height)
        self.slice_dt_s = float(slice_dt_s)
        self.gain_events_per_px = float(gain_events_per_px)
        self.min_mean_activity = float(min_mean_activity)
        self.gate_on_common_mode = bool(gate_on_common_mode)

        self._areas = np.array([g.area_px(width, height) for g in self.gauges],
                               dtype=np.float64)
        self._filters = [HighPassFilter(highpass_hz, slice_dt_s, highpass_order)
                         for _ in self.gauges]
        self._raw = np.zeros(len(self.gauges), dtype=np.float64)

        self.t_s: List[float] = []
        self.displacement: List[np.ndarray] = []
        self.dropouts: List[np.ndarray] = []
        self._n_slices = 0
        self._n_dropped = np.zeros(len(self.gauges), dtype=np.int64)
        self._activity_total = np.zeros(len(self.gauges), dtype=np.int64)

    # ---- per-slice update -------------------------------------------------- #

    def _signed_image(self, events: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Signed and absolute per-pixel event count images for one slice.

        Built with a single ``bincount`` pass so cost is O(n_events) rather than
        O(n_gauges * n_events).
        """
        n = self.width * self.height
        if events.size == 0:
            zeros = np.zeros((self.height, self.width), dtype=np.float64)
            return zeros, zeros.astype(np.int64)
        flat = events["y"].astype(np.int64) * self.width + events["x"].astype(np.int64)
        signed = np.bincount(flat, weights=events["p"].astype(np.float64), minlength=n)
        total = np.bincount(flat, minlength=n)
        return signed.reshape(self.height, self.width), total.reshape(self.height, self.width)

    def update(self, ts_us: int, events: np.ndarray, guarded=None) -> List[GaugeSample]:
        """Advance every gauge by one slice.

        ``guarded`` is an optional :class:`~glare_guard.GuardedSlice`; when given,
        its common-mode field is subtracted and its mask and contamination flag
        gate the gauges.
        """
        signed_img, total_img = self._signed_image(events)
        contaminated = bool(
            self.gate_on_common_mode and guarded is not None
            and guarded.common_mode.contaminated
        )

        samples: List[GaugeSample] = []
        disp = np.zeros(len(self.gauges), dtype=np.float64)
        drop = np.zeros(len(self.gauges), dtype=bool)

        for i, gauge in enumerate(self.gauges):
            x0, x1, y0, y1 = gauge.bounds(self.width, self.height)
            signed = float(signed_img[y0:y1, x0:x1].sum())
            activity = int(total_img[y0:y1, x0:x1].sum())

            common = 0.0
            masked = False
            if guarded is not None:
                cell = guarded.counts.cell_size_px
                common = guarded.common_mode.at(gauge.x_px, gauge.y_px, cell) * self._areas[i]
                r = int(np.clip(gauge.y_px // cell, 0, guarded.mask.shape[0] - 1))
                c = int(np.clip(gauge.x_px // cell, 0, guarded.mask.shape[1] - 1))
                masked = bool(guarded.mask[r, c])

            corrected = signed - common
            # A slice may legitimately carry no events: at 0.08 Hz the per-slice
            # travel is a small fraction of a pixel, so a zero count means zero
            # motion, not a failed gauge. Only invalid readings are dropped, and
            # starved gauges are caught by the record-level health metric instead.
            dropout = contaminated or masked
            increment = 0.0 if dropout else gauge.sign * corrected / self.gain_events_per_px

            self._raw[i] += increment
            disp[i] = self._filters[i](self._raw[i])
            drop[i] = dropout
            self._activity_total[i] += activity
            if dropout:
                self._n_dropped[i] += 1

            samples.append(GaugeSample(
                increment_px=increment,
                displacement_px=disp[i],
                signed_events=signed,
                common_mode_events=common,
                activity=activity,
                dropout=dropout,
            ))

        self.t_s.append(ts_us * 1e-6)
        self.displacement.append(disp.copy())
        self.dropouts.append(drop)
        self._n_slices += 1
        return samples

    # ---- results ----------------------------------------------------------- #

    @property
    def names(self) -> List[str]:
        return [g.name for g in self.gauges]

    @property
    def xi(self) -> np.ndarray:
        return np.array([g.xi for g in self.gauges], dtype=np.float64)

    @property
    def warmup_s(self) -> float:
        """Interval that modal analysis should discard while the drift filter settles."""
        return self._filters[0].warmup_s if self._filters else 0.0

    def series(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(t_s, displacement_px)`` with displacement shaped (n_gauges, n_samples)."""
        if not self.displacement:
            return np.zeros(0), np.zeros((len(self.gauges), 0))
        return np.array(self.t_s), np.stack(self.displacement, axis=1)

    def dropout_fraction(self) -> np.ndarray:
        if self._n_slices == 0:
            return np.zeros(len(self.gauges))
        return self._n_dropped / float(self._n_slices)

    def mean_activity(self) -> np.ndarray:
        """Mean events per slice per gauge, averaged over the whole record."""
        if self._n_slices == 0:
            return np.zeros(len(self.gauges))
        return self._activity_total / float(self._n_slices)

    def healthy(self, max_dropout_fraction: float = 0.35) -> np.ndarray:
        """A gauge is healthy if it is both sufficiently valid and sufficiently fed."""
        return ((self.dropout_fraction() <= max_dropout_fraction)
                & (self.mean_activity() >= self.min_mean_activity))

    def calibrate_gain(self, reference_px: np.ndarray,
                       gauge_index: int = 0) -> float:
        """Least-squares rescale of the gain against a reference displacement.

        Returns the new gain. Frequencies and damping are unaffected; this only
        fixes absolute amplitude.
        """
        _, disp = self.series()
        if disp.shape[1] != reference_px.size:
            raise ValueError("reference length does not match the recorded series")
        measured = disp[gauge_index]
        denom = float(np.dot(measured, measured))
        if denom <= 0.0:
            return self.gain_events_per_px
        scale = float(np.dot(measured, reference_px)) / denom
        if abs(scale) < 1e-12:
            return self.gain_events_per_px
        self.gain_events_per_px /= scale
        for i in range(len(self.gauges)):
            self.displacement = [d * scale for d in self.displacement]
            break
        return self.gain_events_per_px

    @classmethod
    def from_config(cls, cfg: dict, width: int, height: int, *, slice_dt_s: float,
                    scale: int = 1, contrast_sign: float = 1.0) -> "VibrometerArray":
        v = cfg.get("vibrometer", {})
        scene = cfg.get("scene", {})
        s = max(int(scale), 1)
        half_w = max(1, int(v.get("gauge_half_width_px", 40)) // s)
        gauges = place_gauges(
            v.get("gauge_xi", [0.18, 0.34, 0.50, 0.66, 0.82, 0.96]),
            root_x_px=scene.get("root_x_px", 90) / s,
            tip_x_px=scene.get("tip_x_px", 1210) / s,
            centre_y_px=scene.get("centre_y_px", 300) / s,
            chord_px=scene.get("chord_px", 150) / s,
            half_width_px=half_w,
            half_height_px=max(1, int(v.get("gauge_half_height_px", 18)) // s),
            edges=v.get("edges", ["top", "bottom"]),
            contrast_sign=contrast_sign,
        )
        log_contrast = math.log(
            max(scene.get("panel_reflectance", 0.075), 1e-9)
            / max(scene.get("background_reflectance", 0.004), 1e-9)
        )
        gain = estimate_gain_events_per_px(
            roi_width_px=2 * half_w + 1,
            log_contrast=log_contrast,
            contrast_threshold=cfg.get("sensor", {}).get("contrast_threshold_on", 0.2),
        )
        return cls(gauges, width, height, slice_dt_s=slice_dt_s,
                   gain_events_per_px=gain,
                   highpass_hz=v.get("highpass_hz", 0.01),
                   highpass_order=int(v.get("highpass_order", 2)),
                   min_mean_activity=float(v.get("min_mean_activity_events", 0.5)))


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #

def _self_test() -> int:
    ok = True

    def check(label: str, condition: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= bool(condition)
        print(f"  [{'PASS' if condition else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))

    print("place_gauges / GaugeSpec")
    gauges = place_gauges([0.25, 0.75], root_x_px=90, tip_x_px=1210,
                          centre_y_px=300, chord_px=150)
    check("two stations x two edges", len(gauges) == 4, f"{[g.name for g in gauges]}")
    top = [g for g in gauges if g.edge == "top"]
    bot = [g for g in gauges if g.edge == "bottom"]
    check("top edge sits above centre", all(g.y_px < 300 for g in top))
    check("bottom edge sits below centre", all(g.y_px > 300 for g in bot))
    check("edges carry opposite sign",
          all(g.sign < 0 for g in top) and all(g.sign > 0 for g in bot))

    print("HighPassFilter")
    dt = 1.0 / 200.0
    t = np.arange(0, 600.0, dt)
    tone = np.sin(2 * math.pi * 0.082 * t)
    drifting = tone + 3.0 * t                      # 3 px/s of integration drift

    hp1, hp2 = HighPassFilter(0.01, dt, order=1), HighPassFilter(0.01, dt, order=2)
    first = np.array([hp1(v) for v in drifting])
    second = np.array([hp2(v) for v in drifting])
    tail = t > hp2.warmup_s
    print(f"    warm-up {hp2.warmup_s:.0f} s; residual offset after a 3 px/s ramp: "
          f"1st order {float(np.mean(first[tail])):+.2f} px | "
          f"2nd order {float(np.mean(second[tail])):+.4f} px")
    check("first order cannot reject a ramp", abs(float(np.mean(first[tail]))) > 10.0)
    check("second order rejects the ramp", abs(float(np.mean(second[tail]))) < 0.05)
    amp = float(np.percentile(np.abs(second[tail]), 95))
    check("modal amplitude preserved", 0.9 < amp < 1.1,
          f"{amp:.3f} of unit input at 0.082 Hz")

    print("VibrometerArray - synthetic edge motion")
    w, h, cell = 320, 180, 16
    dt_s = 1.0 / 200.0
    spec = [GaugeSpec("G0", x_px=160, y_px=90, half_width_px=10,
                      half_height_px=6, xi=0.5, edge="bottom", sign=1.0)]
    array = VibrometerArray(spec, w, h, slice_dt_s=dt_s, gain_events_per_px=100.0,
                            highpass_hz=0.01)

    dtype = np.dtype([("x", "<u2"), ("y", "<u2"), ("p", "<i2"), ("t", "<i8")])
    rng = np.random.default_rng(11)
    freq, amp_px = 0.5, 2.0
    n = 4000
    truth, meas = [], []
    prev = 0.0
    for k in range(n):
        t_s = k * dt_s
        pos = amp_px * math.sin(2 * math.pi * freq * t_s)
        delta = pos - prev
        prev = pos
        # A moving edge emits 100 signed events per pixel of travel.
        count = int(round(abs(delta) * 100.0))
        ev = np.empty(count, dtype=dtype)
        if count:
            ev["x"] = rng.integers(150, 171, count)
            ev["y"] = rng.integers(84, 97, count)
            ev["p"] = 1 if delta > 0 else -1
            ev["t"] = rng.integers(0, int(dt_s * 1e6), count)
        s = array.update(int(t_s * 1e6), ev)
        truth.append(pos)
        meas.append(s[0].displacement_px)

    truth = np.array(truth)
    meas = np.array(meas)
    warm = n // 4
    err = float(np.sqrt(np.mean((meas[warm:] - truth[warm:]) ** 2)))
    corr = float(np.corrcoef(meas[warm:], truth[warm:])[0, 1])
    print(f"    tracked {amp_px:.1f} px at {freq} Hz from a mean of "
          f"{float(array.mean_activity()[0]):.1f} events/slice: "
          f"RMSE {err:.4f} px, r = {corr:.5f}")
    check("displacement tracked", err < 0.15 and corr > 0.999)
    check("low per-slice counts are not treated as dropouts",
          float(array.dropout_fraction()[0]) == 0.0)
    check("gauge reported healthy", bool(array.healthy()[0]))

    print("Symmetric texture rejection")
    # A symmetric ON/OFF pair inside the gauge must cancel; an asymmetric step
    # must not. This is what makes the gauge selective for the panel boundary.
    arr2 = VibrometerArray(spec, w, h, slice_dt_s=dt_s, gain_events_per_px=100.0,
                           highpass_hz=0.0)
    sym = np.empty(400, dtype=dtype)
    sym["x"] = rng.integers(150, 171, 400)
    sym["y"] = rng.integers(84, 97, 400)
    sym["p"] = np.where(np.arange(400) % 2 == 0, 1, -1)
    sym["t"] = 0
    s_sym = arr2.update(0, sym)[0]
    check("symmetric texture cancels", abs(s_sym.signed_events) < 1e-9,
          f"signed sum {s_sym.signed_events:+.1f} from 400 events")

    print("Common-mode bias removal")
    try:
        from .glare_guard import GlareGuard
    except ImportError:
        from glare_guard import GlareGuard  # type: ignore[no-redef]

    guard = GlareGuard(w, h, cell_size_px=cell, erc_enabled=False,
                       mask_enabled=False)
    arr3 = VibrometerArray(spec, w, h, slice_dt_s=dt_s, gain_events_per_px=100.0,
                           highpass_hz=0.0, gate_on_common_mode=False)
    # Uniform ON burst of exactly 3 events per pixel: pure common mode.
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    flat_n = w * h * 3
    burst = np.empty(flat_n, dtype=dtype)
    burst["x"] = np.tile(xs.ravel(), 3).astype(np.uint16)
    burst["y"] = np.tile(ys.ravel(), 3).astype(np.uint16)
    burst["p"] = 1
    burst["t"] = 0
    guarded = guard.process(0, int(dt_s * 1e6), burst)
    s_cm = arr3.update(0, guarded.events, guarded)[0]
    area = spec[0].area_px(w, h)
    print(f"    gauge area {area} px, raw signed {s_cm.signed_events:.0f}, "
          f"common-mode estimate {s_cm.common_mode_events:.0f}")
    check("common mode fully removed",
          abs(s_cm.signed_events - s_cm.common_mode_events) < 1e-6,
          f"residual {s_cm.signed_events - s_cm.common_mode_events:+.3f} events")
    check("no displacement from a pure ramp", abs(s_cm.increment_px) < 1e-6)

    print("\nSELF-TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
