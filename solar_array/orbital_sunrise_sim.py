"""Synthetic orbital-sunrise event generator with ground truth.

Renders a cantilevered flexible solar array vibrating in front of a rising sun and
converts the scene into an event stream using the standard logarithmic threshold
model. Because the modal content is prescribed, the output carries exact ground
truth for scoring the estimator — no camera and no flight data required.

What is modelled
----------------
* Sub-pixel panel edge motion from :class:`~space_environment.FlexibleArrayModel`,
  including torsion (top and bottom edges move in anti-phase).
* Solar-cell interconnect striping, so the panel interior also carries contrast.
* A four-to-six decade global irradiance ramp from
  :class:`~space_environment.OrbitalSunriseProfile`.
* The solar disk itself rising into frame with a Gaussian veiling-glare halo.
* The bright Earth limb entering from below.
* Per-pixel contrast-threshold mismatch, refractory period, leak and shot noise.
* Two distinct loss mechanisms, which is the point of the experiment:
    - ``erc_target_evs``  : graceful, spatially uniform pseudo-random decimation,
      as performed by the on-sensor Event Rate Controller.
    - ``readout_limit_evs``: ungoverned FIFO overflow, which drops *time-contiguous
      blocks* and therefore destroys phase rather than merely adding variance.

Timestamps within a slice are distributed uniformly across each pixel's event
count. This is exact for a linear log-intensity ramp within the slice and is a
good approximation at the default 5 ms slice, but it does mean the stream should
not be used to study sub-slice timing statistics.

NumPy only — no SDK, no scipy. Self-test::

    python orbital_sunrise_sim.py
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

try:
    from .space_environment import (
        FlexibleArrayModel,
        OrbitalSunriseProfile,
        ThermalSnap,
        VibrationMode,
    )
except ImportError:  # running as a script from inside the package directory
    from space_environment import (  # type: ignore[no-redef]
        FlexibleArrayModel,
        OrbitalSunriseProfile,
        ThermalSnap,
        VibrationMode,
    )

#: Binary layout of a Metavision ``EventCD`` record, replicated so this module
#: stays importable without the SDK.
EVENT_CD_DTYPE = np.dtype([("x", "<u2"), ("y", "<u2"), ("p", "<i2"), ("t", "<i8")])

#: Radiance floor (W/m^2/sr) preventing log(0) — represents sensor dark current.
DARK_RADIANCE_FLOOR = 1.0e-7


# --------------------------------------------------------------------------- #
# Scene description
# --------------------------------------------------------------------------- #

@dataclass
class SolarDiskParams:
    enabled: bool = True
    centre_x_px: float = 980.0
    final_y_px: float = 620.0
    travel_y_px: float = 190.0
    radius_px: float = 26.0
    halo_sigma_px: float = 95.0
    peak_radiance_gain: float = 4.0e4


@dataclass
class EarthLimbParams:
    enabled: bool = True
    horizon_y_px: float = 688.0
    softness_px: float = 34.0
    radiance_gain: float = 55.0


@dataclass
class SceneGeometry:
    """Projection of the array into the image plane."""

    root_x_px: float = 90.0
    tip_x_px: float = 1210.0
    centre_y_px: float = 300.0
    chord_px: float = 150.0
    metres_per_pixel: float = 0.0304
    panel_reflectance: float = 0.075
    background_reflectance: float = 0.004
    cell_stripe_contrast: float = 0.55
    n_cell_stripes: int = 26
    n_cell_rows: int = 5
    solar_disk: SolarDiskParams = field(default_factory=SolarDiskParams)
    earth_limb: EarthLimbParams = field(default_factory=EarthLimbParams)

    def scaled(self, scale: int) -> "SceneGeometry":
        """Return a copy with all pixel dimensions divided by ``scale``."""
        if scale == 1:
            return self
        s = float(scale)
        disk = SolarDiskParams(
            enabled=self.solar_disk.enabled,
            centre_x_px=self.solar_disk.centre_x_px / s,
            final_y_px=self.solar_disk.final_y_px / s,
            travel_y_px=self.solar_disk.travel_y_px / s,
            radius_px=self.solar_disk.radius_px / s,
            halo_sigma_px=self.solar_disk.halo_sigma_px / s,
            peak_radiance_gain=self.solar_disk.peak_radiance_gain,
        )
        limb = EarthLimbParams(
            enabled=self.earth_limb.enabled,
            horizon_y_px=self.earth_limb.horizon_y_px / s,
            softness_px=self.earth_limb.softness_px / s,
            radiance_gain=self.earth_limb.radiance_gain,
        )
        return SceneGeometry(
            root_x_px=self.root_x_px / s,
            tip_x_px=self.tip_x_px / s,
            centre_y_px=self.centre_y_px / s,
            chord_px=self.chord_px / s,
            metres_per_pixel=self.metres_per_pixel * s,
            panel_reflectance=self.panel_reflectance,
            background_reflectance=self.background_reflectance,
            cell_stripe_contrast=self.cell_stripe_contrast,
            n_cell_stripes=self.n_cell_stripes,
            n_cell_rows=self.n_cell_rows,
            solar_disk=disk,
            earth_limb=limb,
        )


@dataclass
class SensorParams:
    """DVS pixel model. Threshold values are in natural-log contrast units."""

    contrast_threshold_on: float = 0.20
    contrast_threshold_off: float = 0.20
    threshold_mismatch_std: float = 0.15
    refractory_period_us: float = 100.0
    leak_rate_hz: float = 0.01
    shot_noise_rate_hz: float = 0.05
    erc_target_evs: Optional[float] = None
    readout_limit_evs: Optional[float] = None


@dataclass
class SimSlice:
    """One time slice of simulator output."""

    ts_us: int
    dt_us: int
    events: np.ndarray            # EVENT_CD_DTYPE, sorted by t
    truth: Dict[str, object]
    stats: Dict[str, float]


# --------------------------------------------------------------------------- #
# Simulator
# --------------------------------------------------------------------------- #

class OrbitalSunriseSimulator:
    """Generates an event stream of a vibrating array through a terminator crossing."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        *,
        scale: int = 1,
        scene: Optional[SceneGeometry] = None,
        sensor: Optional[SensorParams] = None,
        sun: Optional[OrbitalSunriseProfile] = None,
        array: Optional[FlexibleArrayModel] = None,
        slice_dt_us: int = 5000,
        seed: int = 20260204,
    ) -> None:
        if scale < 1:
            raise ValueError("scale must be >= 1")
        self.scale = scale
        self.width = int(width // scale)
        self.height = int(height // scale)
        self.slice_dt_us = int(slice_dt_us)
        self.scene = (scene or SceneGeometry()).scaled(scale)
        self.sensor = sensor or SensorParams()
        self.sun = sun or OrbitalSunriseProfile()
        self.array = array or FlexibleArrayModel()
        self.rng = np.random.default_rng(seed)

        self._build_static_fields()
        self.reset()

    # ---- static precomputation -------------------------------------------- #

    def _build_static_fields(self) -> None:
        sc = self.scene
        w, h = self.width, self.height

        self._xs = np.arange(w, dtype=np.float64)
        self._ys = np.arange(h, dtype=np.float64)[:, None]

        span_px = max(sc.tip_x_px - sc.root_x_px, 1e-9)
        self._xi = (self._xs - sc.root_x_px) / span_px
        self._in_span = (self._xi >= 0.0) & (self._xi <= 1.0)
        xi_c = np.clip(self._xi, 0.0, 1.0)

        # Mode shapes depend only on xi, so evaluate them once.
        self._shape_bending: List[np.ndarray] = []
        self._shape_torsion: List[np.ndarray] = []
        for mode in self.array.modes:
            phi = mode.shape(xi_c)
            if mode.kind == "torsion":
                self._shape_bending.append(np.zeros_like(phi))
                self._shape_torsion.append(phi)
            else:
                self._shape_bending.append(phi)
                self._shape_torsion.append(np.zeros_like(phi))

        # Solar-cell interconnect grid. The spanwise stripes are static under
        # vertical panel motion, so it is the *chordwise* cell rows that make the
        # panel interior sensitive to bending -- without them the only vibration
        # signal comes from the two outer edges and it sits below the leak floor.
        stripe_span = np.tanh(4.0 * np.sin(2.0 * math.pi * sc.n_cell_stripes * xi_c))
        self._stripe_span = stripe_span

        # Radial distance to the solar disk centre (x component is static).
        self._dx_sun = self._xs - sc.solar_disk.centre_x_px

        # Per-pixel threshold mismatch, fixed for the life of the sensor.
        mism = self.sensor.threshold_mismatch_std
        self._c_on = np.clip(
            self.sensor.contrast_threshold_on
            * (1.0 + mism * self.rng.standard_normal((h, w))),
            0.02, None,
        )
        self._c_off = np.clip(
            self.sensor.contrast_threshold_off
            * (1.0 + mism * self.rng.standard_normal((h, w))),
            0.02, None,
        )

        self._max_events_per_slice_per_px = max(
            1, int(self.slice_dt_us / max(self.sensor.refractory_period_us, 1e-6))
        )

    def reset(self) -> None:
        """Reset pixel reference levels to the eclipse-adapted state at t = 0."""
        self._t_s = 0.0
        self._ref = self.log_radiance(0.0)
        self._blackout_until_us = -1
        self._last_bg = None
        self._noise_cdf = None

    # ---- scene rendering -------------------------------------------------- #

    def edge_positions_px(self, t_s: float) -> Tuple[np.ndarray, np.ndarray]:
        """Image-row position of the panel's top and bottom edges at time ``t_s``."""
        sc = self.scene
        bend = np.zeros_like(self._xi)
        twist = np.zeros_like(self._xi)
        for k, mode in enumerate(self.array.modes):
            q = self.array.modal_coordinate(mode, t_s)
            bend = bend + q * self._shape_bending[k]
            twist = twist + q * self._shape_torsion[k]

        centre_px = sc.centre_y_px + bend / sc.metres_per_pixel
        twist_px = twist / sc.metres_per_pixel
        half = 0.5 * sc.chord_px
        return centre_px - half - twist_px, centre_px + half + twist_px

    def _panel_coverage(self, t_s: float,
                        rows: Optional[slice] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Sub-pixel vertical coverage of the panel and the chordwise coordinate.

        Returns ``(coverage, eta)`` where coverage is in [0, 1] and ``eta`` in
        [0, 1] runs across the chord. Both have shape (len(rows), W).
        """
        ys = self._ys if rows is None else self._ys[rows]
        top, bot = self.edge_positions_px(t_s)
        upper = np.minimum(ys + 0.5, bot[None, :])
        lower = np.maximum(ys - 0.5, top[None, :])
        cov = np.clip(upper - lower, 0.0, 1.0) * self._in_span[None, :]
        chord = np.maximum(bot - top, 1e-6)[None, :]
        eta = np.clip((ys - top[None, :]) / chord, 0.0, 1.0)
        return cov, eta

    def log_radiance(self, t_s: float, rows: Optional[slice] = None) -> np.ndarray:
        """Natural-log scene radiance seen by each pixel, shape (len(rows), W)."""
        sc = self.scene
        ys = self._ys if rows is None else self._ys[rows]
        irradiance = float(self.sun.irradiance(t_s))
        disk_frac = float(self.sun.disk_fraction(t_s))

        cov, eta = self._panel_coverage(t_s, rows)
        stripe_chord = np.tanh(4.0 * np.sin(2.0 * math.pi * sc.n_cell_rows * eta))
        grid = 0.5 * (self._stripe_span[None, :] + stripe_chord)
        panel_refl = sc.panel_reflectance * (1.0 + sc.cell_stripe_contrast * grid)
        refl = sc.background_reflectance + cov * (panel_refl - sc.background_reflectance)
        radiance = irradiance * refl / math.pi

        if sc.earth_limb.enabled:
            limb = sc.earth_limb
            glow = 1.0 / (1.0 + np.exp(-(ys - limb.horizon_y_px) / limb.softness_px))
            radiance = radiance + irradiance * limb.radiance_gain * glow / math.pi

        if sc.solar_disk.enabled and disk_frac > 0.0:
            disk = sc.solar_disk
            y_sun = disk.final_y_px + disk.travel_y_px * (1.0 - disk_frac)
            dy = ys - y_sun
            r2 = dy * dy + (self._dx_sun * self._dx_sun)[None, :]
            core = 1.0 / (1.0 + np.exp((np.sqrt(r2) - disk.radius_px) / 1.5))
            halo = np.exp(-0.5 * r2 / (disk.halo_sigma_px ** 2))
            peak = irradiance * disk.peak_radiance_gain / math.pi
            radiance = radiance + peak * disk_frac * (core + 0.05 * halo)

        return np.log(radiance + DARK_RADIANCE_FLOOR)

    def _active_rows(self, t_s: float) -> slice:
        """Rows that can possibly change when the background is static.

        Once the disk has fully risen the illumination is bit-identical from slice
        to slice, so every pixel outside the panel's own row band has zero
        log-intensity change and cannot emit. Restricting the render and the
        threshold arithmetic to that band is the dominant speed-up for the long
        post-sunrise observation, which is most of the record.
        """
        top, bot = self.edge_positions_px(t_s)
        r0 = max(0, int(math.floor(float(top.min()))) - 2)
        r1 = min(self.height, int(math.ceil(float(bot.max()))) + 3)
        if r1 <= r0:
            return slice(0, self.height)
        return slice(r0, r1)

    # ---- event generation ------------------------------------------------- #

    def _emit(self, counts: np.ndarray, polarity: int, t0_us: int,
              row_offset: int = 0) -> np.ndarray:
        """Expand a per-pixel event count map into an EventCD array."""
        flat = counts.ravel()
        active = np.nonzero(flat)[0]
        if active.size == 0:
            return np.empty(0, dtype=EVENT_CD_DTYPE)

        n = flat[active].astype(np.int64)
        total = int(n.sum())
        ends = np.cumsum(n)
        starts = ends - n
        ordinal = np.arange(total, dtype=np.int64) - np.repeat(starts, n)
        frac = (ordinal + 1.0) / np.repeat(n + 1.0, n)

        width = counts.shape[1]
        out = np.empty(total, dtype=EVENT_CD_DTYPE)
        out["x"] = np.repeat((active % width).astype(np.uint16), n)
        out["y"] = np.repeat((active // width + row_offset).astype(np.uint16), n)
        out["p"] = polarity
        out["t"] = t0_us + (frac * self.slice_dt_us).astype(np.int64)
        return out

    def _noise_events(self, t0_us: int, log_l: np.ndarray,
                      refresh: bool) -> np.ndarray:
        """Leak (ON-biased) and shot noise (stronger where it is dark)."""
        dt_s = self.slice_dt_us * 1e-6
        n_px = self.width * self.height
        out = []

        lam_leak = self.sensor.leak_rate_hz * dt_s * n_px
        lam_shot = self.sensor.shot_noise_rate_hz * dt_s * n_px
        if lam_leak > 0:
            k = int(self.rng.poisson(lam_leak))
            if k:
                idx = self.rng.integers(0, n_px, size=k)
                ev = np.empty(k, dtype=EVENT_CD_DTYPE)
                ev["x"] = (idx % self.width).astype(np.uint16)
                ev["y"] = (idx // self.width).astype(np.uint16)
                ev["p"] = 1
                ev["t"] = t0_us + self.rng.integers(0, self.slice_dt_us, size=k)
                out.append(ev)
        if lam_shot > 0:
            # Shot noise per pixel scales inversely with photocurrent. Sampling via
            # a cached cumulative distribution avoids rebuilding it every slice,
            # which profiling showed dominated the quiescent-slice cost.
            if refresh or self._noise_cdf is None:
                weight = np.exp(-(log_l.ravel() - float(log_l.max())))
                cdf = np.cumsum(weight)
                self._noise_cdf = cdf / cdf[-1] if cdf[-1] > 0 else None
            k = int(self.rng.poisson(lam_shot))
            if k and self._noise_cdf is not None:
                idx = np.searchsorted(self._noise_cdf, self.rng.random(k))
                idx = np.minimum(idx, self._noise_cdf.size - 1)
                ev = np.empty(k, dtype=EVENT_CD_DTYPE)
                ev["x"] = (idx % self.width).astype(np.uint16)
                ev["y"] = (idx // self.width).astype(np.uint16)
                ev["p"] = np.where(self.rng.random(k) < 0.5, 1, -1)
                ev["t"] = t0_us + self.rng.integers(0, self.slice_dt_us, size=k)
                out.append(ev)

        if not out:
            return np.empty(0, dtype=EVENT_CD_DTYPE)
        return np.concatenate(out)

    def _apply_loss(self, events: np.ndarray, t0_us: int) -> Tuple[np.ndarray, Dict[str, float]]:
        """Apply on-sensor ERC decimation and/or ungoverned FIFO overflow."""
        stats = {"n_generated": float(events.size), "n_dropped_erc": 0.0,
                 "n_dropped_overflow": 0.0, "blackout": 0.0}
        dt_s = self.slice_dt_us * 1e-6

        if self.sensor.erc_target_evs is not None:
            budget = int(self.sensor.erc_target_evs * dt_s)
            if events.size > budget > 0:
                # Uniform pseudo-random decimation: unbiased in space and phase.
                keep = self.rng.choice(events.size, size=budget, replace=False)
                keep.sort()
                stats["n_dropped_erc"] = float(events.size - budget)
                events = events[keep]

        if self.sensor.readout_limit_evs is not None:
            budget = int(self.sensor.readout_limit_evs * dt_s)
            if events.size > budget > 0:
                # FIFO overflow: the tail of the slice is lost wholesale, and the
                # pipeline stays blind while the buffer drains.
                order = np.argsort(events["t"], kind="stable")
                events = events[order][:budget]
                overflow = stats["n_generated"] - budget
                stats["n_dropped_overflow"] = float(overflow)
                drain_us = int(self.slice_dt_us * min(4.0, overflow / max(budget, 1)))
                self._blackout_until_us = t0_us + self.slice_dt_us + drain_us
                stats["blackout"] = 1.0

        return events, stats

    def step(self, t0_us: int) -> SimSlice:
        """Generate one slice starting at ``t0_us``."""
        t1_s = (t0_us + self.slice_dt_us) * 1e-6

        # The background is bit-identical once the disk has fully risen, so only
        # the panel band needs evaluating from then on.
        signature = (float(self.sun.irradiance(t1_s)), float(self.sun.disk_fraction(t1_s)))
        static_bg = signature == self._last_bg
        self._last_bg = signature
        rows = self._active_rows(t1_s) if static_bg else slice(0, self.height)

        log_l = self.log_radiance(t1_s, rows)
        ref = self._ref[rows]
        delta = log_l - ref
        c_on, c_off = self._c_on[rows], self._c_off[rows]

        cap = self._max_events_per_slice_per_px
        n_on = np.floor(np.maximum(delta, 0.0) / c_on).astype(np.int32)
        n_off = np.floor(np.maximum(-delta, 0.0) / c_off).astype(np.int32)
        np.minimum(n_on, cap, out=n_on)
        np.minimum(n_off, cap, out=n_off)
        self._ref[rows] = ref + n_on * c_on - n_off * c_off

        offset = rows.start or 0
        parts = [self._emit(n_on, 1, t0_us, offset),
                 self._emit(n_off, -1, t0_us, offset),
                 self._noise_events(t0_us, log_l, refresh=not static_bg)]
        events = np.concatenate([p for p in parts if p.size])
        if events.size:
            events = events[np.argsort(events["t"], kind="stable")]

        events, stats = self._apply_loss(events, t0_us)

        if t0_us < self._blackout_until_us:
            stats["n_dropped_overflow"] += float(events.size)
            stats["blackout"] = 1.0
            events = np.empty(0, dtype=EVENT_CD_DTYPE)

        top, bot = self.edge_positions_px(t1_s)
        truth = {
            "t_s": t1_s,
            "irradiance_w_m2": float(self.sun.irradiance(t1_s)),
            "log_rate_per_s": float(self.sun.log_rate(t1_s)),
            "disk_fraction": float(self.sun.disk_fraction(t1_s)),
            "tip_displacement_m": self.array.tip_displacement(t1_s),
            "modal_coordinates_m": self.array.modal_coordinates(t1_s).tolist(),
            "edge_top_px": top,
            "edge_bottom_px": bot,
        }
        stats["event_rate_evs"] = events.size / (self.slice_dt_us * 1e-6)
        return SimSlice(ts_us=t0_us, dt_us=self.slice_dt_us, events=events,
                        truth=truth, stats=stats)

    def iterate(self, duration_s: float) -> Iterator[SimSlice]:
        """Yield consecutive slices covering ``duration_s`` seconds."""
        n = int(round(duration_s * 1e6 / self.slice_dt_us))
        for i in range(n):
            yield self.step(i * self.slice_dt_us)

    # ---- construction from config ----------------------------------------- #

    @classmethod
    def from_config(cls, cfg: dict, *, scale: Optional[int] = None,
                    seed: Optional[int] = None) -> "OrbitalSunriseSimulator":
        """Build a simulator from a parsed ``solar_array_config.json``."""
        sensor_cfg = cfg.get("sensor", {})
        scene_cfg = dict(cfg.get("scene", {}))
        run_cfg = cfg.get("run", {})
        array_cfg = cfg.get("array", {})

        scene_cfg.pop("_comment", None)
        disk_cfg = dict(scene_cfg.pop("solar_disk", {}))
        limb_cfg = dict(scene_cfg.pop("earth_limb", {}))
        scene = SceneGeometry(
            solar_disk=SolarDiskParams(**disk_cfg),
            earth_limb=EarthLimbParams(**limb_cfg),
            **scene_cfg,
        )

        sensor = SensorParams(
            contrast_threshold_on=sensor_cfg.get("contrast_threshold_on", 0.20),
            contrast_threshold_off=sensor_cfg.get("contrast_threshold_off", 0.20),
            threshold_mismatch_std=sensor_cfg.get("threshold_mismatch_std", 0.15),
            refractory_period_us=sensor_cfg.get("refractory_period_us", 100.0),
            leak_rate_hz=sensor_cfg.get("leak_rate_hz", 0.01),
            shot_noise_rate_hz=sensor_cfg.get("shot_noise_rate_hz", 0.05),
        )

        illum = dict(cfg.get("illumination", {}))
        illum.pop("_comment", None)
        sun = OrbitalSunriseProfile(**illum)

        modes = [VibrationMode(**m) for m in array_cfg.get("modes", [])]
        snap_cfg = array_cfg.get("thermal_snap", {})
        array = FlexibleArrayModel(
            span_m=array_cfg.get("span_m", 34.0),
            chord_m=array_cfg.get("chord_m", 11.6),
            modes=modes or FlexibleArrayModel().modes,
            snap=ThermalSnap(**snap_cfg) if snap_cfg else ThermalSnap(),
        )

        return cls(
            width=sensor_cfg.get("width", 1280),
            height=sensor_cfg.get("height", 720),
            scale=scale if scale is not None else run_cfg.get("sim_scale", 4),
            scene=scene,
            sensor=sensor,
            sun=sun,
            array=array,
            slice_dt_us=run_cfg.get("slice_dt_us", 5000),
            seed=seed if seed is not None else run_cfg.get("random_seed", 20260204),
        )


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #

def _self_test() -> int:
    ok = True

    def check(label: str, condition: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= bool(condition)
        print(f"  [{'PASS' if condition else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))

    print("OrbitalSunriseSimulator (320x180, 5 ms slices)")
    sim = OrbitalSunriseSimulator(scale=4, slice_dt_us=5000)
    check("geometry scaled", (sim.width, sim.height) == (320, 180),
          f"{sim.width}x{sim.height}")

    # Edge motion, measured peak-to-peak over the post-snap response.
    tip = int(sim.scene.tip_x_px) - 1
    root = int(sim.scene.root_x_px) + 1
    probe = np.array([sim.edge_positions_px(t)[0] for t in np.linspace(8.0, 20.0, 400)])
    swing_tip = float(probe[:, tip].max() - probe[:, tip].min())
    swing_root = float(probe[:, root].max() - probe[:, root].min())
    check("panel tip sweeps a resolvable distance", swing_tip > 1.5,
          f"{swing_tip:.2f} px peak-to-peak at 320x180 "
          f"({swing_tip * sim.scale:.1f} px at full resolution)")
    check("root is clamped", swing_root < 0.05 * swing_tip,
          f"{swing_root:.3f} px at the root")

    # Single forward pass — the pixel model is stateful, so never seek backwards.
    quiet, storm, post = [], [], []
    balance_storm, balance_post = [], []
    peak_rate = 0.0
    probe_slice = None
    for sl in sim.iterate(duration_s=20.0):
        rate = sl.stats["event_rate_evs"]
        t_s = sl.truth["t_s"]
        peak_rate = max(peak_rate, rate)
        pol = sl.events["p"]
        bal = float((np.sum(pol > 0) - np.sum(pol < 0)) / pol.size) if pol.size else 0.0
        if t_s < 1.5:
            quiet.append(rate)
        elif 1.8 < t_s < 4.0:
            storm.append(rate)
            balance_storm.append(bal)
            if probe_slice is None and pol.size:
                probe_slice = sl
        elif t_s > 18.5:
            post.append(rate)
            if pol.size:
                balance_post.append(bal)

    q, s, p = np.mean(quiet), np.mean(storm), np.mean(post)
    print(f"    mean rate: eclipse {q / 1e3:.1f} kev/s | ramp {s / 1e6:.2f} Mev/s | "
          f"post-sunrise {p / 1e3:.1f} kev/s | peak {peak_rate / 1e6:.2f} Mev/s")
    check("ramp produces an event storm", s > 50 * max(q, 1.0),
          f"{s / max(q, 1.0):.0f}x over eclipse baseline")
    check("storm subsides after sunrise", p < 0.1 * s)

    # Scale the 320x180 storm to the full array and compare with the independent
    # analytic budget from space_environment.
    scaled_peak = peak_rate * (1280 * 720) / (sim.width * sim.height)
    analytic = sim.sun.event_storm_budget(1280 * 720,
                                          contrast_threshold=sim.sensor.contrast_threshold_on)
    print(f"    peak scaled to 1280x720: {scaled_peak / 1e6:.0f} Mev/s "
          f"(analytic budget {analytic['peak_rate_evs'] / 1e6:.0f} Mev/s)")
    check("simulated peak agrees with the analytic budget within 3x",
          0.33 < scaled_peak / analytic["peak_rate_evs"] < 3.0,
          f"ratio {scaled_peak / analytic['peak_rate_evs']:.2f}")

    ev = probe_slice.events
    check("events time-ordered", bool(np.all(np.diff(ev["t"]) >= 0)))
    check("events in bounds",
          bool(ev["x"].max() < sim.width and ev["y"].max() < sim.height))
    check("polarity is +-1", set(np.unique(ev["p"]).tolist()) <= {-1, 1})

    # Polarity balance is the discriminant the glare guard relies on: the ramp is
    # common mode (ON-dominated), quiescent vibration is differential (balanced).
    b_storm = float(np.mean(balance_storm))
    b_post = float(np.mean(balance_post))
    print(f"    polarity balance: ramp {b_storm:+.3f} | vibrating in full sun {b_post:+.3f}")
    check("ramp is ON-dominated (common mode)", b_storm > 0.9)
    check("vibration is balanced (differential)", abs(b_post) < 0.5)

    print("\nSELF-TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
