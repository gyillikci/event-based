"""Saturation mitigation for event-based sensing through an orbital sunrise.

Three layers, applied in order:

L0  Sensor      :class:`EventRateGovernor`, :class:`BiasScheduler`
    Cap the readout budget and retune the pixel front end as the scene brightens.
    The governor decimates *uniformly at random*, which is what the on-sensor ERC
    does; unlike FIFO overflow this is unbiased in space and in phase, so it costs
    variance but not accuracy. That property is what makes a rate cap safe here.

L1  Common mode :class:`CommonModeRejector`
    A global irradiance ramp shifts every pixel's log-intensity by the same amount,
    so every pixel emits the same polarity: the disturbance is *common mode*. A
    vibrating structural edge drives its pixels alternately ON and OFF and is
    localised: the signal is *differential*. Estimating the per-pixel signed event
    count attributable to the ramp (robustly, so the localised signal cannot bias
    it) and subtracting it is exactly the common-mode rejection of a differential
    pair, and it is quantified the same way, as a CMRR in dB.

L2  Spatial     :class:`SolarDiskMask`
    The solar disk and its veiling-glare halo, and the bright Earth limb, are
    persistently hot regardless of the global ramp. They are detected by persistence
    of an extreme, polarity-pinned event rate and masked with a halo margin.

:class:`GlareGuard` runs all three and is the single entry point used by both the
live application and the offline benchmark.

NumPy only -- no SDK, no scipy. The optional hardware handles (``erc_module``,
``biases``) are duck-typed against the Metavision HAL facilities so this module
stays importable without the SDK. Self-test::

    python glare_guard.py
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

EVENT_CD_DTYPE = np.dtype([("x", "<u2"), ("y", "<u2"), ("p", "<i2"), ("t", "<i8")])


# --------------------------------------------------------------------------- #
# Binning
# --------------------------------------------------------------------------- #

@dataclass
class CellCounts:
    """Per-cell ON/OFF event counts for one time slice.

    ``px_map`` holds the true number of sensor pixels in each cell. Cells at the
    right and bottom borders are partial whenever the sensor dimensions are not an
    exact multiple of the cell size, and normalising those by the nominal cell area
    injects a large false residual into the common-mode estimate.
    """

    on: np.ndarray            # (n_rows, n_cols) int32
    off: np.ndarray           # (n_rows, n_cols) int32
    cell_size_px: int
    px_map: np.ndarray        # (n_rows, n_cols) int64

    @property
    def total(self) -> np.ndarray:
        return self.on + self.off

    @property
    def signed(self) -> np.ndarray:
        return self.on.astype(np.float64) - self.off.astype(np.float64)

    @property
    def signed_per_px(self) -> np.ndarray:
        """Signed event count per sensor pixel -- the common-mode observable."""
        return self.signed / self.px_map

    @property
    def density(self) -> np.ndarray:
        """Total events per sensor pixel -- comparable across partial cells."""
        return self.total / self.px_map

    def balance(self, min_events: int = 4) -> np.ndarray:
        """Polarity balance per cell in [-1, 1]; NaN where there is too little data."""
        tot = self.total
        out = np.full(tot.shape, np.nan, dtype=np.float64)
        ok = tot >= min_events
        out[ok] = self.signed[ok] / tot[ok]
        return out


class EventSliceBinner:
    """Bins an EventCD slice into a coarse ON/OFF cell grid."""

    def __init__(self, width: int, height: int, cell_size_px: int = 16) -> None:
        if cell_size_px < 1:
            raise ValueError("cell_size_px must be >= 1")
        self.width = int(width)
        self.height = int(height)
        self.cell_size_px = int(cell_size_px)
        self.n_cols = int(math.ceil(self.width / self.cell_size_px))
        self.n_rows = int(math.ceil(self.height / self.cell_size_px))
        self.n_cells = self.n_rows * self.n_cols

        col_px = np.minimum(self.cell_size_px,
                            self.width - self.cell_size_px * np.arange(self.n_cols))
        row_px = np.minimum(self.cell_size_px,
                            self.height - self.cell_size_px * np.arange(self.n_rows))
        self.px_map = (row_px[:, None] * col_px[None, :]).astype(np.int64)

    def cell_index(self, events: np.ndarray) -> np.ndarray:
        cy = events["y"].astype(np.int64) // self.cell_size_px
        cx = events["x"].astype(np.int64) // self.cell_size_px
        return cy * self.n_cols + cx

    def bin(self, events: np.ndarray) -> CellCounts:
        shape = (self.n_rows, self.n_cols)
        if events.size == 0:
            zeros = np.zeros(shape, dtype=np.int32)
            return CellCounts(zeros, zeros.copy(), self.cell_size_px, self.px_map)
        idx = self.cell_index(events)
        pos = events["p"] > 0
        on = np.bincount(idx[pos], minlength=self.n_cells).astype(np.int32)
        off = np.bincount(idx[~pos], minlength=self.n_cells).astype(np.int32)
        return CellCounts(on.reshape(shape), off.reshape(shape),
                          self.cell_size_px, self.px_map)


# --------------------------------------------------------------------------- #
# L0 - sensor level
# --------------------------------------------------------------------------- #

class EventRateGovernor:
    """Caps the event rate, mirroring the on-sensor Event Rate Controller.

    When a hardware ``erc_module`` (Metavision ``I_ErcModule``) is supplied the cap
    is enforced on-sensor and :meth:`apply` is a pass-through. Otherwise the same
    policy is emulated in software so that replay and simulation see the identical
    decimation statistics.

    Decimation is uniform without replacement. For a periodic edge signal this
    leaves the estimated phase unbiased and inflates its variance by roughly the
    decimation factor, which is why a rate cap is preferable to letting the readout
    FIFO overflow -- overflow removes time-contiguous blocks and destroys phase.
    """

    def __init__(
        self,
        target_evs: float = 20e6,
        *,
        erc_module: object = None,
        seed: int = 7,
        enabled: bool = True,
    ) -> None:
        self.target_evs = float(target_evs)
        self.enabled = bool(enabled)
        self.rng = np.random.default_rng(seed)
        self.erc_module = erc_module
        self.hardware = False
        self.last_rate_evs = 0.0
        self.last_kept = 0
        self.last_dropped = 0

        if erc_module is not None and enabled:
            try:
                erc_module.set_cd_event_rate(int(self.target_evs))
                erc_module.enable(True)
                self.hardware = True
            except Exception as exc:            # pragma: no cover - hardware path
                print(f"[EventRateGovernor] hardware ERC unavailable ({exc}); "
                      f"falling back to software decimation")

    @property
    def decimation_factor(self) -> float:
        kept = max(self.last_kept, 1)
        return (self.last_kept + self.last_dropped) / kept

    def apply(self, events: np.ndarray, dt_us: int) -> np.ndarray:
        dt_s = max(dt_us * 1e-6, 1e-12)
        self.last_rate_evs = events.size / dt_s
        self.last_kept = int(events.size)
        self.last_dropped = 0

        if not self.enabled or self.hardware:
            return events

        budget = int(self.target_evs * dt_s)
        if budget <= 0 or events.size <= budget:
            return events

        keep = self.rng.choice(events.size, size=budget, replace=False)
        keep.sort()
        self.last_kept = budget
        self.last_dropped = int(events.size - budget)
        return events[keep]


class BiasScheduler:
    """Switches pixel bias profiles as the scene crosses the terminator.

    Profiles are selected from the measured event rate with hysteresis. Raising the
    contrast thresholds (``bias_diff_on`` / ``bias_diff_off``) during the ramp cuts
    the storm count by roughly ``ln(1 + dC) / ln(1 + C)``; the trade is a coarser
    displacement quantum, which is acceptable because the ramp window is short
    relative to the modal periods being measured.

    WARNING: bias codes are unit-specific. The values in ``solar_array_config.json``
    are starting points and MUST be swept on the actual EVK4 before flight-like use.
    """

    ORDER = ("eclipse", "terminator", "full_sun")

    def __init__(
        self,
        profiles: Optional[Dict[str, Dict[str, int]]] = None,
        *,
        enter_rate_evs: float = 8e6,
        exit_rate_evs: float = 2e6,
        biases: object = None,
        dwell_slices: int = 4,
    ) -> None:
        self.profiles = {k: v for k, v in (profiles or {}).items()
                         if not k.startswith("_")}
        self.enter_rate_evs = float(enter_rate_evs)
        self.exit_rate_evs = float(exit_rate_evs)
        self.biases = biases
        self.dwell_slices = int(dwell_slices)
        self.profile = "eclipse"
        self.transitions: List[Dict[str, object]] = []
        self._pending: Optional[str] = None
        self._pending_count = 0
        self._sunrise_seen = False

    def _target(self, rate_evs: float) -> str:
        if rate_evs >= self.enter_rate_evs:
            self._sunrise_seen = True
            return "terminator"
        if self._sunrise_seen and rate_evs <= self.exit_rate_evs:
            return "full_sun"
        return self.profile if self._sunrise_seen else "eclipse"

    def update(self, rate_evs: float, ts_us: int = 0) -> str:
        """Advance the scheduler with the latest measured rate; returns the profile."""
        target = self._target(rate_evs)
        if target == self.profile:
            self._pending, self._pending_count = None, 0
            return self.profile

        if target == self._pending:
            self._pending_count += 1
        else:
            self._pending, self._pending_count = target, 1

        if self._pending_count >= self.dwell_slices:
            previous = self.profile
            self.profile = target
            self._pending, self._pending_count = None, 0
            self.transitions.append({
                "ts_us": int(ts_us),
                "from": previous,
                "to": self.profile,
                "rate_evs": float(rate_evs),
            })
            self._push()
        return self.profile

    def _push(self) -> None:
        values = self.profiles.get(self.profile)
        if not values or self.biases is None:
            return
        for name, code in values.items():          # pragma: no cover - hardware path
            if name.startswith("_"):
                continue
            try:
                self.biases.set(name, int(code))
            except Exception as exc:
                print(f"[BiasScheduler] could not set {name}={code}: {exc}")

    @property
    def active_biases(self) -> Dict[str, int]:
        return {k: v for k, v in self.profiles.get(self.profile, {}).items()
                if not k.startswith("_")}


# --------------------------------------------------------------------------- #
# L1 - common-mode rejection
# --------------------------------------------------------------------------- #

@dataclass
class CommonModeEstimate:
    """Result of one common-mode estimation."""

    per_px: np.ndarray        # (n_rows, n_cols) signed events per pixel, common mode
    index: float              # common-mode index in [0, 1]
    balance: float            # robust global polarity balance in [-1, 1]
    dispersion: float         # spatial MAD of the per-cell balance
    coverage: float           # fraction of cells carrying enough events to judge
    contaminated: bool        # index exceeded the gate threshold

    def at(self, x_px: float, y_px: float, cell_size_px: int) -> float:
        """Common-mode signed events per pixel at an image location."""
        r = int(np.clip(y_px // cell_size_px, 0, self.per_px.shape[0] - 1))
        c = int(np.clip(x_px // cell_size_px, 0, self.per_px.shape[1] - 1))
        return float(self.per_px[r, c])


class CommonModeRejector:
    """Separates a globally correlated illumination transient from local vibration.

    ``mode`` selects the spatial model of the common-mode field:

    ``"median"``
        A single scalar, the spatial median of the per-pixel signed count. Maximally
        robust; correct when the ramp is uniform. Breaks down only if the structure
        fills more than half the frame.
    ``"plane"``
        A trimmed least-squares plane, which additionally absorbs the illumination
        *gradient* produced by the rising limb and the glare halo. More accurate
        during the ramp, but it can partially absorb genuine low-order mode shapes,
        so it is reported as an ablation rather than the default.
    ``"none"``
        Disabled, for the ablation baseline.
    """

    def __init__(
        self,
        mode: str = "median",
        *,
        index_threshold: float = 0.55,
        min_events_per_cell: int = 4,
        dispersion_scale: float = 0.5,
        trim_fraction: float = 0.1,
    ) -> None:
        if mode not in ("median", "plane", "none"):
            raise ValueError(f"unknown common-mode mode: {mode!r}")
        self.mode = mode
        self.index_threshold = float(index_threshold)
        self.min_events_per_cell = int(min_events_per_cell)
        self.dispersion_scale = float(dispersion_scale)
        self.trim_fraction = float(trim_fraction)
        self._grid: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def _plane_fit(self, signed_px: np.ndarray, valid: np.ndarray) -> np.ndarray:
        rows, cols = signed_px.shape
        if self._grid is None or self._grid[0].shape != signed_px.shape:
            rr, cc = np.meshgrid(np.arange(rows, dtype=np.float64),
                                 np.arange(cols, dtype=np.float64), indexing="ij")
            self._grid = (rr / max(rows - 1, 1), cc / max(cols - 1, 1))
        rr, cc = self._grid

        keep = valid.copy()
        coeffs = np.zeros(3)
        for _ in range(2):
            if keep.sum() < 8:
                break
            a = np.stack([np.ones(int(keep.sum())), rr[keep], cc[keep]], axis=1)
            coeffs, *_ = np.linalg.lstsq(a, signed_px[keep], rcond=None)
            resid = np.abs(signed_px - (coeffs[0] + coeffs[1] * rr + coeffs[2] * cc))
            cutoff = np.quantile(resid[keep], 1.0 - self.trim_fraction)
            keep = valid & (resid <= max(cutoff, 1e-12))
        return coeffs[0] + coeffs[1] * rr + coeffs[2] * cc

    def estimate(self, counts: CellCounts,
                 mask: Optional[np.ndarray] = None) -> CommonModeEstimate:
        """Estimate the common-mode field from one slice of cell counts.

        The index combines three necessary conditions for a global illumination
        transient. All three must hold, so they multiply:

        * **polarity pinning** -- ``|median balance|`` near 1;
        * **spatial uniformity** -- small dispersion of the balance across cells;
        * **spatial coverage** -- most of the array is active.

        Coverage is what separates the ramp from a localised disturbance. A
        vibrating panel edge can be strongly polarity-pinned within its own cells,
        but it only ever lights up the fraction of the array it occupies, whereas
        the sunrise lights up all of it.
        """
        signed_px = counts.signed_per_px
        active = counts.total >= self.min_events_per_cell
        valid = active.copy()
        if mask is not None:
            valid &= ~mask
            judged = ~mask
        else:
            judged = np.ones_like(active)

        bal = counts.balance(self.min_events_per_cell)
        if mask is not None:
            bal = np.where(mask, np.nan, bal)
        finite = bal[np.isfinite(bal)]
        if finite.size:
            balance = float(np.median(finite))
            dispersion = float(np.median(np.abs(finite - balance)))
        else:
            balance, dispersion = 0.0, 0.0

        n_judged = int(judged.sum())
        coverage = float(valid.sum() / n_judged) if n_judged else 0.0
        uniformity = 1.0 - min(1.0, dispersion / max(self.dispersion_scale, 1e-9))
        index = abs(balance) * uniformity * coverage

        if self.mode == "none" or not valid.any():
            per_px = np.zeros_like(signed_px)
        elif self.mode == "plane":
            per_px = self._plane_fit(signed_px, valid)
        else:
            per_px = np.full_like(signed_px, float(np.median(signed_px[valid])))

        return CommonModeEstimate(
            per_px=per_px,
            index=float(index),
            balance=balance,
            dispersion=dispersion,
            coverage=coverage,
            contaminated=bool(index >= self.index_threshold),
        )


# --------------------------------------------------------------------------- #
# L2 - spatial masking
# --------------------------------------------------------------------------- #

def _box_dilate(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    """Separable box dilation of a boolean cell mask (NumPy only)."""
    if radius_cells <= 0 or not mask.any():
        return mask
    out = mask.copy()
    for axis in (0, 1):
        acc = out.copy()
        for shift in range(1, radius_cells + 1):
            acc |= np.roll(out, shift, axis=axis)
            acc |= np.roll(out, -shift, axis=axis)
        out = acc
    return out


class SolarDiskMask:
    """Masks the solar disk, its glare halo and the Earth limb.

    Detection is by *persistence* of an extreme, polarity-pinned event rate relative
    to the robust scene median. Persistence is what distinguishes these sources from
    the global ramp: during the ramp every cell is hot, but only the disk and the
    limb stay hot once the ramp has passed.
    """

    def __init__(
        self,
        n_rows: int,
        n_cols: int,
        *,
        rate_sigma: float = 6.0,
        persistence: int = 4,
        halo_px: int = 48,
        cell_size_px: int = 16,
        release: int = 2,
    ) -> None:
        self.rate_sigma = float(rate_sigma)
        self.persistence = int(persistence)
        self.halo_cells = max(0, int(round(halo_px / max(cell_size_px, 1))))
        self.release = int(release)
        self._hits = np.zeros((n_rows, n_cols), dtype=np.int32)
        self.mask = np.zeros((n_rows, n_cols), dtype=bool)

    def update(self, counts: CellCounts) -> np.ndarray:
        density = counts.density
        # Partial border cells carry few pixels, so their density is noisy. Judge
        # the scene statistics on whole cells only and never let a partial cell
        # seed a detection on its own.
        nominal = counts.cell_size_px ** 2
        eligible = counts.px_map >= 0.5 * nominal
        reference = density[eligible] if eligible.any() else density

        median = float(np.median(reference))
        mad = float(np.median(np.abs(reference - median)))
        # 1.4826 converts MAD to a Gaussian-equivalent sigma.
        sigma = max(1.4826 * mad, 1e-3)
        hot = (density > median + self.rate_sigma * sigma) & eligible

        self._hits = np.where(hot, self._hits + 1,
                              np.maximum(self._hits - self.release, 0))
        core = self._hits >= self.persistence
        self.mask = _box_dilate(core, self.halo_cells)
        return self.mask

    def masked_fraction(self) -> float:
        return float(self.mask.mean()) if self.mask.size else 0.0


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

@dataclass
class GuardedSlice:
    """Output of :meth:`GlareGuard.process` for one time slice."""

    ts_us: int
    dt_us: int
    events: np.ndarray
    counts: CellCounts
    common_mode: CommonModeEstimate
    mask: np.ndarray
    profile: str
    stats: Dict[str, float] = field(default_factory=dict)


class GlareGuard:
    """Runs the sensor, common-mode and spatial layers over an event slice."""

    def __init__(
        self,
        width: int,
        height: int,
        *,
        cell_size_px: int = 16,
        common_mode: str = "median",
        index_threshold: float = 0.55,
        solar_mask_rate_sigma: float = 6.0,
        solar_mask_persistence: int = 4,
        solar_mask_halo_px: int = 48,
        erc_target_evs: float = 20e6,
        erc_enabled: bool = True,
        erc_module: object = None,
        bias_profiles: Optional[Dict[str, Dict[str, int]]] = None,
        terminator_enter_rate_evs: float = 8e6,
        terminator_exit_rate_evs: float = 2e6,
        biases: object = None,
        mask_enabled: bool = True,
        seed: int = 7,
    ) -> None:
        self.binner = EventSliceBinner(width, height, cell_size_px)
        self.governor = EventRateGovernor(erc_target_evs, erc_module=erc_module,
                                          seed=seed, enabled=erc_enabled)
        self.scheduler = BiasScheduler(bias_profiles,
                                       enter_rate_evs=terminator_enter_rate_evs,
                                       exit_rate_evs=terminator_exit_rate_evs,
                                       biases=biases)
        self.rejector = CommonModeRejector(common_mode, index_threshold=index_threshold)
        self.mask_enabled = bool(mask_enabled)
        self.masker = SolarDiskMask(
            self.binner.n_rows, self.binner.n_cols,
            rate_sigma=solar_mask_rate_sigma,
            persistence=solar_mask_persistence,
            halo_px=solar_mask_halo_px,
            cell_size_px=cell_size_px,
        )

    def process(self, ts_us: int, dt_us: int, events: np.ndarray) -> GuardedSlice:
        raw_n = int(events.size)
        events = self.governor.apply(events, dt_us)
        rate_evs = self.governor.last_rate_evs
        profile = self.scheduler.update(rate_evs, ts_us)

        counts = self.binner.bin(events)
        mask = (self.masker.update(counts) if self.mask_enabled
                else np.zeros((self.binner.n_rows, self.binner.n_cols), dtype=bool))

        if self.mask_enabled and mask.any() and events.size:
            idx = self.binner.cell_index(events)
            events = events[~mask.ravel()[idx]]
            counts = self.binner.bin(events)

        cm = self.rejector.estimate(counts, mask if self.mask_enabled else None)

        return GuardedSlice(
            ts_us=int(ts_us), dt_us=int(dt_us), events=events, counts=counts,
            common_mode=cm, mask=mask, profile=profile,
            stats={
                "raw_events": float(raw_n),
                "kept_events": float(events.size),
                "raw_rate_evs": float(rate_evs),
                "erc_dropped": float(self.governor.last_dropped),
                "decimation_factor": float(self.governor.decimation_factor),
                "masked_cell_fraction": self.masker.masked_fraction(),
                "common_mode_index": cm.index,
                "polarity_balance": cm.balance,
                "balance_dispersion": cm.dispersion,
                "active_coverage": cm.coverage,
            },
        )

    @classmethod
    def from_config(cls, cfg: dict, width: int, height: int, *, scale: int = 1,
                    erc_module: object = None, biases: object = None,
                    common_mode: Optional[str] = None) -> "GlareGuard":
        g = cfg.get("glare_guard", {})
        return cls(
            width, height,
            # The cell grid is a statistical binning choice, not a sensor
            # dimension, so it does NOT scale with resolution. Shrinking it
            # alongside the sensor starves each cell of the events it needs to
            # estimate a polarity balance.
            cell_size_px=int(g.get("cell_size_px", 16)),
            common_mode=common_mode or ("median" if g.get("median_subtraction", True) else "none"),
            index_threshold=g.get("common_mode_index_threshold", 0.55),
            solar_mask_rate_sigma=g.get("solar_mask_rate_sigma", 6.0),
            solar_mask_persistence=g.get("solar_mask_persistence", 4),
            # The glare halo IS physical, so this one does scale.
            solar_mask_halo_px=max(1, int(g.get("solar_mask_halo_px", 48)) // max(scale, 1)),
            erc_target_evs=g.get("erc_target_evs", 20e6),
            erc_module=erc_module,
            bias_profiles=g.get("bias_profiles"),
            terminator_enter_rate_evs=g.get("terminator_enter_rate_evs", 8e6),
            terminator_exit_rate_evs=g.get("terminator_exit_rate_evs", 2e6),
            biases=biases,
        )


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #

def _expand(counts: np.ndarray, polarity: int, rng: np.random.Generator,
            dt_us: int = 5000) -> np.ndarray:
    """Expand a per-pixel count map into an EventCD array."""
    flat = counts.ravel()
    active = np.nonzero(flat)[0]
    if active.size == 0:
        return np.empty(0, dtype=EVENT_CD_DTYPE)
    n = flat[active].astype(np.int64)
    total = int(n.sum())
    height, width = counts.shape
    out = np.empty(total, dtype=EVENT_CD_DTYPE)
    out["x"] = np.repeat((active % width).astype(np.uint16), n)
    out["y"] = np.repeat((active // width).astype(np.uint16), n)
    out["p"] = polarity
    out["t"] = rng.integers(0, dt_us, total)
    return out


def _synthetic_slice(width: int, height: int, *, common_per_px: int,
                     mismatch: float, band_rows: Optional[Tuple[int, int]],
                     band_per_px: float, rng: np.random.Generator) -> np.ndarray:
    """Global illumination step plus a localised, polarity-split moving edge.

    The common mode is *deterministic*, not Poisson: a global irradiance ramp shifts
    every pixel's log-intensity identically, so every pixel emits the same count.
    The only scatter is fixed-pattern contrast-threshold mismatch, which is the true
    physical floor on how well the common mode can be estimated.

    The band models a moving edge: pixels on one side brighten while pixels on the
    other darken, so the disturbance is polarity-split and localised.
    """
    shape = (height, width)
    parts = []

    if common_per_px:
        base = np.full(shape, abs(common_per_px), dtype=np.int64)
        if mismatch > 0:
            base += (rng.random(shape) < mismatch).astype(np.int64)
        parts.append(_expand(base, 1 if common_per_px > 0 else -1, rng))

    if band_rows is not None and band_per_px:
        r0, r1 = band_rows
        mid = (r0 + r1) // 2
        upper = np.zeros(shape, dtype=np.int64)
        lower = np.zeros(shape, dtype=np.int64)
        upper[r0:mid, :] = int(abs(band_per_px))
        lower[mid:r1, :] = int(abs(band_per_px))
        parts.append(_expand(upper, 1, rng))
        parts.append(_expand(lower, -1, rng))

    parts = [p for p in parts if p.size]
    if not parts:
        return np.empty(0, dtype=EVENT_CD_DTYPE)
    out = np.concatenate(parts)
    return out[np.argsort(out["t"], kind="stable")]


def _self_test() -> int:
    ok = True

    def check(label: str, condition: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= bool(condition)
        print(f"  [{'PASS' if condition else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))

    rng = np.random.default_rng(3)
    w, h = 320, 180
    cell = 16
    # Band spans exactly two cell rows so the expected per-cell value is exact.
    band = (3 * cell, 5 * cell)

    print("EventSliceBinner")
    binner = EventSliceBinner(w, h, cell_size_px=cell)
    ev = _synthetic_slice(w, h, common_per_px=1, mismatch=0.0,
                          band_rows=None, band_per_px=0.0, rng=rng)
    counts = binner.bin(ev)
    check("grid shape", counts.on.shape == (binner.n_rows, binner.n_cols),
          f"{counts.on.shape}")
    check("all events binned", int(counts.total.sum()) == ev.size)
    check("one event per pixel", int(counts.total.sum()) == w * h)
    check("uniform burst is ON-pinned", abs(float(np.nanmedian(counts.balance())) - 1.0) < 1e-9)

    print("CommonModeRejector")
    rej = CommonModeRejector("median")
    ev = _synthetic_slice(w, h, common_per_px=2, mismatch=0.15,
                          band_rows=band, band_per_px=3, rng=rng)
    counts = binner.bin(ev)
    est = rej.estimate(counts)
    check("common mode recovered", abs(est.per_px[0, 0] - 2.15) < 0.1,
          f"estimated {est.per_px[0, 0]:.3f} /px, true 2.150 /px")

    residual_on = float(counts.signed_per_px[3].mean() - est.per_px[3].mean())
    residual_off = float(counts.signed_per_px[4].mean() - est.per_px[4].mean())
    check("differential signal survives with sign intact",
          residual_on > 2.5 and residual_off < -2.5,
          f"ON side {residual_on:+.2f}, OFF side {residual_off:+.2f} signed events/px "
          f"(true +-3.00)")

    # The index must separate a global ramp from a localised moving edge.
    ramp = binner.bin(_synthetic_slice(w, h, common_per_px=2, mismatch=0.15,
                                       band_rows=None, band_per_px=0.0, rng=rng))
    vib = binner.bin(_synthetic_slice(w, h, common_per_px=0, mismatch=0.0,
                                      band_rows=band, band_per_px=3, rng=rng))
    e_ramp, e_vib = rej.estimate(ramp), rej.estimate(vib)
    print(f"    ramp      : index {e_ramp.index:.3f}  balance {e_ramp.balance:+.2f}  "
          f"coverage {e_ramp.coverage:.2f}")
    print(f"    vibration : index {e_vib.index:.3f}  balance {e_vib.balance:+.2f}  "
          f"coverage {e_vib.coverage:.2f}")
    check("index flags the ramp", e_ramp.index > 0.9)
    check("index clears localised vibration", e_vib.index < 0.3)
    check("ramp gate trips", e_ramp.contaminated and not e_vib.contaminated)

    # CMRR against a pure ramp: the floor is fixed-pattern threshold mismatch.
    cm_before = float(np.mean(np.abs(ramp.signed_per_px)))
    cm_after = float(np.mean(np.abs(ramp.signed_per_px - e_ramp.per_px)))
    cmrr_db = 20.0 * math.log10(cm_before / max(cm_after, 1e-12))
    print(f"    common-mode residual {cm_before:.3f} -> {cm_after:.4f} /px  "
          f"(CMRR {cmrr_db:.1f} dB)")
    check("rejection exceeds 30 dB", cmrr_db > 30.0)

    # The plane model must do better on a ramp carrying a spatial gradient.
    grad = ramp.signed.astype(np.float64).copy()
    grad += np.linspace(0, 400, grad.shape[0])[:, None]
    grad_counts = CellCounts(np.maximum(grad, 0).astype(np.int32),
                             np.zeros_like(ramp.off), cell, ramp.px_map)
    med_res = np.abs(grad_counts.signed_per_px
                     - CommonModeRejector("median").estimate(grad_counts).per_px).mean()
    pln_res = np.abs(grad_counts.signed_per_px
                     - CommonModeRejector("plane").estimate(grad_counts).per_px).mean()
    print(f"    graded ramp residual: median {med_res:.4f} /px | plane {pln_res:.4f} /px")
    check("plane model beats median on a graded ramp", pln_res < 0.5 * med_res,
          f"{med_res / max(pln_res, 1e-12):.1f}x better")

    print("EventRateGovernor")
    gov = EventRateGovernor(target_evs=1e6, seed=1)
    big = _synthetic_slice(w, h, common_per_px=1, mismatch=0.0,
                           band_rows=None, band_per_px=0.0, rng=rng)
    kept = gov.apply(big, dt_us=5000)
    check("decimated to budget", kept.size == int(1e6 * 5000e-6),
          f"{big.size} -> {kept.size} events")
    check("kept slice stays time-ordered", bool(np.all(np.diff(kept["t"]) >= 0)))
    frac_top = float(np.mean(kept["y"] < h / 2))
    check("decimation is spatially uniform", abs(frac_top - 0.5) < 0.02,
          f"{frac_top:.3f} of kept events in the top half")
    check("decimation factor reported", abs(gov.decimation_factor - big.size / kept.size) < 1e-9,
          f"{gov.decimation_factor:.2f}x")

    print("BiasScheduler")
    sched = BiasScheduler({"eclipse": {"bias_diff_on": 0}, "terminator": {"bias_diff_on": 45},
                           "full_sun": {"bias_diff_on": 20}},
                          enter_rate_evs=8e6, exit_rate_evs=2e6, dwell_slices=2)
    seq = [1e5] * 5 + [3e7] * 6 + [5e5] * 6
    seen = [sched.update(r, ts_us=i * 5000) for i, r in enumerate(seq)]
    check("starts in eclipse", seen[0] == "eclipse")
    check("enters terminator profile", "terminator" in seen)
    check("settles in full_sun", seen[-1] == "full_sun", f"{seen[-1]}")
    check("hysteresis limits thrash", len(sched.transitions) == 2,
          f"{len(sched.transitions)} transitions")

    print("SolarDiskMask")
    masker = SolarDiskMask(binner.n_rows, binner.n_cols, rate_sigma=4.0,
                           persistence=3, halo_px=32, cell_size_px=cell)
    # A uniform scene produces uniform event *density*, so scale the background
    # counts by the true pixel count of each cell.
    hot = np.round(0.02 * binner.px_map).astype(np.int32)
    hot[3:5, 10:12] = 5000
    for _ in range(4):
        mask = masker.update(CellCounts(hot, np.zeros_like(hot), cell, binner.px_map))
    check("hot region masked", bool(mask[3:5, 10:12].all()))
    check("halo applied", bool(mask[1, 10]) and not bool(mask[0, 0]))
    check("masked fraction bounded", 0.0 < masker.masked_fraction() < 0.35,
          f"{masker.masked_fraction():.3f}")

    print("GlareGuard end to end")
    guard = GlareGuard(w, h, cell_size_px=cell, erc_target_evs=5e6)
    out = guard.process(0, 5000, _synthetic_slice(w, h, common_per_px=2, mismatch=0.15,
                                                  band_rows=band, band_per_px=3, rng=rng))
    check("slice carries all layers",
          out.events.size > 0 and out.counts.on.size > 0
          and 0.0 <= out.common_mode.index <= 1.0)
    check("stats populated", {"raw_events", "common_mode_index", "active_coverage",
                              "masked_cell_fraction"} <= set(out.stats))

    print("\nSELF-TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
