"""Space illumination and flexible-array structural models.

Two independent physical models used by the rest of the package:

1. :class:`OrbitalSunriseProfile` — scene irradiance as the solar disk clears the
   Earth limb. Modelled as the geometric circular-segment occultation of a uniform
   solar disk rising at the orbital rate, with an optional refracted/scattered
   precursor glow. Provides d(ln E)/dt, which is the quantity that sets the event
   rate of a logarithmic sensor and therefore the size of the event storm.

2. :class:`FlexibleArrayModel` — out-of-plane displacement of a cantilevered
   flexible array (ISS SAW / ROSA class). Euler-Bernoulli cantilever bending mode
   shapes plus fixed-free torsion, excited by a thermal-snap impulse at terminator
   crossing on top of a low ambient level.

All values that are *specification claims* rather than model parameters are marked
CITE-REQUIRED and must be sourced before publication.

NumPy only — no SDK, no scipy. Self-test::

    python space_environment.py
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# Physical constants
# --------------------------------------------------------------------------- #

SOLAR_CONSTANT_W_M2 = 1361.0          # AM0 total solar irradiance  [CITE-REQUIRED]
SOLAR_ANGULAR_DIAMETER_DEG = 0.533    # apparent diameter at 1 AU   [CITE-REQUIRED]
LEO_PERIOD_S = 5580.0                 # ~93 min at 400 km           [CITE-REQUIRED]

#: Roots of ``cos(x)cosh(x) + 1 = 0`` — the cantilever (fixed-free) eigenvalues.
CANTILEVER_BETA_L = (1.875104069, 4.694091133, 7.854757438, 10.995540734)


# --------------------------------------------------------------------------- #
# Illumination
# --------------------------------------------------------------------------- #

@dataclass
class OrbitalSunriseProfile:
    """Scene irradiance through a terminator crossing.

    The solar disk is treated as a uniform circle of angular radius ``R`` whose
    centre climbs linearly from ``-R`` to ``+R`` relative to the Earth limb over
    ``rise_duration_s``. The visible fraction is then the exact circular-segment
    area ratio, which gives a smooth S-curve with zero slope at both ends — unlike
    an ad-hoc sigmoid it has the right tails, and the tails are what dominate the
    logarithmic event rate.

    Parameters
    ----------
    eclipse_irradiance_w_m2:
        Illumination floor during eclipse. Earthshine over a sunlit Earth is the
        dominant term; the default corresponds to roughly five decades below full
        sun, a deliberately optimistic (i.e. hard) case for the sensor.
    rise_duration_s:
        Time for the disk to go from first contact to fully clear. At the LEO
        orbital rate (~0.065 deg/s) a 0.533 deg disk clears in ~8 s; refraction
        and limb scattering stretch this, hence the 12 s default.
    refraction_glow_frac:
        Fractional irradiance contributed by a scattered/refracted precursor that
        ramps in over ``refraction_lead_s`` before geometric first contact. The
        ramp has compact support (exactly zero earlier), so the eclipse floor is
        well defined. Set to 0 to disable.
    """

    solar_irradiance_w_m2: float = SOLAR_CONSTANT_W_M2
    eclipse_irradiance_w_m2: float = 1.4e-2
    t_first_contact_s: float = 6.0
    rise_duration_s: float = 12.0
    refraction_glow_frac: float = 0.03
    refraction_lead_s: float = 4.0

    # ---- disk geometry ---------------------------------------------------- #

    @staticmethod
    def _visible_disk_fraction(h_over_r: np.ndarray) -> np.ndarray:
        """Fraction of a unit disk above a chord at signed height ``-h`` from centre.

        ``h_over_r`` is the disk-centre height above the limb, in disk radii.
        Returns 0 at ``h/R = -1`` (fully occulted), 0.5 at 0, 1 at ``+1``.
        """
        u = np.clip(np.asarray(h_over_r, dtype=np.float64), -1.0, 1.0)
        return (np.arccos(-u) + u * np.sqrt(np.maximum(0.0, 1.0 - u * u))) / math.pi

    def disk_fraction(self, t_s):
        """Visible solar-disk fraction at time ``t_s`` (scalar or array)."""
        t = np.asarray(t_s, dtype=np.float64)
        # Disk centre sweeps -R -> +R linearly across the rise window.
        u = 2.0 * (t - self.t_first_contact_s) / self.rise_duration_s - 1.0
        return self._visible_disk_fraction(u)

    # ---- irradiance ------------------------------------------------------- #

    def irradiance(self, t_s):
        """Scene irradiance in W/m^2 at time ``t_s`` (scalar or array)."""
        t = np.asarray(t_s, dtype=np.float64)
        direct = self.disk_fraction(t)

        glow = 0.0
        if self.refraction_glow_frac > 0.0:
            # Smoothstep with compact support: zero before the precursor window.
            lead = max(self.refraction_lead_s, 1e-6)
            s = np.clip((t - (self.t_first_contact_s - lead)) / lead, 0.0, 1.0)
            glow = self.refraction_glow_frac * s * s * (3.0 - 2.0 * s)

        span = self.solar_irradiance_w_m2 - self.eclipse_irradiance_w_m2
        e = self.eclipse_irradiance_w_m2 + span * np.clip(direct + glow, 0.0, 1.0)
        return e if e.ndim else float(e)

    def log_irradiance(self, t_s):
        """Natural log of irradiance — the quantity a DVS pixel actually tracks."""
        return np.log(np.asarray(self.irradiance(t_s), dtype=np.float64))

    def log_rate(self, t_s, dt_s: float = 1e-3):
        """d(ln E)/dt in 1/s, by central difference."""
        t = np.asarray(t_s, dtype=np.float64)
        fwd = self.log_irradiance(t + 0.5 * dt_s)
        bwd = self.log_irradiance(t - 0.5 * dt_s)
        rate = (fwd - bwd) / dt_s
        return rate if rate.ndim else float(rate)

    # ---- derived quantities ----------------------------------------------- #

    @property
    def contrast_decades(self) -> float:
        """Total irradiance swing, in decades."""
        return math.log10(self.solar_irradiance_w_m2 / self.eclipse_irradiance_w_m2)

    @property
    def dynamic_range_db(self) -> float:
        """Total irradiance swing in dB, image-sensor convention (20*log10).

        Directly comparable with the >120 dB figure quoted for the IMX636 and with
        the ~60-70 dB typical of a scientific CMOS frame sensor. [CITE-REQUIRED]
        """
        return 20.0 * math.log10(self.solar_irradiance_w_m2 / self.eclipse_irradiance_w_m2)

    def event_storm_budget(
        self,
        n_pixels: int,
        contrast_threshold: float = 0.2,
        t_start_s: float = 0.0,
        t_end_s: float = 30.0,
        n_samples: int = 20000,
    ) -> dict:
        """Order-of-magnitude event budget for a global irradiance ramp.

        A logarithmic pixel emits one event per ``contrast_threshold`` of ln-intensity
        change, so the *total* count is set by the overall swing and the *peak rate*
        by the steepest part of the ramp.

        Returns a dict with ``total_events``, ``mean_rate_evs``, ``peak_rate_evs``
        and ``peak_time_s``.
        """
        t = np.linspace(t_start_s, t_end_s, n_samples)
        rate_per_px = np.abs(self.log_rate(t)) / contrast_threshold
        total_per_px = np.log(
            self.irradiance(t_end_s) / self.irradiance(t_start_s)
        ) / contrast_threshold
        duration = max(t_end_s - t_start_s, 1e-9)
        peak_idx = int(np.argmax(rate_per_px))
        return {
            "contrast_threshold": contrast_threshold,
            "n_pixels": n_pixels,
            "crossings_per_pixel": float(abs(total_per_px)),
            "total_events": float(abs(total_per_px) * n_pixels),
            "mean_rate_evs": float(abs(total_per_px) * n_pixels / duration),
            "peak_rate_evs": float(rate_per_px[peak_idx] * n_pixels),
            "peak_time_s": float(t[peak_idx]),
            "crest_factor": float(
                rate_per_px[peak_idx] * duration / max(abs(total_per_px), 1e-12)
            ),
        }


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #

@dataclass
class VibrationMode:
    """One structural mode of the array.

    ``kind`` selects the spatial shape:
      * ``"bending"`` — Euler-Bernoulli cantilever, ``order`` = 1, 2, 3, 4
      * ``"torsion"`` — fixed-free shaft, ``order`` = 1, 2, ...  (rotation about span)
    """

    name: str
    freq_hz: float
    damping_ratio: float
    amplitude_m: float
    kind: str = "bending"
    order: int = 1
    phase_rad: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in ("bending", "torsion"):
            raise ValueError(f"unknown mode kind: {self.kind!r}")
        if self.kind == "bending" and not 1 <= self.order <= len(CANTILEVER_BETA_L):
            raise ValueError(f"bending order {self.order} out of tabulated range")
        if self.damping_ratio <= 0.0 or self.damping_ratio >= 1.0:
            raise ValueError("damping_ratio must be in (0, 1)")

    @property
    def omega_n(self) -> float:
        return 2.0 * math.pi * self.freq_hz

    @property
    def omega_d(self) -> float:
        """Damped natural frequency (rad/s)."""
        return self.omega_n * math.sqrt(1.0 - self.damping_ratio ** 2)

    @property
    def damped_freq_hz(self) -> float:
        return self.omega_d / (2.0 * math.pi)

    def shape(self, xi) -> np.ndarray:
        """Mass-normalised-ish mode shape on the normalised span ``xi`` in [0, 1].

        Scaled so ``max|phi| == 1``. For ``"torsion"`` the returned value is the
        twist angle profile; the caller multiplies by the chordwise offset.
        """
        x = np.clip(np.asarray(xi, dtype=np.float64), 0.0, 1.0)
        if self.kind == "torsion":
            return np.sin((2 * self.order - 1) * math.pi * x / 2.0)

        bl = CANTILEVER_BETA_L[self.order - 1]
        sigma = (math.cosh(bl) + math.cos(bl)) / (math.sinh(bl) + math.sin(bl))
        b = bl * x
        phi = (np.cosh(b) - np.cos(b)) - sigma * (np.sinh(b) - np.sin(b))
        peak = np.max(np.abs(phi)) if phi.size else 1.0
        return phi / peak if peak > 0 else phi


@dataclass
class ThermalSnap:
    """Impulsive excitation at terminator crossing.

    Before ``t_snap_s`` the array oscillates at ``ambient_fraction`` of the modal
    amplitude (residual reaction-wheel / gimbal disturbance). At ``t_snap_s`` the
    rapid thermal gradient across the boom snaps the structure and each mode is
    excited to full amplitude, then decays at its own modal damping.
    """

    t_snap_s: float = 8.0
    ambient_fraction: float = 0.06
    rise_time_s: float = 0.15


@dataclass
class FlexibleArrayModel:
    """Cantilevered flexible solar array with a handful of lightly damped modes.

    Defaults are representative of a large deployable wing (ISS SAW / ROSA class):
    sub-hertz fundamental, damping well below 1%. Exact figures are
    [CITE-REQUIRED] and should be replaced with sourced values for publication.
    """

    span_m: float = 34.0
    chord_m: float = 11.6
    modes: List[VibrationMode] = field(default_factory=lambda: [
        VibrationMode("B1", freq_hz=0.082, damping_ratio=0.0035,
                      amplitude_m=0.145, kind="bending", order=1),
        VibrationMode("T1", freq_hz=0.310, damping_ratio=0.0050,
                      amplitude_m=0.048, kind="torsion", order=1, phase_rad=1.1),
        VibrationMode("B2", freq_hz=0.515, damping_ratio=0.0060,
                      amplitude_m=0.032, kind="bending", order=2, phase_rad=0.4),
        VibrationMode("B3", freq_hz=1.440, damping_ratio=0.0090,
                      amplitude_m=0.009, kind="bending", order=3, phase_rad=2.2),
    ])
    snap: ThermalSnap = field(default_factory=ThermalSnap)

    # ---- temporal envelope ------------------------------------------------ #

    def modal_coordinate(self, mode: VibrationMode, t_s: float) -> float:
        """Generalised coordinate q_k(t) in metres of tip displacement."""
        amb = self.snap.ambient_fraction * mode.amplitude_m
        q = amb * math.sin(mode.omega_n * t_s + mode.phase_rad)

        tau = t_s - self.snap.t_snap_s
        if tau >= 0.0:
            # Smooth switch-on over rise_time_s avoids a non-physical step.
            gate = 1.0 - math.exp(-tau / max(self.snap.rise_time_s, 1e-6))
            env = mode.amplitude_m * math.exp(-mode.damping_ratio * mode.omega_n * tau)
            q += gate * env * math.sin(mode.omega_d * tau + mode.phase_rad)
        return q

    def modal_coordinates(self, t_s: float) -> np.ndarray:
        return np.array([self.modal_coordinate(m, t_s) for m in self.modes])

    # ---- spatial field ---------------------------------------------------- #

    def displacement(self, xi, t_s: float, eta: float = 0.0) -> np.ndarray:
        """Out-of-plane displacement in metres.

        Parameters
        ----------
        xi:
            Normalised span coordinate in [0, 1] (0 = root/bus, 1 = tip).
        t_s:
            Time in seconds.
        eta:
            Normalised chord coordinate in [-0.5, 0.5]; only affects torsion modes.
        """
        x = np.asarray(xi, dtype=np.float64)
        total = np.zeros_like(x)
        for mode in self.modes:
            q = self.modal_coordinate(mode, t_s)
            if mode.kind == "torsion":
                total = total + q * mode.shape(x) * (eta * self.chord_m) / max(self.chord_m * 0.5, 1e-9)
            else:
                total = total + q * mode.shape(x)
        return total

    def tip_displacement(self, t_s: float) -> float:
        return float(self.displacement(np.array([1.0]), t_s)[0])

    # ---- reference values ------------------------------------------------- #

    def ground_truth(self) -> List[dict]:
        """Ground-truth modal table for scoring the estimator."""
        return [
            {
                "name": m.name,
                "kind": m.kind,
                "order": m.order,
                "freq_hz": m.freq_hz,
                "damped_freq_hz": m.damped_freq_hz,
                "damping_ratio": m.damping_ratio,
                "amplitude_m": m.amplitude_m,
            }
            for m in self.modes
        ]


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #

def _self_test() -> int:
    ok = True

    def check(label: str, condition: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= bool(condition)
        print(f"  [{'PASS' if condition else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))

    print("OrbitalSunriseProfile")
    sun = OrbitalSunriseProfile()
    check("eclipse floor exact before precursor",
          abs(sun.irradiance(0.0) - sun.eclipse_irradiance_w_m2) < 1e-12,
          f"E(0) = {sun.irradiance(0.0):.4g} W/m^2")
    check("full sun after rise",
          sun.irradiance(30.0) > 0.99 * sun.solar_irradiance_w_m2,
          f"E(30) = {sun.irradiance(30.0):.1f} W/m^2")
    check("half disk at mid-rise",
          abs(sun.disk_fraction(sun.t_first_contact_s + sun.rise_duration_s / 2) - 0.5) < 1e-9)
    check("monotonic ramp",
          bool(np.all(np.diff(sun.irradiance(np.linspace(0, 30, 3000))) >= -1e-9)))
    check("contrast decades", 4.0 < sun.contrast_decades < 6.0,
          f"{sun.contrast_decades:.2f} decades / {sun.dynamic_range_db:.0f} dB")

    budget = sun.event_storm_budget(n_pixels=1280 * 720)
    print(f"    storm budget @1280x720, C=0.2: "
          f"{budget['crossings_per_pixel']:.0f} crossings/px, "
          f"{budget['total_events'] / 1e6:.0f} Mev total, "
          f"peak {budget['peak_rate_evs'] / 1e6:.0f} Mev/s at t={budget['peak_time_s']:.2f} s")
    check("peak rate exceeds mean", budget["peak_rate_evs"] > 5 * budget["mean_rate_evs"],
          f"crest factor {budget['crest_factor']:.1f}")

    print("FlexibleArrayModel")
    array = FlexibleArrayModel()
    xi = np.linspace(0.0, 1.0, 200)
    for mode in array.modes:
        phi = mode.shape(xi)
        check(f"mode {mode.name} shape normalised", abs(np.max(np.abs(phi)) - 1.0) < 1e-9)
        check(f"mode {mode.name} clamped at root", abs(phi[0]) < 1e-9)

    b1 = array.modes[0]
    pre = abs(array.modal_coordinate(b1, array.snap.t_snap_s - 1.0))
    post = max(abs(array.modal_coordinate(b1, array.snap.t_snap_s + dt))
               for dt in np.linspace(0.5, 6.0, 60))
    check("thermal snap amplifies response", post > 5 * pre,
          f"pre {pre * 1e3:.2f} mm -> post {post * 1e3:.1f} mm")

    late = max(abs(array.modal_coordinate(b1, array.snap.t_snap_s + dt))
               for dt in np.linspace(300.0, 306.0, 60))
    check("free decay reduces amplitude", late < 0.6 * post,
          f"{post * 1e3:.1f} mm -> {late * 1e3:.1f} mm after 300 s")

    check("ground truth table complete", len(array.ground_truth()) == len(array.modes))

    print("\nSELF-TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
