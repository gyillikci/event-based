"""
Event-Based Vibration Monitoring of Flexible Solar Arrays Through Orbital Sunrise
================================================================================

IAC 2026 study branch.

The dominant excitation of a large flexible solar array is *thermal snap* at the
terminator crossing: the structure is shocked the instant the spacecraft leaves
eclipse. That is precisely the instant a conventional frame camera is unusable —
the scene irradiance climbs four to six decades in seconds, far beyond the ~60-70
dB intrinsic dynamic range of a CMOS imager, and auto-exposure needs hundreds of
milliseconds to reconverge.

A Prophesee EVK4 (IMX636) does not photo-saturate: its logarithmic front end has
>120 dB of dynamic range. It does, however, suffer *readout* saturation. A global
irradiance ramp drives every pixel across its contrast threshold tens of times,
producing an event storm that is three to five decades larger than the vibration
signal we are trying to measure.

The approach taken here treats the ON/OFF polarity channel as a differential pair:

  * the sunrise ramp is **common mode** — every pixel, same polarity, spatially
    uniform;
  * panel vibration is **differential** — spatially localised to structural edges,
    balanced in ON/OFF, and anti-correlated across the edge normal.

Rejecting the common mode leaves a clean sub-pixel displacement signal from which
modal frequencies, damping ratios and operational deflection shapes are recovered.

Modules
-------
- space_environment      : Orbital sunrise irradiance model + flexible-array modal model
- orbital_sunrise_sim    : Synthetic event generator with ground truth (no camera needed)
- glare_guard            : Sensor-level and algorithmic saturation mitigation
- event_vibrometer       : Sub-pixel displacement gauges from signed event integration
- modal_analysis         : Welch PSD, half-power damping, log-decrement, deflection shapes
- monitor_solar_array    : Main application (live EVK4 / recording / synthetic)
- evaluate_glare_rejection : Benchmark harness producing paper-ready metrics and figures

Only ``monitor_solar_array`` imports the Metavision SDK. Everything else is
NumPy-only so the algorithms can be unit-tested on any interpreter.

References
----------
See LITERATURE_REVIEW.md and TECHNICAL_EVALUATION.md.
"""

__version__ = "0.1.0"
