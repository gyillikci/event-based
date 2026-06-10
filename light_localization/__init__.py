"""
Light Localization using Fluorescent Light Flicker Frequency Fingerprints
=========================================================================

Indoor positioning system that exploits the unique frequency characteristics
of fluorescent (and other) indoor lights as active markers. Each ceiling light
emits a characteristic flicker signature determined by its ballast type,
component tolerances, aging, and thermal state. By capturing these signatures
with a Prophesee event camera (DVS) and matching them against a calibrated
fingerprint database, the system estimates the receiver's indoor position.

Modules
-------
- light_fingerprint       : Calibration — capture and store per-light frequency signatures
- light_localizer         : Runtime — identify visible lights and estimate position
- light_localization_utils: Shared utilities — FFT, ROI extraction, geometry, matching
- capture_flicker_map     : Experiment script — record ceiling frequency maps
- evaluate_separability   : Analysis script — test if lights are distinguishable

References
----------
See LITERATURE_REVIEW.md and TECHNICAL_EVALUATION.md for detailed background.
"""

__version__ = "0.1.0"
