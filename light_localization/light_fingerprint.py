"""
Light Fingerprint — Calibration module for ceiling light frequency signatures.

Captures, stores, and manages frequency fingerprints of individual ceiling lights.
Each fingerprint contains the light's frequency characteristics (fundamental frequency,
harmonic ratios, spectral bandwidth) along with its known 3D position in the room.

Usage:
    # Build a fingerprint database
    db = FingerprintDatabase()
    fp = LightFingerprint(
        light_id="room1_light_A",
        position_3d=(2.5, 3.0, 2.8),
        room_id="room1",
    )
    fp.set_frequency_features(features_dict)
    db.add(fp)
    db.save("calibration_data/room1.json")

    # Load and query
    db2 = FingerprintDatabase.load("calibration_data/room1.json")
    matches = db2.find_nearest(query_features, top_k=3)
"""

import json
import os
import time
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional

from light_localization_utils import frequency_distance


# ---------------------------------------------------------------------------
#  LightFingerprint dataclass
# ---------------------------------------------------------------------------

@dataclass
class LightFingerprint:
    """Frequency fingerprint of a single ceiling light fixture.

    Attributes:
        light_id: Unique identifier string (e.g., "room1_light_A").
        position_3d: Known (x, y, z) position in meters (room coordinate frame).
        room_id: Room identifier for multi-room deployments.
        pixel_centroid: (cx, cy) in the calibration image (for reference).
        fundamental_freq: Dominant flicker frequency in Hz.
        harmonic_ratios: Amplitude ratios of harmonics relative to fundamental.
        spectral_bandwidth: 3dB bandwidth of the fundamental peak in Hz.
        peak_snr: Signal-to-noise ratio of the fundamental peak.
        peak_magnitude: Absolute magnitude of the fundamental peak.
        freq_band: Categorical label — 'low_magnetic', 'low_led', 'high_electronic'.
        duty_cycle: Ratio of positive to total events per flicker cycle.
        pixel_count: Number of active pixels in the calibration frequency map.
        calibration_timestamp: Unix timestamp of when the fingerprint was captured.
        notes: Free-text field for operator notes.
    """
    light_id: str
    position_3d: tuple = (0.0, 0.0, 0.0)
    room_id: str = ""
    pixel_centroid: tuple = (0.0, 0.0)

    # Frequency features (populated after calibration)
    fundamental_freq: float = 0.0
    harmonic_ratios: list = field(default_factory=list)
    spectral_bandwidth: float = 0.0
    peak_snr: float = 0.0
    peak_magnitude: float = 0.0
    freq_band: str = "unknown"
    duty_cycle: float = 0.5
    pixel_count: int = 0

    # Metadata
    calibration_timestamp: float = 0.0
    notes: str = ""

    def set_frequency_features(self, features):
        """Populate frequency features from a FrequencyFeatureExtractor result.

        Args:
            features: Dict from FrequencyFeatureExtractor.extract_features().
        """
        self.fundamental_freq = features.get("fundamental_freq", 0.0)
        self.harmonic_ratios = features.get("harmonic_ratios", [])
        self.spectral_bandwidth = features.get("spectral_bandwidth", 0.0)
        self.peak_snr = features.get("peak_snr", 0.0)
        self.peak_magnitude = features.get("peak_magnitude", 0.0)
        self.calibration_timestamp = time.time()

        # Classify frequency band
        f = self.fundamental_freq
        if f <= 0:
            self.freq_band = "unknown"
        elif f < 500:
            self.freq_band = "low_magnetic"  # magnetic ballast or LED PWM
        elif f < 15000:
            self.freq_band = "low_led"       # LED driver switching
        else:
            self.freq_band = "high_electronic"  # electronic ballast fluorescent

    def set_from_roi(self, roi):
        """Populate basic fields from a LightROIExtractor result.

        Args:
            roi: Dict from LightROIExtractor.extract().
        """
        self.pixel_centroid = (roi["centroid_x"], roi["centroid_y"])
        self.pixel_count = roi["pixel_count"]
        self.fundamental_freq = roi.get("median_freq", 0.0)

    def feature_vector(self):
        """Return a compact feature vector for distance computation.

        Returns:
            Dict compatible with frequency_distance().
        """
        return {
            "fundamental_freq": self.fundamental_freq,
            "harmonic_ratios": self.harmonic_ratios,
            "spectral_bandwidth": self.spectral_bandwidth,
        }

    def is_valid(self):
        """Check if this fingerprint has been properly calibrated."""
        return self.fundamental_freq > 0 and self.peak_snr > 0

    def to_dict(self):
        """Serialize to a JSON-compatible dict."""
        return {
            "light_id": self.light_id,
            "position_3d": list(self.position_3d),
            "room_id": self.room_id,
            "pixel_centroid": list(self.pixel_centroid),
            "fundamental_freq": self.fundamental_freq,
            "harmonic_ratios": list(self.harmonic_ratios),
            "spectral_bandwidth": self.spectral_bandwidth,
            "peak_snr": self.peak_snr,
            "peak_magnitude": self.peak_magnitude,
            "freq_band": self.freq_band,
            "duty_cycle": self.duty_cycle,
            "pixel_count": self.pixel_count,
            "calibration_timestamp": self.calibration_timestamp,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d):
        """Deserialize from a dict (e.g., loaded from JSON)."""
        return cls(
            light_id=d["light_id"],
            position_3d=tuple(d.get("position_3d", (0, 0, 0))),
            room_id=d.get("room_id", ""),
            pixel_centroid=tuple(d.get("pixel_centroid", (0, 0))),
            fundamental_freq=d.get("fundamental_freq", 0.0),
            harmonic_ratios=d.get("harmonic_ratios", []),
            spectral_bandwidth=d.get("spectral_bandwidth", 0.0),
            peak_snr=d.get("peak_snr", 0.0),
            peak_magnitude=d.get("peak_magnitude", 0.0),
            freq_band=d.get("freq_band", "unknown"),
            duty_cycle=d.get("duty_cycle", 0.5),
            pixel_count=d.get("pixel_count", 0),
            calibration_timestamp=d.get("calibration_timestamp", 0.0),
            notes=d.get("notes", ""),
        )


# ---------------------------------------------------------------------------
#  FingerprintDatabase — persistent collection of light fingerprints
# ---------------------------------------------------------------------------

class FingerprintDatabase:
    """In-memory database of light fingerprints with JSON persistence.

    Supports:
      - Add/remove individual fingerprints
      - Save/load to/from JSON files
      - Nearest-neighbor query in frequency feature space
      - Filter by room, frequency band, etc.

    Typical workflow:
      1. Calibration phase: add fingerprints with known positions
      2. Save database to JSON
      3. Localization phase: load database, query against observed features
    """

    def __init__(self):
        self.fingerprints = {}   # light_id → LightFingerprint
        self.metadata = {
            "version": "1.0",
            "created": time.time(),
            "description": "",
        }

    def add(self, fp):
        """Add or update a fingerprint.

        Args:
            fp: LightFingerprint instance.
        """
        if not isinstance(fp, LightFingerprint):
            raise TypeError(f"Expected LightFingerprint, got {type(fp)}")
        self.fingerprints[fp.light_id] = fp

    def remove(self, light_id):
        """Remove a fingerprint by ID."""
        self.fingerprints.pop(light_id, None)

    def get(self, light_id):
        """Get a fingerprint by ID, or None."""
        return self.fingerprints.get(light_id)

    def list_ids(self):
        """Return list of all light_id strings."""
        return list(self.fingerprints.keys())

    def list_by_room(self, room_id):
        """Return fingerprints for a specific room."""
        return [fp for fp in self.fingerprints.values() if fp.room_id == room_id]

    def list_by_band(self, freq_band):
        """Return fingerprints in a specific frequency band."""
        return [fp for fp in self.fingerprints.values() if fp.freq_band == freq_band]

    def find_nearest(self, query_features, top_k=3, room_id=None,
                     freq_weight=0.5, harmonic_weight=0.3, bandwidth_weight=0.2):
        """Find the top-k nearest fingerprints to a query.

        Args:
            query_features: Dict with keys fundamental_freq, harmonic_ratios,
                            spectral_bandwidth (from FrequencyFeatureExtractor).
            top_k: Number of nearest neighbors to return.
            room_id: If set, restrict search to this room.
            freq_weight, harmonic_weight, bandwidth_weight: Distance metric weights.

        Returns:
            List of (LightFingerprint, distance) tuples, sorted by distance ascending.
        """
        candidates = self.fingerprints.values()
        if room_id is not None:
            candidates = [fp for fp in candidates if fp.room_id == room_id]

        distances = []
        for fp in candidates:
            if not fp.is_valid():
                continue
            dist = frequency_distance(
                query_features, fp.feature_vector(),
                freq_weight=freq_weight,
                harmonic_weight=harmonic_weight,
                bandwidth_weight=bandwidth_weight,
            )
            distances.append((fp, dist))

        distances.sort(key=lambda x: x[1])
        return distances[:top_k]

    def find_by_frequency(self, freq_hz, tolerance_hz=500.0, room_id=None):
        """Find fingerprints within a frequency tolerance.

        Args:
            freq_hz: Target frequency in Hz.
            tolerance_hz: Maximum absolute frequency difference.
            room_id: If set, restrict search to this room.

        Returns:
            List of LightFingerprint instances within tolerance.
        """
        candidates = self.fingerprints.values()
        if room_id is not None:
            candidates = [fp for fp in candidates if fp.room_id == room_id]

        return [fp for fp in candidates
                if fp.is_valid() and abs(fp.fundamental_freq - freq_hz) <= tolerance_hz]

    def compute_distance_matrix(self):
        """Compute pairwise distance matrix between all fingerprints.

        Returns:
            (ids, matrix): List of light_ids and NxN distance matrix.
        """
        fps = [fp for fp in self.fingerprints.values() if fp.is_valid()]
        ids = [fp.light_id for fp in fps]
        n = len(fps)
        matrix = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            for j in range(i + 1, n):
                d = frequency_distance(fps[i].feature_vector(), fps[j].feature_vector())
                matrix[i, j] = d
                matrix[j, i] = d

        return ids, matrix

    def save(self, filepath):
        """Save the database to a JSON file.

        Args:
            filepath: Path to the output JSON file.
        """
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        data = {
            "metadata": self.metadata,
            "fingerprints": [fp.to_dict() for fp in self.fingerprints.values()],
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath):
        """Load a database from a JSON file.

        Args:
            filepath: Path to the input JSON file.

        Returns:
            FingerprintDatabase instance.
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        db = cls()
        db.metadata = data.get("metadata", db.metadata)
        for fp_dict in data.get("fingerprints", []):
            fp = LightFingerprint.from_dict(fp_dict)
            db.add(fp)
        return db

    def summary(self):
        """Return a human-readable summary string."""
        n = len(self.fingerprints)
        valid = sum(1 for fp in self.fingerprints.values() if fp.is_valid())
        rooms = set(fp.room_id for fp in self.fingerprints.values() if fp.room_id)
        bands = {}
        for fp in self.fingerprints.values():
            bands[fp.freq_band] = bands.get(fp.freq_band, 0) + 1

        lines = [
            f"FingerprintDatabase: {n} lights ({valid} valid)",
            f"  Rooms: {sorted(rooms) if rooms else 'none'}",
            f"  Bands: {bands}",
        ]
        if valid > 0:
            freqs = [fp.fundamental_freq for fp in self.fingerprints.values() if fp.is_valid()]
            lines.append(f"  Freq range: {min(freqs):.1f} – {max(freqs):.1f} Hz")
        return "\n".join(lines)

    def __len__(self):
        return len(self.fingerprints)

    def __repr__(self):
        return f"FingerprintDatabase({len(self)} lights)"
