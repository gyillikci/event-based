# Light Localization — Indoor Positioning via Ceiling Light Flicker Fingerprinting

Indoor localization system that exploits the unique frequency characteristics of ceiling
lights (fluorescent, LED, CFL) as active markers. Each light fixture emits a characteristic
flicker signature determined by its ballast/driver electronics, enabling infrastructure-free
positioning using a Prophesee event camera (DVS).

## Concept

```
  ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ Light A  │    │ Light B  │    │ Light C  │    ← Ceiling lights (unmodified)
  │ f=40.1kHz│    │ f=38.7kHz│    │ f=42.3kHz│       Each has unique flicker frequency
  └──┬───────┘    └──┬───────┘    └──┬───────┘
     │ flicker       │ flicker       │ flicker
     ▼               ▼               ▼
  ┌──────────────────────────────────────────┐
  │         Prophesee Event Camera           │    ← Per-pixel µs-resolution
  │         (pointing at ceiling)            │       flicker detection
  └──────────────────┬───────────────────────┘
                     │ events
                     ▼
  ┌──────────────────────────────────────────┐
  │   FrequencyMapAsyncAlgorithm (SDK)       │    ← Per-pixel frequency map
  └──────────────────┬───────────────────────┘
                     │ freq_map
                     ▼
  ┌──────────────────────────────────────────┐
  │   Light Segmentation → Fingerprinting    │    ← Identify individual fixtures
  │   → Database Matching → Position Est.    │       from frequency signatures
  └──────────────────┬───────────────────────┘
                     │
                     ▼
               (x, y) position
```

## Why It Works

1. **Fluorescent lights flicker** at frequencies set by their ballast electronics (20–60 kHz
   for electronic ballasts, 100/120 Hz for magnetic ballasts)
2. **Manufacturing tolerances** (±5–10% on LC components) make each fixture's frequency unique
3. **Event cameras detect flicker natively** — each pixel reports µs-resolution intensity changes
4. **Sunlight/DC sources are invisible** to the event camera → natural ambient immunity

## Project Structure

```
light_localization/
├── README.md                      # This file
├── LITERATURE_REVIEW.md           # Survey of 15 key papers
├── TECHNICAL_EVALUATION.md        # Feasibility analysis & design decisions
├── __init__.py                    # Package init
│
├── light_localization_utils.py    # Shared utilities: ROI extraction, FFT, geometry
├── light_fingerprint.py           # LightFingerprint + FingerprintDatabase classes
├── light_localizer.py             # LightLocalizer: main runtime positioning facade
│
├── capture_flicker_map.py         # Calibration tool: record ceiling frequency maps
├── evaluate_separability.py       # Go/no-go experiment: can we distinguish lights?
│
├── calibration_data/              # Stored fingerprint databases (JSON)
├── recordings/                    # Raw event recordings
└── results/                       # Output plots, metrics, reports
```

## Quick Start

### 1. Activate Environment

```powershell
$env:PATH = "C:\Users\z003n5uc\Desktop\event-based\bin;" + $env:PATH
.\.venv\Scripts\Activate.ps1
```

### 2. Capture Ceiling Light Fingerprints

Point the event camera at the ceiling and run:

```powershell
# Magnetic ballast fluorescent (100-500 Hz)
python light_localization\capture_flicker_map.py --room-id office1

# Electronic ballast fluorescent (20-60 kHz) — requires faster sampling
python light_localization\capture_flicker_map.py --room-id office1 \
    --min-freq 15000 --max-freq 70000 --delta-t 5

# From a recording
python light_localization\capture_flicker_map.py -i recordings\ceiling.raw --room-id lab
```

Press **S** to save fingerprints, **Q** to quit. Fingerprints are saved to
`calibration_data/<room_id>_fingerprints.json`.

### 3. Evaluate Separability (Go/No-Go)

```powershell
# From saved fingerprints
python light_localization\evaluate_separability.py \
    --database calibration_data\office1_fingerprints.json

# Live capture + analyze
python light_localization\evaluate_separability.py --capture-seconds 15
```

This produces a report with Fisher discriminant ratio, pairwise distance matrix, and a
GO/MARGINAL/NO-GO verdict.

### 4. Run Localization (After Calibration)

```python
from light_localization.light_localizer import LightLocalizer

localizer = LightLocalizer(
    database_path="calibration_data/office1_fingerprints.json",
    width=1280, height=720,
    min_freq=50, max_freq=500,
)

# In your event processing loop:
# result = localizer.process_frequency_map(freq_map)
# print(localizer.get_status())
```

## Key Classes

| Class | Module | Role |
|-------|--------|------|
| `LightROIExtractor` | `light_localization_utils.py` | Segments light fixtures from frequency maps |
| `FrequencyFeatureExtractor` | `light_localization_utils.py` | FFT-based harmonic feature extraction |
| `LightFingerprint` | `light_fingerprint.py` | Dataclass for per-light frequency signature |
| `FingerprintDatabase` | `light_fingerprint.py` | JSON-persistent collection of fingerprints |
| `LightMatcher` | `light_localizer.py` | Matches observed ROIs to known fingerprints |
| `PositionEstimator` | `light_localizer.py` | Weighted centroid / AOA triangulation |
| `LightLocalizer` | `light_localizer.py` | Main facade combining all components |

## Dependencies

All dependencies are already in the project's virtual environment:
- numpy, scipy, opencv-python
- Prophesee Metavision SDK (metavision_core, metavision_sdk_analytics, etc.)
- matplotlib (optional, for publication-quality plots in evaluate_separability)

## Literature

See [LITERATURE_REVIEW.md](LITERATURE_REVIEW.md) for a comprehensive survey of 15 papers.
Key references:
- **LiTell** (Zhang & Zhang, MobiCom 2016) — 215 citations — proved fluorescent lights have unique frequencies
- **Chen et al.** (IEEE Sensors 2020) — 61 citations — first event-camera VLP system
- **Munir & Dyo** (IEEE Sensors 2019) — 96% identification accuracy for same-model CFLs
- **Wang et al.** (ICRA 2022) — characterized fluorescent flicker in event camera data

## Status

- [x] Literature review & technical evaluation
- [x] Core utilities (ROI extraction, FFT, geometry)
- [x] Fingerprint data model & database persistence
- [x] Calibration capture tool
- [x] Separability analysis tool
- [x] Runtime localization engine
- [ ] End-to-end validation with real ceiling recordings
- [ ] Position accuracy benchmarking vs. ground truth
- [ ] Multi-room handoff
- [ ] Real-time Kalman filter tracking
