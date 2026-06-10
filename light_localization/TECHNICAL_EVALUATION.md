# Technical Evaluation: Event-Camera Indoor Localization via Light Flicker Fingerprinting

## 1. Executive Summary

This document evaluates the technical feasibility of using a **Prophesee event camera (DVS)**
to perform indoor localization by fingerprinting the flicker characteristics of unmodified
ceiling lights (fluorescent and LED). Based on literature evidence and the capabilities of our
Metavision SDK, the approach is **technically feasible** with several clear advantages over
prior work, but faces challenges that must be addressed through careful system design.

**Verdict: GO** — with the following caveats:
- Separability between fixtures must be empirically validated in the target environment
- Mixed fluorescent/LED environments need adaptive frequency range scanning
- Calibration workflow must be practical (< 10 min per room)

---

## 2. Signal Existence & Uniqueness

### 2.1 Fluorescent Lights — Why They Flicker

**Magnetic ballast fluorescent tubes** operate directly on mains AC (50/60 Hz). The gas
discharge extinguishes and re-ignites twice per cycle, producing strong flicker at **100 Hz
(Europe) or 120 Hz (USA)**. Each tube has a slightly different re-ignition delay depending on
gas composition aging, electrode wear, and tube temperature, creating unique harmonic profiles.

**Electronic ballast fluorescent tubes** convert mains AC to high-frequency AC (typically
**20–60 kHz**) using an LC resonant inverter circuit. The exact resonance frequency is:

```
f_res = 1 / (2π √(LC))
```

where L and C are the inductor and capacitor values. Manufacturing tolerances of ±5–10%
on each component yield inter-fixture frequency variations of:

```
Δf ≈ f_res × (ΔL/2L + ΔC/2C) ≈ f_res × 5–10%
```

For a 40 kHz ballast, this means **2–4 kHz variation** between fixtures — easily resolvable
by the event camera's µs-level temporal resolution.

**LED panels with drivers** may operate in several modes:
- **Constant-current DC drivers:** No flicker → invisible to our system (these become "holes"
  in the frequency map, which is itself a landmark pattern)
- **PWM-dimmed LEDs:** Flicker at the PWM frequency (100 Hz – 10 kHz), with unique duty cycles
- **Switch-mode drivers:** Emit EMI-induced flicker at the switching frequency (50–200 kHz)

### 2.2 Empirical Evidence of Uniqueness

From the literature (Section references to LITERATURE_REVIEW.md):

| Source | Finding |
|--------|---------|
| LiTell (Zhang & Zhang, 2016) | Adjacent same-model fluorescent fixtures differ by 200–2000 Hz |
| Munir & Dyo (2019) | 96% identification accuracy for same-manufacturer CFLs using harmonic structure |
| iLAMP (Zhu & Zhang, 2017) | Even "non-flickering" LED panels have detectable hidden frequency features |
| Wang et al. (ICRA 2022) | Fluorescent flicker produces rich harmonic structure in event streams |

**Assessment: HIGH confidence** that individual fixtures are distinguishable in frequency
domain, particularly when using harmonic ratios as additional discriminators.

### 2.3 Frequency Stability

| Factor | Impact | Mitigation |
|--------|--------|------------|
| Temperature | ~50–200 Hz/10°C drift for electronic ballasts | Use harmonic ratios (temperature-invariant) |
| Aging | Slow drift over months/years | Periodic re-calibration (quarterly) |
| Mains voltage | <0.1% frequency variation for electronic ballasts | Negligible |
| Warm-up | 1–5 min settling after switch-on | Calibrate only after warm-up |
| Same-model variance | 200–3000 Hz at 40 kHz fundamental | Sufficient for discrimination |

---

## 3. Event Camera Advantage Analysis

### 3.1 Comparison with Prior Receiver Technologies

| Property | Photodiode | Smartphone (RS) | Event Camera (DVS) |
|----------|-----------|-----------------|-------------------|
| Temporal resolution | ~1 MHz | ~30 kHz (via RS) | ~1 MHz (per-pixel) |
| Spatial resolution | 1 point | Megapixels | 640×480 – 1280×720 |
| Simultaneous lights | 1 (mixed signal) | Multiple (if visible) | Multiple (per-pixel) |
| Dynamic range | ~60 dB | ~60 dB | **>120 dB** |
| Power consumption | <10 mW | ~2 W | ~150 mW |
| Ambient immunity | Low (needs filtering) | Low | **High** (DC = no events) |
| Motion tolerance | N/A | Low (blur) | **High** (no motion blur) |
| Cost | < $10 | Commodity | ~$3,000 (EVK4) |

### 3.2 Why DVS Is Ideal for This Application

1. **Per-pixel frequency extraction:** Each pixel independently reports intensity changes.
   When pointing at a ceiling, each light fixture occupies a cluster of pixels. The DVS
   naturally performs spatial-spectral decomposition — each pixel cluster's event rate
   directly encodes the flicker frequency of the corresponding light.

2. **Natural ambient light immunity:** Sunlight and other constant light sources produce
   zero events (no intensity change). Only flickering sources generate events. This
   means the DVS acts as a **natural bandpass filter** that suppresses DC ambient — a
   critical advantage in environments with windows.

3. **Wide dynamic range (>120 dB):** Can detect faint flicker from distant lights while
   simultaneously monitoring bright nearby fixtures. Frame-based cameras saturate or
   under-expose in mixed conditions.

4. **Microsecond latency:** Position updates can occur within a single flicker cycle
   (< 10 ms for 100 Hz lights, < 50 µs for 20 kHz ballasts).

### 3.3 Prophesee EVK4 Specifications Relevant to This Application

| Parameter | EVK4 Value | Requirement | Margin |
|-----------|-----------|-------------|--------|
| Pixel resolution | 1280 × 720 | 320×240 sufficient | 4× |
| Temporal resolution | 1 µs | 50 µs for 20 kHz | 50× |
| Dynamic range | 124 dB | 90 dB for mixed indoor | 34 dB |
| Bandwidth | >10 kHz per pixel | 60 kHz for electronic ballasts | OK |
| Event throughput | 1.066 Gev/s | ~100 Mev/s ceiling scenario | 10× |
| Latency | < 100 µs | < 10 ms for real-time positioning | 100× |

**Note on bandwidth:** The event camera's per-pixel bandwidth determines the maximum
detectable flicker frequency. The EVK4's contrast sensitivity threshold of ~15% at
>10 kHz means electronic ballast frequencies (20–60 kHz) are within range, but the
signal-to-noise ratio decreases at higher frequencies. For magnetic ballast fixtures
(100–500 Hz), SNR is excellent.

---

## 4. SDK Capability Assessment

### 4.1 Core Algorithm: FrequencyMapAsyncAlgorithm

The Metavision SDK provides `FrequencyMapAsyncAlgorithm` — a hardware-accelerated per-pixel
frequency estimator. This is the **primary building block** for our system.

**How it works:**
- Maintains a per-pixel state machine that tracks ON/OFF event transitions
- Estimates the period between consecutive same-polarity events
- Applies temporal filtering (configurable `filter_length` = number of consistent periods
  required to confirm a frequency)
- Outputs a 2D frequency map (float32, Hz) at configurable update rate

**Configuration for ceiling light detection:**

```python
# Magnetic ballast fluorescent (100-500 Hz range)
freq_algo_low = FrequencyMapAsyncAlgorithm(
    width, height,
    filter_length=7,         # more periods for robust estimation
    min_freq=50,             # Hz — below 2× mains
    max_freq=500,            # Hz — covers harmonics
    diff_thresh_us=5000,     # relaxed jitter — older ballasts
)

# Electronic ballast fluorescent (20-60 kHz range)
freq_algo_high = FrequencyMapAsyncAlgorithm(
    width, height,
    filter_length=10,        # many cycles available at kHz rates
    min_freq=15000,          # Hz
    max_freq=70000,          # Hz
    diff_thresh_us=5,        # tight jitter — stable oscillators
)
```

**Dual-range strategy:** Since the target environment has mixed/unknown light types, we
run **two frequency map instances** — one for the low-frequency regime (magnetic ballasts,
LED PWM) and one for the high-frequency regime (electronic ballasts). This avoids the
Nyquist constraint of a single algorithm spanning 50 Hz to 70 kHz.

### 4.2 Supporting SDK Components

| Component | Role in Our System |
|-----------|-------------------|
| `DominantValueMapAlgorithm` | Extract dominant frequency per segmented light ROI |
| `HeatMapFrameGeneratorAlgorithm` | Visualize frequency map as color-coded ceiling image |
| `PeriodicFrameGenerationAlgorithm` | Generate event visualiation for debugging |
| `EventsIterator` | Primary event source (camera or recording file) |
| `LiveReplayEventsIterator` | Reproduce recordings at real-time speed for testing |

### 4.3 Custom Signal Processing Pipeline

For features beyond what the SDK provides (harmonic ratios, waveform shape), we build a
complementary FFT pipeline following the pattern established in `propeller_utils.py`:

```
Events → Per-ROI event accumulation → Time-series (event rate per Δt)
→ Hanning window → FFT → Peak detection → Harmonic extraction → Fingerprint vector
```

This runs on top of the SDK's spatial segmentation but uses numpy FFT for the
spectral analysis, giving us full control over the feature extraction.

---

## 5. System Architecture

### 5.1 Two-Phase Pipeline

```
╔══════════════════════════════════════════════════════════════╗
║                   PHASE 1: CALIBRATION                       ║
║                                                              ║
║  Camera → [FrequencyMapAsyncAlgorithm] → Frequency Map      ║
║              ↓                                               ║
║  Frequency Map → [Spatial Segmentation] → Light ROIs         ║
║              ↓                                               ║
║  Per-ROI Events → [FFT Pipeline] → Harmonic Features         ║
║              ↓                                               ║
║  Features + Known 3D Position → [FingerprintDatabase]        ║
║              ↓                                               ║
║  Save to calibration_data/ (JSON)                            ║
╚══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════╗
║                    PHASE 2: LOCALIZATION                     ║
║                                                              ║
║  Camera → [FrequencyMapAsyncAlgorithm] → Frequency Map      ║
║              ↓                                               ║
║  Frequency Map → [Spatial Segmentation] → Light ROIs         ║
║              ↓                                               ║
║  Per-ROI Features → [FingerprintMatcher] → Light IDs         ║
║              ↓                                               ║
║  Light IDs + Angular Position → [PositionEstimator]          ║
║              ↓                                               ║
║  Weighted Centroid / Triangulation → (x, y) estimate         ║
╚══════════════════════════════════════════════════════════════╝
```

### 5.2 Fingerprint Feature Vector

Each light fixture produces a fingerprint vector:

```
F = [f₀, a₁/a₀, a₂/a₀, a₃/a₀, a₄/a₀, duty_cycle, bandwidth, freq_band]
```

Where:
- `f₀` — fundamental frequency (Hz)
- `aₖ/a₀` — k-th harmonic amplitude relative to fundamental
- `duty_cycle` — ratio of positive to negative events per cycle
- `bandwidth` — spectral width of the fundamental peak (quality factor Q)
- `freq_band` — categorical: {low_magnetic, low_led, high_electronic}

### 5.3 Position Estimation Methods

**Method 1 — Weighted Centroid (Simple, Robust):**
```
pos = Σ wᵢ · posᵢ / Σ wᵢ
```
where wᵢ = similarity score × signal strength for light i. Requires ≥ 1 identified light
with known 3D position.

**Method 2 — Angle-of-Arrival Triangulation (High Precision):**
If camera intrinsics are calibrated, each identified light's pixel centroid gives an angular
bearing. With ≥ 2 lights at known ceiling positions, intersection of bearing lines yields
position. Expected accuracy: **10–30 cm** depending on ceiling height and angular separation.

**Method 3 — Fingerprint k-NN (RF Machine Learning):**
Full frequency map compared against database of calibration-point frequency maps using
distance metric in feature space. Requires dense calibration grid but handles complex
multi-light environments.

---

## 6. Challenge Analysis & Mitigations

### 6.1 Frequency Resolution vs. Update Rate

The frequency resolution of FFT-based analysis is:

```
Δf = 1 / T_observation
```

For 1 Hz resolution, we need 1 second of data. For 100 Hz resolution (sufficient to separate
electronic ballasts), T_obs = 10 ms. The SDK's `FrequencyMapAsyncAlgorithm` uses period
measurement (not FFT), achieving frequency estimates within a few cycles — much faster.

| Light Type | f₀ | Cycles in 100ms | Achievable Resolution |
|------------|-----|-----------------|---------------------|
| Magnetic FL | 100 Hz | 10 | ~10 Hz (period-based) |
| LED PWM | 1 kHz | 100 | ~10 Hz |
| Electronic FL | 40 kHz | 4000 | ~0.1 Hz |

**Assessment:** Frequency resolution is excellent for electronic ballasts, adequate for
magnetic. For magnetic ballasts, harmonic ratios provide the additional discrimination.

### 6.2 Spatial Segmentation of Light Sources

Each ceiling light occupies a roughly rectangular region of pixels. Segmentation approach:

1. Threshold the frequency map: pixels with valid frequency readings → binary mask
2. Morphological operations (dilate → erode) to clean up noise
3. Connected component analysis → individual light ROIs
4. ROI filtering: minimum area, aspect ratio constraints

This follows the established pattern in `FrequencyMapAnalyzer._find_clusters()` from the
propeller detector.

**Challenge:** Overlapping light footprints where two fixtures illuminate the same surface area.
**Mitigation:** Unlike surface-reflected light, our camera looks **directly at the ceiling**,
where each fixture is a spatially distinct emitting source. Even if their light cones overlap
on the floor, the fixtures themselves are physically separated points on the ceiling.

### 6.3 Mixed Light Environments

In environments with mixed light types (fluorescent + LED + natural):

| Light Type | Flicker | Event Camera Response | Treatment |
|------------|---------|----------------------|-----------|
| Magnetic ballast FL | 100/120 Hz + harmonics | Strong periodic events | Primary landmark |
| Electronic ballast FL | 20–60 kHz | Periodic events at kHz | Primary landmark |
| PWM-dimmed LED | 100 Hz – 10 kHz | Periodic events | Primary landmark |
| DC-driven LED | None | No events | Negative landmark (dark hole) |
| CFL | 50/100/120 Hz | Periodic events | Primary landmark |
| Natural daylight (window) | None | No events | Ignored (DC) |
| Computer monitor | 60–240 Hz | Periodic events | Filtered by position (below ceiling) |

**Key insight:** Even non-flickering (DC) lights provide information — their absence in the
frequency map creates a characteristic "fingerprint pattern" of the room's light layout.

### 6.4 Calibration Workflow

**Target: < 10 minutes per room**

1. **Walk-around capture** (3 min): Slowly walk through room pointing camera at ceiling.
   Record continuous event stream.
2. **Automatic segmentation** (30 sec): Process recording to identify and segment light fixtures.
3. **Semi-automatic labeling** (5 min): Operator confirms light positions on a floor plan.
   Alternatively, if a floor plan with light positions exists (common in facility management
   systems), this step is automatic.
4. **Fingerprint extraction** (1 min): Compute frequency fingerprint for each segmented fixture.
5. **Database storage** (instant): Save fingerprints with 3D coordinates to JSON.

---

## 7. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Insufficient frequency variation between fixtures | Low | High | Use harmonic ratios; fall back to intentional modulation |
| Frequency drift invalidates calibration | Medium | Medium | Periodic re-calibration; temperature-normalized features |
| Electronic ballast kHz exceeds DVS bandwidth | Low | High | Verify EVK4 contrast sensitivity at 20–60 kHz |
| All lights are DC-driven LEDs (no flicker) | Medium | High | Negative landmark pattern; intentional modulation as fallback |
| Computational cost too high for real-time | Low | Medium | SDK algorithm is C++/optimized; per-pixel frequency map is efficient |
| Calibration too labor-intensive | Low | Medium | Semi-automatic workflow; one-time per room |

---

## 8. Implementation Roadmap

### Phase 1: Signal Validation (This Sprint)
- [ ] Record ceiling event data in 2–3 rooms with different light types
- [ ] Visualize per-pixel frequency maps
- [ ] Verify that individual fixtures produce distinct frequency clusters
- [ ] Compute separability metrics (Fisher discriminant, confusion matrix)

### Phase 2: Fingerprinting System
- [ ] Implement automatic light fixture segmentation
- [ ] Build FFT-based harmonic feature extraction
- [ ] Create fingerprint database with persistence
- [ ] Test identification accuracy with leave-one-out cross-validation

### Phase 3: Localization
- [ ] Implement weighted centroid position estimator
- [ ] Add AOA triangulation for calibrated camera setups
- [ ] Evaluate positioning accuracy vs. ground truth
- [ ] Compare against WiFi RSSI baseline

### Phase 4: Optimization & Robustness
- [ ] Real-time performance optimization
- [ ] Temporal tracking with Kalman filter
- [ ] Multi-room handoff
- [ ] Handle light switching (fixtures turning on/off)

---

## 9. Conclusion

The proposed approach of using a Prophesee event camera to identify unmodified ceiling lights
by their flicker frequency characteristics is **technically sound and well-supported by
literature**. The key strengths are:

1. **Proven signal:** Multiple papers confirm fluorescent lights have unique, stable frequencies
2. **Proven receiver:** Chen et al. (2020) validated DVS for frequency-based VLP
3. **SDK support:** FrequencyMapAsyncAlgorithm provides the core per-pixel frequency extraction
4. **Novel contribution:** No existing work combines DVS + unmodified lights + fingerprinting

The primary risk is that the target environment may contain predominantly DC-driven LED panels
with insufficient flicker. This can be mitigated by the evaluate_separability.py experiment
script, which serves as a **go/no-go gate** before investing in the full localization pipeline.
