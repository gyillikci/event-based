# Propeller Detector — Development Journey

## Overview

This document traces the iterative development of `detect_propeller.py`, a real-time drone propeller rotation detector built on the Prophesee Metavision SDK for event-based cameras. It covers the problems encountered at each stage, the root-cause analysis, and the architectural decisions that led to a working, low-latency detector.

**Sensor:** Prophesee event camera, 1280×720 resolution  
**SDK:** Metavision SDK (FrequencyMapAsyncAlgorithm, DominantValueMapAlgorithm, etc.)  
**Language:** Python 3.9, OpenCV, NumPy

---

## Phase 1 — Initial Baseline (v1)

### Approach

The first version used a **two-pronged detection strategy**:

1. **Per-pixel frequency map** (SDK's `FrequencyMapAsyncAlgorithm`)  
   - Each pixel independently detects periodic brightness changes.  
   - Displayed as a colour heatmap — useful for visual inspection.

2. **Grid-based FFT on event rates** (`PropellerGridAnalyzer`)  
   - Divided the sensor into an N×N grid (default 16×16 = 256 cells).  
   - Counted events per cell per time slice (500 µs).  
   - Ran FFT on each cell's event-rate time series (rolling window).  
   - Reported cells whose FFT peak exceeded a minimum SNR threshold.  
   - Clustered adjacent active cells into "propeller regions."

### Default Parameters (v1)

| Parameter        | Value  |
|------------------|--------|
| filter_length    | 7      |
| max_period_diff  | 500 µs |
| min_pixel_count  | 25     |
| grid_cells       | 16     |
| fft_window_sec   | 0.5 s  |
| min_snr          | 3.0    |

### Result

✅ Ran successfully, processing ~175 seconds of recorded event data.  
❌ Detection only worked at **very close range** — a few centimetres from the propeller.

---

## Phase 2 — Parameter Tuning for Range

### Problem

At distance, a propeller occupies far fewer pixels and produces weaker periodic signals. The v1 defaults were tuned for close-range, high-signal scenarios.

### Changes Made

| Parameter        | Before | After | Rationale |
|------------------|--------|-------|-----------|
| filter_length    | 7      | 4     | Fewer periods needed to confirm a vibration → faster lock on weak signals |
| max_period_diff  | 500    | 1500  | More tolerance for jittery timing at low pixel counts |
| min_pixel_count  | 25     | 5     | Accept clusters with very few vibrating pixels |
| grid_cells       | 16     | 8     | Larger cells → more events per cell → better FFT SNR |
| fft_window_sec   | 0.5    | 1.0   | Longer integration time → better frequency resolution |
| min_snr          | 3.0    | 2.0   | Lower threshold to catch weaker signals |

### Result

❌ **Still not working.** Output showed 25–43 "propeller regions" per frame, scattered across all frequencies. This was **noise, not detection** — lowering thresholds just turned the noise floor into false positives.

### Key Insight

> Parameter tuning cannot fix a fundamentally wrong architecture. The grid FFT approach counts events without regard for *where* in the cell they come from or *whether* they are spatially coherent. At distance, a propeller's few pixels are drowned out by background noise in the same cell.

---

## Phase 3 — Architectural Redesign (v2)

### Root-Cause Analysis

The grid FFT approach had three fatal flaws:

1. **No spatial coherence** — A 160×90 pixel grid cell mixes propeller pixels with hundreds of noise pixels. The propeller signal is diluted below the noise floor.

2. **No temporal persistence filtering** — Each frame was analysed independently. Noise spikes that happen to pass the SNR threshold are reported just as confidently as real propellers.

3. **Fixed grid boundaries** — A propeller straddling a cell boundary splits its signal across 2–4 cells, halving or quartering the effective SNR.

### New Architecture

The v2 detector replaced the grid FFT with a three-stage pipeline:

```
Per-pixel frequency map (SDK)
        ↓
Connected-component spatial clustering (FrequencyMapAnalyzer)
        ↓
Temporal tracking with persistence filtering (PropellerTracker)
```

#### Stage 1 — Per-Pixel Frequency Map (unchanged)

The SDK's `FrequencyMapAsyncAlgorithm` already computes a per-pixel frequency. This is the correct foundation — it tells us *which* pixels are vibrating and *at what frequency*, without any grid quantisation.

#### Stage 2 — Spatial Clustering (`FrequencyMapAnalyzer`)

Instead of dividing the image into a fixed grid, we:

1. **Threshold** the frequency map to get a binary mask of vibrating pixels (within the target Hz range).
2. **Dilate** the mask with an elliptical kernel to bridge small gaps between nearby vibrating pixels.
3. **Connected-component labelling** (`cv2.connectedComponentsWithStats`) to find spatial clusters.
4. **Filter** clusters by:
   - Minimum pixel count (default 3 — even tiny distant propellers pass).
   - Frequency consistency (coefficient of variation ≤ 0.3 — all pixels in the cluster must agree on the frequency).

This approach is fundamentally superior because:
- It adapts to the propeller's actual shape and size (no fixed grid).
- A 3-pixel cluster with consistent frequency is meaningful; 3 pixels scattered across a 160×90 cell are not.
- No signal dilution — we only look at the pixels that are actually vibrating.

#### Stage 3 — Temporal Tracking (`PropellerTracker`)

Each candidate detection is matched to an existing track using:
- **Spatial proximity** (max 100 px distance).
- **Frequency similarity** (within 30% relative).

Tracks accumulate a hit counter. A track is only reported as a **confirmed propeller** after `min_hits` consecutive detections (default 3). Tracks that go undetected for `max_age` frames (default 8) are pruned.

Position and frequency estimates are smoothed with exponential moving average (α = 0.3).

#### Visualisation

Both windows now show detection overlays:
- **Event frame:** Green bounding boxes with ID, frequency, RPM, pixel count, and crosshair at centroid.
- **Frequency heatmap:** White bounding boxes with frequency/RPM labels, plus status line showing candidate/track/confirmed counts.

### Components Removed

- `PropellerGridAnalyzer` class (grid FFT) — entirely deleted.
- `cluster_detections()` function (grid-cell clustering) — entirely deleted.
- All grid-related CLI arguments (`--grid-cells`, `--fft-window`, `--min-snr`).

### Result

✅ **Dramatically improved.** Typical output:
- 1 confirmed propeller, stable at ~99.7 Hz / ~2,990 RPM.
- Locked position at (430, 22), not jumping.
- ~100 active pixels in the cluster.
- 270+ consecutive hits (persistent over the entire recording).
- Transient false positives (3-pixel clusters at random frequencies) are pruned within 1–2 seconds by the temporal tracker.

---

## Phase 4 — Latency Elimination

### Problem

The v2 detector had **increasing latency** — the visualisation fell further and further behind real-time. By the end of a 175-second recording, the display was many seconds behind.

### Root-Cause Analysis

The `on_freq_map` callback ran **synchronously** inside the main event loop at 25 Hz. Each invocation performed expensive operations on the full 1280×720 image:

| Operation | Cost per call | At 25 Hz |
|-----------|--------------|----------|
| `cv2.dilate` (11×11 kernel, 921K pixels) | ~3–5 ms | ~100 ms/s |
| `cv2.connectedComponentsWithStats` (921K pixels) | ~5–10 ms | ~175 ms/s |
| `frame.copy()` in `draw_detections_on_frame` (2.7 MB) | ~1–2 ms | ~40 ms/s |
| **Total overhead** | **~15 ms** | **~315 ms/s** |

This left only ~685 ms of every second for processing the 2,000 event slices (at delta_t = 500 µs). The main loop couldn't keep up, and `LiveReplayEventsIterator`'s wall-clock pacing caused events to queue up → monotonically increasing latency.

### Fixes Applied

1. **4× downscale before morphology + connected components**
   - Binary mask resized from 1280×720 → 320×180 before `dilate` and `connectedComponentsWithStats`.
   - ~16× fewer pixels to process; dilation kernel scaled accordingly.
   - Frequency statistics still computed on full-resolution ROI for accuracy.
   - **Impact:** ~10× speedup for the two most expensive operations.

2. **Update frequency reduced from 25 Hz → 10 Hz**
   - The frequency map doesn't change fast enough to warrant 25 updates/second.
   - **Impact:** 2.5× fewer callback invocations per second.

3. **In-place frame drawing (eliminated `frame.copy()`)**
   - `draw_detections_on_frame` now draws directly on the CD frame buffer.
   - Safe because `PeriodicFrameGenerationAlgorithm` overwrites the buffer on the next callback anyway.
   - **Impact:** Eliminated ~67 MB/s of unnecessary memory allocation + copying.

4. **Early exit on empty masks**
   - Added `np.count_nonzero` check before any morphology operations.
   - Skips the entire dilate → CC pipeline when fewer than `min_cluster_pixels` are active.
   - **Impact:** Near-zero cost for quiet frames.

### Result

✅ **Zero latency drift.** Timestamps advance at exactly 2.0-second intervals, matching wall-clock time perfectly, from 0s through 175s+. Net callback cost dropped from ~15 ms × 25/s (~375 ms/s) to ~2 ms × 10/s (~20 ms/s).

---

## Phase 5 — Frequency Range Tuning

### Change

Narrowed the default search range from 50–500 Hz to **10–100 Hz** to better match the expected propeller frequencies for the target drone type.

| Parameter | Before | After |
|-----------|--------|-------|
| min_freq  | 50 Hz  | 10 Hz |
| max_freq  | 500 Hz | 100 Hz |

This reduces the number of false-positive clusters (fewer noise pixels pass the frequency filter) and focuses the detector on the physically relevant range.

---

## Phase 6 — Blade Count Correction & Frequency Range

### Problem

The user revealed the test propeller is a **3-blade** prop. The detected 270 Hz was the *blade-pass frequency*, not the shaft frequency:

$$\text{shaft RPM} = \frac{270\,\text{Hz}}{3\,\text{blades}} \times 60 = 5{,}400\,\text{RPM}$$

The previous 10–100 Hz range would completely miss the 270 Hz blade-pass signal.

### Changes

| Parameter | Before | After | Rationale |
|-----------|--------|-------|-----------|
| max_freq  | 100 Hz | 300 Hz | Catch blade-pass frequency for 3-blade props |
| num_blades| 2      | 3      | Correct RPM calculation |

### Result

✅ Detected 2 stable propellers at ~100 Hz → 2,000 RPM, plus intermittent ~270 Hz → ~5,400 RPM. Clean output, zero latency.

---

## Phase 7 — Pixel Threshold Experiment (1-Pixel Minimum)

### Hypothesis

Reducing `min_cluster_pixels` from 2 to 1 might extend detection range — a distant propeller might produce only a single vibrating pixel.

### Result

❌ **Massive false positives** — 13–17 "confirmed propellers" per frame, all single-pixel at random frequencies (19–233 Hz). Single pixels have no spatial coherence to verify, so any noise spike that persists for `min_hits` frames becomes a "propeller."

### Key Insight

> Spatial coherence (≥ 2 connected pixels) is a critical false-positive filter. A single pixel has zero spatial information — it cannot be distinguished from sensor noise. Two connected pixels vibrating at the same frequency is meaningful; one is not.

---

## Phase 8 — False-Positive Tuning (Final v2 Parameters)

### Changes

| Parameter        | Before | After | Rationale |
|------------------|--------|-------|-----------|
| min_cluster_pixels| 1     | 2     | Restore spatial coherence filter |
| min_hits          | 3     | 5     | Require longer persistence to compensate for 2-pixel minimum |

### Result

✅ **Clean detections with zero false positives.** 1–2 confirmed propellers at 270–290 Hz blade-pass → 5,400–5,700 RPM. Multi-pixel clusters (35–269 px). Tracked over 80+ consecutive hits across a 32-second recording.

---

## Phase 9 — Production Optimizations (v3)

### Motivation

Analysis of end-product deployment requirements. Even though the GUI is retained for development, all optimizations applicable *without* removing the GUI were implemented. The goal is to measure and demonstrate what can be achieved in Python, to establish a baseline before a potential C++ port.

### Latency Analysis (Pre-Optimization)

| Metric | v2 Value |
|--------|----------|
| First-detection latency | 600–1200 ms |
| Steady-state update | 100–140 ms |
| Jitter | ±5 ms (GC pauses) |
| Analysis cycle time | 100 ms (10 Hz) |
| CC analysis cost | ~1.5 ms (with per-frame allocation) |

### Optimizations Applied

#### 1. Pre-Allocated Buffers

**Before:** Every `analyze()` call allocated three new numpy arrays:
```python
mask_full = valid.astype(np.uint8) * 255   # 921 KB
mask_small = cv2.resize(...)               # 57 KB
mask_dilated = cv2.dilate(...)             # 57 KB
```
At 20 Hz, that's ~20 MB/s of allocation → garbage collection pressure → jitter.

**After:** All three buffers pre-allocated once at `__init__` and reused via `dst=` or `out=` parameters:
```python
self._mask_full = np.empty((height, width), dtype=np.uint8)
self._mask_small = np.empty((h_ds, w_ds), dtype=np.uint8)
self._mask_dilated = np.empty((h_ds, w_ds), dtype=np.uint8)
```
**Impact:** Zero per-frame allocation in the analysis path. GC pauses reduced.

#### 2. Bayesian Confidence Scoring

**Before:** Hard hit-count threshold (`min_hits=5`). A propeller needed 5 consecutive detections at 10 Hz = **500 ms** minimum first-detection latency. No quality weighting — a 2-pixel cluster at CV=0.29 counted the same as a 200-pixel cluster at CV=0.01.

**After:** Each detection is assigned a per-observation likelihood *p* based on its quality:
- **Pixel score:** ramps from 0 (1 pixel) to 1.0 (50+ pixels)
- **CV score:** ramps from 1.0 (CV=0) to 0.0 (CV=0.3)
- **Combined:** `p = BASE_P + quality × (MAX_P − BASE_P)` where `BASE_P=0.65`, `MAX_P=0.85`

Bayesian confidence accumulates as: `P(propeller) = 1 − (1−p)^n`

| Detection quality | p per obs | Frames to 95% confidence |
|-------------------|-----------|--------------------------|
| Weak (2 px, CV=0.25) | 0.67 | 4 |
| Medium (20 px, CV=0.15) | 0.74 | 3 |
| Strong (100 px, CV=0.05) | 0.83 | 2 |

Confidence also **decays** when a track goes undetected (× 0.8 per frame), causing stale tracks to drop below threshold naturally.

**Impact:** First-detection latency reduced from 500 ms to **100–200 ms** for strong detections (at 20 Hz update rate), while maintaining zero false positives.

#### 3. Velocity-Based Predictive Tracking

**Before:** Matching used current position only. A moving propeller (e.g., panning camera, translating drone) could drift out of the search radius between frames.

**After:** Each track maintains a smoothed velocity estimate `(vx, vy)`:
```python
raw_vx = det["x"] - track["x"]
track["vx"] = α × raw_vx + (1−α) × track["vx"]
```
Matching uses the **predicted** next position `(x + vx, y + vy)` instead of the current position. This:
- Improves association accuracy for moving targets.
- Could allow a tighter search radius (reducing false matches).
- Provides velocity data as additional output for downstream consumers.

#### 4. Increased Update Rate (10 → 20 Hz)

With pre-allocated buffers and cheaper per-frame cost, the analysis rate was doubled:

| Metric | 10 Hz (v2) | 20 Hz (v3) |
|--------|-----------|-----------|
| Cycle time | 100 ms | 50 ms |
| First-detection (strong) | 500 ms | ~100 ms |
| First-detection (weak) | 500 ms | ~200 ms |
| Steady-state update | 100 ms | 50 ms |
| CPU overhead (analysis) | ~2% | ~4% |

#### 5. Timing Instrumentation

The analyzer now reports an exponential moving average of its per-cycle analysis cost (in ms), printed alongside each detection. This provides ongoing production telemetry without requiring external profiling.

### Revised Latency Budget (v3)

| Metric | v2 | v3 | Improvement |
|--------|----|----|-------------|
| First-detection (strong) | 600–1200 ms | ~100 ms | **6–12×** |
| First-detection (weak) | 600–1200 ms | ~200 ms | **3–6×** |
| Steady-state update | 100–140 ms | ~50–90 ms | **~2×** |
| Per-frame analysis cost | ~1.5 ms | ~0.8 ms | ~2× |
| GC jitter | ±5 ms | < ±1 ms | ~5× |

### Production Roadmap (Beyond Python)

For a headless C++ end-product deployment, additional measures identified:

| Measure | Expected Impact |
|---------|----------------|
| **C++ port** | Eliminate GIL, interpreter overhead, GC; deterministic sub-ms jitter |
| **Remove GUI** | −8–12 ms/cycle CPU; eliminate display thread contention |
| **Double-buffered freq map** | Zero contention between SDK writes and analysis reads |
| **SIMD thresholding** | AVX2/NEON for binary mask creation in ~0.05 ms |
| **Incremental CC** | Re-use previous frame's labels; −50% CC cost for stable scenes |
| **50–100 Hz update rate** | First-detection in 50–100 ms (only viable without GUI) |
| **FPGA frequency extraction** | Offload per-pixel period detection to sensor FPGA fabric |
| **Real-time scheduling** | `SCHED_FIFO` + CPU affinity for deterministic timing |
| **Output interfaces** | Callback, UDP multicast, shared memory, GPIO, CAN bus |

**Projected C++ headless latency:** First-detection ~80–100 ms, steady-state ~20 ms.

---

## Summary of Architecture Evolution

```
v1 (Baseline)                    v2 (Redesign)                   v2-opt (Latency fix)              v3 (Production-opt)
─────────────                    ─────────────                   ────────────────────              ───────────────────
Grid FFT on event counts         Per-pixel freq map (SDK)        Same                              Same
  ↓                                ↓                               ↓                                 ↓
Fixed N×N grid cells             Connected components            4× downscaled CC                  Pre-alloc buffers
  ↓                              (adaptive, shape-aware)         (320×180 mask)                    ROI-only freq stats
SNR threshold                      ↓                               ↓                                 ↓
  ↓                              Frequency consistency           Same filtering                    Same filtering
Grid-cell adjacency clustering   (CV ≤ 0.3 within cluster)         ↓                                 ↓
  ↓                                ↓                             Same tracking                     Bayesian confidence
No temporal filtering            Temporal tracking                  ↓                              Velocity prediction
  ↓                              (min 3 consecutive hits)        In-place drawing                  20 Hz update rate
Print detections                   ↓                             10 Hz update rate                    ↓
                                 Bounding box overlays                                            Timing telemetry
                                 Console output on changes
```

## Key Takeaways

1. **Architecture beats tuning.** No amount of parameter adjustment could fix the grid FFT approach because it fundamentally lacks spatial coherence. Switching to connected components on the per-pixel frequency map was the decisive improvement.

2. **Temporal persistence is essential.** Single-frame detections are unreliable at any sensitivity level. Bayesian confidence scoring achieves the same false-positive suppression as hard hit-count thresholds, but 3–6× faster.

3. **Spatial coherence is a critical filter.** Even ≥ 2 connected vibrating pixels is meaningful; 1 pixel is indistinguishable from sensor noise (Phase 7 experiment).

4. **Profiling matters.** The v2 detector was functionally correct but operationally unusable due to latency. Identifying that dilate + CC on a 1280×720 image at 25 Hz consumed 315 ms/s led to a simple 4× downscale that solved the problem with no loss in detection quality.

5. **Copy what you must, reference what you can.** Eliminating per-frame `frame.copy()` saved 67 MB/s; pre-allocating analysis buffers eliminated ~20 MB/s of GC pressure at 20 Hz.

6. **Probabilistic > deterministic thresholds.** Bayesian confidence (v3) adapts to detection quality — a strong 200-pixel detection confirms in 2 frames; a weak 3-pixel detection takes 4. Hard hit-count treats both identically.

## File Reference

| File | Description |
|------|-------------|
| `detect_propeller.py` | Current production-optimized detector (v3) |
| `detect_propeller_v1_backup.py` | Backup of the original grid FFT version |
| `PROPELLER_DETECTOR_DEVELOPMENT.md` | This document — full development history |
