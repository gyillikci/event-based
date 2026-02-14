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

## Summary of Architecture Evolution

```
v1 (Baseline)                    v2 (Redesign)                   v2-opt (Latency fix)
─────────────                    ─────────────                   ────────────────────
Grid FFT on event counts         Per-pixel freq map (SDK)        Same
  ↓                                ↓                               ↓
Fixed N×N grid cells             Connected components            4× downscaled CC
  ↓                              (adaptive, shape-aware)         (320×180 mask)
SNR threshold                      ↓                               ↓
  ↓                              Frequency consistency           Same filtering
Grid-cell adjacency clustering   (CV ≤ 0.3 within cluster)         ↓
  ↓                                ↓                             Same tracking
No temporal filtering            Temporal tracking                  ↓
  ↓                              (min 3 consecutive hits)        In-place drawing
Print detections                   ↓                             10 Hz update rate
                                 Bounding box overlays
                                 Console output on changes
```

## Key Takeaways

1. **Architecture beats tuning.** No amount of parameter adjustment could fix the grid FFT approach because it fundamentally lacks spatial coherence. Switching to connected components on the per-pixel frequency map was the decisive improvement.

2. **Temporal persistence is essential.** Single-frame detections are unreliable at any sensitivity level. Requiring 3+ consecutive hits eliminates virtually all false positives while adding only ~300 ms of latency.

3. **Profiling matters.** The v2 detector was functionally correct but operationally unusable due to latency. Identifying that dilate + CC on a 1280×720 image at 25 Hz consumed 315 ms/s led to a simple 4× downscale that solved the problem with no loss in detection quality.

4. **Copy what you must, reference what you can.** Eliminating the per-frame `frame.copy()` saved 67 MB/s of allocation overhead — a trivial code change with outsized impact.

## File Reference

| File | Description |
|------|-------------|
| `detect_propeller.py` | Current optimised detector (v2) |
| `detect_propeller_v1_backup.py` | Backup of the original grid FFT version |
