# Passive Drone Detection via Neuromorphic Vision and Acoustic Sensor Fusion

**Comprehensive research record — methods attempted, results, and lessons learned.**

This document is structured to be a direct source for composing an academic paper.
It records the full experimental trajectory of the project: every major approach
that was tried, why it was tried, what worked, what failed, and the quantitative
or qualitative evidence behind each verdict. Sections map cleanly onto a typical
paper structure (Abstract → Introduction → Related Work → System/Methods →
Experiments → Results → Discussion → Limitations → Conclusion).

---

## 1. Abstract / One-Paragraph Summary

We investigate **passive, RF-silent drone detection** by exploiting the fact that
a spinning propeller emits a characteristic **blade-pass frequency (BPF)**
simultaneously in two physical channels: **optical flicker** (sensed by a
neuromorphic event camera) and **acoustic pressure** (sensed by a microphone
array). We built a working prototype that (i) extracts per-pixel flicker
frequency from a Prophesee EVK4-HD event camera and isolates propeller regions
via connected-component spatial clustering and temporal tracking, (ii) extracts
the acoustic BPF and a direction-of-arrival (DOA) estimate from a ReSpeaker
4-microphone array using FFT and GCC-PHAT, and (iii) fuses the two streams with a
**sequential Bayesian updater** that includes **harmonic-aware frequency
cross-validation**. The central finding is methodological: **detection
architecture, not parameter tuning, determines success.** A grid-FFT baseline
failed at range regardless of tuning; replacing it with per-pixel frequency +
connected-component clustering produced stable, low-false-positive detection.
Acoustic detection of the BPF and cross-modal frequency agreement worked
reliably; cross-modal *spatial* (azimuth) calibration did not converge to a
usable model under our test conditions.

---

## 2. Motivation and Problem Statement

Consumer drones are cheap, increasingly autonomous (GPS waypoint navigation), and
can fly with **no active RF link**, defeating the dominant RF-scanning detection
technology. Existing modalities each have a structural blind spot:

| Modality | Mechanism | Structural weakness |
|---|---|---|
| RF scanning | Detect drone radio protocol | Blind to autonomous/RF-silent drones |
| Radar | Reflected RF | Cannot separate small drones from birds; high cost |
| RGB/thermal camera | Frame imaging (30–60 fps) | Motion blur, ~60 dB dynamic range, fails in low light/fog |
| Acoustic-only | Propeller sound | Short range, 10–20 % false-positive rate in noise |

**Research question.** Can a *passive* sensor combination exploit the
*physics-guaranteed* propeller BPF to detect drones with a false-alarm rate far
below acoustic-only systems, without RF emission?

**Hypothesis.** The BPF appears in both optical and acoustic domains. Requiring
*cross-modal frequency agreement* should suppress the dominant false-alarm
sources (each of which usually lives in only one modality), driving the joint
false-positive probability sharply down.

---

## 3. Hardware and Experimental Platform

### 3.1 Event camera (visual)
- **Prophesee EVK4-HD**, Sony **IMX636** sensor, **1280×720**.
- ~1 µs per-pixel temporal resolution; >120 dB dynamic range; <100 µs pixel
  latency @ 1000 lux.
- Lenses used during the study: a **220° fisheye** (hemispheric coverage) and a
  narrower **CCTV/C-mount rectilinear lens** (~63.8° HFOV) used for the later
  azimuth experiments.
- SDK: **Prophesee Metavision** — key algorithms used:
  `FrequencyMapAsyncAlgorithm`, `DominantValueMapAlgorithm`,
  `HeatMapFrameGeneratorAlgorithm`, `PeriodicFrameGenerationAlgorithm`.

### 3.2 Microphone array (acoustic)
- **ReSpeaker 4-Mic USB Array v2.0**, 4× MEMS mics in a circular array,
  **32 mm radius** (→ 64 mm baseline for opposing pairs 0–2 and 1–3).
- 16 kHz sampling, INT16, via PyAudio.
- On-board **XMOS XVF-3000** DSP providing factory DOA + voice-activity detection
  (readable over a vendor USB interface; also drives a 12-LED ring).

### 3.3 Software environment
- **Python 3.9** (the only interpreter for which the Metavision binaries load;
  important reproducibility note — 3.12/3.14 cannot load the SDK `.pyd` files).
- NumPy (FFT/statistics), OpenCV (morphology, connected components, drawing),
  PyAudio (capture), pyusb/libusb (LED ring).

### 3.4 Test sources
- A bench **fan / propeller** used as a controlled rotating source. Acoustic
  calibration measured its dominant tone at **162.7 Hz** (SNR ≈ 24 dB), with
  harmonics at 149.5 Hz and 100 Hz.
- A synthetic **vibration pattern generator** (on-screen shapes flickering at
  15–120 Hz) used to validate the event-camera frequency pipeline end-to-end
  without a live drone.

> **Limitation up front:** all reported detection results are from **indoor,
> short (~30 s), single-source lab recordings**. No outdoor / multi-drone /
> GPS-ground-truth validation was performed. See §10.

---

## 4. Visual Detection — Methods Tried and Outcomes

The visual pipeline went through nine documented iterations. The crucial story is
the **architectural replacement** in Phase 3.

### 4.1 v1 — Grid-based FFT on event rates  ❌ FAILED at range
**Approach.** Divide the sensor into an N×N grid (16×16 = 256 cells), count events
per cell per 500 µs slice, run a rolling-window FFT per cell, threshold on SNR,
and cluster adjacent active cells into "propeller regions."

Default parameters: `filter_length=7`, `max_period_diff=500 µs`,
`min_pixel_count=25`, `grid_cells=16`, `fft_window=0.5 s`, `min_snr=3.0`.

**Result.** Worked **only at ~centimetre range**. At distance the propeller
occupies a handful of pixels inside a 160×90 cell crowded with noise pixels, so
its signal is diluted below the cell noise floor.

### 4.2 v2 (attempt) — Parameter tuning  ❌ FAILED (key negative result)
Lowered thresholds to chase range (`filter_length 7→4`, `max_period_diff
500→1500 µs`, `min_pixel_count 25→5`, `grid_cells 16→8`, `fft_window 0.5→1.0 s`,
`min_snr 3.0→2.0`).

**Result.** 25–43 "propeller regions" per frame — **pure noise promoted to false
positives.** 

> **Lesson (quotable):** *Parameter tuning cannot repair a fundamentally wrong
> architecture.* The grid FFT has no spatial coherence: it never asks whether the
> active pixels in a cell are spatially or spectrally consistent.

### 4.3 v2 — Per-pixel frequency + connected components  ✅ BREAKTHROUGH
**Architecture.**
```
SDK per-pixel frequency map  →  threshold to binary mask
   →  morphological dilation  →  connected-component labelling
   →  per-cluster frequency-consistency filter  →  temporal tracking
```
Key design choices:
- **Spatial clustering instead of a fixed grid** (`FrequencyMapAnalyzer`): the
  detector adapts to the propeller's true shape; even 2–3 connected pixels at a
  consistent frequency are meaningful, while the same pixels scattered in a grid
  cell are not.
- **Frequency-consistency gate:** within-cluster coefficient of variation
  CV ≤ 0.3 (all pixels must agree on frequency).
- **Temporal tracking** (`PropellerTracker`): greedy match on spatial proximity
  (<100 px) and frequency similarity (<30 %); confirm only after `min_hits`
  consecutive detections; prune after `max_age=8` idle frames; EMA smoothing
  (α=0.3) on position and frequency.

**Result.** Single stable propeller at ~99.7 Hz / ~2,990 RPM, locked position,
~100 active pixels, 270+ consecutive hits over the recording. Transient
single-/few-pixel false positives pruned within 1–2 s.

### 4.4 Phase 4 — Latency elimination  ✅ FIXED real-time drift
**Problem.** The synchronous `on_freq_map` callback ran full-resolution
`dilate` + `connectedComponentsWithStats` + `frame.copy()` at 25 Hz, costing
~315 ms of CPU per wall-second and causing monotonically growing display lag.

**Fixes:** (1) **4× downscale** (1280×720→320×180) before morphology/CC; (2)
update rate 25→10 Hz; (3) **in-place** frame drawing (removed `frame.copy()`);
(4) **early-exit** on near-empty masks. **Result:** zero latency drift; per-frame
callback cost ~375 ms/s → ~20 ms/s.

### 4.5 Phases 5–6 — Frequency range and blade count  ✅
Realised the test prop is **3-blade**, so the observed 270 Hz was the *blade-pass*
(not shaft) frequency (shaft = 270/3·60 = 5,400 RPM). Widened `max_freq` to
300 Hz and set `num_blades=3` for correct RPM reporting.

### 4.6 Phase 7 — Single-pixel minimum experiment  ❌ FAILED (validates a rule)
Setting `min_cluster_pixels=1` produced 13–17 "confirmed propellers"/frame at
random frequencies.

> **Lesson:** **Spatial coherence (≥2 connected pixels) is the primary
> false-positive filter.** A single pixel carries no spatial information and is
> indistinguishable from sensor noise.

### 4.7 Phase 8 — Final v2 tuning  ✅
`min_cluster_pixels=2`, `min_hits=5` → clean detections, zero false positives,
multi-pixel clusters (35–269 px) tracked over 80+ hits in a 32 s recording.

### 4.8 Phase 9 — v3 production optimisation  ✅
- **Pre-allocated buffers** (`_mask_full/_mask_small/_mask_dilated`) → zero
  per-frame allocation → no GC jitter.
- **ROI-only frequency statistics** (read only each component's bounding box).
- **Bayesian confidence scoring** replacing the hard hit-count threshold:
  per-observation likelihood `p = 0.65 + quality·(0.85−0.65)` from pixel-count and
  CV sub-scores; accumulated as `P = 1 − (1−p)^n`; decays ×0.8 per idle frame.
- **Velocity-predictive tracking** (match on predicted next position).
- Update rate doubled to 20 Hz.

**Measured effect (from dev logs):**

| Metric | v2 | v3 | Improvement |
|---|---|---|---|
| First-detect (strong) | 600–1200 ms | ~100 ms | 6–12× |
| First-detect (weak) | 600–1200 ms | ~200 ms | 3–6× |
| Steady-state update | 100–140 ms | 50–90 ms | ~2× |
| Per-frame analysis | ~1.5 ms | ~0.8 ms | ~2× |
| GC jitter | ±5 ms | <±1 ms | ~5× |

---

## 5. Acoustic Detection — Methods Tried and Outcomes

### 5.1 BPF extraction via multi-channel FFT  ✅
Rolling 0.2 s window (3200 samples @ 16 kHz) per channel, Hann-windowed
`rfft`; magnitudes **incoherently averaged across the 4 channels** (~6 dB SNR
gain); peak picked in the target band; SNR = peak / median noise floor; reject
below `min_snr` (3.0). Reliable on the sustained propeller tone.

### 5.2 DOA via GCC-PHAT  ✅ (azimuth only)
Per opposing pair (0–2 and 1–3, 64 mm baseline): phase-transform cross-spectrum,
**frequency-band weighting** emphasising the target band, **4× zero-padded IFFT**
for sub-sample TDOA, **parabolic peak interpolation**, and a **peak/second-peak
confidence ratio** gate (`doa_min_confidence=1.5`). Pair estimates fused
(averaged when both confident); a 7-sample **median** temporal filter reduces
jitter, with a spread metric used to gate unstable bearings.

**Result.** Usable azimuth on steady tones. The planar 32 mm array **cannot
resolve elevation** and has limited angular resolution (~±10° at range).

### 5.3 ODAS (Open embeddeD Audition System)  ❌ FAILED for propellers
Considerable effort went into integrating ODAS via WSL, including a custom
**socket relay** (`third_party/odas_wsl_relay.py`) to work around WSL bridged
networking being one-way: audio PCM Windows→WSL:5000→odaslive, and SSL/SST
results piped back to Windows for ODAS Studio.

- **Worked:** robust DOA on **speech** (sustained, 500+ ms utterances).
- **Failed:** propeller acoustics are comparatively **impulsive/broadband**;
  ODAS DOA was unreliable on short bursts. This motivated the **custom GCC-PHAT**
  in `AcousticProcessor` instead.

### 5.4 Firmware DSP DOA  ◐ available, not integrated
The XMOS firmware exposes its own DOA + VAD over USB
(`respeaker_led_control.py`). Its VAD is tuned for speech, so it is unreliable on
steady tones; it remains a candidate auxiliary input but was not fused.

---

## 6. Sensor Fusion — Methods and Outcomes

### 6.1 Harmonic-aware frequency cross-validation  ✅
`cross_validate_frequency()` searches harmonic pairs (n_v, n_a) up to the 4th and
declares a match if `|f_v/n_v − f_a/n_a| / (f_a/n_a) < 0.05`, returning confidence
`1/(n_v·n_a)`. This handles the common case of one modality reporting the
blade-pass tone while the other reports the shaft fundamental.

### 6.2 Sequential Bayesian updating  ✅
`BayesianDroneDetector` starts at prior `P=0.001` and updates per modality with
sigmoid likelihoods:
- Visual: `P(V|D)=σ(SNR−5)`, `P(V|¬D)=0.01·e^(−0.5·SNR)`.
- Acoustic: `P(A|D)=σ(SNR−4)`, `P(A|¬D)=0.05` (0.005 if elevated).
- Frequency-match bonus: `P(F|D)=0.99·conf`, `P(F|¬D)=0.001`.
- **Decay** toward prior (×0.995/frame) when no evidence arrives.
Detection at `P>0.8`.

### 6.3 Fusion gating (what made fusion trustworthy)  ✅
To prevent overconfident fusion from coincidental evidence, several gates were
added and proved necessary in practice:
- **Visual quality gate** (pixels ≥ threshold, track confidence ≥ 0.98).
- **Acoustic stability gate** (persistence hits, frequency jitter ≤ 25 Hz).
- **Temporal association gate** (|t_visual − t_acoustic| ≤ ~200 ms).
- **Angle-residual gate** (|visual azimuth − acoustic DOA| ≤ 35°).
- **Audio-only gates** (higher SNR ≥ 6 dB *and* bounded DOA spread) before an
  audio-only update may raise `P(drone)`.

**Observed behaviour (live run, audio-only path):** the detector repeatedly
latched `P(drone)→~1.0` on strong ~140–200 Hz tones and decayed back when the
source stopped — i.e. the Bayesian accumulate/decay dynamics behaved as designed.
In that particular run the **visual channel never confirmed** (no live propeller
in view), so every confirmation was acoustic-only — a useful reminder that
single-modality dominance is possible and the gates matter.

---

## 7. Calibration — Methods and Outcomes

### 7.1 Acoustic source-frequency calibration  ✅
`calibrate_fan_frequency.py`: 10 s capture, channel-averaged spectrum, top-8 peak
report. Output (`fan_calibration.json`): dominant 162.7 Hz, recommended band
122–203 Hz; used to set detector frequency limits.

### 7.2 Visual→acoustic azimuth calibration  ❌ DID NOT CONVERGE (key negative result)
`AzimuthCalibrationCollector` sweeps a co-located source laterally, bins samples
by visual azimuth (robust per-bin medians), and fits
`acoustic_doa = gain·visual_az + offset` with iterative MAD outlier rejection and
an R² "solidity" check (threshold 0.6).

Best achieved fit (`azimuth_calib.json`, CCTV lens, HFOV 63.8°):
`gain=−0.011`, `offset=0.37°`, **R²=0.137**, RMSE 0.70°, **solid=false**. The
acoustic DOA only spanned ~±1.2° while the visual sweep spanned ~±25°.

> **Interpretation / what to fix:** the acoustic DOA did not swing enough to
> constrain the line — consistent with reverberation, an insufficiently wide/slow
> sweep, or low GCC-PHAT confidence during the session. Cross-modal *spatial*
> fusion is therefore **not yet validated**; cross-modal *frequency* fusion is.

### 7.3 Camera↔LED-ring reference  ✅ (utility)
`align_camera_led.py` / `camera_led_ref.json`: records which ring LED points along
the camera +z axis (LED 11 @ 330°), enabling the ring to physically indicate the
detector azimuth. Pixel→azimuth uses a pinhole model
`f_px=(W/2)/tan(HFOV/2); az=atan2(x−W/2, f_px)`.

---

## 8. Exploratory Sub-Project — Light-Flicker Localization

A separate module (`light_localization/`) investigated using **ceiling-light
flicker frequencies as location fingerprints** (electronic ballasts 20–60 kHz, LED
PWM, magnetic ballast harmonics), inspired by LiTell (MobiCom 2016), iLAMP
(MobiSys 2017), and the first event-camera VLP work (Wang et al., IEEE Sensors
2020).

- **Done:** literature review (15 papers), feasibility verdict **GO**, utility
  classes (`LightROIExtractor`, `FrequencyFeatureExtractor`, `LightFingerprint`,
  `FingerprintDatabase`, `LightLocalizer`), and calibration/separability tools.
- **Not done:** end-to-end real-world validation and 3D-accuracy benchmarking.

Status: **theory validated, implementation incomplete** — candidate for a
companion paper or a "future work" thread, not a current result.

---

## 9. Consolidated Results — What Worked vs. What Failed

### ✅ Worked
1. Per-pixel flicker-frequency extraction (Metavision `FrequencyMapAsyncAlgorithm`).
2. Connected-component spatial clustering (adaptive, shape-aware) — the decisive
   architectural change.
3. Temporal tracking + Bayesian confidence (persistence kills ~95 % of transient
   noise).
4. Real-time latency engineering (downscale, pre-alloc buffers, in-place draw).
5. Acoustic BPF via multi-channel FFT (incoherent averaging ≈ 6 dB gain).
6. GCC-PHAT **azimuth** on sustained tones.
7. Harmonic-aware frequency cross-validation.
8. Sequential Bayesian fusion with explicit quality/association/angle gates.

### ❌ Failed or incomplete
1. v1 grid-FFT architecture (signal dilution, grid quantisation, no persistence).
2. Pure parameter tuning to extend range (turned noise into false positives).
3. `min_cluster_pixels=1` (massive false positives — proves the coherence rule).
4. ODAS for propeller acoustics (designed for sustained speech, not impulsive
   broadband).
5. Visual→acoustic **azimuth calibration** (R²=0.137, not solid).
6. Light-localization end-to-end (feasibility proven only).
7. Elevation DOA (planar array — physically out of scope).

---

## 10. Limitations and Threats to Validity

| Area | Limitation | Consequence for claims |
|---|---|---|
| Dataset | Indoor, single source, ~30 s recordings | Generalisation unproven |
| Range | No outdoor / GPS-ground-truth tests | Range figures are projections, not measurements |
| Spatial fusion | Azimuth calibration weak (R²=0.14) | Cross-modal *spatial* gating unvalidated |
| DOA | 32 mm planar array | ~±10° azimuth error, no elevation |
| Multi-target | Tracker assumes one source per cluster | Overlapping BPFs may merge |
| Sync | No hardware inter-sensor clock | ~100 ms association uncertainty |
| Harmonic ambiguity | 2-blade@100 Hz ≡ 4-blade@50 Hz BPF | Needs external metadata to disambiguate |
| FPR claim | "<0.1 %" is design-derived, not measured on a labelled set | Must be empirically established for publication |

---

## 11. Reproducibility Notes

- Use **Python 3.9** (`.venv39`); Metavision `.pyd` binaries do **not** load under
  3.12/3.14.
- Visual Nyquist constraint: `max_freq ≤ 1e6/(2·delta_t)`; default `delta_t=500 µs`
  → 1 kHz Nyquist.
- Representative invocations:
  - Visual only: `python detect_drone_fused.py -i <event_file>`
  - Fused: `python detect_drone_fused.py -i <event_file> --enable-audio`
  - Live: `python detect_drone_fused.py --enable-audio`
- Session traces are logged as JSONL (`--session-log`) with per-interval
  `p_drone`, visual/acoustic evidence, association lag, and gate states — suitable
  as the raw data appendix for a paper.

---

## 12. Suggested Paper Structure

1. **Introduction** — drone threat, RF-silent gap, passive-detection need (§2).
2. **Related Work** — event-based vision survey (Gallego 2020), DVS (Lichtsteiner
   2008), acoustic/RF/radar counter-UAS; explicit gap: no event-camera + mic-array
   fusion exists.
3. **Physical Principle** — BPF in optical and acoustic domains; cross-modal
   argument for low FPR (§2 hypothesis).
4. **System & Methods** — visual pipeline (§4.3–4.8), acoustic pipeline (§5),
   fusion (§6).
5. **Implementation** — hardware (§3), real-time engineering (§4.4, §4.8).
6. **Experiments** — ablations already run: grid-FFT vs CC (§4.1–4.3),
   `min_pixels` sweep (§4.6), latency before/after (§4.4, §4.8), ODAS vs GCC-PHAT
   (§5.3), calibration outcome (§7.2).
7. **Results** — §4.8 table, §6.3 fusion dynamics, §7 calibration.
8. **Discussion / Lessons** — "architecture > tuning"; spatial coherence as the
   FPR filter; modality-dominance failure mode.
9. **Limitations & Future Work** — §10 plus light-localization (§8).
10. **Conclusion**.

### Honest framing for reviewers
Lead with the **validated** contributions (architecture comparison, real-time
engineering, harmonic frequency cross-validation, Bayesian fusion dynamics) and
present range/FPR numbers as **design targets requiring outdoor validation**.
Report the **azimuth-calibration failure** as a finding, not an omission — it is a
genuine, instructive negative result.

---

## 13. Source Map (for citing internal artifacts)

| Topic | File(s) |
|---|---|
| Visual detector (v3) | `detect_propeller.py` |
| Legacy grid-FFT (v1) | `detect_propeller_v1_backup.py`, `propeller_utils.py` |
| Fusion + acoustic + calibration collector | `detect_drone_fused.py` |
| Acoustic range profiling | `acoustic_range_profiler.py` |
| Source-frequency calibration | `calibrate_fan_frequency.py`, `fan_calibration.json` |
| Azimuth calibration | `calibration/calibrate_camera_mic_azimuth.py`, `azimuth_calib.json` |
| Camera/LED reference | `align_camera_led.py`, `camera_led_ref.json`, `respeaker_led_control.py` |
| ODAS integration | `third_party/odas_wsl_relay.py`, `odas_audio_bridge.py`, `third_party/run_odaslive.sh` |
| Light localization | `light_localization/` (README, LITERATURE_REVIEW, TECHNICAL_EVALUATION, `*.py`) |
| Sound localization demo | `sound localization/listen.py` |
| Verification workflow | `verify_propeller_sound.py`, `propeller_verification_session.jsonl` |
| Dev narrative | `PROPELLER_DETECTOR_DEVELOPMENT.md`, `PRESENTATION_DOCUMENTATION.md` |
| Session traces | `logs/*.jsonl` |
```
