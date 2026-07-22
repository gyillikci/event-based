# NeuraSense — C++ Port

C++ port of the Python detection applications, motivated by the performance
roadmap in `PROPELLER_DETECTOR_DEVELOPMENT.md` §"Production Roadmap": removing
the Python interpreter, GIL, and garbage-collector overheads that bounded the
v3 detector's latency and caused timing jitter.

## What was ported

| C++ app | Python original | Purpose |
|---|---|---|
| `detect_propeller` | `detect_propeller.py` | Visual propeller detector (v3): per-pixel frequency map → connected-component clustering → temporal tracking with Bayesian confidence |
| `detect_drone_fused` | `detect_drone_fused.py` | Audio-visual Bayesian fusion with harmonic frequency cross-validation and quality/association/angle gates |
| `detect_hand_shake` | `detect_hand_shake.py` | Centroid-FFT hand-shake frequency/magnitude estimator |
| `calibrate_fan_frequency` | `calibrate_fan_frequency.py` | Acoustic source-frequency calibration (top peaks, SNR, RPM, recommended band) |

The reusable core is split into a portable static library `ndrone_core`:

| Module | Ported class / function |
|---|---|
| `propeller_detector.{hpp,cpp}` | `FrequencyMapAnalyzer`, `PropellerTracker` |
| `bayesian_fusion.{hpp,cpp}` | `BayesianDroneDetector`, `cross_validate_frequency` |
| `acoustic_processor.{hpp,cpp}` | `AcousticProcessor` (multi-channel FFT BPF, GCC-PHAT DOA) |
| `azimuth.{hpp,cpp}` | pixel↔azimuth mapping, linear calibration load/apply |
| `fft.{hpp,cpp}` | numpy-equivalent `rfft`/`irfft`/`rfftfreq`/`hanning` (arbitrary length, no FFTW) |
| `synthetic.{hpp,cpp}` | synthetic frequency-map / shake / tone sources for offline runs and tests |

## Design: SDK-independent core

The per-pixel frequency map is the only input the visual pipeline needs, and it
is produced by Metavision's `FrequencyMapAsyncAlgorithm`. The port keeps that
input identical to the Python version, so **the SDK is the only platform-specific
part**. Everything downstream (clustering, tracking, fusion, acoustics) is pure
C++/OpenCV and builds and runs anywhere.

* Build **without** the SDK (default): synthetic sources drive the identical
  pipeline — useful for development, CI, and benchmarking. This is what runs in
  the Linux sandbox where the Windows SDK binaries cannot load.
* Build **with** the SDK (`-DWITH_METAVISION=ON`): `detect_propeller.cpp`
  contains the real Metavision C++ glue (camera → `FrequencyMapAsyncAlgorithm`
  → `set_output_callback` → the same analyzer/tracker). Confirm the exact SDK
  component/type names against your installed Metavision version.

## Build

```bash
cd cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure     # runs the known-answer core tests
```

Dependencies: CMake ≥ 3.16, a C++17 compiler, OpenCV (core, imgproc). PortAudio
is auto-detected for live ReSpeaker capture (`WITH_PORTAUDIO`, on by default);
without it the acoustic code still runs offline on sample buffers.

To build against the event-camera SDK:

```bash
cmake -S . -B build -DWITH_METAVISION=ON
```

## Run (synthetic, no hardware)

```bash
./build/detect_propeller --synthetic-frames 40
./build/detect_drone_fused --synthetic-frames 30
./build/detect_hand_shake --shake-freq 6 --shake-amp 10 --synthetic-frames 3000
./build/calibrate_fan_frequency --synthetic-tone 162.7
```

With the SDK built in, pass `-i <recording.raw>` or omit it for a live camera,
exactly like the Python CLIs. The Python flags are mirrored (`--min-freq`,
`--max-freq`, `--num-blades`, `--min-cluster-pixels`, `--confidence-threshold`,
`--enable-audio`, `--camera-hfov-deg`, `--azimuth-calibration`, …).

## Performance

Hot path (`FrequencyMapAnalyzer::analyze` + `PropellerTracker::update`) over
2000 frames at 1280×720 with a ~253-pixel propeller, `-O3 -march=native`:

| Metric | Python v3 (reported) | C++ port (measured) |
|---|---|---|
| Per-frame analysis | ~0.8 ms | **0.63 ms mean** |
| Timing jitter | ±5 ms (v2) → <±1 ms (v3, GC) | **0.28 ms** (p99−p50), no GC pauses |

The elimination of interpreter overhead and garbage-collection pauses gives
deterministic per-frame timing — the property the Python version could not
guarantee. This matches the roadmap's rationale for the C++ port and is the
foundation for the further gains it lists (headless operation, SIMD
thresholding, 50–100 Hz update rates).

## Verification

`ctest` runs `test_core`, which checks against known answers:
FFT round-trip and tone localization; propeller confirmation at 270 Hz / 5400 RPM
(3 blades) plus rejection of single-pixel noise; harmonic cross-validation
(including the 2×/1× harmonic match and a true non-match); Bayesian
accumulate-then-decay dynamics; acoustic tone recovery at 162.7 Hz; and
pixel↔azimuth round-tripping.
