# Native (C++) event-based detectors

Two command-line tools built directly on the Metavision SDK C++ API. They exist
because they use SDK algorithms that **have no Python bindings**, so the existing
Python scripts in this workspace cannot call them.

| Tool | C++-only algorithm(s) used | Python equivalent |
|------|----------------------------|-------------------|
| `active_marker_detector` | `ModulatedLightDetectorAlgorithm`, `ActiveMarkerTrackerAlgorithm` | none (no wrapper) |
| `propeller_detector` | `ProximityFilterAlgorithm` (focus) + `FrequencyAlgorithm` + `FrequencyClusteringAlgorithm` | `detect_propeller.py --use-sdk-clustering` (no proximity focus) |
| `drone_propeller_detector` | `FrequencyMapAsyncAlgorithm` + `ProximityFilterAlgorithm` (focus) | `detect_propeller.py` (default v3 path) |

## `drone_propeller_detector` vs `propeller_detector`

The two propeller tools implement the two clustering strategies of
`detect_propeller.py`:

- `propeller_detector` — the SDK path (`FrequencyAlgorithm` +
  `FrequencyClusteringAlgorithm`), i.e. `--use-sdk-clustering`. Console only.
- `drone_propeller_detector` — the **default v3 production path**: per-pixel
  `FrequencyMapAsyncAlgorithm`, then a custom connected-component analyzer
  (threshold → 4× downscale → dilate → components → ROI-only frequency stats →
  fragment consolidation) and a Bayesian tracker
  (`P = 1 − (1 − p)^hits`, velocity-predicted greedy matching). Shows the event
  window with detection overlays plus the frequency heat map.

CLI names and defaults match `detect_propeller.py` so results can be compared
directly, with three deliberate differences:

- `--delta-t` is dropped — event batching is driver-controlled in C++, there is no
  `EventsIterator` slice size. The Python Nyquist check tied to it is dropped too:
  `FrequencyMapAsyncAlgorithm` measures per-pixel periods, so the batch size does
  not limit the detectable frequency.
- `--replay-factor` becomes the boolean `--realtime-playback-speed`; the SDK has no
  fractional replay factor.
- `--use-sdk-clustering` is not ported — that mode is `propeller_detector`.

Extras not in the Python script: `--session-log` (JSONL trace), `--benchmark`
(headless timing summary) and `--self-test` (synthetic frequency map, no camera).

## Which SDK binaries had no Python wrapper?

Confirmed by introspecting the installed `metavision_sdk_cv` / `metavision_sdk_cv3d`
Python modules (these classes are **absent** from Python):

- `ModulatedLightDetectorAlgorithm` (metavision_sdk_cv) — decode LED IDs from blink timing
- `ActiveMarkerTrackerAlgorithm` (metavision_sdk_cv) — track active-marker LEDs
- `ProximityFilterAlgorithm` (metavision_sdk_cv) — keep events near a point
- `ActiveMarkerPoseEstimatorAlgorithm` (metavision_sdk_cv3d) — 6-DoF marker pose
- calibration module algorithms: `BlinkingDotsGridDetectorAlgorithm`,
  `DftHighFreqScorerAlgorithm` (metavision_sdk_calibration)

`FrequencyAlgorithm` and `FrequencyClusteringAlgorithm` **do** have Python bindings
(that is what `detect_propeller.py --use-sdk-clustering` uses); they are included
here so the propeller tool is a fully native pipeline that can be combined with the
C++-only `ProximityFilterAlgorithm` focus stage.

## Prerequisites

- Metavision SDK (developer install) — headers, import libs and CMake configs.
  Default install: `C:\Program Files\Prophesee`.
- CMake >= 3.16 and an MSVC toolchain (Visual Studio Build Tools).

## Build

```powershell
cmake -S cpp -B cpp/build -DCMAKE_PREFIX_PATH="C:/Program Files/Prophesee"
cmake --build cpp/build --config Release
```

Executables are written to `cpp/build/active_marker_detector/Release/` and
`cpp/build/propeller_detector/Release/`.

## Run

At runtime the full SDK DLL set must be on `PATH`. Use the workspace `bin/`, which
bundles the third-party runtime dependencies (OpenCV, HDF5, Boost, zlib, …). The
installed SDK's `bin` on its own is **not** enough — it ships only the Metavision
DLLs, so loading fails with `0xC0000139` (entry-point-not-found) unless those
third-party DLLs are also reachable.

```powershell
$env:PATH = "C:\Users\z003n5uc\Desktop\event-based\bin;" + $env:PATH

# active markers — from a recording, IDs listed in a JSON file
cpp\build\active_marker_detector\Release\active_marker_detector.exe `
    --am-json cpp\active_marker_detector\active_marker_sample.json -i recording.raw

# active markers — live camera, IDs given inline
cpp\build\active_marker_detector\Release\active_marker_detector.exe --led-ids 1,2,3,4

# propeller — from a recording, focus on a region
cpp\build\propeller_detector\Release\propeller_detector.exe `
    -i recording.raw --num-blades 3 --focus-x 320 --focus-y 240 --focus-radius 120

# drone propeller (v3 pipeline) — live camera, two windows, q/ESC to quit
cpp\build\drone_propeller_detector\Release\drone_propeller_detector.exe --num-blades 3

# drone propeller — offline benchmark against the Python detector, with a JSONL trace
cpp\build\drone_propeller_detector\Release\drone_propeller_detector.exe `
    -i recording.raw --benchmark --session-log results\cpp_propeller.jsonl

# drone propeller — self test, no camera needed
cpp\build\drone_propeller_detector\Release\drone_propeller_detector.exe --self-test
```

Run any tool with `--help` for the full option list.

Always quit `drone_propeller_detector` with `q` or `ESC`. Killing the process
mid-stream can leave an EVK4 in `LIBUSB_ERROR_TIMEOUT`, which only a physical
USB replug clears.

## Important: active-marker firmware

`active_marker_detector` decodes **ID-encoded modulated light**: each LED transmits
an ID via inter-blink timing (base period `p`: `p`=0, `2p`=1, `3p`=start bit). This
is **different** from the workspace's `detect_active_markers.py`, which detects LEDs
that each blink at a **fixed frequency** (see `active_markers.json`). To use this C++
tool the markers must run modulated-light (ID) firmware. `--am-json` accepts either
the SDK `{"active marker":[{"led":{"id":N}}]}` format or the workspace
`{"markers":[{"id":N,...}]}` format — it simply collects the `id` values.
