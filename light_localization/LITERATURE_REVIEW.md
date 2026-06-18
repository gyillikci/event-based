# Literature Review: Indoor Localization via Fluorescent Light Flicker Fingerprinting

## 1. Overview

This document surveys the scientific literature on indoor positioning systems that exploit the
frequency characteristics (flicker) of indoor lighting — particularly fluorescent lamps — as
location landmarks. We focus on three intersecting research threads:

1. **Visible Light Positioning (VLP) using unmodified lights** — no VLC hardware required
2. **Flicker fingerprinting** — passive identification of individual light sources from their temporal signatures
3. **Event-camera-based VLP** — using neuromorphic vision sensors (DVS) for flicker detection

---

## 2. Foundational Work: VLP with Unmodified Fluorescent Lights

### 2.1 LiTell — Zhang & Zhang (MobiCom 2016)

> C. Zhang and X. Zhang, "LiTell: Robust indoor localization using unmodified light fixtures,"
> Proc. 22nd Annual International Conference on Mobile Computing and Networking (MobiCom), 2016.
> **215 citations.**

**Core Contribution:**
LiTell is the seminal work proving that **unmodified fluorescent lights** possess unique frequency
signatures usable as location landmarks. The key insight: electronic ballasts in fluorescent
tubes convert 50/60 Hz mains AC to high-frequency (20–60 kHz) AC to drive the tube. The exact
resonance frequency depends on the ballast's LC circuit components, which vary due to
**manufacturing tolerances, component aging, and temperature**. This makes each fixture's
operating frequency effectively unique.

**Method:**
- Captures light flicker using a **smartphone camera in rolling-shutter mode**
- The rolling shutter samples the light at ~30 kHz effective rate, producing visible banding
  in the image proportional to the light's flicker frequency
- Extracts the **characteristic frequency (CF)** of each visible light from the banding pattern
- Fingerprinting-based localization: pre-recorded CF database is matched against observed CFs

**Results:**
- Room-level accuracy (correct room identification) > 90% in multi-room office environments
- Demonstrated that CFs remain **stable over weeks** but shift measurably with temperature
- Works with commodity smartphones (no hardware modification)

**Relevance to Our Approach:**
LiTell validates the fundamental premise — fluorescent lights have unique, stable frequency
signatures. However, LiTell is limited by rolling-shutter decoding bandwidth (~30 kHz) and
cannot resolve sub-room positioning. Our event camera approach offers:
- **MHz-level temporal resolution** → can resolve much finer frequency differences
- **Per-pixel spatial resolution** → simultaneously fingerprint all visible lights with known
  angular positions
- **No motion artifacts** → rolling shutter requires controlled camera motion; DVS does not

---

### 2.2 iLAMP — Zhu & Zhang (MobiSys 2017)

> S. Zhu and X. Zhang, "Enabling high-precision visible light localization in today's buildings,"
> Proc. 15th Annual International Conference on Mobile Systems, Applications, and Services
> (MobiSys), 2017. **137 citations.**

**Core Contribution:**
Extends LiTell to **sub-meter precision** by exploiting "hidden visual features" of conventional
LEDs and fluorescent lamps. Key insight: even lamps that appear steady to human eyes exhibit
imperceptible flicker at frequencies determined by their driver electronics.

**Method:**
- Uses smartphone camera to capture these hidden features
- Combines frequency-domain features with spatial light distribution models
- Machine-learning-based matching for high-precision localization

**Results:**
- Median error of **0.4 m** in real office environments
- Works with both fluorescent tubes and LED panels
- Demonstrates that feature extraction is possible even from "non-flickering" LED drivers

**Relevance:**
Proves sub-meter accuracy is achievable with unmodified lights. The event camera's superior
temporal resolution should enable even finer frequency discrimination, potentially pushing
accuracy below 0.4 m. Also confirms that **LED panels** (not just fluorescents) carry usable
frequency signatures — important since our target environment has mixed/unknown light types.

---

### 2.3 Visible Light Localization Using Conventional Light Fixtures — Zhang & Zhang (IEEE TMC 2018)

> C. Zhang and X. Zhang, "Visible light localization using conventional light fixtures and
> smartphones," IEEE Transactions on Mobile Computing, vol. 18, no. 11, pp. 2736–2749, 2018.
> **38 citations.**

**Core Contribution:**
Journal extension of LiTell. Provides deeper analysis of:
- **Frequency stability** over time (days/weeks)
- **Temperature dependence** of ballast resonance frequency
- **Fingerprinting vs. triangulation** trade-offs
- **Scalability** to large indoor environments with hundreds of fixtures

**Key Data Points:**
- Electronic ballast resonance frequencies typically span **25–60 kHz** range
- Inter-fixture frequency separation: typically **200–2000 Hz** between adjacent fixtures
- Same-model fixtures can differ by **500–3000 Hz** due to component tolerances
- Frequency drift with temperature: approximately **50–200 Hz per 10°C**

---

## 3. Passive Flicker Fingerprinting

### 3.1 Munir & Dyo (IEEE Sensors Journal 2019)

> B. Munir and V. Dyo, "Passive localization through light flicker fingerprinting,"
> IEEE Sensors Journal, vol. 19, no. 20, pp. 9548–9558, 2019. **15 citations.**

**Core Contribution:**
Demonstrates that a **simple photodiode** can identify individual light sources purely from
their flicker **waveform** (time-domain shape), not just frequency. Tested on compact
fluorescent lights (CFLs) and standard incandescent bulbs.

**Method:**
- Samples flicker waveform at high speed using a photodiode + ADC
- Extracts features: dominant frequency, harmonic ratios, waveform shape descriptors
- k-NN classifier for source identification
- Position estimated from known light positions + identification confidence

**Results:**
- **96% identification accuracy** for individual CFL bulbs from the same manufacturer
- Proved that even bulbs of the same make/model have distinguishable waveforms
- Harmonic structure (2nd, 3rd, 4th harmonic amplitudes) is the most discriminative feature

**Key Insight for Our Approach:**
The harmonic structure — not just the fundamental frequency — is critical for distinguishing
same-model fixtures. Our system should extract **harmonic ratio vectors** as part of the
fingerprint, using the event camera's high temporal resolution to resolve higher harmonics.

---

### 3.2 Spectral-Loc — Wang et al. (ACM MobiSys 2023)

> Y. Wang, J. Hu, H. Jia, W. Hu, M. Hassan, et al., "Spectral-Loc: Indoor localization using
> light spectral information," Proc. ACM MobiSys, 2023. **16 citations.**

**Core Contribution:**
Goes beyond temporal flicker to use the **spectral distribution** (wavelength composition)
of received light as a location fingerprint. Different locations receive different spectral
mixes due to surface reflections from nearby colored materials.

**Method:**
- Uses a spectrometer or multi-channel color sensor
- Captures spectral fingerprint at each calibration point
- Location estimation via spectral fingerprint matching

**Results:**
- Median positioning error of **0.5 m**
- Confirms that Zhang & Zhang's (2016) resonance frequency approach and spectral approaches
  are complementary

**Relevance:**
Spectral features are orthogonal to temporal frequency features. A future extension of our
system could fuse frequency fingerprinting (from event camera) with spectral fingerprinting
(from an additional color sensor) for improved robustness.

---

## 4. Event-Camera-Based Visible Light Positioning

### 4.1 Chen et al. (IEEE Sensors Journal 2020)

> G. Chen, W. Chen, Q. Yang, Z. Xu, L. Yang, J. Conradt, and A. Knoll, "A novel visible light
> positioning system with event-based neuromorphic vision sensor," IEEE Sensors Journal, vol. 20,
> no. 17, pp. 10211–10219, 2020. **61 citations.**

**Core Contribution:**
**First VLP system using an event camera (DVS) as the light receiver.** Demonstrates that DVS
can identify multiple simultaneously flickering LEDs by their distinct frequencies in the
asynchronous event stream.

**Method:**
- LEDs modulated at different frequencies (1–10 kHz range) serve as VLC beacons
- DVS captures events triggered by each LED's ON/OFF transitions
- Per-pixel frequency extraction via inter-event-interval (IEI) analysis
- LED identification from frequency → position estimation via triangulation

**Results:**
- Successfully identifies **5+ LEDs simultaneously** in the field of view
- Positioning error < **5 cm** at 2 m range
- Works under high ambient light conditions where traditional cameras fail
- **50× lower latency** than frame-based VLP systems

**Critical Relevance:**
This paper validates the core technical approach of our system — using a DVS to extract
per-pixel light flicker frequencies for positioning. The key difference: Chen et al. use
**intentionally modulated** LEDs with known frequencies, while we aim to use
**naturally occurring** flicker from unmodified fluorescent/LED fixtures. This makes our
approach **infrastructure-free** but requires the additional step of frequency fingerprinting.

---

### 4.2 Temporal Feature Markers for Event Cameras — You et al. (JRTIP 2024)

> Y. You, M. Zhu, B. He, Y. Wang, "Temporal feature markers for event cameras,"
> Journal of Real-Time Image Processing, vol. 21, 2024. **2 citations.**

**Core Contribution:**
Designs active fiducial markers that flicker at controlled frequencies specifically to be
detected by event cameras. Investigates the error of positioning static markers using their
flicker frequency.

**Relevance:**
While our approach targets unmodified lights, this work provides valuable insight into:
- Event camera frequency detection precision as a function of distance
- Optimal frequency ranges for reliable detection
- Error models for frequency-based position estimation
- This represents the "enhanced mode" — if natural fluorescent frequencies aren't sufficiently
  distinct, adding small frequency-shift circuits to each light is a viable fallback

---

### 4.3 Structured Light with Flickering Patterns — Fujimoto et al. (IEEE 2022)

> Y. Fujimoto, T. Sawabe, M. Kanbara, et al., "Structured light of flickering patterns having
> different frequencies for a projector-event-camera system," IEEE Conference, 2022. **9 citations.**

**Core Contribution:**
Proves that event cameras can reliably **separate spatially overlapping signals** at different
flicker frequencies. Uses a projector to create structured light patterns where adjacent
regions flicker at different rates.

**Relevance:**
Directly applicable to our scenario where multiple ceiling lights with overlapping illumination
cones are visible simultaneously. The event camera's per-pixel frequency extraction provides
natural spatial separation even when light footprints physically overlap on surfaces below.

---

## 5. Event Camera Flicker Characterization & Removal

### 5.1 Linear Comb Filter for Event Flicker Removal — Wang et al. (ICRA 2022)

> Z. Wang, D. Yuan, Y. Ng, R. Mahony, "A linear comb filter for event flicker removal,"
> IEEE Int. Conf. on Robotics and Automation (ICRA), 2022. **30 citations.**

**Core Contribution:**
Characterizes fluorescent and LED light flicker as it appears in event camera data streams.
Proposes a linear comb filter to remove flicker events (which are noise for robotics SLAM).

**Key Technical Details:**
- Fluorescent lights produce **periodic ON/OFF event bursts** at 100/120 Hz (2× mains)
- Electronic ballast harmonics visible at 200/240, 300/360, 400/480 Hz in the event stream
- Each pixel under a flickering light generates a **near-periodic event train**
- The filter models flicker as a sum of harmonics: f(t) = Σ aₖ · cos(2πkf₀t + φₖ)

**Paradoxical Relevance:**
This paper treats flicker as **noise to be removed**. For our application, flicker is the
**signal to be exploited**. Understanding the flicker's structure in the event domain — as
characterized by this work — directly informs our feature extraction pipeline. The harmonic
model f(t) = Σ aₖ · cos(2πkf₀t + φₖ) with per-fixture unique (f₀, {aₖ}, {φₖ}) is exactly
the fingerprint we aim to capture.

---

### 5.2 PINK: Polarity-based Anti-flicker — Im et al. (CVPRW 2023)

> G. Im, K. Park, J. Kim, B. Son, S. Shin, et al., "Live demonstration: PINK: Polarity-based
> anti-flicker for event cameras," CVPR Workshop, 2023. **8 citations.**

**Core Contribution:**
Proposes hardware/firmware-level flicker filtering using event polarity patterns.
Demonstrates that flicker events have a **characteristic alternating ON-OFF polarity pattern**
that can be detected and suppressed in real-time.

**Relevance:**
The polarity pattern described (alternating positive/negative events at the flicker frequency)
is another discriminative feature we can add to our fingerprint vector. Different ballast
types produce different ON/OFF duty cycles, creating distinct polarity ratio signatures.

---

### 5.3 Identifying Light Interference in Event-Based Vision — Shi et al. (IEEE Trans. 2023)

> C. Shi, Y. Li, N. Song, B. Wei, Y. Zhang, et al., "Identifying light interference in
> event-based vision," IEEE Transactions on Instrumentation and Measurement, 2023. **14 citations.**

**Core Contribution:**
Proposes methods to detect and classify different types of light interference (flicker) in
event camera data. Categorizes interference by frequency band and temporal structure.

**Relevance:**
Provides a taxonomy of light interference types that maps directly to our light-type
classification problem:
- **Band 1 (50–120 Hz):** Magnetic ballast fluorescent, incandescent on AC dimmers
- **Band 2 (100–500 Hz):** LED drivers with low-cost regulators
- **Band 3 (1–60 kHz):** Electronic ballast fluorescent
- **Band 4 (60+ kHz):** High-end LED drivers, switch-mode power supplies

---

## 6. Complementary VLP Surveys & Tutorials

### 6.1 A Survey of Positioning Systems Using Visible LED Lights — Zhuang et al. (IEEE COMST 2018)

> Y. Zhuang, L. Hua, L. Qi, J. Yang, P. Cao, et al., "A survey of positioning systems using
> visible LED lights," IEEE Communications Surveys & Tutorials, vol. 20, no. 3, 2018.
> **754 citations.**

Comprehensive survey covering VLP taxonomy: RSS-based, AOA-based, TDOA-based, fingerprint-based.
Includes comparison of receiver types (photodiode, camera, solar cell). Notes that fluorescent
lamp VLP is under-explored compared to LED-VLC approaches.

### 6.2 VLP as Next-Generation Indoor Positioning — Bastiaens et al. (IEEE COMST 2024)

> S. Bastiaens, M. Alijani, W. Joseph, et al., "Visible light positioning as a next-generation
> indoor positioning technology: A tutorial," IEEE Communications Surveys & Tutorials, 2024.
> **92 citations.**

Most recent comprehensive tutorial. Discusses near-zero electromagnetic signature of VLP
compared to RF, and the transition from fluorescent to LED infrastructure. Notes that
frequency-division multiplexing for VLP aligns with our approach of using natural frequency
variation as an identifier.

### 6.3 Unmodulated VLP — Shi et al. (Sensors 2020)

> C. Shi, X. Niu, T. Li, S. Li, C. Huang, Q. Niu, "Exploring fast fingerprint construction
> algorithm for unmodulated visible light indoor localization," Sensors, 2020. **8 citations.**

Proposes reverse fingerprint collection for unmodulated VLP using color polarizers. Achieves
**< 10 cm accuracy in 95% of cases** in a 4×4×2 m room. Demonstrates that fingerprint
databases collected in one environment can be reused in different environments.

---

## 7. Summary: State of the Art & Research Gap

| Approach | Best Accuracy | Sensor | Infrastructure | Limitations |
|----------|--------------|--------|---------------|-------------|
| LiTell (2016) | Room-level | Smartphone rolling shutter | None (unmodified FL) | Low resolution, requires motion |
| iLAMP (2017) | 0.4 m | Smartphone camera | None (unmodified FL/LED) | Computationally expensive |
| Munir & Dyo (2019) | Room-level | Photodiode | None (unmodified CFL) | No spatial info, single point |
| Chen et al. (2020) | < 5 cm | Event camera (DVS) | Modified (modulated LEDs) | Requires LED modulation HW |
| Spectral-Loc (2023) | 0.5 m | Spectrometer | None | Requires spectrometer sensor |
| Shi et al. (2020) | < 10 cm | Smartphone camera | Color polarizers on lights | Requires physical light modification |

### Research Gap — Our Contribution

**No existing work combines:**
1. ✅ Event camera (DVS) as receiver — µs temporal resolution, per-pixel spatial resolution
2. ✅ Unmodified fluorescent/LED lights — zero infrastructure cost
3. ✅ Frequency fingerprinting — exploiting natural frequency variation between fixtures

Our approach fills this gap by applying the DVS's unique capabilities (validated by Chen et al.
for modulated VLP) to the unmodified-light fingerprinting paradigm (validated by LiTell and
Munir & Dyo). The expected advantages:
- **Higher frequency resolution** than rolling-shutter methods (µs vs. ms)
- **Spatial + temporal** information simultaneously (vs. photodiode = temporal only)
- **No infrastructure modification** required (vs. Chen et al.'s modulated LEDs)
- **Robust to ambient conditions** — sunlight is DC → zero flicker → natural immunity

---

## 8. References (Chronological)

1. C. Zhang and X. Zhang, "LiTell: Robust indoor localization using unmodified light fixtures," MobiCom 2016.
2. S. Zhu and X. Zhang, "Enabling high-precision visible light localization in today's buildings," MobiSys 2017.
3. Y. Zhuang et al., "A survey of positioning systems using visible LED lights," IEEE COMST, 2018.
4. C. Zhang and X. Zhang, "Visible light localization using conventional light fixtures and smartphones," IEEE TMC, 2018.
5. B. Munir and V. Dyo, "Passive localization through light flicker fingerprinting," IEEE Sensors J., 2019.
6. G. Chen et al., "A novel visible light positioning system with event-based neuromorphic vision sensor," IEEE Sensors J., 2020.
7. C. Shi et al., "Exploring fast fingerprint construction algorithm for unmodulated visible light indoor localization," Sensors, 2020.
8. Y. Fujimoto et al., "Structured light of flickering patterns having different frequencies for a projector-event-camera system," IEEE Conf., 2022.
9. Z. Wang et al., "A linear comb filter for event flicker removal," ICRA, 2022.
10. G. Im et al., "Live demonstration: PINK: Polarity-based anti-flicker for event cameras," CVPRW, 2023.
11. C. Shi et al., "Identifying light interference in event-based vision," IEEE Trans. Instrum. Meas., 2023.
12. Y. Wang et al., "Spectral-Loc: Indoor localization using light spectral information," ACM MobiSys, 2023.
13. Y. You et al., "Temporal feature markers for event cameras," J. Real-Time Image Processing, 2024.
14. S. Bastiaens et al., "Visible light positioning as a next-generation indoor positioning technology: A tutorial," IEEE COMST, 2024.
15. Y. Wang, "Exploiting Visible Light for High-Precision Indoor Localization and Robust Communication," PhD Thesis, UNSW, 2025.
