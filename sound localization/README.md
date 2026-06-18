# ReSpeaker Mic Array v2.0 — Speech Direction Detection

Real-time speech detection with **Direction of Arrival (DOA)** using the ReSpeaker 4-Mic Array v2.0.

---

## Features

- **Speech-only logging** — Only logs when speech frequencies (300-3400 Hz) detected
- **DOA estimation** — Calculates direction using GCC-PHAT algorithm on 4-mic array
- **LED ring visualization** — Shows which of the 12 LEDs corresponds to speech direction
- **Cardinal directions** — Displays N, NE, E, SE, S, SW, W, NW

---

## Quick Start

```powershell
# 1. Create virtual environment (first time only)
python -m venv .venv

# 2. Activate and install dependencies
.\.venv\Scripts\Activate.ps1
pip install pyaudio numpy

# 3. Run the listener
python app\listen.py
```

---

## Example Output

```
✔ Found device [1]: ReSpeaker 4 Mic Array (UAC1.0)

Listening... (channel 1, resolution ~3.9 Hz/bin)
Logs only when speech detected (300-3400 Hz)
────────────────────────────────────────────────────────────────────────────────────────────────────
TIME        | DIR    | LED | RMS   | SNR  | PITCH | VOICE | CONF | FREQ
────────────────────────────────────────────────────────────────────────────────────────────────────
[15:47:05.329]  40.4° NE  |  1  | -19.5 | +13.3 | 229Hz |   F   | █░░░░ | 469Hz, 672Hz  [·●··········]
[15:47:13.259]  21.3° NNE |  1  | -19.1 | +1.6  | 154Hz |   M   | █░░░░ | 422Hz         [·●··········]
[15:47:16.333] 279.7° W   |  9  | -12.5 | +17.9 | 229Hz |   F   | █░░░░ | 457Hz         [·········●··]
```

### Log Columns

| Column | Description |
|--------|-------------|
| **TIME** | Timestamp (HH:MM:SS.ms) |
| **DIR** | DOA angle (degrees) + cardinal direction |
| **LED** | LED index (0-11) for visual direction |
| **RMS** | Loudness in dB (higher = louder) |
| **SNR** | Signal-to-noise ratio (higher = cleaner) |
| **PITCH** | Fundamental frequency (F0) in Hz |
| **VOICE** | Voice type: M=Male, F=Female, C=Child |
| **CONF** | DOA confidence (█████ = high confidence) |
| **FREQ** | Dominant speech frequencies |

---

## Project Structure

```
C:\D\Respeaker\
├── README.md           # This file
├── .venv/              # Python virtual environment
└── app/
    └── listen.py       # Speech detection + DOA + LED display
```

---

## Requirements

- **Windows 10/11**
- **Python 3.10+**
- **ReSpeaker 4-Mic Array v2.0** (VID `0x2886`, PID `0x0018`)

### Python Packages

```
pyaudio
numpy
```

---

## Hardware

| Spec | Value |
|------|-------|
| Microphones | 4x MEMS (circular array, 46mm diameter) |
| Sample Rate | 16 kHz |
| Channels | 6 (4 raw mics + 2 DSP processed) |
| LEDs | 12 RGB (APA102) |
| Interface | USB Audio Class 1.0 |

---

## How It Works

1. **Audio Capture** — Reads 6-channel audio at 16 kHz via PyAudio
2. **Frequency Analysis** — FFT to find dominant frequencies
3. **Speech Detection** — Filters for 300-3400 Hz band
4. **DOA Estimation** — GCC-PHAT cross-correlation between opposite mic pairs
5. **Pitch Detection** — Autocorrelation to find fundamental frequency (F0)
6. **Voice Classification** — Male (<165Hz), Female (165-255Hz), Child (>255Hz)
7. **SNR Calculation** — Compares speech band energy to noise floor
8. **Direction Mapping** — Converts angle to cardinal direction + LED index

---

## Metrics Reference

### DOA (Direction of Arrival)

| Metric | Range | Description |
|--------|-------|-------------|
| **Angle** | 0° - 360° | Direction of sound source relative to microphone array |
| **Cardinal** | N, NE, E, SE, S, SW, W, NW | 16-point compass direction (includes NNE, ENE, etc.) |
| **LED** | 0 - 11 | Which LED on the ring corresponds to direction |
| **Confidence** | 0% - 100% | GCC-PHAT correlation strength (higher = more reliable) |

**Algorithm:** GCC-PHAT (Generalized Cross-Correlation with Phase Transform)
- Uses time delay between opposite mic pairs (0↔2, 1↔3)
- Smoothed over 5 frames to reduce jitter

### Audio Level

| Metric | Range | Description |
|--------|-------|-------------|
| **RMS** | -60 to 0 dB | Root Mean Square loudness |
| **SNR** | -10 to +40 dB | Signal-to-Noise Ratio |

**Typical RMS Values:**
| Level | dB Range | Example |
|-------|----------|---------|
| Silence | < -50 | Empty room |
| Whisper | -45 to -35 | Quiet speech |
| Normal | -35 to -20 | Conversational |
| Loud | -20 to -10 | Raised voice |
| Shouting | > -10 | Very loud |

**SNR Interpretation:**
| SNR | Quality |
|-----|---------|
| < 0 dB | Very noisy, speech may be masked |
| 0-10 dB | Noisy environment |
| 10-20 dB | Moderate noise |
| 20-30 dB | Clean audio |
| > 30 dB | Very clean, quiet environment |

### Pitch (Fundamental Frequency)

| Metric | Range | Description |
|--------|-------|-------------|
| **F0** | 75 - 400 Hz | Fundamental frequency of voice |
| **Voice Type** | M / F / C | Classification based on pitch |

**Voice Classification:**
| Type | Code | F0 Range | Description |
|------|------|----------|-------------|
| Male | M | 85 - 165 Hz | Adult male voice |
| Female | F | 165 - 255 Hz | Adult female voice |
| Child | C | > 255 Hz | Child or high-pitched voice |
| Unvoiced | --- | N/A | No clear pitch detected |

**Algorithm:** Autocorrelation pitch detection
- Finds periodicity in the waveform
- Only reports pitch if correlation strength > 30%

### Frequency Analysis

| Metric | Description |
|--------|-------------|
| **Dominant Freq** | Top frequencies in speech band (300-3400 Hz) |
| **dB Level** | Amplitude of each frequency peak |

**Frequency Bands:**
| Band | Range | Typical Content |
|------|-------|-----------------|
| Sub-bass | 20-60 Hz | Rumble, HVAC noise |
| Bass | 60-250 Hz | Male voice fundamentals |
| Low-mid | 250-500 Hz | Voice body, warmth |
| Mid | 500-2000 Hz | Voice clarity, consonants |
| High-mid | 2000-4000 Hz | Presence, sibilance |
| High | 4000-8000 Hz | Air, brightness |

---

## Configuration

Edit these constants in `app/listen.py`:

```python
RATE       = 16000   # Sample rate (Hz)
CHANNELS   = 6       # Number of channels
CHUNK      = 4096    # Samples per frame (affects frequency resolution)
MIC_CH     = 1       # Which mic channel to analyze (0=DSP, 1-4=raw mics)
```

**Frequency Resolution:** `RATE / CHUNK` = ~3.9 Hz per bin at default settings
