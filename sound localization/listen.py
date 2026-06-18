#!/usr/bin/env python3
"""
ReSpeaker Mic Array v2.0 — Real-time Frequency Analysis

Captures audio from the ReSpeaker and displays dominant frequencies.
"""

import pyaudio
import numpy as np
import time
from datetime import datetime
from collections import deque

# ─── Config ─────────────────────────────────────────────────
RATE       = 16000
CHANNELS   = 6       # 6-channel firmware; use 1 for single-channel firmware
CHUNK      = 4096    # larger = better frequency resolution (resolution = RATE/CHUNK = ~3.9 Hz)
FORMAT     = pyaudio.paInt16
MIC_CH     = 1       # raw mic channel (1–4); 0 = DSP-processed
SPEED_OF_SOUND = 343.0  # m/s

# ─── Find ReSpeaker ──────────────────────────────────────────
def find_respeaker():
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        name = info['name'].lower()
        if 'respeaker' in name or 'seeed' in name or 'usb audio' in name:
            print(f"✔ Found device [{i}]: {info['name']}")
            p.terminate()
            return i
    p.terminate()
    # fallback: list all inputs for manual selection
    p = pyaudio.PyAudio()
    print("\nNo ReSpeaker found. Available input devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"  [{i}] {info['name']} ({int(info['maxInputChannels'])}ch)")
    p.terminate()
    idx = int(input("Enter device index manually: "))
    return idx

# ─── DOA Estimation (GCC-PHAT) ───────────────────────────────
def gcc_phat(sig1, sig2, fs, max_tau=None):
    """GCC-PHAT cross-correlation for time delay estimation"""
    n = sig1.shape[0] + sig2.shape[0]
    SIG1 = np.fft.rfft(sig1, n=n)
    SIG2 = np.fft.rfft(sig2, n=n)
    R = SIG1 * np.conj(SIG2)
    R = R / (np.abs(R) + 1e-10)
    cc = np.fft.irfft(R, n=n)
    cc = np.concatenate((cc[-len(sig1)+1:], cc[:len(sig2)]))
    
    if max_tau:
        max_shift = int(max_tau * fs)
        center = len(sig1) - 1
        cc = cc[center - max_shift:center + max_shift + 1]
    
    shift = np.argmax(np.abs(cc)) - (len(cc) // 2)
    return shift / fs

def estimate_doa(samples, fs=RATE):
    """Estimate direction of arrival from 4-mic array"""
    # Deinterleave all 4 mic channels
    mics = [samples[i::CHANNELS].astype(np.float32) / 32768.0 for i in range(4)]
    
    # Max delay based on 46mm mic spacing
    max_tau = 0.023 * 2 / SPEED_OF_SOUND
    
    # TDOA between opposite mics
    tau_02 = gcc_phat(mics[0], mics[2], fs, max_tau)
    tau_13 = gcc_phat(mics[1], mics[3], fs, max_tau)
    
    # Convert to angle
    angle = np.arctan2(tau_13, tau_02) * 180 / np.pi
    angle = (angle + 360) % 360
    
    return angle

def angle_to_direction(angle):
    """Convert angle to cardinal direction"""
    dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
            'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    return dirs[int((angle + 11.25) / 22.5) % 16]

def angle_to_led(angle):
    """Convert angle to LED index (0-11)"""
    return int(round(angle / 30)) % 12

# ─── Audio Analysis Functions ────────────────────────────────
def estimate_pitch(frame, fs, min_hz=75, max_hz=400):
    """
    Estimate fundamental frequency (F0) using autocorrelation.
    Returns pitch in Hz or 0 if unvoiced/no pitch detected.
    """
    # Autocorrelation method
    corr = np.correlate(frame, frame, mode='full')
    corr = corr[len(corr)//2:]  # Keep positive lags only
    
    # Find first peak after min_lag
    min_lag = int(fs / max_hz)
    max_lag = int(fs / min_hz)
    
    if max_lag > len(corr):
        max_lag = len(corr) - 1
    
    # Find the highest peak in the valid lag range
    segment = corr[min_lag:max_lag]
    if len(segment) == 0:
        return 0
    
    peak_idx = np.argmax(segment) + min_lag
    
    # Check if it's a valid pitch (correlation strength)
    if corr[peak_idx] < 0.3 * corr[0]:
        return 0
    
    return fs / peak_idx

def calculate_rms_db(frame):
    """Calculate RMS level in dB"""
    rms = np.sqrt(np.mean(frame ** 2))
    if rms < 1e-10:
        return -100
    return 20 * np.log10(rms)

def calculate_snr(frame, fs, speech_band=(300, 3400)):
    """Estimate SNR by comparing speech band energy to total"""
    window = np.hanning(len(frame))
    fft_mag = np.abs(np.fft.rfft(frame * window)) ** 2
    freqs = np.fft.rfftfreq(len(frame), d=1.0/fs)
    
    # Speech band energy
    speech_mask = (freqs >= speech_band[0]) & (freqs <= speech_band[1])
    speech_energy = np.sum(fft_mag[speech_mask])
    
    # Noise band (below and above speech)
    noise_mask = (freqs < speech_band[0]) | (freqs > speech_band[1])
    noise_energy = np.sum(fft_mag[noise_mask])
    
    if noise_energy < 1e-10:
        return 40  # Very clean
    
    snr = 10 * np.log10(speech_energy / noise_energy + 1e-10)
    return max(-10, min(40, snr))  # Clamp to reasonable range

def estimate_doa_with_confidence(samples, fs=RATE):
    """Estimate DOA with confidence score based on correlation peak"""
    mics = [samples[i::CHANNELS].astype(np.float32) / 32768.0 for i in range(4)]
    max_tau = 0.023 * 2 / SPEED_OF_SOUND
    
    # Get correlation peaks for confidence
    def gcc_phat_with_peak(sig1, sig2, fs, max_tau):
        n = sig1.shape[0] + sig2.shape[0]
        SIG1 = np.fft.rfft(sig1, n=n)
        SIG2 = np.fft.rfft(sig2, n=n)
        R = SIG1 * np.conj(SIG2)
        R = R / (np.abs(R) + 1e-10)
        cc = np.fft.irfft(R, n=n)
        cc = np.concatenate((cc[-len(sig1)+1:], cc[:len(sig2)]))
        
        max_shift = int(max_tau * fs)
        center = len(sig1) - 1
        cc = cc[center - max_shift:center + max_shift + 1]
        
        peak_val = np.max(np.abs(cc))
        shift = np.argmax(np.abs(cc)) - (len(cc) // 2)
        return shift / fs, peak_val
    
    tau_02, conf_02 = gcc_phat_with_peak(mics[0], mics[2], fs, max_tau)
    tau_13, conf_13 = gcc_phat_with_peak(mics[1], mics[3], fs, max_tau)
    
    angle = np.arctan2(tau_13, tau_02) * 180 / np.pi
    angle = (angle + 360) % 360
    
    # Confidence is average of correlation peaks (0-1 scale)
    confidence = (conf_02 + conf_13) / 2
    confidence = min(1.0, max(0.0, confidence))
    
    return angle, confidence

def pitch_to_voice_type(pitch):
    """Classify voice type based on pitch"""
    if pitch == 0:
        return "---"
    elif pitch < 165:
        return "M"   # Male typical: 85-180 Hz
    elif pitch < 255:
        return "F"   # Female typical: 165-255 Hz
    else:
        return "C"   # Child: > 255 Hz

# ─── Top-N Frequency Picker ──────────────────────────────────
def top_n_frequencies(frame, fs, n=3, min_hz=40, max_hz=8000,
                       min_db=-60, min_distance_hz=50):
    """
    Returns the top-n dominant frequencies with their amplitudes.

    min_distance_hz: minimum separation between peaks (avoids
                     returning harmonics of the same tone as separate peaks)
    """
    window  = np.hanning(len(frame))
    fft_mag = np.abs(np.fft.rfft(frame * window))
    freqs   = np.fft.rfftfreq(len(frame), d=1.0 / fs)
    db      = 20 * np.log10(fft_mag / (len(frame) / 2) + 1e-10)

    # Band-limit
    mask        = (freqs >= min_hz) & (freqs <= max_hz)
    freqs_band  = freqs[mask]
    db_band     = db[mask]

    # Simple peak-picking with minimum distance constraint
    peaks = []
    db_copy = db_band.copy()
    freq_res = freqs_band[1] - freqs_band[0]          # Hz per bin
    min_bins = int(min_distance_hz / freq_res)

    for _ in range(n):
        idx = np.argmax(db_copy)
        amplitude = db_copy[idx]
        if amplitude < min_db:
            break
        peaks.append((freqs_band[idx], amplitude))

        # Zero out nearby bins so next peak is distinct
        lo = max(0, idx - min_bins)
        hi = min(len(db_copy), idx + min_bins)
        db_copy[lo:hi] = -np.inf

    return peaks

# ─── Simple ASCII Bar ────────────────────────────────────────
def bar(db_val, min_db=-60, max_db=0, width=20):
    ratio = (db_val - min_db) / (max_db - min_db)
    ratio = max(0.0, min(1.0, ratio))
    filled = int(ratio * width)
    return '█' * filled + '░' * (width - filled)

# ─── Main ────────────────────────────────────────────────────
def listen(n_peaks=3, channel=MIC_CH):
    idx = find_respeaker()
    p   = pyaudio.PyAudio()

    stream = p.open(
        format            = FORMAT,
        channels          = CHANNELS,
        rate              = RATE,
        input             = True,
        input_device_index= idx,
        frames_per_buffer = CHUNK,
    )

    print(f"\nListening... (channel {channel}, resolution ~{RATE/CHUNK:.1f} Hz/bin)")
    print("Logs only when speech detected (300-3400 Hz)")
    print("─" * 100)
    print("TIME        | DIR    | LED | RMS   | SNR  | PITCH | VOICE | CONF | FREQ")
    print("─" * 100)

    # DOA smoothing
    doa_history = deque(maxlen=5)
    last_speech_time = 0
    speech_cooldown = 0.3  # seconds between logs
    speech_start = None    # Track speech duration
    
    # Background noise estimation (first second)
    noise_floor = -50

    try:
        while True:
            raw     = stream.read(CHUNK, exception_on_overflow=False)
            samples = np.frombuffer(raw, dtype=np.int16)

            # Deinterleave: pick chosen mic channel for freq analysis
            frame   = samples[channel::CHANNELS].astype(np.float32)
            frame  /= 32768.0

            peaks = top_n_frequencies(
                frame, RATE,
                n               = n_peaks,
                min_hz          = 40,
                max_hz          = 8000,
                min_db          = -55,
                min_distance_hz = 60,
            )

            # Check for speech frequencies (300-3400 Hz)
            speech_peaks = [(f, a) for f, a in peaks if 300 <= f <= 3400]
            
            if speech_peaks:
                now = time.time()
                
                # Track speech duration
                if speech_start is None:
                    speech_start = now
                
                # Estimate DOA with confidence
                doa, confidence = estimate_doa_with_confidence(samples, RATE)
                doa_history.append(doa)
                
                # Circular mean for smoothed angle
                sin_sum = np.sum([np.sin(np.radians(a)) for a in doa_history])
                cos_sum = np.sum([np.cos(np.radians(a)) for a in doa_history])
                smooth_doa = np.degrees(np.arctan2(sin_sum, cos_sum)) % 360
                
                direction = angle_to_direction(smooth_doa)
                led = angle_to_led(smooth_doa)
                
                # Calculate audio metrics
                rms_db = calculate_rms_db(frame)
                snr = calculate_snr(frame, RATE)
                pitch = estimate_pitch(frame, RATE)
                voice_type = pitch_to_voice_type(pitch)
                
                # Only log if cooldown passed
                if now - last_speech_time > speech_cooldown:
                    last_speech_time = now
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    
                    # Format frequencies
                    freq_str = ", ".join([f"{f:.0f}Hz" for f, a in speech_peaks[:2]])
                    
                    # LED ring visualization (12 LEDs)
                    ring = ['·'] * 12
                    ring[led] = '●'
                    ring_str = ''.join(ring)
                    
                    # Confidence bar (5 chars)
                    conf_bars = int(confidence * 5)
                    conf_str = '█' * conf_bars + '░' * (5 - conf_bars)
                    
                    # Pitch display
                    pitch_str = f"{pitch:3.0f}Hz" if pitch > 0 else "  ---"
                    
                    print(f"[{timestamp}] {smooth_doa:5.1f}° {direction:3s} | "
                          f"{led:2d}  | {rms_db:+5.1f} | {snr:+4.1f} | {pitch_str} |   {voice_type}   | "
                          f"{conf_str} | {freq_str:20s} [{ring_str}]")
            else:
                # Reset speech tracking when silence
                speech_start = None

    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == "__main__":
    listen(n_peaks=3)
