"""
Fan calibration utility: captures audio from ReSpeaker 4-mic array and identifies
the fan's blade-pass frequency.

Usage:
    python calibrate_fan_frequency.py
    
    Turn on your fan when prompted. The script will display real-time status
    and identify the dominant frequency peaks.
"""

import numpy as np
import time
import threading
import json
import sys

try:
    import pyaudio
except ImportError:
    print("[Init] PyAudio not available. Install with: pip install pyaudio")
    exit(1)


class FanCalibrator:
    """Captures audio and identifies fan blade-pass frequency."""
    
    def __init__(self, sample_rate=16000, chunk_size=1024, n_channels=4,
                 capture_duration_sec=10.0, device_index=None):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.n_channels = n_channels
        self.capture_duration_sec = capture_duration_sec
        self.device_index = device_index
        
        max_samples = int(capture_duration_sec * sample_rate)
        self.buffers = [np.zeros(max_samples, dtype=np.float64)
                       for _ in range(n_channels)]
        self.sample_count = 0
        self.latest_spectrum = None
        self.latest_freqs = None
        self._running = False
        self._thread = None
        self.capture_complete = False
    
    def start(self):
        """Start audio capture thread."""
        self._running = True
        self.capture_complete = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"[Audio] Capture starting... (duration: {self.capture_duration_sec}s)")
    
    def stop(self):
        """Stop capture and wait for thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=15.0)
    
    def _run(self):
        """Background audio capture thread."""
        pa = pyaudio.PyAudio()
        
        # List available devices
        print(f"[Audio] Available devices:")
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            print(f"  [{i}] {info['name']} ({info['maxInputChannels']} in)")
        
        try:
            device_idx = self.device_index
            if device_idx is None:
                # Try to find ReSpeaker
                for i in range(pa.get_device_count()):
                    info = pa.get_device_info_by_index(i)
                    if 'respeaker' in info['name'].lower():
                        device_idx = i
                        print(f"[Audio] Found ReSpeaker at device {i}")
                        break
            
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=self.n_channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                input_device_index=device_idx,
            )
        except Exception as e:
            print(f"[Audio] Failed to open stream: {e}")
            pa.terminate()
            return
        
        print(f"[Audio] Opened stream: {self.sample_rate} Hz, {self.n_channels} ch, "
              f"chunk={self.chunk_size}")
        print(f"[Audio] === TURN ON YOUR FAN NOW ===")
        print()
        
        try:
            start_time = time.time()
            chunk_count = 0
            total_samples_needed = int(self.capture_duration_sec * self.sample_rate)
            
            while self._running and self.sample_count < total_samples_needed:
                try:
                    raw = stream.read(self.chunk_size, exception_on_overflow=False)
                except Exception as e:
                    print(f"[Audio] Read error: {e}")
                    continue
                
                # Deinterleave
                samples = np.frombuffer(raw, dtype=np.int16).reshape(-1, self.n_channels)
                
                # Store in buffers
                end_idx = min(self.sample_count + len(samples),
                             total_samples_needed)
                n_store = end_idx - self.sample_count
                
                for ch in range(self.n_channels):
                    self.buffers[ch][self.sample_count:end_idx] = \
                        samples[:n_store, ch].astype(np.float64)
                
                self.sample_count = end_idx
                chunk_count += 1
                
                # Real-time progress
                elapsed = time.time() - start_time
                progress = self.sample_count / total_samples_needed
                print(f"\r[Audio] {elapsed:.1f}s - {progress*100:.0f}% "
                      f"({self.sample_count}/{total_samples_needed} samples)  ",
                      end="", flush=True)
                
                if self.sample_count >= total_samples_needed:
                    break
            
            print()  # newline after progress
            print(f"[Audio] Capture complete: {self.sample_count} samples, "
                  f"{chunk_count} chunks")
            
        except Exception as e:
            print(f"[Audio] Capture error: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()
            self.capture_complete = True
    
    def analyze(self):
        """Compute FFT and identify dominant frequencies."""
        if self.sample_count < self.sample_rate * 0.5:
            print(f"[Analysis] Insufficient samples: {self.sample_count} "
                  f"(need at least {int(self.sample_rate * 0.5)})")
            return None
        
        print(f"[Analysis] Analyzing {self.sample_count} samples from {self.n_channels} channels...")
        
        # Average across all channels
        all_spectra = []
        for ch in range(self.n_channels):
            sig = self.buffers[ch][:self.sample_count]
            sig_dc = sig - np.mean(sig)
            window = np.hanning(len(sig))
            spectrum = np.fft.rfft(sig_dc * window)
            all_spectra.append(np.abs(spectrum))
        
        avg_spectrum = np.mean(all_spectra, axis=0) * 2.0 / len(sig)
        freqs = np.fft.rfftfreq(len(sig), d=1.0 / self.sample_rate)
        
        self.latest_spectrum = avg_spectrum
        self.latest_freqs = freqs
        
        # Find peaks in 30-500 Hz range (typical fan range)
        valid = (freqs >= 30.0) & (freqs <= 500.0)
        if not np.any(valid):
            print("[Analysis] No energy in 30-500 Hz range")
            return None
        
        valid_mags = avg_spectrum[valid]
        valid_freqs = freqs[valid]
        
        # Find top 8 peaks
        peak_indices = np.argsort(valid_mags)[-8:][::-1]
        peaks = []
        median_mag = np.median(valid_mags)
        
        for idx in peak_indices:
            peak_mag = valid_mags[idx]
            snr_db = 20 * np.log10(peak_mag / (median_mag + 1e-9))
            
            peaks.append({
                "freq_hz": float(valid_freqs[idx]),
                "magnitude": float(peak_mag),
                "snr_db": snr_db
            })
        
        return peaks
    
    def print_spectrum(self, num_peaks=8):
        """Print frequency peaks with RPM conversions."""
        if self.latest_spectrum is None:
            print("[Analysis] No spectrum available")
            return
        
        peaks = self.analyze()
        if peaks is None:
            return
        
        print("\n" + "=" * 80)
        print("FAN FREQUENCY CALIBRATION RESULTS")
        print("=" * 80)
        print(f"\nTop {num_peaks} frequency peaks detected:")
        print()
        print("  #    Frequency      Magnitude       SNR      RPM(2-blade) RPM(3-blade)")
        print("  " + "-" * 76)
        
        for i, peak in enumerate(peaks[:num_peaks]):
            rpm_2blade = (peak["freq_hz"] / 2) * 60
            rpm_3blade = (peak["freq_hz"] / 3) * 60
            print(f"  {i+1:2d}   {peak['freq_hz']:7.1f} Hz   {peak['magnitude']:.2e}    "
                  f"{peak['snr_db']:+6.1f} dB    {rpm_2blade:7.0f}     {rpm_3blade:7.0f}")
        
        print("\n" + "=" * 80)
        print("DETECTOR RECOMMENDATION")
        print("=" * 80)
        
        dominant_freq = peaks[0]["freq_hz"]
        min_freq = max(30, int(dominant_freq * 0.75))
        max_freq = min(500, int(dominant_freq * 1.25))
        
        print(f"\n✓ Dominant frequency: {dominant_freq:.1f} Hz")
        print(f"✓ Use these parameters with the detector:\n")
        print(f"    --min-freq {min_freq}")
        print(f"    --max-freq {max_freq}")
        print(f"    --num-blades 2  (or 3 if it's a 3-blade fan)")
        print(f"    --audio-min-snr 3.0")
        print(f"\n✓ Example full command:\n")
        print(f"    python detect_drone_fused.py --enable-audio \\")
        print(f"      --min-freq {min_freq} --max-freq {max_freq} \\")
        print(f"      --session-log logs/fan_session.jsonl")
        
        print(f"\n✓ Calibration saved to: fan_calibration.json\n")
        
        with open("fan_calibration.json", "w") as f:
            json.dump({
                "dominant_freq_hz": dominant_freq,
                "recommended_min_freq": min_freq,
                "recommended_max_freq": max_freq,
                "all_peaks": peaks,
                "sample_rate": self.sample_rate,
                "n_channels": self.n_channels,
                "samples_captured": self.sample_count,
                "capture_duration_sec": self.capture_duration_sec,
            }, f, indent=2)
        
        print("=" * 80)
        return peaks


def main():
    print("\n" + "=" * 80)
    print("FAN FREQUENCY CALIBRATION TOOL")
    print("=" * 80)
    print("\nThis tool will:")
    print("  1. Start audio capture from ReSpeaker 4-mic array")
    print("  2. Give you 10 seconds to turn on your fan")
    print("  3. Analyze the captured audio to find blade-pass frequency")
    print("  4. Suggest detector parameters tuned to your fan")
    print("\nBest results:")
    print("  • Place fan within 1-2 meters of microphone")
    print("  • Use a steady fan speed (not oscillating)")
    print("  • Minimize background noise")
    print()
    
    input("Press Enter to START the 10-second capture...")
    print()
    
    calibrator = FanCalibrator(capture_duration_sec=10.0)
    calibrator.start()
    
    # Wait for capture to complete
    while not calibrator.capture_complete:
        time.sleep(0.1)
    
    calibrator.stop()
    
    print("\n[Analysis] Computing FFT spectrum...")
    peaks = calibrator.analyze()
    
    if peaks:
        calibrator.print_spectrum(num_peaks=8)
    else:
        print("\n[Analysis] No peaks detected. Troubleshooting:")
        print("  1. Increase capture duration (edit script, change capture_duration_sec)")
        print("  2. Place fan closer to microphone")
        print("  3. Check if ReSpeaker is receiving audio (look at device list above)")
        print("  4. Try using a different audio device with --device-index")


if __name__ == "__main__":
    main()
