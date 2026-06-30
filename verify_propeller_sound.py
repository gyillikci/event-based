#!/usr/bin/env python3
"""
Propeller Sound Verification Tool with Real-time Spectrum Analysis
- Real-time audio capture from ReSpeaker 4-mic array
- FFT-based frequency spectrum visualization
- Interactive approval/rejection workflow
- Session logging of verified sounds
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import pyaudio
import threading
import json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.signal import find_peaks


class PropellerSoundVerifier:
    def __init__(self, root):
        self.root = root
        self.root.title("Propeller Sound Verification Tool")
        self.root.geometry("1400x900")
        
        # Audio parameters
        self.SAMPLE_RATE = 16000
        self.CHUNK_SIZE = 2048
        self.RECORD_DURATION = 5.0  # seconds per capture
        self.MIN_FREQ = 50
        self.MAX_FREQ = 500
        self.THRESHOLD_DB = -40  # noise floor
        
        # ReSpeaker parameters
        self.RESPEAKER_INDEX = 1
        self.NUM_CHANNELS = 1  # Use mic 0 (left) for simplicity
        
        # State
        self.is_recording = False
        self.audio_data = None
        self.spectrum_freqs = None
        self.spectrum_db = None
        self.detected_peaks = []
        self.verified_sounds = []
        self.stream = None
        self.pa = None
        
        # Build UI
        self._build_ui()
        
        # Load or initialize session
        self.session_file = Path("propeller_verification_session.jsonl")
        self._load_session()
        
    def _build_ui(self):
        """Build the GUI layout"""
        # Top frame: Controls
        control_frame = ttk.LabelFrame(self.root, text="Recording Controls", padding=10)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Record button (use tk.Button so foreground color changes work)
        self.record_btn = tk.Button(
            control_frame, text="🎤 START RECORDING", command=self._toggle_recording,
            font=("Segoe UI", 10, "bold"), bg="#e0e0e0", activebackground="#d0d0d0"
        )
        self.record_btn.pack(side=tk.LEFT, padx=5)
        
        # Status label (use tk.Label so foreground color changes work)
        self.status_label = tk.Label(
            control_frame, text="Status: Ready", fg="green", font=("Segoe UI", 10, "bold")
        )
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        # Recording time
        self.time_label = ttk.Label(control_frame, text="Time: 0.0s")
        self.time_label.pack(side=tk.LEFT, padx=5)
        
        # Parameters frame
        params_frame = ttk.LabelFrame(self.root, text="Detection Parameters", padding=10)
        params_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Min freq
        ttk.Label(params_frame, text="Min Frequency (Hz):").pack(side=tk.LEFT, padx=5)
        self.min_freq_var = tk.IntVar(value=122)
        ttk.Spinbox(
            params_frame, from_=20, to=200, textvariable=self.min_freq_var, width=5
        ).pack(side=tk.LEFT, padx=5)
        
        # Max freq
        ttk.Label(params_frame, text="Max Frequency (Hz):").pack(side=tk.LEFT, padx=5)
        self.max_freq_var = tk.IntVar(value=203)
        ttk.Spinbox(
            params_frame, from_=200, to=1000, textvariable=self.max_freq_var, width=5
        ).pack(side=tk.LEFT, padx=5)
        
        # Expected blades
        ttk.Label(params_frame, text="Expected Blades:").pack(side=tk.LEFT, padx=5)
        self.num_blades_var = tk.IntVar(value=2)
        ttk.Spinbox(
            params_frame, from_=1, to=6, textvariable=self.num_blades_var, width=5
        ).pack(side=tk.LEFT, padx=5)
        
        # Main content: Spectrum plot
        plot_frame = ttk.LabelFrame(self.root, text="Frequency Spectrum", padding=5)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.fig = Figure(figsize=(12, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("Magnitude (dB)")
        self.ax.set_title("Real-time Frequency Spectrum")
        self.ax.grid(True, alpha=0.3)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Detection results frame
        results_frame = ttk.LabelFrame(self.root, text="Detection Results", padding=10)
        results_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Results text
        self.results_text = tk.Text(results_frame, height=6, width=80)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.config(yscrollcommand=scrollbar.set)
        
        # Action frame
        action_frame = ttk.Frame(self.root)
        action_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.approve_btn = ttk.Button(
            action_frame, text="✓ APPROVE SOUND", command=self._approve_sound, state=tk.DISABLED
        )
        self.approve_btn.pack(side=tk.LEFT, padx=5)
        
        self.reject_btn = ttk.Button(
            action_frame, text="✗ REJECT SOUND", command=self._reject_sound, state=tk.DISABLED
        )
        self.reject_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Separator(action_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)
        
        ttk.Button(
            action_frame, text="📊 Export Results", command=self._export_results
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            action_frame, text="🔄 Clear Session", command=self._clear_session
        ).pack(side=tk.LEFT, padx=5)
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(self.root, text="Session Statistics", padding=10)
        stats_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.stats_label = ttk.Label(stats_frame, text="Approved: 0 | Rejected: 0")
        self.stats_label.pack(side=tk.LEFT, padx=20)
        
    def _initialize_audio(self):
        """Initialize PyAudio and ReSpeaker stream"""
        try:
            self.pa = pyaudio.PyAudio()
            
            # List available devices
            print("\n" + "="*60)
            print("Available Audio Devices:")
            print("="*60)
            for i in range(self.pa.get_device_count()):
                info = self.pa.get_device_info_by_index(i)
                print(f"[{i}] {info['name']}")
                print(f"    Channels: {info['maxInputChannels']}, "
                      f"Sample Rate: {info['defaultSampleRate']}")
            
            # Try to open ReSpeaker
            print(f"\nAttempting to open ReSpeaker at index {self.RESPEAKER_INDEX}...")
            device_info = self.pa.get_device_info_by_index(self.RESPEAKER_INDEX)
            print(f"Device: {device_info['name']}")
            print(f"Channels: {device_info['maxInputChannels']}")
            
            self.stream = self.pa.open(
                format=pyaudio.paFloat32,
                channels=self.NUM_CHANNELS,
                rate=self.SAMPLE_RATE,
                input=True,
                input_device_index=self.RESPEAKER_INDEX,
                frames_per_buffer=self.CHUNK_SIZE
            )
            print("✓ Audio stream opened successfully")
            return True
        except Exception as e:
            messagebox.showerror("Audio Error", f"Failed to initialize audio: {e}")
            print(f"✗ Audio initialization failed: {e}")
            return False
    
    def _toggle_recording(self):
        """Toggle recording on/off"""
        if not self.is_recording:
            # Start recording
            if not self._initialize_audio():
                return
            
            self.is_recording = True
            self.record_btn.config(text="🛑 STOP RECORDING", fg="red")
            self.status_label.config(text="Status: Recording...", fg="red")
            self.approve_btn.config(state=tk.DISABLED)
            self.reject_btn.config(state=tk.DISABLED)
            
            # Start recording thread
            thread = threading.Thread(target=self._record_audio, daemon=True)
            thread.start()
        else:
            # Stop recording
            self.is_recording = False
            self.record_btn.config(text="🎤 START RECORDING", fg="black")
            self.status_label.config(text="Status: Processing...", fg="orange")
    
    def _record_audio(self):
        """Record audio in a separate thread"""
        try:
            frames = []
            num_chunks = int(self.SAMPLE_RATE / self.CHUNK_SIZE * self.RECORD_DURATION)
            
            for i in range(num_chunks):
                if not self.is_recording:
                    break
                
                # Read chunk (ignore overflow to avoid crashes on USB mics)
                data = self.stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                frames.append(np.frombuffer(data, dtype=np.float32))
                
                # Update time label
                elapsed = (i + 1) * self.CHUNK_SIZE / self.SAMPLE_RATE
                self.root.after(0, lambda t=elapsed: self.time_label.config(text=f"Time: {t:.1f}s"))
            
            # Combine all frames
            self.audio_data = np.concatenate(frames)
            
            # Close stream
            self.stream.stop_stream()
            self.stream.close()
            self.pa.terminate()
            
            # Process and display
            self.root.after(0, self._process_audio)
            
        except Exception as e:
            print(f"Recording error: {e}")
            messagebox.showerror("Recording Error", str(e))
    
    def _process_audio(self):
        """Process recorded audio and compute spectrum"""
        if self.audio_data is None or len(self.audio_data) == 0:
            messagebox.showerror("Error", "No audio data recorded")
            return
        
        try:
            # Compute FFT
            fft = np.fft.rfft(self.audio_data)
            freqs = np.fft.rfftfreq(len(self.audio_data), 1.0 / self.SAMPLE_RATE)
            magnitude = np.abs(fft)
            
            # Convert to dB
            magnitude_db = 20 * np.log10(magnitude + 1e-10)
            
            # Crop to frequency range
            mask = (freqs >= self.MIN_FREQ) & (freqs <= self.MAX_FREQ)
            self.spectrum_freqs = freqs[mask]
            self.spectrum_db = magnitude_db[mask]
            
            # Find peaks
            min_freq = self.min_freq_var.get()
            max_freq = self.max_freq_var.get()
            freq_mask = (self.spectrum_freqs >= min_freq) & (self.spectrum_freqs <= max_freq)
            freq_region_db = self.spectrum_db[freq_mask]
            freq_region = self.spectrum_freqs[freq_mask]
            
            # Find peaks with minimum prominence
            peaks, properties = find_peaks(
                freq_region_db,
                height=-20,
                distance=10,
                prominence=5
            )
            
            self.detected_peaks = []
            for peak_idx in peaks:
                freq = freq_region[peak_idx]
                magnitude = freq_region_db[peak_idx]
                self.detected_peaks.append({
                    "frequency": float(freq),
                    "magnitude_db": float(magnitude)
                })
            
            # Sort by magnitude
            self.detected_peaks.sort(key=lambda x: x["magnitude_db"], reverse=True)
            
            # Display
            self._display_spectrum()
            self._display_results()
            
            self.status_label.config(text="Status: Ready", fg="green")
            self.record_btn.config(text="🎤 START RECORDING", fg="black")
            self.approve_btn.config(state=tk.NORMAL)
            self.reject_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            print(f"Processing error: {e}")
            messagebox.showerror("Processing Error", str(e))
    
    def _display_spectrum(self):
        """Display the frequency spectrum"""
        self.ax.clear()
        
        # Plot full spectrum
        self.ax.plot(self.spectrum_freqs, self.spectrum_db, "b-", linewidth=1, label="Spectrum")
        
        # Highlight target frequency range
        min_freq = self.min_freq_var.get()
        max_freq = self.max_freq_var.get()
        self.ax.axvspan(min_freq, max_freq, alpha=0.2, color="green", label="Target Range")
        
        # Plot detected peaks
        if self.detected_peaks:
            peak_freqs = [p["frequency"] for p in self.detected_peaks]
            peak_mags = [p["magnitude_db"] for p in self.detected_peaks]
            self.ax.scatter(peak_freqs, peak_mags, color="red", s=100, marker="x", linewidth=2, label="Peaks")
            
            # Label top 3 peaks
            for i, peak in enumerate(self.detected_peaks[:3]):
                self.ax.annotate(
                    f"{peak['frequency']:.1f} Hz\n{peak['magnitude_db']:.1f} dB",
                    xy=(peak["frequency"], peak["magnitude_db"]),
                    xytext=(10, 10),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7),
                    fontsize=8
                )
        
        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("Magnitude (dB)")
        self.ax.set_title(f"Frequency Spectrum ({self.SAMPLE_RATE} Hz, {len(self.audio_data)} samples)")
        self.ax.set_xlim(self.MIN_FREQ, self.MAX_FREQ)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        self.canvas.draw()
    
    def _display_results(self):
        """Display detection results"""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        output = "DETECTION RESULTS\n"
        output += "=" * 80 + "\n\n"
        
        min_freq = self.min_freq_var.get()
        max_freq = self.max_freq_var.get()
        num_blades = self.num_blades_var.get()
        
        output += f"Target Range: {min_freq}-{max_freq} Hz (for {num_blades}-blade propeller)\n"
        output += f"Total Samples: {len(self.audio_data)}\n"
        output += f"Duration: {len(self.audio_data) / self.SAMPLE_RATE:.2f}s\n\n"
        
        output += "DETECTED PEAKS (sorted by magnitude):\n"
        output += "-" * 80 + "\n"
        output += f"{'#':<3} {'Frequency (Hz)':<20} {'Magnitude (dB)':<20} {'In Range':<15} {'Harmonic':<20}\n"
        output += "-" * 80 + "\n"
        
        if not self.detected_peaks:
            output += "No peaks detected above threshold\n"
        else:
            for i, peak in enumerate(self.detected_peaks[:10]):
                freq = peak["frequency"]
                mag = peak["magnitude_db"]
                in_range = "✓ YES" if (min_freq <= freq <= max_freq) else "✗ NO"
                
                # Calculate harmonics
                harmonics = []
                for h in range(1, 5):
                    fundamental = freq / h
                    if min_freq <= fundamental <= max_freq:
                        harmonics.append(f"1/{h}×{freq:.1f}={fundamental:.1f}Hz")
                
                harmonic_str = " | ".join(harmonics) if harmonics else "—"
                
                output += f"{i+1:<3} {freq:<20.1f} {mag:<20.1f} {in_range:<15} {harmonic_str:<20}\n"
        
        output += "\n" + "=" * 80 + "\n"
        output += "VERIFICATION:\n"
        output += "-" * 80 + "\n"
        
        if self.detected_peaks:
            top_freq = self.detected_peaks[0]["frequency"]
            top_mag = self.detected_peaks[0]["magnitude_db"]
            in_range = min_freq <= top_freq <= max_freq
            
            output += f"✓ Primary Frequency: {top_freq:.1f} Hz ({top_mag:.1f} dB)\n"
            output += f"{'✓' if in_range else '✗'} In Target Range: {in_range}\n"
            
            # RPM calculation
            rpm_2blade = top_freq * 60 / 2
            rpm_3blade = top_freq * 60 / 3
            output += f"  → ~{rpm_2blade:.0f} RPM (2-blade) or ~{rpm_3blade:.0f} RPM (3-blade)\n"
        else:
            output += "✗ No valid propeller sound detected\n"
        
        self.results_text.insert(tk.END, output)
        self.results_text.config(state=tk.DISABLED)
    
    def _approve_sound(self):
        """Approve the current sound"""
        if self.audio_data is None:
            messagebox.showwarning("Warning", "No audio to approve")
            return
        
        if not self.detected_peaks:
            messagebox.showwarning("Warning", "No peaks detected. Cannot approve.")
            return
        
        approval = {
            "timestamp": datetime.now().isoformat(),
            "status": "APPROVED",
            "duration_sec": len(self.audio_data) / self.SAMPLE_RATE,
            "detected_peaks": self.detected_peaks,
            "top_frequency_hz": self.detected_peaks[0]["frequency"],
            "parameters": {
                "min_freq": self.min_freq_var.get(),
                "max_freq": self.max_freq_var.get(),
                "num_blades": self.num_blades_var.get()
            }
        }
        
        self.verified_sounds.append(approval)
        self._save_session()
        self._update_stats()
        messagebox.showinfo("Approved", f"Sound approved! Frequency: {approval['top_frequency_hz']:.1f} Hz")
        
        self.audio_data = None
        self.spectrum_freqs = None
        self.spectrum_db = None
        self.ax.clear()
        self.canvas.draw()
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state=tk.DISABLED)
    
    def _reject_sound(self):
        """Reject the current sound"""
        if self.audio_data is None:
            messagebox.showwarning("Warning", "No audio to reject")
            return
        
        approval = {
            "timestamp": datetime.now().isoformat(),
            "status": "REJECTED",
            "duration_sec": len(self.audio_data) / self.SAMPLE_RATE,
            "detected_peaks": self.detected_peaks
        }
        
        self.verified_sounds.append(approval)
        self._save_session()
        self._update_stats()
        messagebox.showinfo("Rejected", "Sound rejected")
        
        self.audio_data = None
        self.spectrum_freqs = None
        self.spectrum_db = None
        self.ax.clear()
        self.canvas.draw()
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state=tk.DISABLED)
    
    def _update_stats(self):
        """Update statistics display"""
        approved = sum(1 for s in self.verified_sounds if s["status"] == "APPROVED")
        rejected = sum(1 for s in self.verified_sounds if s["status"] == "REJECTED")
        self.stats_label.config(text=f"Approved: {approved} | Rejected: {rejected}")
    
    def _save_session(self):
        """Save session to JSONL file"""
        try:
            with open(self.session_file, "a") as f:
                for sound in self.verified_sounds[len(self.verified_sounds)-1:]:
                    f.write(json.dumps(sound) + "\n")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save session: {e}")
    
    def _load_session(self):
        """Load previous session"""
        try:
            if self.session_file.exists():
                with open(self.session_file, "r") as f:
                    for line in f:
                        if line.strip():
                            self.verified_sounds.append(json.loads(line))
                self._update_stats()
                print(f"Loaded {len(self.verified_sounds)} previous approvals")
        except Exception as e:
            print(f"Failed to load session: {e}")
    
    def _export_results(self):
        """Export results to CSV"""
        if not self.verified_sounds:
            messagebox.showwarning("Warning", "No verified sounds to export")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        
        if not file_path:
            return
        
        try:
            import csv
            with open(file_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp", "Status", "Duration (s)", "Top Frequency (Hz)",
                    "Magnitude (dB)", "Min Freq", "Max Freq", "Num Blades"
                ])
                
                for sound in self.verified_sounds:
                    top_peak = sound["detected_peaks"][0] if sound.get("detected_peaks") else {}
                    params = sound.get("parameters", {})
                    writer.writerow([
                        sound["timestamp"],
                        sound["status"],
                        f"{sound['duration_sec']:.2f}",
                        f"{sound.get('top_frequency_hz', 'N/A')}",
                        f"{top_peak.get('magnitude_db', 'N/A')}",
                        params.get("min_freq", "N/A"),
                        params.get("max_freq", "N/A"),
                        params.get("num_blades", "N/A")
                    ])
            
            messagebox.showinfo("Success", f"Results exported to {file_path}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))
    
    def _clear_session(self):
        """Clear session"""
        if messagebox.askyesno("Confirm", "Clear all verified sounds?"):
            self.verified_sounds = []
            self.session_file.unlink(missing_ok=True)
            self._update_stats()
            messagebox.showinfo("Cleared", "Session cleared")


if __name__ == "__main__":
    root = tk.Tk()
    app = PropellerSoundVerifier(root)
    root.mainloop()
