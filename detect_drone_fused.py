"""
Audio-visual drone detection: fuses event camera propeller detection with
ReSpeaker 4-mic array acoustic detection using Bayesian sensor fusion.

System components:
  - Prophesee EVK4-HD (1280x720, IMX636) with 220-degree fisheye lens
  - ReSpeaker 4-Mic Array (USB or Pi HAT) via PyAudio

Detection pipeline:
  1. Visual: per-pixel frequency map + grid FFT (from detect_propeller.py)
  2. Acoustic: mic array FFT for blade-pass frequency + GCC-PHAT for DOA
  3. Fusion: sequential Bayesian updating with frequency cross-validation

Usage:
    python detect_drone_fused.py -i <event_file>                  # visual only
    python detect_drone_fused.py -i <event_file> --enable-audio   # fused mode
    python detect_drone_fused.py --enable-audio                   # live camera + mic

Latency budget (typical DJI Mavic-class drone at 189 Hz BPF):
    Visual first detect  :  ~40-60 ms  (7 periods @ 189 Hz + overhead)
    Acoustic first detect:  ~50-120 ms (FFT window fill + USB buffer)
    Fused high confidence: ~100-200 ms (both modalities agree)
"""

import argparse
import threading
import time
import numpy as np
from collections import deque

from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator, is_live_camera
from metavision_sdk_analytics import FrequencyMapAsyncAlgorithm, DominantValueMapAlgorithm, \
    HeatMapFrameGeneratorAlgorithm
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction, UIKeyEvent
import cv2


# ---------------------------------------------------------------------------
# Frequency cross-validation
# ---------------------------------------------------------------------------

def cross_validate_frequency(visual_freq_hz, acoustic_freq_hz,
                             tolerance_ratio=0.05, max_harmonic=4):
    """Check if visual and acoustic frequencies are consistent.

    Accounts for one sensor possibly detecting a harmonic of the fundamental.

    Returns:
        (is_consistent, confidence, (harmonic_v, harmonic_a))
    """
    for n_v in range(1, max_harmonic + 1):
        for n_a in range(1, max_harmonic + 1):
            fundamental_v = visual_freq_hz / n_v
            fundamental_a = acoustic_freq_hz / n_a
            ratio = abs(fundamental_v - fundamental_a) / max(fundamental_a, 1e-6)
            if ratio < tolerance_ratio:
                confidence = 1.0 / (n_v * n_a)
                return True, confidence, (n_v, n_a)
    return False, 0.0, (0, 0)


# ---------------------------------------------------------------------------
# Sequential Bayesian detector
# ---------------------------------------------------------------------------

class BayesianDroneDetector:
    """Sequential Bayesian fusion for real-time drone detection.

    Updates P(drone) as new visual and acoustic evidence arrives.
    """

    def __init__(self, prior=0.001, decay_rate=0.995,
                 min_freq=50, max_freq=500):
        self.p_drone = prior
        self.base_prior = prior
        self.decay_rate = decay_rate
        self.min_freq = min_freq
        self.max_freq = max_freq

    def _freq_in_range(self, freq_hz):
        return self.min_freq <= freq_hz <= self.max_freq

    def _visual_likelihood(self, snr, freq_hz):
        """P(V|D) and P(V|~D)."""
        if not self._freq_in_range(freq_hz):
            return 0.1, 0.3
        p_v_d = 1.0 / (1.0 + np.exp(-(snr - 5.0)))
        p_v_nd = 0.01 * np.exp(-0.5 * snr)
        return p_v_d, max(p_v_nd, 1e-9)

    def _acoustic_likelihood(self, snr, freq_hz, elevation_deg=None):
        """P(A|D) and P(A|~D)."""
        if not self._freq_in_range(freq_hz):
            return 0.05, 0.2
        p_a_d = 1.0 / (1.0 + np.exp(-(snr - 4.0)))
        if elevation_deg is not None and elevation_deg > 10:
            p_a_nd = 0.005  # few elevated non-drone sources
        else:
            p_a_nd = 0.05
        return p_a_d, max(p_a_nd, 1e-9)

    def update(self, visual_evidence=None, acoustic_evidence=None):
        """Bayesian update with new evidence.

        visual_evidence:  dict(snr, freq_hz)  or None
        acoustic_evidence: dict(snr, freq_hz, doa_azimuth, doa_elevation) or None

        Returns updated P(drone).
        """
        p_d = self.p_drone
        p_nd = 1.0 - p_d

        # Visual update
        if visual_evidence is not None:
            pv_d, pv_nd = self._visual_likelihood(
                visual_evidence["snr"], visual_evidence["freq_hz"])
            denom = pv_d * p_d + pv_nd * p_nd
            p_d = pv_d * p_d / denom
            p_nd = 1.0 - p_d

        # Acoustic update
        if acoustic_evidence is not None:
            pa_d, pa_nd = self._acoustic_likelihood(
                acoustic_evidence["snr"], acoustic_evidence["freq_hz"],
                acoustic_evidence.get("doa_elevation"))
            denom = pa_d * p_d + pa_nd * p_nd
            p_d = pa_d * p_d / denom
            p_nd = 1.0 - p_d

        # Frequency cross-validation bonus
        if visual_evidence and acoustic_evidence:
            matched, conf, _ = cross_validate_frequency(
                visual_evidence["freq_hz"], acoustic_evidence["freq_hz"])
            if matched:
                pf_d = 0.99 * conf
                pf_nd = 0.001
                denom = pf_d * p_d + pf_nd * p_nd
                p_d = pf_d * p_d / denom

        self.p_drone = np.clip(p_d, 1e-9, 1.0 - 1e-9)
        return self.p_drone

    def decay(self):
        """Without new evidence, belief drifts toward prior."""
        self.p_drone = (self.decay_rate * self.p_drone
                        + (1 - self.decay_rate) * self.base_prior)

    @property
    def detected(self):
        return self.p_drone > 0.8

    @property
    def confidence(self):
        return self.p_drone


# ---------------------------------------------------------------------------
# Acoustic processing thread (ReSpeaker 4-mic array)
# ---------------------------------------------------------------------------

class AcousticProcessor:
    """Processes audio from the ReSpeaker 4-mic array in a background thread.

    Computes:
      - FFT on each mic channel to find blade-pass frequency peaks
      - Simple GCC-PHAT DOA estimation between mic pairs

    ReSpeaker USB Mic Array v2.0 geometry (circular, 32 mm radius):
        Mic 0: (-32, 0, 0) mm
        Mic 1: ( 0, -32, 0) mm
        Mic 2: (+32, 0, 0) mm
        Mic 3: ( 0, +32, 0) mm
    """

    # Mic positions in meters (USB Mic Array v2.0)
    MIC_POSITIONS = np.array([
        [-0.032, 0.0, 0.0],
        [0.0, -0.032, 0.0],
        [0.032, 0.0, 0.0],
        [0.0, 0.032, 0.0],
    ])
    SPEED_OF_SOUND = 343.0  # m/s

    def __init__(self, sample_rate=16000, chunk_size=1024, n_channels=4,
                 fft_window_sec=0.2, min_freq=50, max_freq=500, min_snr=3.0,
                 device_index=None):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.n_channels = n_channels
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.min_snr = min_snr
        self.device_index = device_index

        fft_window_samples = int(fft_window_sec * sample_rate)
        self.buffer_maxlen = fft_window_samples
        # Per-channel rolling audio buffers
        self.buffers = [deque(maxlen=fft_window_samples) for _ in range(n_channels)]

        # Latest results (thread-safe via GIL for simple reads)
        self.latest_result = None  # dict or None
        self._running = False
        self._thread = None

    def start(self):
        """Start the audio capture + analysis thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self):
        """Background thread: capture audio and run analysis."""
        try:
            import pyaudio
        except ImportError:
            print("[Audio] PyAudio not available. Acoustic detection disabled.")
            return

        pa = pyaudio.PyAudio()
        try:
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=self.n_channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                input_device_index=self.device_index,
            )
        except Exception as e:
            print(f"[Audio] Failed to open mic stream: {e}")
            pa.terminate()
            return

        print(f"[Audio] Streaming from ReSpeaker ({self.sample_rate} Hz, "
              f"{self.n_channels} ch, chunk={self.chunk_size})")

        try:
            while self._running:
                try:
                    raw = stream.read(self.chunk_size, exception_on_overflow=False)
                except Exception:
                    continue

                # Deinterleave into per-channel samples
                samples = np.frombuffer(raw, dtype=np.int16).reshape(-1, self.n_channels)
                for ch in range(self.n_channels):
                    self.buffers[ch].extend(samples[:, ch].astype(np.float64))

                # Analyze when buffer is full
                if len(self.buffers[0]) >= self.buffer_maxlen:
                    self.latest_result = self._analyze()
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

    def _analyze(self):
        """Run FFT + DOA on buffered audio."""
        n = len(self.buffers[0])
        if n < 64:
            return None

        # Average spectrum across all channels for frequency detection
        window = np.hanning(n)
        all_spectra = []
        channel_data = []
        for ch in range(self.n_channels):
            sig = np.array(self.buffers[ch])
            channel_data.append(sig)
            sig_dc = sig - np.mean(sig)
            spectrum = np.fft.rfft(sig_dc * window)
            all_spectra.append(np.abs(spectrum))

        avg_spectrum = np.mean(all_spectra, axis=0) * 2.0 / n
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sample_rate)

        # Find peak in target range
        valid = (freqs >= self.min_freq) & (freqs <= self.max_freq)
        if not np.any(valid):
            return None

        valid_mags = avg_spectrum[valid]
        valid_freqs = freqs[valid]
        peak_idx = np.argmax(valid_mags)
        peak_freq = float(valid_freqs[peak_idx])
        peak_mag = float(valid_mags[peak_idx])

        noise_floor = float(np.median(valid_mags))
        snr = peak_mag / noise_floor if noise_floor > 0 else peak_mag

        if snr < self.min_snr:
            return None

        # GCC-PHAT DOA estimation (using mic pair 0-2, x-axis)
        doa_azimuth = self._gcc_phat_doa(channel_data[0], channel_data[2],
                                          mic_dist=0.064)  # 64 mm apart

        return {
            "freq_hz": peak_freq,
            "snr": snr,
            "magnitude": peak_mag,
            "doa_azimuth": doa_azimuth,
            "doa_elevation": None,  # planar array cannot resolve elevation
        }

    def _gcc_phat_doa(self, sig1, sig2, mic_dist):
        """Estimate DOA angle using GCC-PHAT between two mics.

        Returns azimuth in degrees (0 = broadside, +-90 = endfire).
        """
        n = len(sig1)
        S1 = np.fft.rfft(sig1)
        S2 = np.fft.rfft(sig2)
        cross = S1 * np.conj(S2)
        magnitude = np.abs(cross)
        magnitude[magnitude < 1e-10] = 1e-10
        gcc = np.fft.irfft(cross / magnitude, n=n)

        # Maximum delay in samples
        max_delay_samples = int(mic_dist / self.SPEED_OF_SOUND * self.sample_rate) + 1
        # Search in valid delay range
        search_range = min(max_delay_samples + 2, n // 2)
        gcc_shifted = np.concatenate([gcc[-search_range:], gcc[:search_range + 1]])
        peak = np.argmax(np.abs(gcc_shifted))
        delay_samples = peak - search_range

        delay_sec = delay_samples / self.sample_rate
        # Clamp to physical limits
        max_delay = mic_dist / self.SPEED_OF_SOUND
        delay_sec = np.clip(delay_sec, -max_delay, max_delay)

        sin_theta = delay_sec * self.SPEED_OF_SOUND / mic_dist
        sin_theta = np.clip(sin_theta, -1.0, 1.0)
        return float(np.degrees(np.arcsin(sin_theta)))


# ---------------------------------------------------------------------------
# Grid-based visual FFT analyzer (shared with detect_propeller.py)
# ---------------------------------------------------------------------------

class PropellerGridAnalyzer:
    """Divides sensor into grid cells, runs FFT on event counts per cell."""

    def __init__(self, width, height, grid_cells, fft_window_samples,
                 min_freq, max_freq, num_blades, min_snr):
        self.width = width
        self.height = height
        self.grid_cells = grid_cells
        self.cell_w = width / grid_cells
        self.cell_h = height / grid_cells
        self.num_blades = num_blades
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.min_snr = min_snr
        n_cells = grid_cells * grid_cells
        self.buffers = [deque(maxlen=fft_window_samples) for _ in range(n_cells)]

    def accumulate(self, events):
        if events.size == 0:
            for buf in self.buffers:
                buf.append(0)
            return
        cols = np.clip((events["x"] / self.cell_w).astype(int), 0, self.grid_cells - 1)
        rows = np.clip((events["y"] / self.cell_h).astype(int), 0, self.grid_cells - 1)
        indices = rows * self.grid_cells + cols
        counts = np.bincount(indices, minlength=self.grid_cells * self.grid_cells)
        for i, buf in enumerate(self.buffers):
            buf.append(int(counts[i]))

    def analyze(self, fs):
        detections = []
        for r in range(self.grid_cells):
            for c in range(self.grid_cells):
                idx = r * self.grid_cells + c
                buf = self.buffers[idx]
                n = len(buf)
                if n < 16:
                    continue
                signal = np.array(buf, dtype=np.float64)
                if np.mean(signal) < 1.0:
                    continue
                sig = signal - np.mean(signal)
                window = np.hanning(n)
                spectrum = np.fft.rfft(sig * window)
                freqs = np.fft.rfftfreq(n, d=1.0 / fs)
                magnitudes = np.abs(spectrum) * 2.0 / n
                valid = (freqs >= self.min_freq) & (freqs <= self.max_freq)
                if not np.any(valid):
                    continue
                valid_mags = magnitudes[valid]
                valid_freqs = freqs[valid]
                peak_idx = np.argmax(valid_mags)
                peak_mag = valid_mags[peak_idx]
                peak_freq = valid_freqs[peak_idx]
                noise_floor = np.median(valid_mags)
                snr = peak_mag / noise_floor if noise_floor > 0 else peak_mag
                if snr >= self.min_snr:
                    rpm = (peak_freq / self.num_blades) * 60
                    detections.append({
                        "row": r, "col": c,
                        "freq_hz": float(peak_freq),
                        "rpm": float(rpm),
                        "snr": float(snr),
                    })
        return detections


def cluster_detections(detections):
    """Group adjacent grid cells with similar frequencies into propeller clusters."""
    if not detections:
        return []
    used = [False] * len(detections)
    clusters = []
    for i, det in enumerate(detections):
        if used[i]:
            continue
        cluster = [det]
        used[i] = True
        queue = [i]
        while queue:
            ci = queue.pop(0)
            cd = detections[ci]
            for j, other in enumerate(detections):
                if used[j]:
                    continue
                if abs(cd["row"] - other["row"]) <= 1 and abs(cd["col"] - other["col"]) <= 1:
                    freq_ratio = (max(cd["freq_hz"], other["freq_hz"])
                                  / max(min(cd["freq_hz"], other["freq_hz"]), 1e-6))
                    if freq_ratio <= 1.2:
                        cluster.append(other)
                        used[j] = True
                        queue.append(j)
        freqs = [d["freq_hz"] for d in cluster]
        rpms = [d["rpm"] for d in cluster]
        snrs = [d["snr"] for d in cluster]
        rows = [d["row"] for d in cluster]
        cols = [d["col"] for d in cluster]
        clusters.append({
            "center_row": float(np.mean(rows)),
            "center_col": float(np.mean(cols)),
            "freq_hz": float(np.median(freqs)),
            "rpm": float(np.median(rpms)),
            "n_cells": len(cluster),
            "avg_snr": float(np.mean(snrs)),
        })
    clusters.sort(key=lambda c: c["n_cells"], reverse=True)
    return clusters


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Audio-visual drone detection with Bayesian sensor fusion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Input
    parser.add_argument("-i", "--input-event-file", dest="event_file_path", default="",
                        help="Path to event file (RAW/DAT/HDF5). Empty = live camera.")

    # Frequency range
    parser.add_argument("--min-freq", dest="min_freq", type=float, default=50)
    parser.add_argument("--max-freq", dest="max_freq", type=float, default=500)
    parser.add_argument("--num-blades", dest="num_blades", type=int, default=2)

    # Visual tuning
    parser.add_argument("--filter-length", dest="filter_length", type=int, default=7)
    parser.add_argument("--max-period-diff", dest="max_period_diff", type=int, default=500)
    parser.add_argument("--freq-precision", dest="freq_precision", type=float, default=5.0)
    parser.add_argument("--min-pixel-count", dest="min_pixel_count", type=int, default=25)
    parser.add_argument("--grid-cells", dest="grid_cells", type=int, default=16)
    parser.add_argument("--fft-window", dest="fft_window_sec", type=float, default=0.5)
    parser.add_argument("--min-snr", dest="min_snr", type=float, default=3.0)
    parser.add_argument("--delta-t", dest="delta_t", type=int, default=500,
                        help="Event slice (us). 500 -> 2 kHz sampling -> 1 kHz Nyquist.")

    # Audio
    parser.add_argument("--enable-audio", dest="enable_audio", action="store_true",
                        help="Enable ReSpeaker 4-mic array acoustic detection.")
    parser.add_argument("--audio-rate", dest="audio_rate", type=int, default=16000,
                        help="Audio sampling rate (Hz). ReSpeaker supports 16000/48000.")
    parser.add_argument("--audio-chunk", dest="audio_chunk", type=int, default=1024,
                        help="Audio chunk size in samples.")
    parser.add_argument("--audio-channels", dest="audio_channels", type=int, default=4)
    parser.add_argument("--audio-device", dest="audio_device", type=int, default=None,
                        help="PyAudio device index for ReSpeaker.")
    parser.add_argument("--audio-fft-window", dest="audio_fft_window", type=float, default=0.2,
                        help="Audio FFT window duration (seconds).")
    parser.add_argument("--audio-min-snr", dest="audio_min_snr", type=float, default=3.0)

    # Fusion
    parser.add_argument("--prior", dest="prior", type=float, default=0.001,
                        help="Bayesian prior P(drone).")
    parser.add_argument("--detection-threshold", dest="detection_threshold", type=float,
                        default=0.8, help="P(drone) threshold for confirmed detection.")

    # Timing
    parser.add_argument("--update-freq", dest="update_freq", type=float, default=25)
    parser.add_argument("--print-interval", dest="print_interval_sec", type=float, default=0.5)
    parser.add_argument("-f", "--replay-factor", dest="replay_factor", type=float, default=1)

    args = parser.parse_args()

    nyquist = 1e6 / args.delta_t / 2
    if args.max_freq > nyquist:
        parser.error(
            f"--max-freq ({args.max_freq} Hz) > Nyquist ({nyquist:.0f} Hz). "
            f"Decrease --delta-t to at least {int(1e6 / (2 * args.max_freq))} us.")
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    delta_t = args.delta_t
    fs = 1e6 / delta_t
    fft_window_samples = int(args.fft_window_sec * fs)
    print_interval_samples = max(1, int(args.print_interval_sec * fs))

    # --- Event camera setup ---
    mv_iterator = EventsIterator(input_path=args.event_file_path, delta_t=delta_t)
    height, width = mv_iterator.get_size()
    if not is_live_camera(args.event_file_path):
        mv_iterator = LiveReplayEventsIterator(mv_iterator, replay_factor=args.replay_factor)

    # SDK frequency map
    freq_algo = FrequencyMapAsyncAlgorithm(
        width=width, height=height,
        filter_length=args.filter_length,
        min_freq=args.min_freq, max_freq=args.max_freq,
        diff_thresh_us=args.max_period_diff)
    freq_algo.update_frequency = args.update_freq

    dominant_algo = DominantValueMapAlgorithm(
        args.min_freq, args.max_freq, args.freq_precision, args.min_pixel_count)

    heat_gen = HeatMapFrameGeneratorAlgorithm(
        args.min_freq, args.max_freq, args.freq_precision, width, height, "Hz")
    freq_img = heat_gen.get_output_image()
    freq_full_height = heat_gen.full_height

    # Grid FFT analyzer
    grid_analyzer = PropellerGridAnalyzer(
        width, height, args.grid_cells, fft_window_samples,
        args.min_freq, args.max_freq, args.num_blades, args.min_snr)

    # --- Audio setup ---
    audio_proc = None
    if args.enable_audio:
        audio_proc = AcousticProcessor(
            sample_rate=args.audio_rate,
            chunk_size=args.audio_chunk,
            n_channels=args.audio_channels,
            fft_window_sec=args.audio_fft_window,
            min_freq=args.min_freq, max_freq=args.max_freq,
            min_snr=args.audio_min_snr,
            device_index=args.audio_device)
        audio_proc.start()

    # --- Bayesian detector ---
    bayesian = BayesianDroneDetector(
        prior=args.prior, min_freq=args.min_freq, max_freq=args.max_freq)

    # --- Windows ---
    ev_window = MTWindow(title="Events - Fused Drone Detector",
                         width=width, height=height,
                         mode=BaseWindow.RenderMode.BGR, open_directly=True)
    freq_window = MTWindow(title="Frequency Map",
                           width=width, height=freq_full_height,
                           mode=BaseWindow.RenderMode.BGR, open_directly=True)

    def keyboard_cb(key, scancode, action, mods):
        if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
            ev_window.set_close_flag()
            freq_window.set_close_flag()

    ev_window.set_keyboard_callback(keyboard_cb)
    freq_window.set_keyboard_callback(keyboard_cb)

    event_frame_gen = PeriodicFrameGenerationAlgorithm(
        sensor_width=width, sensor_height=height, fps=25, palette=ColorPalette.Dark)

    def on_cd_frame_cb(ts, cd_frame):
        ev_window.show_async(cd_frame)

    event_frame_gen.set_output_callback(on_cd_frame_cb)

    # Frequency map callback
    latest_freq_map = [None]

    def on_freq_map(ts, freq_map):
        latest_freq_map[0] = freq_map.copy()
        heat_gen.generate_bgr_heat_map(freq_map, freq_img)

        success, dom_freq = dominant_algo.compute_dominant_value(freq_map)
        if success:
            rpm = (dom_freq / args.num_blades) * 60
            label = f"Visual: {dom_freq:.0f} Hz ({rpm:.0f} RPM)"
            cv2.putText(freq_img, label, (10, height - 30),
                        cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)

        # Show Bayesian confidence
        status = "DRONE DETECTED" if bayesian.detected else "scanning"
        color = (0, 0, 255) if bayesian.detected else (0, 255, 0)
        cv2.putText(freq_img, f"P(drone)={bayesian.confidence:.3f}  [{status}]",
                    (10, height - 10), cv2.FONT_HERSHEY_PLAIN, 1.0, color, 1)

        freq_window.show_async(freq_img)

    freq_algo.set_output_callback(on_freq_map)

    # --- Print header ---
    nyquist = fs / 2
    print("=" * 78)
    print("  Audio-Visual Drone Detector (Bayesian Fusion)")
    print(f"  Sensor          : {width}x{height} + 220-deg fisheye")
    print(f"  Visual sampling : {fs:.0f} Hz  (delta_t={delta_t} us, Nyquist={nyquist:.0f} Hz)")
    print(f"  Freq range      : {args.min_freq}-{args.max_freq} Hz  ({args.num_blades} blades)")
    print(f"  Audio           : {'ENABLED' if args.enable_audio else 'DISABLED'}")
    if args.enable_audio:
        print(f"    Sample rate   : {args.audio_rate} Hz")
        print(f"    FFT window    : {args.audio_fft_window}s")
        print(f"    DOA method    : GCC-PHAT (4-mic, ~32 mm radius)")
    print(f"  Fusion          : Sequential Bayesian (prior={args.prior})")
    print(f"  Detection thresh: P(drone) > {args.detection_threshold}")
    print("=" * 78)
    print()

    sample_count = 0
    t_start = time.monotonic()

    for evs in mv_iterator:
        EventLoop.poll_and_dispatch()
        event_frame_gen.process_events(evs)
        freq_algo.process_events(evs)
        grid_analyzer.accumulate(evs)
        sample_count += 1

        if ev_window.should_close() or freq_window.should_close():
            break

        # --- Periodic fusion + printout ---
        if sample_count % print_interval_samples != 0:
            continue

        elapsed = time.monotonic() - t_start
        wall_time_ms = elapsed * 1000

        # Visual evidence
        visual_evidence = None
        detections = grid_analyzer.analyze(fs)
        clusters = cluster_detections(detections)
        if clusters:
            best = clusters[0]
            visual_evidence = {"freq_hz": best["freq_hz"], "snr": best["avg_snr"]}

        # Acoustic evidence
        acoustic_evidence = None
        if audio_proc and audio_proc.latest_result:
            r = audio_proc.latest_result
            acoustic_evidence = {
                "freq_hz": r["freq_hz"],
                "snr": r["snr"],
                "doa_azimuth": r["doa_azimuth"],
                "doa_elevation": r.get("doa_elevation"),
            }

        # Bayesian update
        if visual_evidence or acoustic_evidence:
            p = bayesian.update(visual_evidence, acoustic_evidence)
        else:
            bayesian.decay()
            p = bayesian.confidence

        # Print status
        sim_time = sample_count * delta_t / 1e6
        v_str = (f"freq={visual_evidence['freq_hz']:6.1f} Hz, "
                 f"SNR={visual_evidence['snr']:.1f}"
                 if visual_evidence else "---")
        a_str = (f"freq={acoustic_evidence['freq_hz']:6.1f} Hz, "
                 f"SNR={acoustic_evidence['snr']:.1f}, "
                 f"DOA={acoustic_evidence['doa_azimuth']:.0f} deg"
                 if acoustic_evidence else "---")

        # Frequency cross-validation
        xval_str = ""
        if visual_evidence and acoustic_evidence:
            matched, conf, harmonics = cross_validate_frequency(
                visual_evidence["freq_hz"], acoustic_evidence["freq_hz"])
            if matched:
                xval_str = f"  FREQ MATCH (h={harmonics}, conf={conf:.2f})"

        status = "*** DRONE ***" if bayesian.detected else ""
        print(f"[{sim_time:7.2f}s] P={p:.4f} "
              f" V:[{v_str}]  A:[{a_str}]{xval_str}  {status}")

        if clusters:
            for i, cl in enumerate(clusters[:3]):
                print(f"           Propeller {i+1}: "
                      f"{cl['freq_hz']:.1f} Hz, {cl['rpm']:.0f} RPM, "
                      f"{cl['n_cells']} cells")

    # Cleanup
    if audio_proc:
        audio_proc.stop()
    ev_window.destroy()
    freq_window.destroy()
    print("\nDone.")


if __name__ == "__main__":
    main()
