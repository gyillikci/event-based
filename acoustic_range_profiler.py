#!/usr/bin/env python3
"""
Acoustic range profiler for the ReSpeaker 4-Mic Array.

Goal: answer "how far can we detect the fan?" quantitatively, and squeeze the
maximum usable range out of the existing hardware.

Why this is more sensitive than detect_drone_fused.py's acoustic path:
  * HARMONIC SUMMATION. A fan/propeller emits a comb of tones at the blade-pass
    fundamental f0 and its multiples (2 f0, 3 f0, ...). The current detector
    scores a single FFT bin; this tool coherently sums energy across the first
    N harmonics, which typically buys +6..+12 dB of effective SNR and roughly
    doubles the detection distance.
  * MULTI-MIC INCOHERENT AVERAGING. Averaging the magnitude spectra of all 4
    mics lowers the noise floor by ~10*log10(4) = 6 dB versus one mic.
  * MEDIAN NOISE FLOOR with a guard band, so a strong tone does not inflate its
    own noise estimate.

It also helps you build an SNR-vs-distance curve: tag the current distance with
a keypress and the tool records (distance, SNR_dB) so you can see the real
detection envelope and extrapolate the maximum range via the inverse-square law.

Controls (in the matplotlib window):
  d : tag a measurement at the distance typed in the box (adds to the curve)
  c : clear the distance/SNR curve
  s : save curve + a snapshot to acoustic_range_profile.csv / .png

Requires: pyaudio, numpy, scipy, matplotlib  (all already in .venv39)
"""

import argparse
import csv
import time
from collections import deque

import numpy as np

SPEED_OF_SOUND = 343.0


# ----------------------------------------------------------------------------
# LED ring direction indicator (reads firmware DOA, lights the ring at it)
# ----------------------------------------------------------------------------
class RingDirection:
    """Reads the ReSpeaker firmware DOA and lights the ring at the heard bearing.

    The XMOS DSP runs its own (very steady) direction-of-arrival estimator. We
    read that angle over the vendor USB channel and drive the 12-LED ring to
    point at it: red when our harmonic-sum detector also confirms the fan,
    cyan while the card merely hears something. USB writes are throttled to
    10-degree bins so we never flood the control endpoint.
    """

    def __init__(self, brightness=12, led0_offset=0.0, clockwise=True):
        self.led0_offset = led0_offset
        self.clockwise = clockwise
        self.tuning = None
        self.ring = None
        self.available = False
        self.error = None
        self.last_bin = None
        self.last_lit = False
        try:
            import respeaker_led_control as rc
            dev = rc.find_device()
            self.tuning = rc.Tuning(dev)
            self.ring = rc.PixelRing(dev)
            self.ring.set_brightness(int(max(0, min(31, brightness))))
            self.available = True
        except Exception as e:  # pragma: no cover - hardware/driver dependent
            self.error = str(e)

    def update(self, detected):
        """Read firmware DOA + VAD and light the ring. Returns (doa, vad)."""
        if not self.available:
            return None, None
        try:
            doa = self.tuning.direction
            vad = self.tuning.voice_activity
        except Exception as e:
            self.error = str(e)
            return None, None
        active = bool(detected) or bool(vad)
        if active:
            bin10 = int(round(doa / 10.0))
            if bin10 != self.last_bin or not self.last_lit:
                rgb = (255, 30, 0) if detected else (0, 255, 255)
                self.ring.point_at(doa, rgb=rgb,
                                   led0_offset_deg=self.led0_offset,
                                   clockwise=self.clockwise)
                self.last_bin = bin10
                self.last_lit = True
        elif self.last_lit:
            self.ring.off()
            self.last_lit = False
            self.last_bin = None
        return doa, vad

    def release(self):
        if self.available and self.ring is not None:
            try:
                self.ring.trace()
            except Exception:
                pass


# ----------------------------------------------------------------------------
# Harmonic-sum detector
# ----------------------------------------------------------------------------
class HarmonicSumDetector:
    """Estimates fan-tone presence by summing energy across blade harmonics."""

    def __init__(self, sample_rate, fft_samples, f_min, f_max,
                 n_harmonics, guard_bins=3):
        self.sample_rate = sample_rate
        self.n = fft_samples
        self.f_min = f_min
        self.f_max = f_max
        self.n_harmonics = n_harmonics
        self.guard_bins = guard_bins
        self.freqs = np.fft.rfftfreq(self.n, d=1.0 / sample_rate)
        self.window = np.hanning(self.n)
        self.bin_hz = sample_rate / self.n

    def _avg_spectrum(self, channels):
        """Incoherently average magnitude spectra across all mic channels."""
        acc = None
        for sig in channels:
            sig = sig - np.mean(sig)
            mag = np.abs(np.fft.rfft(sig * self.window))
            acc = mag if acc is None else acc + mag
        return acc / len(channels)

    def analyze(self, channels):
        """Return dict with best f0, harmonic-sum SNR (dB), per-harmonic detail."""
        spec = self._avg_spectrum(channels)
        # Global noise floor from the median of the in-band spectrum.
        band = (self.freqs >= self.f_min) & (self.freqs <= self.f_max * 1.05)
        if not np.any(band):
            return None
        noise_floor = float(np.median(spec[band]) + 1e-9)

        # Candidate fundamentals: every bin in [f_min, f_max].
        cand_mask = (self.freqs >= self.f_min) & (self.freqs <= self.f_max)
        cand_idx = np.where(cand_mask)[0]
        if cand_idx.size == 0:
            return None

        best = None
        nyq = self.sample_rate / 2.0
        for i in cand_idx:
            f0 = self.freqs[i]
            harmonic_power = 0.0
            harmonic_noise = 0.0
            used = 0
            detail = []
            for h in range(1, self.n_harmonics + 1):
                fh = f0 * h
                if fh > nyq:
                    break
                k = int(round(fh / self.bin_hz))
                if k <= 0 or k >= spec.size:
                    break
                # Peak within +/-1 bin to tolerate slight detuning.
                lo = max(0, k - 1)
                hi = min(spec.size, k + 2)
                peak = float(np.max(spec[lo:hi]))
                # Local noise just outside a guard band around this harmonic.
                g0 = max(0, k - self.guard_bins - 2)
                g1 = min(spec.size, k + self.guard_bins + 3)
                local = np.concatenate([spec[g0:lo], spec[hi:g1]])
                local_noise = float(np.median(local)) if local.size else noise_floor
                harmonic_power += peak * peak
                harmonic_noise += (local_noise * local_noise) + 1e-12
                used += 1
                detail.append((h, fh, peak, local_noise))
            if used == 0:
                continue
            # Coherent-ish ratio: total harmonic energy vs total local noise.
            snr_lin = harmonic_power / harmonic_noise
            snr_db = 10.0 * np.log10(snr_lin + 1e-12)
            if best is None or snr_db > best["snr_db"]:
                best = {
                    "f0_hz": float(f0),
                    "snr_db": float(snr_db),
                    "n_harmonics_used": used,
                    "noise_floor": noise_floor,
                    "harmonics": detail,
                }
        return best


# ----------------------------------------------------------------------------
# Audio capture
# ----------------------------------------------------------------------------
def open_stream(pa, rate, channels, chunk, device_index):
    import pyaudio
    return pa.open(format=pyaudio.paInt16, channels=channels, rate=rate,
                   input=True, frames_per_buffer=chunk,
                   input_device_index=device_index)


def find_respeaker(pa):
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if "respeaker" in info.get("name", "").lower() and info.get("maxInputChannels", 0) >= 4:
            return i
    return None


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="ReSpeaker acoustic range profiler")
    p.add_argument("--rate", type=int, default=16000)
    p.add_argument("--channels", type=int, default=4)
    p.add_argument("--chunk", type=int, default=1024)
    p.add_argument("--device", type=int, default=None,
                   help="PyAudio device index (auto-detect ReSpeaker if omitted).")
    p.add_argument("--fft-sec", type=float, default=0.5,
                   help="FFT window length in seconds. Longer = finer bins, more "
                        "sensitivity, slower update. 0.5 s -> 2 Hz bins.")
    p.add_argument("--f-min", type=float, default=120.0,
                   help="Lowest fan fundamental to search (Hz).")
    p.add_argument("--f-max", type=float, default=220.0,
                   help="Highest fan fundamental to search (Hz).")
    p.add_argument("--harmonics", type=int, default=6,
                   help="Number of harmonics to sum (incl. fundamental).")
    p.add_argument("--detect-db", type=float, default=6.0,
                   help="Harmonic-sum SNR (dB) above which we declare 'detected'.")
    p.add_argument("--no-gui", action="store_true",
                   help="Console-only mode (no matplotlib window).")
    p.add_argument("--no-led", action="store_true",
                   help="Do not drive the ReSpeaker LED ring direction indicator.")
    p.add_argument("--led-brightness", type=int, default=12,
                   help="LED ring brightness 0-31.")
    p.add_argument("--led0-offset", type=float, default=60.0,
                   help="Physical mounting offset of LED index 0 (deg). "
                        "Calibrated default 60 (from --mode align).")
    p.add_argument("--led-counter-clockwise", action="store_true",
                   help="Reverse ring direction if the dot moves the wrong way.")
    args = p.parse_args()

    try:
        import pyaudio
    except ImportError:
        raise SystemExit("PyAudio not installed in this environment.")

    pa = pyaudio.PyAudio()
    dev = args.device if args.device is not None else find_respeaker(pa)
    if dev is None:
        print("[warn] ReSpeaker not auto-detected; using default input device.")
    fft_samples = int(args.fft_sec * args.rate)

    detector = HarmonicSumDetector(
        sample_rate=args.rate, fft_samples=fft_samples,
        f_min=args.f_min, f_max=args.f_max, n_harmonics=args.harmonics)

    buffers = [deque(maxlen=fft_samples) for _ in range(args.channels)]
    stream = open_stream(pa, args.rate, args.channels, args.chunk, dev)

    print("=" * 70)
    print("  Acoustic Range Profiler  (ReSpeaker 4-mic, harmonic-sum)")
    print(f"  Device index : {dev}")
    print(f"  FFT window   : {args.fft_sec:.2f} s  ({fft_samples} samp, "
          f"{args.rate/fft_samples:.2f} Hz bins)")
    print(f"  Search band  : {args.f_min:.0f}-{args.f_max:.0f} Hz fundamental")
    print(f"  Harmonics    : {args.harmonics}  summed")
    print(f"  Detect thresh: {args.detect_db:.1f} dB harmonic-sum SNR")
    print("=" * 70)

    # SNR-vs-distance curve storage.
    curve = []  # list of (distance_m, snr_db)
    snr_hist = deque(maxlen=200)

    # LED ring direction indicator (firmware DOA -> ring).
    led = None
    if not args.no_led:
        led = RingDirection(brightness=args.led_brightness,
                            led0_offset=args.led0_offset,
                            clockwise=not args.led_counter_clockwise)
        if led.available:
            print(f"  LED ring     : ENABLED (firmware DOA -> ring, "
                  f"offset {args.led0_offset:.0f} deg)")
        else:
            print(f"  LED ring     : unavailable ({led.error})")
            print("                 (bind Interface 3 with Zadig/WinUSB to enable)")
            led = None
    else:
        print("  LED ring     : disabled (--no-led)")

    gui = None
    if not args.no_gui:
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt
            gui = plt
        except Exception as e:
            print(f"[warn] GUI unavailable ({e}); falling back to console.")
            gui = None

    try:
        if gui is not None:
            _run_gui(gui, stream, buffers, detector, args, curve, snr_hist, led)
        else:
            _run_console(stream, buffers, detector, args, snr_hist, led)
    finally:
        if led is not None:
            led.release()

    stream.stop_stream()
    stream.close()
    pa.terminate()


def _read_into(buffers, stream, chunk, channels):
    try:
        raw = stream.read(chunk, exception_on_overflow=False)
    except Exception:
        return False
    samples = np.frombuffer(raw, dtype=np.int16).reshape(-1, channels)
    for ch in range(channels):
        buffers[ch].extend(samples[:, ch].astype(np.float64))
    return len(buffers[0]) >= buffers[0].maxlen


def _run_console(stream, buffers, detector, args, snr_hist, led=None):
    print("\nConsole mode. Ctrl+C to stop.\n")
    try:
        while True:
            if not _read_into(buffers, stream, args.chunk, args.channels):
                continue
            chans = [np.array(b) for b in buffers]
            res = detector.analyze(chans)
            if res is None:
                continue
            snr_hist.append(res["snr_db"])
            detected = res["snr_db"] >= args.detect_db
            det = "DETECT" if detected else "  -   "
            doa_str = ""
            if led is not None:
                doa, vad = led.update(detected)
                if doa is not None:
                    doa_str = f"  DOA={doa:3d} deg  VAD={vad}"
            print(f"\r[{det}] f0={res['f0_hz']:6.1f} Hz  "
                  f"SNR={res['snr_db']:6.2f} dB  "
                  f"(harm {res['n_harmonics_used']}){doa_str}      ",
                  end="", flush=True)
    except KeyboardInterrupt:
        print("\nStopped.")


def _run_gui(plt, stream, buffers, detector, args, curve, snr_hist, led=None):
    import matplotlib.pyplot as _plt
    from matplotlib.widgets import TextBox

    fig, (ax_spec, ax_curve) = _plt.subplots(2, 1, figsize=(9, 7))
    fig.subplots_adjust(bottom=0.18, hspace=0.4)
    fig.canvas.manager.set_window_title("Acoustic Range Profiler")

    # Large on-screen direction readout (top-right of the figure).
    doa_text = fig.text(0.985, 0.965, "DOA --", ha="right", va="top",
                        fontsize=13, fontweight="bold", color="gray",
                        family="monospace")

    # Distance entry box.
    ax_box = fig.add_axes([0.15, 0.04, 0.15, 0.05])
    dist_box = TextBox(ax_box, "Distance (m) ", initial="1.0")

    state = {"last": None}

    def on_key(event):
        if event.key == "d" and state["last"] is not None:
            try:
                dist = float(dist_box.text)
            except ValueError:
                return
            curve.append((dist, state["last"]["snr_db"]))
            print(f"\n[tag] {dist:.2f} m -> {state['last']['snr_db']:.2f} dB")
        elif event.key == "c":
            curve.clear()
            print("\n[curve cleared]")
        elif event.key == "s":
            _save(curve, fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    def update():
        if not _read_into(buffers, stream, args.chunk, args.channels):
            return
        chans = [np.array(b) for b in buffers]
        res = detector.analyze(chans)
        if res is None:
            return
        state["last"] = res
        snr_hist.append(res["snr_db"])
        detected = res["snr_db"] >= args.detect_db

        # --- LED ring + on-screen direction readout ---
        if led is not None:
            doa, vad = led.update(detected)
            if doa is not None:
                tag = "DRONE" if detected else ("sound" if vad else "quiet")
                color = "red" if detected else ("#0aa0a0" if vad else "gray")
                doa_text.set_text(f"DOA {doa:3d} deg  [{tag}]")
                doa_text.set_color(color)
                print(f"\r[DOA] {doa:3d} deg   VAD={vad}   "
                      f"SNR={res['snr_db']:5.1f} dB   "
                      f"{'DETECTED' if detected else '        '}   ",
                      end="", flush=True)

        # --- Spectrum + harmonic markers ---
        ax_spec.clear()
        spec = detector._avg_spectrum(chans)
        ax_spec.semilogy(detector.freqs, spec + 1e-6, color="#3377cc", lw=0.8)
        for (h, fh, peak, ln) in res["harmonics"]:
            ax_spec.axvline(fh, color="orange", alpha=0.5, lw=1)
            ax_spec.text(fh, peak, f"{h}", fontsize=7, color="darkorange")
        ax_spec.set_xlim(0, min(detector.f_max * (args.harmonics + 1),
                                detector.sample_rate / 2))
        det = res["snr_db"] >= args.detect_db
        ax_spec.set_title(
            f"f0={res['f0_hz']:.1f} Hz   harmonic-sum SNR={res['snr_db']:.2f} dB"
            f"   [{'DETECTED' if det else 'no detect'}]",
            color="green" if det else "gray")
        ax_spec.set_xlabel("Hz")
        ax_spec.set_ylabel("avg |FFT|")

        # --- SNR vs distance curve ---
        ax_curve.clear()
        if curve:
            ds = np.array([c[0] for c in curve])
            ss = np.array([c[1] for c in curve])
            order = np.argsort(ds)
            ax_curve.plot(ds[order], ss[order], "o-", color="#cc3333")
            # Inverse-square reference fit (SNR_dB ~ -20 log10(d) + C).
            if len(ds) >= 2:
                C = np.mean(ss + 20.0 * np.log10(ds + 1e-9))
                dd = np.linspace(max(0.2, ds.min() * 0.5), ds.max() * 3, 100)
                ax_curve.plot(dd, -20.0 * np.log10(dd) + C, "--",
                              color="gray", alpha=0.7, label="inverse-square fit")
                # Estimated max range where fit crosses the detect threshold.
                d_max = 10.0 ** ((C - args.detect_db) / 20.0)
                ax_curve.axhline(args.detect_db, color="green", ls=":", lw=1)
                ax_curve.set_title(f"Estimated max detection range ~ {d_max:.2f} m "
                                   f"(at {args.detect_db:.0f} dB)")
                ax_curve.legend(fontsize=8)
        else:
            ax_curve.set_title("Tag distances with 'd' to build the range curve")
        ax_curve.set_xlabel("distance (m)")
        ax_curve.set_ylabel("harmonic-sum SNR (dB)")
        ax_curve.grid(True, alpha=0.3)

        fig.canvas.draw_idle()

    timer = fig.canvas.new_timer(interval=120)
    timer.add_callback(update)
    timer.start()
    print("\nGUI running. Keys: d=tag distance, c=clear, s=save. Close window to exit.")
    _plt.show()


def _save(curve, fig):
    with open("acoustic_range_profile.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["distance_m", "snr_db"])
        for d, s in curve:
            w.writerow([f"{d:.3f}", f"{s:.3f}"])
    fig.savefig("acoustic_range_profile.png", dpi=120)
    print("\n[saved] acoustic_range_profile.csv + .png")


if __name__ == "__main__":
    main()
