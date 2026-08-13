"""GPU-accelerated event-based vibration spectral analysis.

Turns an event stream into a per-region modal-frequency map by:
  1. binning polarity-signed events into a space-time tensor  [T, Hs, Ws]
     (time bins x spatially-pooled cells),
  2. scattering that tensor onto the GPU,
  3. running an FFT along the time axis (torch.fft.rfft) on the GPU,
  4. peak-picking the dominant flutter frequency per cell and globally.

This is the "Event-Based Vibration Monitoring of Flexible Solar Arrays" pipeline,
accelerated on the RTX (Blackwell / sm_120) via PyTorch cu128.

Event sources (pick one):
  --synth                 generate a synthetic vibrating panel (known frequencies)
  --csv  FILE             CSV with columns t_us,x,y,p   (video_to_event output)
  --input FILE.raw|.hdf5  a Metavision recording (needs metavision_core)

Examples:
  python gpu_vibration_spectra.py --synth
  python gpu_vibration_spectra.py --csv ..\\results\\events.csv --block 16 --bin-us 1000
"""
import argparse
import sys
import time

import numpy as np

try:
    import torch
except Exception as e:  # pragma: no cover
    print("PyTorch is required. Install: pip install torch --index-url "
          "https://download.pytorch.org/whl/cu128\n  ->", e)
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Event loading
# --------------------------------------------------------------------------- #
def load_csv(path):
    """Load t_us,x,y,p from a CSV (header auto-detected)."""
    arr = np.genfromtxt(path, delimiter=",", names=True)
    names = arr.dtype.names
    t = arr[names[0]].astype(np.int64)
    x = arr[names[1]].astype(np.int32)
    y = arr[names[2]].astype(np.int32)
    p = arr[names[3]].astype(np.int8)
    W = int(x.max()) + 1
    H = int(y.max()) + 1
    return t, x, y, p, W, H


def load_raw(path, max_us):
    """Load events from a Metavision .raw/.hdf5 recording."""
    from metavision_core.event_io import EventsIterator
    ts, xs, ys, ps = [], [], [], []
    it = EventsIterator(path, delta_t=100000, max_duration=max_us or None)
    H = it.get_size()[0]
    W = it.get_size()[1]
    for evs in it:
        if evs.size == 0:
            continue
        ts.append(evs["t"].astype(np.int64))
        xs.append(evs["x"].astype(np.int32))
        ys.append(evs["y"].astype(np.int32))
        ps.append(evs["p"].astype(np.int8))
    if not ts:
        raise RuntimeError("no events read from " + path)
    return (np.concatenate(ts), np.concatenate(xs), np.concatenate(ys),
            np.concatenate(ps), W, H)


def make_synth(duration_s=8.0, W=320, H=240, seed=0):
    """Two vibrating vertical edges at known frequencies -> real event tuples.

    Left region oscillates at F1 Hz, right region at F2 Hz.  Events are emitted
    where each edge crosses a pixel column as it sweeps, with polarity set by the
    direction of motion (a faithful, if idealized, event stream).
    """
    rng = np.random.default_rng(seed)
    F1, F2 = 7.0, 18.0            # Hz (target modal frequencies)
    A = 6.0                        # px amplitude
    fs = 2000.0                    # internal sampling (Hz)
    n = int(duration_s * fs)
    tt = np.arange(n) / fs
    ts, xs, ys, ps = [], [], [], []
    for (f, x0, y0, y1) in ((F1, W * 0.30, 0, H // 2), (F2, W * 0.68, H // 2, H)):
        xe = x0 + A * np.sin(2 * np.pi * f * tt)
        col = np.round(xe).astype(np.int32)
        dirn = np.sign(np.gradient(xe))
        ys_span = np.arange(y0, y1, 2, dtype=np.int32)
        for k in range(1, n):
            if col[k] == col[k - 1]:
                continue
            c = col[k]
            m = ys_span.size
            ts.append(np.full(m, int(tt[k] * 1e6), dtype=np.int64))
            xs.append(np.full(m, c, dtype=np.int32))
            ys.append(ys_span)
            ps.append(np.full(m, 1 if dirn[k] > 0 else 0, dtype=np.int8))
    # sprinkle noise events
    nn = n * 3
    ts.append(rng.integers(0, int(duration_s * 1e6), nn).astype(np.int64))
    xs.append(rng.integers(0, W, nn).astype(np.int32))
    ys.append(rng.integers(0, H, nn).astype(np.int32))
    ps.append(rng.integers(0, 2, nn).astype(np.int8))
    t = np.concatenate(ts); x = np.concatenate(xs)
    y = np.concatenate(ys); p = np.concatenate(ps)
    order = np.argsort(t)
    return t[order], x[order], y[order], p[order], W, H


# --------------------------------------------------------------------------- #
# GPU binning + FFT
# --------------------------------------------------------------------------- #
def bin_to_voxels(t, x, y, p, W, H, block, bin_us, device):
    """Scatter signed events into a [T, Hs, Ws] tensor on `device`."""
    t0 = int(t.min())
    Ws = (W + block - 1) // block
    Hs = (H + block - 1) // block
    tb = torch.as_tensor((t - t0) // bin_us, dtype=torch.long, device=device)
    T = int(tb.max().item()) + 1
    xs = torch.as_tensor(x // block, dtype=torch.long, device=device)
    ys = torch.as_tensor(y // block, dtype=torch.long, device=device)
    val = torch.as_tensor(p.astype(np.float32) * 2.0 - 1.0, device=device)  # -1/+1
    flat = (tb * Hs + ys) * Ws + xs
    vox = torch.zeros(T * Hs * Ws, dtype=torch.float32, device=device)
    vox.index_add_(0, flat, val)
    return vox.view(T, Hs, Ws), T, Hs, Ws


def spectra(vox, T, bin_us, min_hz, device):
    """FFT along time -> magnitude spectrum, freq axis (both on `device`)."""
    vox = vox - vox.mean(dim=0, keepdim=True)                  # remove DC drift
    win = torch.hann_window(T, device=device)
    vox = vox * win[:, None, None]
    spec = torch.fft.rfft(vox, dim=0).abs()                    # [F, Hs, Ws]
    freqs = torch.fft.rfftfreq(T, d=bin_us * 1e-6).to(device)  # Hz
    lo = int(torch.searchsorted(freqs, torch.tensor(min_hz, device=device)).item())
    lo = max(lo, 1)
    return spec, freqs, lo


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--synth", action="store_true", help="synthetic vibrating panel")
    src.add_argument("--csv", help="CSV t_us,x,y,p")
    src.add_argument("--input", help="Metavision .raw/.hdf5")
    ap.add_argument("--block", type=int, default=16, help="spatial pool size px (default 16)")
    ap.add_argument("--bin-us", type=int, default=1000, help="time bin us (default 1000 -> 1kHz)")
    ap.add_argument("--min-hz", type=float, default=2.0, help="ignore below this Hz (default 2)")
    ap.add_argument("--max-us", type=int, default=0, help="cap recording duration us (0=all)")
    ap.add_argument("--topk", type=int, default=5, help="global peaks to report (default 5)")
    ap.add_argument("--cpu", action="store_true", help="force CPU (skip GPU)")
    ap.add_argument("--heatmap", default="", help="save dominant-freq heatmap PNG to this path")
    args = ap.parse_args()

    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
        dev_name = "CPU"
    else:
        device = torch.device("cuda")
        dev_name = torch.cuda.get_device_name(0)

    print("=== GPU event vibration spectra ===")
    print(f"  torch    : {torch.__version__}")
    print(f"  device   : {dev_name}  (cuda_available={torch.cuda.is_available()})")

    # Warm up CUDA (context init + cuFFT plan cache) so timings below are
    # steady-state, not dominated by one-time lazy initialization.
    if device.type == "cuda":
        w = torch.fft.rfft(torch.randn(4096, 64, 64, device=device), dim=0)
        torch.cuda.synchronize()
        del w

    # ---- load events ----
    if args.csv:
        t, x, y, p, W, H = load_csv(args.csv)
        srcdesc = f"csv:{args.csv}"
    elif args.input:
        t, x, y, p, W, H = load_raw(args.input, args.max_us)
        srcdesc = f"raw:{args.input}"
    else:
        t, x, y, p, W, H = make_synth()
        srcdesc = "synthetic (F1=7Hz left, F2=18Hz right)"
    dur = (t.max() - t.min()) / 1e6
    print(f"  source   : {srcdesc}")
    print(f"  events   : {t.size:,}   sensor {W}x{H}   duration {dur:.2f}s")

    # ---- bin (GPU) ----
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_bin = time.perf_counter()
    vox, T, Hs, Ws = bin_to_voxels(t, x, y, p, W, H, args.block, args.bin_us, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_bin = time.perf_counter() - t_bin
    fs = 1e6 / args.bin_us
    print(f"  voxels   : T={T} x {Hs} x {Ws}  (fs={fs:.0f} Hz, Nyquist={fs/2:.0f} Hz)  "
          f"bin {t_bin*1e3:.1f} ms")

    # ---- FFT (GPU) ---- (median of a few runs, post-warmup)
    def _fft_time(v):
        vv = v - v.mean(dim=0, keepdim=True)
        win = torch.hann_window(vv.shape[0], device=vv.device)
        vv = vv * win[:, None, None]
        t0 = time.perf_counter()
        s = torch.fft.rfft(vv, dim=0).abs()
        if vv.device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - t0, s

    fft_times = [_fft_time(vox)[0] for _ in range(3)]
    t_fft = float(np.median(fft_times))
    spec, freqs, lo = spectra(vox, T, args.bin_us, args.min_hz, device)
    print(f"  fft      : {spec.shape[0]} bins x {Hs}x{Ws} on {device.type.upper()}  "
          f"{t_fft*1e3:.1f} ms (median of 3)")

    # ---- global modal peaks ----
    g = spec[lo:].sum(dim=(1, 2))
    fr = freqs[lo:]
    # simple local-maxima peak pick
    gm = g.detach().cpu().numpy()
    frm = fr.detach().cpu().numpy()
    peaks = [i for i in range(1, gm.size - 1) if gm[i] > gm[i-1] and gm[i] >= gm[i+1]]
    peaks.sort(key=lambda i: gm[i], reverse=True)
    peaks = peaks[:args.topk]
    tot = gm.sum() + 1e-9
    print("  --- dominant modal frequencies (global) ---")
    for i in peaks:
        print(f"    {frm[i]:6.2f} Hz    power {gm[i]/tot*100:5.1f}%")

    # ---- per-cell dominant-frequency map ----
    idx = spec[lo:].argmax(dim=0) + lo
    domfreq = freqs[idx]                      # [Hs, Ws]
    dm = domfreq.detach().cpu().numpy()

    # ---- optional GPU-vs-CPU FFT timing (fair: both warmed up, median of 3) ----
    if device.type == "cuda":
        voxc = vox.cpu()
        _ = _fft_time(voxc)  # warm up CPU / FFT plan
        cpu_times = [_fft_time(voxc)[0] for _ in range(3)]
        t_cpu = float(np.median(cpu_times))
        ratio = t_cpu / max(t_fft, 1e-6)
        verdict = f"{ratio:.1f}x faster" if ratio >= 1 else f"{1/ratio:.1f}x slower"
        print(f"  speedup  : GPU FFT {t_fft*1e3:.1f} ms vs CPU {t_cpu*1e3:.1f} ms -> GPU {verdict}")

    # ---- save heatmap ----
    if args.heatmap:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 4))
            im = plt.imshow(dm, cmap="turbo", interpolation="nearest")
            plt.colorbar(im, label="dominant freq (Hz)")
            plt.title("Event vibration: dominant frequency per cell")
            plt.tight_layout()
            plt.savefig(args.heatmap, dpi=130)
            print(f"  heatmap  : {args.heatmap}")
        except Exception as e:
            np.save(args.heatmap + ".npy", dm)
            print(f"  heatmap  : matplotlib unavailable ({e}); saved {args.heatmap}.npy")

    print("=== done ===")


if __name__ == "__main__":
    main()
