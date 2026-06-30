"""
Fit a practical linear azimuth calibration model between visual azimuth estimates
(from event-camera cluster position) and acoustic DOA (mic array).

Input: JSONL session log produced by detect_drone_fused.py --session-log
Output: JSON model with keys: gain, offset_deg, camera_hfov_deg, metadata

Usage:
  python calibration/calibrate_camera_mic_azimuth.py \
      --session-log logs/fusion_session.jsonl \
      --output calibration/azimuth_calibration.json
"""

import argparse
import json
import math
from typing import List, Tuple

import numpy as np


def wrap_angle_deg(x: float) -> float:
    """Wrap angle to [-180, 180)."""
    return ((x + 180.0) % 360.0) - 180.0


def shortest_angle_error_deg(a_deg: float, b_deg: float) -> float:
    """Smallest absolute difference between two azimuth angles in degrees."""
    diff = abs(wrap_angle_deg(a_deg - b_deg))
    if diff > 180.0:
        diff = 360.0 - diff
    return diff


def extract_pairs(
    session_log_path: str,
    min_visual_snr: float,
    min_audio_snr: float,
    require_assoc_ok: bool,
) -> List[Tuple[float, float]]:
    """Extract (visual_azimuth_raw_deg, acoustic_doa_deg) training pairs from log."""
    pairs = []
    with open(session_log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            assoc = rec.get("association", {})
            if require_assoc_ok and not assoc.get("ok", False):
                continue

            v = rec.get("visual") or {}
            a = rec.get("acoustic") or {}

            if "visual_azimuth_raw_deg" not in v:
                continue
            if "doa_azimuth" not in a:
                continue
            if float(v.get("snr", 0.0)) < min_visual_snr:
                continue
            if float(a.get("snr", 0.0)) < min_audio_snr:
                continue

            vis_raw = float(v["visual_azimuth_raw_deg"])
            aud_doa = float(a["doa_azimuth"])
            pairs.append((vis_raw, aud_doa))
    return pairs


def fit_linear_model(pairs: List[Tuple[float, float]], remove_outliers: bool):
    """Fit doa ~= gain * vis + offset with optional residual-based outlier rejection."""
    if len(pairs) < 3:
        raise ValueError("Need at least 3 valid calibration pairs.")

    x = np.array([p[0] for p in pairs], dtype=np.float64)
    y = np.array([p[1] for p in pairs], dtype=np.float64)

    gain, offset = np.polyfit(x, y, deg=1)

    if remove_outliers and len(pairs) >= 8:
        pred = gain * x + offset
        residual = np.array([shortest_angle_error_deg(pred[i], y[i]) for i in range(len(y))])

        q1 = float(np.quantile(residual, 0.25))
        q3 = float(np.quantile(residual, 0.75))
        iqr = q3 - q1
        thresh = q3 + 1.5 * iqr
        keep = residual <= max(thresh, 3.0)

        if np.sum(keep) >= 3 and np.sum(keep) < len(pairs):
            x2 = x[keep]
            y2 = y[keep]
            gain, offset = np.polyfit(x2, y2, deg=1)
            x = x2
            y = y2

    pred = gain * x + offset
    errors = np.array([shortest_angle_error_deg(pred[i], y[i]) for i in range(len(y))])

    metrics = {
        "n_samples": int(len(y)),
        "mae_deg": float(np.mean(errors)),
        "median_err_deg": float(np.median(errors)),
        "max_err_deg": float(np.max(errors)),
        "rmse_deg": float(math.sqrt(np.mean(errors ** 2))),
    }
    return float(gain), float(offset), metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit linear camera-mic azimuth calibration from session logs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--session-log", required=True,
                        help="Path to JSONL session log from detect_drone_fused.py")
    parser.add_argument("--output", required=True,
                        help="Path to output calibration JSON")
    parser.add_argument("--camera-hfov-deg", type=float, default=220.0,
                        help="HFOV used by visual azimuth estimator")
    parser.add_argument("--min-visual-snr", type=float, default=3.0)
    parser.add_argument("--min-audio-snr", type=float, default=3.0)
    parser.add_argument("--allow-stale-association", action="store_true",
                        help="Include records where association.ok is false")
    parser.add_argument("--no-outlier-rejection", action="store_true",
                        help="Disable residual-based outlier rejection")
    return parser.parse_args()


def main():
    args = parse_args()

    pairs = extract_pairs(
        session_log_path=args.session_log,
        min_visual_snr=args.min_visual_snr,
        min_audio_snr=args.min_audio_snr,
        require_assoc_ok=not args.allow_stale_association,
    )

    if len(pairs) < 3:
        raise RuntimeError(
            f"Not enough valid pairs for calibration: {len(pairs)}. "
            "Collect longer sessions or lower SNR thresholds."
        )

    gain, offset_deg, metrics = fit_linear_model(
        pairs,
        remove_outliers=not args.no_outlier_rejection,
    )

    model = {
        "gain": gain,
        "offset_deg": offset_deg,
        "camera_hfov_deg": float(args.camera_hfov_deg),
        "metadata": {
            "source_session_log": args.session_log,
            "min_visual_snr": float(args.min_visual_snr),
            "min_audio_snr": float(args.min_audio_snr),
            "allow_stale_association": bool(args.allow_stale_association),
            "outlier_rejection": not bool(args.no_outlier_rejection),
            "fit_metrics": metrics,
        },
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2)

    print("Calibration model saved:", args.output)
    print(f"  gain={gain:.6f}")
    print(f"  offset_deg={offset_deg:.3f}")
    print("  metrics:")
    for k, v in metrics.items():
        print(f"    {k}: {v:.3f}" if isinstance(v, float) else f"    {k}: {v}")


if __name__ == "__main__":
    main()
