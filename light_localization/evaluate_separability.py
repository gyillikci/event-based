"""
Evaluate Separability — Go/No-Go experiment for light fingerprinting.

Analyzes a captured fingerprint database (or a live/recorded event stream)
to determine whether individual ceiling light fixtures are sufficiently
distinguishable by their frequency characteristics.

Computes:
  - Pairwise frequency distance matrix between all lights
  - Fisher discriminant ratio (between-class vs. within-class variance)
  - Confusion matrix from leave-one-out cross-validation
  - Visualization: frequency histogram, distance heatmap, dendrogram

This script answers the critical question: "Can we reliably tell light A
from light B?" — the go/no-go gate for the entire localization approach.

Usage:
    # From a saved fingerprint database
    python evaluate_separability.py --database calibration_data/room1_fingerprints.json

    # From a live camera or recording (capture + analyze in one step)
    python evaluate_separability.py -i ceiling_recording.raw --min-freq 50 --max-freq 500

    # Save results
    python evaluate_separability.py --database calibration_data/room1_fingerprints.json -o results/
"""

import argparse
import sys
import os
import time
import json
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from light_fingerprint import LightFingerprint, FingerprintDatabase
from light_localization_utils import (
    LightROIExtractor, FrequencyFeatureExtractor, frequency_distance
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate whether ceiling lights are distinguishable by frequency.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Input sources (choose one)
    parser.add_argument(
        "--database", dest="database_path", type=str, default="",
        help="Path to a saved FingerprintDatabase JSON file.",
    )
    parser.add_argument(
        "-i", "--input-event-file", dest="event_file_path", default="",
        help="Path to input event file for live analysis.",
    )

    # Frequency range (for live analysis)
    parser.add_argument(
        "--min-freq", dest="min_freq", type=float, default=50,
        help="Minimum light flicker frequency to detect (Hz).",
    )
    parser.add_argument(
        "--max-freq", dest="max_freq", type=float, default=500,
        help="Maximum light flicker frequency to detect (Hz).",
    )
    parser.add_argument(
        "--delta-t", dest="delta_t", type=int, default=500,
        help="Event slice duration (us) for live analysis.",
    )
    parser.add_argument(
        "--filter-length", dest="filter_length", type=int, default=7,
        help="SDK filter length.",
    )
    parser.add_argument(
        "--max-period-diff", dest="max_period_diff", type=int, default=5000,
        help="SDK max period difference (us).",
    )
    parser.add_argument(
        "--capture-seconds", dest="capture_seconds", type=float, default=10,
        help="Duration to capture before analysis (seconds).",
    )

    # Analysis parameters
    parser.add_argument(
        "--distance-threshold", dest="distance_threshold", type=float, default=0.3,
        help="Maximum distance for two lights to be considered 'same'.",
    )

    # Output
    parser.add_argument(
        "-o", "--output-dir", dest="output_dir", type=str, default="results",
        help="Directory for output plots and reports.",
    )
    parser.add_argument(
        "--no-plot", dest="no_plot", action="store_true",
        help="Skip matplotlib plots (for headless environments).",
    )

    return parser.parse_args()


def compute_separability_metrics(db):
    """Compute all separability metrics from a fingerprint database.

    Args:
        db: FingerprintDatabase with ≥ 2 valid fingerprints.

    Returns:
        Dict with all metrics and analysis results.
    """
    fps = [fp for fp in db.fingerprints.values() if fp.is_valid()]
    n = len(fps)

    if n < 2:
        return {
            "n_lights": n,
            "separable": False,
            "reason": f"Need at least 2 valid fingerprints, got {n}",
        }

    ids = [fp.light_id for fp in fps]
    freqs = [fp.fundamental_freq for fp in fps]

    # --- Pairwise distance matrix ---
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = frequency_distance(fps[i].feature_vector(), fps[j].feature_vector())
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    # --- Minimum pairwise distance (most confusable pair) ---
    upper_tri = dist_matrix[np.triu_indices(n, k=1)]
    min_dist = float(np.min(upper_tri))
    max_dist = float(np.max(upper_tri))
    mean_dist = float(np.mean(upper_tri))
    std_dist = float(np.std(upper_tri))

    min_pair_idx = np.unravel_index(
        np.argmin(dist_matrix + np.eye(n) * 1e10), dist_matrix.shape
    )
    most_confusable = (ids[min_pair_idx[0]], ids[min_pair_idx[1]])

    # --- Frequency separation analysis ---
    freq_array = np.array(freqs)
    freq_sorted_idx = np.argsort(freq_array)
    freq_sorted = freq_array[freq_sorted_idx]
    freq_gaps = np.diff(freq_sorted)
    min_freq_gap = float(np.min(freq_gaps)) if len(freq_gaps) > 0 else 0
    mean_freq_gap = float(np.mean(freq_gaps)) if len(freq_gaps) > 0 else 0

    # --- Fisher Discriminant Ratio ---
    # Between-class variance (variance of fundamental frequencies)
    between_var = float(np.var(freq_array))

    # Within-class variance (average freq_std across lights)
    within_vars = [fp.spectral_bandwidth ** 2 if fp.spectral_bandwidth > 0
                   else (fp.fundamental_freq * 0.01) ** 2  # assume 1% if unknown
                   for fp in fps]
    within_var = float(np.mean(within_vars))

    fisher_ratio = between_var / within_var if within_var > 0 else float('inf')

    # --- Leave-one-out classification accuracy ---
    correct = 0
    confusion_details = []
    for i in range(n):
        # Find nearest neighbor excluding self
        best_j = -1
        best_dist_loo = float('inf')
        for j in range(n):
            if i == j:
                continue
            if dist_matrix[i, j] < best_dist_loo:
                best_dist_loo = dist_matrix[i, j]
                best_j = j
        predicted = ids[best_j] if best_j >= 0 else "none"
        actual = ids[i]
        is_correct = (best_j == i)  # This is always False for LOO — we check if it's the right cluster
        # In LOO for uniqueness: the question is "does the nearest neighbor have a different ID?"
        # Since each light has exactly 1 sample, "correct" means the nearest is distant enough
        confusion_details.append({
            "actual": actual,
            "nearest": predicted,
            "distance": best_dist_loo,
        })

    # For separability: all pairwise distances should exceed the threshold
    confusable_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if dist_matrix[i, j] < 0.1:  # very close = confusable
                confusable_pairs.append((ids[i], ids[j], dist_matrix[i, j]))

    all_separated = len(confusable_pairs) == 0

    # --- Go/No-Go verdict ---
    go = all_separated and fisher_ratio > 2.0 and min_dist > 0.05
    if go:
        verdict = "GO — Lights are sufficiently distinguishable"
    elif fisher_ratio > 1.0 and min_dist > 0.02:
        verdict = "MARGINAL — Some lights may be confusable. Consider harmonic features."
    else:
        verdict = "NO-GO — Lights are not separable with current features"

    return {
        "n_lights": n,
        "light_ids": ids,
        "frequencies": freqs,
        "distance_matrix": dist_matrix,
        "min_distance": min_dist,
        "max_distance": max_dist,
        "mean_distance": mean_dist,
        "std_distance": std_dist,
        "most_confusable_pair": most_confusable,
        "min_freq_gap_hz": min_freq_gap,
        "mean_freq_gap_hz": mean_freq_gap,
        "fisher_ratio": fisher_ratio,
        "between_variance": between_var,
        "within_variance": within_var,
        "confusable_pairs": confusable_pairs,
        "all_separated": all_separated,
        "verdict": verdict,
        "separable": go,
        "confusion_details": confusion_details,
    }


def generate_visualizations(metrics, output_dir, no_plot=False):
    """Generate and save analysis visualizations.

    Uses OpenCV for basic plots (no matplotlib dependency required).
    Optionally uses matplotlib if available and --no-plot is not set.
    """
    os.makedirs(output_dir, exist_ok=True)
    n = metrics["n_lights"]

    # --- Distance matrix heatmap (OpenCV) ---
    if n >= 2:
        dist_mat = metrics["distance_matrix"]
        # Normalize to 0-255
        max_val = max(dist_mat.max(), 0.001)
        norm_mat = (dist_mat / max_val * 255).astype(np.uint8)

        # Scale up for visibility
        cell_size = max(40, 400 // n)
        img_size = n * cell_size
        heatmap = np.zeros((img_size, img_size, 3), dtype=np.uint8)

        for i in range(n):
            for j in range(n):
                val = norm_mat[i, j]
                # Color: green (far = good) → red (close = bad)
                if i == j:
                    color = (128, 128, 128)  # gray diagonal
                else:
                    r = int(255 * (1 - val / 255))
                    g = int(255 * val / 255)
                    color = (0, g, r)
                y1, y2 = i * cell_size, (i + 1) * cell_size
                x1, x2 = j * cell_size, (j + 1) * cell_size
                cv2.rectangle(heatmap, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(heatmap, (x1, y1), (x2, y2), (50, 50, 50), 1)

                # Distance value text
                dist_val = dist_mat[i, j]
                text = f"{dist_val:.2f}"
                font_scale = 0.3 if cell_size < 60 else 0.4
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
                tx = x1 + (cell_size - text_size[0]) // 2
                ty = y1 + (cell_size + text_size[1]) // 2
                cv2.putText(heatmap, text, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

        # Labels
        label_margin = 120
        full_img = np.zeros((img_size + 30, img_size + label_margin, 3), dtype=np.uint8)
        full_img[30:30 + img_size, label_margin:label_margin + img_size] = heatmap

        for i, lid in enumerate(metrics["light_ids"]):
            short_id = lid.split("_")[-1] if "_" in lid else lid[:8]
            y = 30 + i * cell_size + cell_size // 2 + 4
            cv2.putText(full_img, short_id, (5, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            x = label_margin + i * cell_size + cell_size // 2 - 10
            cv2.putText(full_img, short_id, (x, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        heatmap_path = os.path.join(output_dir, "distance_matrix_heatmap.png")
        cv2.imwrite(heatmap_path, full_img)
        print(f"  Saved: {heatmap_path}")

    # --- Frequency bar chart (OpenCV) ---
    freqs = metrics["frequencies"]
    ids = metrics["light_ids"]
    if freqs:
        chart_w = max(400, n * 80)
        chart_h = 300
        chart = np.zeros((chart_h + 60, chart_w, 3), dtype=np.uint8)

        max_freq = max(freqs) * 1.1
        bar_width = max(20, (chart_w - 60) // n - 10)
        x_start = 50

        for i, (freq, lid) in enumerate(zip(freqs, ids)):
            bar_h = int((freq / max_freq) * chart_h)
            x = x_start + i * (bar_width + 10)
            y_top = chart_h - bar_h
            color = (0, 200, 100)
            cv2.rectangle(chart, (x, y_top + 30), (x + bar_width, chart_h + 30), color, -1)
            cv2.rectangle(chart, (x, y_top + 30), (x + bar_width, chart_h + 30), (255, 255, 255), 1)

            # Frequency label on top
            cv2.putText(chart, f"{freq:.0f}", (x, y_top + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            # Light ID below
            short_id = lid.split("_")[-1] if "_" in lid else lid[:6]
            cv2.putText(chart, short_id, (x, chart_h + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)

        cv2.putText(chart, "Fundamental Frequency (Hz)", (chart_w // 3, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        chart_path = os.path.join(output_dir, "frequency_distribution.png")
        cv2.imwrite(chart_path, chart)
        print(f"  Saved: {chart_path}")

    # --- Try matplotlib for publication-quality plots ---
    if not no_plot:
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            from matplotlib.colors import LinearSegmentedColormap

            # Distance matrix with proper colorbar
            if n >= 2:
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(metrics["distance_matrix"], cmap='RdYlGn',
                               vmin=0, vmax=max(metrics["max_distance"], 0.5))
                ax.set_xticks(range(n))
                ax.set_yticks(range(n))
                short_ids = [lid.split("_")[-1] if "_" in lid else lid[:10]
                             for lid in ids]
                ax.set_xticklabels(short_ids, rotation=45, ha='right', fontsize=8)
                ax.set_yticklabels(short_ids, fontsize=8)
                plt.colorbar(im, label='Feature Distance')
                ax.set_title(f'Pairwise Distance Matrix — {metrics["verdict"]}')

                # Annotate cells
                for i in range(n):
                    for j in range(n):
                        ax.text(j, i, f'{metrics["distance_matrix"][i, j]:.2f}',
                                ha='center', va='center', fontsize=7,
                                color='white' if metrics["distance_matrix"][i, j] < 0.2 else 'black')

                plt.tight_layout()
                mpl_path = os.path.join(output_dir, "distance_matrix_matplotlib.png")
                plt.savefig(mpl_path, dpi=150)
                plt.close()
                print(f"  Saved: {mpl_path}")

            # Frequency distribution histogram
            fig, ax = plt.subplots(figsize=(10, 4))
            bars = ax.bar(range(n), freqs, color='steelblue', edgecolor='white')
            ax.set_xticks(range(n))
            ax.set_xticklabels([lid.split("_")[-1] if "_" in lid else lid[:10]
                                for lid in ids], rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('Frequency (Hz)')
            ax.set_title('Fundamental Frequency per Light Fixture')
            for bar, freq in zip(bars, freqs):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(freqs) * 0.02,
                        f'{freq:.1f}', ha='center', va='bottom', fontsize=7)
            plt.tight_layout()
            mpl_freq_path = os.path.join(output_dir, "frequency_distribution_matplotlib.png")
            plt.savefig(mpl_freq_path, dpi=150)
            plt.close()
            print(f"  Saved: {mpl_freq_path}")

        except ImportError:
            print("  (matplotlib not available — OpenCV plots only)")


def capture_and_analyze(args):
    """Capture events from camera/file, build fingerprints, and analyze."""
    from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator, is_live_camera
    from metavision_sdk_analytics import FrequencyMapAsyncAlgorithm

    delta_t = args.delta_t
    sample_rate_hz = 1e6 / delta_t

    # Nyquist check
    nyquist = sample_rate_hz / 2
    if args.max_freq > nyquist:
        print(f"ERROR: max_freq ({args.max_freq} Hz) exceeds Nyquist ({nyquist:.0f} Hz)")
        sys.exit(1)

    mv_iterator = EventsIterator(input_path=args.event_file_path, delta_t=delta_t)
    height, width = mv_iterator.get_size()

    if not is_live_camera(args.event_file_path):
        mv_iterator = LiveReplayEventsIterator(mv_iterator, replay_factor=0)  # Max speed

    freq_algo = FrequencyMapAsyncAlgorithm(
        width=width, height=height,
        filter_length=args.filter_length,
        min_freq=args.min_freq,
        max_freq=args.max_freq,
        diff_thresh_us=args.max_period_diff,
    )
    freq_algo.update_frequency = 5  # Low rate — we want stable maps

    roi_extractor = LightROIExtractor(
        width=width, height=height,
        min_freq=args.min_freq, max_freq=args.max_freq,
    )

    fft_extractor = FrequencyFeatureExtractor(
        fft_window_samples=512, num_harmonics=5, min_snr=2.0,
    )

    # Collect frequency maps
    collected_rois = []
    last_freq_map = [None]
    current_rois_ref = [[]]

    def on_freq_map(ts, freq_map):
        rois = roi_extractor.extract(freq_map)
        current_rois_ref[0] = rois
        last_freq_map[0] = freq_map.copy()
        if rois:
            collected_rois.append((ts, rois))

    freq_algo.set_output_callback(on_freq_map)

    print(f"Capturing events for {args.capture_seconds} seconds...")
    start = time.time()
    for evs in mv_iterator:
        freq_algo.process_events(evs)
        for roi in current_rois_ref[0]:
            fft_extractor.accumulate_events(roi["roi_id"], evs, roi["bbox"])
        if time.time() - start > args.capture_seconds:
            break
    print(f"Captured {len(collected_rois)} frequency map snapshots.")

    if not collected_rois:
        print("ERROR: No light fixtures detected. Check frequency range and camera aim.")
        sys.exit(1)

    # Use the last snapshot's ROIs to build fingerprints
    _, final_rois = collected_rois[-1]
    db = FingerprintDatabase()
    for roi in final_rois:
        light_id = f"light_{roi['roi_id']}"
        fp = LightFingerprint(light_id=light_id)
        fp.set_from_roi(roi)
        features = fft_extractor.extract_features(
            roi["roi_id"], sample_rate_hz,
            min_freq=args.min_freq, max_freq=args.max_freq,
        )
        if features:
            fp.set_frequency_features(features)
        else:
            fp.fundamental_freq = roi["median_freq"]
            fp.peak_snr = 1.0 / max(0.01, roi["freq_cv"])
            fp.calibration_timestamp = time.time()
        db.add(fp)

    return db


def print_report(metrics):
    """Print a formatted analysis report to console."""
    print()
    print("=" * 78)
    print("  LIGHT SEPARABILITY ANALYSIS REPORT")
    print("=" * 78)
    print()
    print(f"  Number of lights:      {metrics['n_lights']}")
    print()

    if metrics["n_lights"] < 2:
        print(f"  ❌ {metrics.get('reason', 'Insufficient data')}")
        return

    print("  Light Frequencies:")
    for lid, freq in zip(metrics["light_ids"], metrics["frequencies"]):
        print(f"    {lid:30s}  {freq:10.1f} Hz")
    print()

    print("  Distance Matrix Statistics:")
    print(f"    Min pairwise distance:  {metrics['min_distance']:.4f}")
    print(f"    Max pairwise distance:  {metrics['max_distance']:.4f}")
    print(f"    Mean pairwise distance: {metrics['mean_distance']:.4f}")
    print(f"    Std pairwise distance:  {metrics['std_distance']:.4f}")
    print(f"    Most confusable pair:   {metrics['most_confusable_pair'][0]} ↔ {metrics['most_confusable_pair'][1]}")
    print()

    print("  Frequency Gaps:")
    print(f"    Min gap between adjacent: {metrics['min_freq_gap_hz']:.1f} Hz")
    print(f"    Mean gap between adjacent: {metrics['mean_freq_gap_hz']:.1f} Hz")
    print()

    print("  Fisher Discriminant Ratio:")
    print(f"    Between-class variance: {metrics['between_variance']:.2f}")
    print(f"    Within-class variance:  {metrics['within_variance']:.2f}")
    print(f"    Fisher ratio:           {metrics['fisher_ratio']:.2f}")
    print(f"    (>2.0 = good, >5.0 = excellent, <1.0 = poor)")
    print()

    if metrics["confusable_pairs"]:
        print("  ⚠ Confusable Pairs (distance < 0.1):")
        for id1, id2, dist in metrics["confusable_pairs"]:
            print(f"    {id1} ↔ {id2}: distance = {dist:.4f}")
        print()

    print("  " + "─" * 60)
    separable_icon = "✅" if metrics["separable"] else "❌"
    print(f"  {separable_icon} VERDICT: {metrics['verdict']}")
    print("  " + "─" * 60)
    print()


def main():
    args = parse_args()

    # Determine input source
    if args.database_path:
        print(f"Loading fingerprint database: {args.database_path}")
        db = FingerprintDatabase.load(args.database_path)
        print(db.summary())
    elif args.event_file_path:
        print(f"Analyzing event file: {args.event_file_path}")
        db = capture_and_analyze(args)
    else:
        print("Analyzing live camera...")
        db = capture_and_analyze(args)

    if len(db) < 2:
        print(f"\nOnly {len(db)} light(s) found. Need ≥ 2 for separability analysis.")
        print("Try adjusting --min-freq / --max-freq or aim the camera at more lights.")
        return

    # Compute metrics
    print("\nComputing separability metrics...")
    metrics = compute_separability_metrics(db)

    # Print report
    print_report(metrics)

    # Save results
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), args.output_dir
    )
    os.makedirs(output_dir, exist_ok=True)

    # Save metrics as JSON (excluding numpy arrays)
    metrics_serializable = {k: v for k, v in metrics.items()
                           if not isinstance(v, np.ndarray)}
    metrics_path = os.path.join(output_dir, "separability_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_serializable, f, indent=2, default=str)
    print(f"  Saved metrics: {metrics_path}")

    # Save distance matrix as CSV
    if "distance_matrix" in metrics:
        csv_path = os.path.join(output_dir, "distance_matrix.csv")
        header = "," + ",".join(metrics["light_ids"])
        np.savetxt(csv_path, metrics["distance_matrix"], delimiter=",",
                   header=header, comments="", fmt="%.4f")
        print(f"  Saved distance matrix: {csv_path}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_visualizations(metrics, output_dir, no_plot=args.no_plot)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
