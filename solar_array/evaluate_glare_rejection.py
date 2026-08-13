"""Benchmark harness producing the IAC 2026 result set.

Three experiments, all driven by the synthetic terminator crossing so that every
number is scored against exact ground truth:

``storm``
    Characterises the event storm itself: rate profile, polarity balance, the
    common-mode index, per-slice common-mode rejection ratio, masked fraction, and
    a head-to-head between a governed rate cap and an ungoverned readout overflow.

``modal``
    Runs the full pipeline end to end and ablates the guard one layer at a time,
    scoring modal frequency error, damping error, deflection-shape MAC and gauge
    dropout against the prescribed modal content.

``erc-sweep``
    Sweeps the rate cap to test the claim that uniform pseudo-random decimation is
    unbiased for frequency estimation while inflating variance.

Results are written as JSON to ``results/`` alongside figures, if matplotlib is
available.

Usage::

    python evaluate_glare_rejection.py --experiment all
    python evaluate_glare_rejection.py --experiment modal --duration 600
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from .event_vibrometer import VibrometerArray
    from .glare_guard import CommonModeRejector, EventSliceBinner, GlareGuard
    from .modal_analysis import (
        ModalIdentifier, modal_assurance_criterion, welch_psd,
    )
    from .orbital_sunrise_sim import OrbitalSunriseSimulator
except ImportError:  # running as a script from inside the package directory
    from event_vibrometer import VibrometerArray                    # type: ignore
    from glare_guard import (                                       # type: ignore
        CommonModeRejector, EventSliceBinner, GlareGuard,
    )
    from modal_analysis import (                                    # type: ignore
        ModalIdentifier, modal_assurance_criterion, welch_psd,
    )
    from orbital_sunrise_sim import OrbitalSunriseSimulator         # type: ignore

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "solar_array_config.json")
RESULTS_DIR = os.path.join(HERE, "results")

#: Guard configurations ablated in the ``modal`` experiment.
ABLATIONS: Sequence[Tuple[str, Dict[str, object]]] = (
    ("none",            {"erc": False, "mask": False, "common_mode": "none",
                         "overflow": True}),
    ("erc",             {"erc": True,  "mask": False, "common_mode": "none"}),
    ("erc+mask",        {"erc": True,  "mask": True,  "common_mode": "none"}),
    ("erc+mask+median", {"erc": True,  "mask": True,  "common_mode": "median"}),
    ("erc+mask+plane",  {"erc": True,  "mask": True,  "common_mode": "plane"}),
)


#: Cell sizes swept to test how common-mode rejection scales with cell area.
CELL_SWEEP: Sequence[int] = (8, 16, 32, 64)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("  (matplotlib unavailable - skipping figures)")
        return None


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #

def ground_truth_shapes(sim: OrbitalSunriseSimulator,
                        vib: VibrometerArray) -> List[np.ndarray]:
    """True deflection shape of each mode, sampled at the gauge stations.

    A bending mode displaces both panel edges identically. A torsion mode rotates
    the section, so the two edges move in anti-phase -- which is exactly why gauge
    signs must be fixed geometrically rather than inferred.
    """
    shapes = []
    for mode in sim.array.modes:
        vals = []
        for gauge in vib.gauges:
            phi = float(mode.shape(np.array([gauge.xi]))[0])
            if mode.kind == "torsion":
                phi *= -1.0 if gauge.edge == "top" else 1.0
            vals.append(phi)
        shapes.append(np.array(vals, dtype=np.float64))
    return shapes


def match_modes(identified, truth: List[dict], tolerance: float = 0.25) -> List[dict]:
    """Pair identified modes with ground truth by nearest relative frequency."""
    rows = []
    used = set()
    for k, ref in enumerate(truth):
        best, best_err = None, tolerance
        for j, mode in enumerate(identified):
            if j in used:
                continue
            err = abs(mode.freq_hz - ref["freq_hz"]) / ref["freq_hz"]
            if err < best_err:
                best, best_err = j, err
        if best is None:
            rows.append({"name": ref["name"], "true_freq_hz": ref["freq_hz"],
                         "found": False})
            continue
        used.add(best)
        mode = identified[best]
        zeta_err = (abs(mode.damping_ratio - ref["damping_ratio"]) / ref["damping_ratio"]
                    if not math.isnan(mode.damping_ratio) else float("nan"))
        rows.append({
            "name": ref["name"],
            "found": True,
            "true_freq_hz": ref["freq_hz"],
            "est_freq_hz": round(mode.freq_hz, 6),
            "freq_error_pct": round(best_err * 100.0, 4),
            "true_damping": ref["damping_ratio"],
            "est_damping": (None if math.isnan(mode.damping_ratio)
                            else round(mode.damping_ratio, 6)),
            "damping_error_pct": (None if math.isnan(zeta_err)
                                  else round(zeta_err * 100.0, 2)),
            "damping_r2": round(mode.damping_r2, 4),
            "half_power_resolved": mode.half_power_resolved,
            "_index": best,
        })
    return rows


# --------------------------------------------------------------------------- #
# Pipeline runner
# --------------------------------------------------------------------------- #

def build_sim(cfg: dict, *, scale: int, slice_dt_us: int,
              overflow: bool = False) -> OrbitalSunriseSimulator:
    sim = OrbitalSunriseSimulator.from_config(cfg, scale=scale)
    sim.slice_dt_us = int(slice_dt_us)
    sim._build_static_fields()
    if overflow:
        full = cfg.get("sensor", {}).get("readout_limit_evs_full_frame", 1.0e8)
        px_ratio = (sim.width * sim.height) / (
            cfg["sensor"]["width"] * cfg["sensor"]["height"])
        sim.sensor.readout_limit_evs = float(full) * px_ratio
    sim.reset()
    return sim


def run_pipeline(
    cfg: dict,
    *,
    duration_s: float,
    slice_dt_us: int,
    scale: int,
    erc: bool = True,
    mask: bool = True,
    common_mode: str = "median",
    overflow: bool = False,
    erc_target_evs: Optional[float] = None,
    collect_trace: bool = False,
) -> dict:
    """Run simulator -> glare guard -> vibrometer for one configuration."""
    sim = build_sim(cfg, scale=scale, slice_dt_us=slice_dt_us, overflow=overflow)

    g = cfg.get("glare_guard", {})
    target = erc_target_evs
    if target is None:
        px_ratio = (sim.width * sim.height) / (
            cfg["sensor"]["width"] * cfg["sensor"]["height"])
        target = float(g.get("erc_target_evs", 20e6)) * px_ratio

    guard = GlareGuard(
        sim.width, sim.height,
        # The cell grid is statistical binning, not a sensor dimension: it must
        # not shrink with the simulated resolution or each cell starves.
        cell_size_px=int(g.get("cell_size_px", 16)),
        common_mode=common_mode,
        index_threshold=g.get("common_mode_index_threshold", 0.55),
        solar_mask_rate_sigma=g.get("solar_mask_rate_sigma", 6.0),
        solar_mask_persistence=g.get("solar_mask_persistence", 4),
        solar_mask_halo_px=max(1, int(g.get("solar_mask_halo_px", 48)) // scale),
        erc_target_evs=target,
        erc_enabled=erc,
        bias_profiles=g.get("bias_profiles"),
        terminator_enter_rate_evs=float(g.get("terminator_enter_rate_evs", 8e6))
        * (sim.width * sim.height) / (cfg["sensor"]["width"] * cfg["sensor"]["height"]),
        terminator_exit_rate_evs=float(g.get("terminator_exit_rate_evs", 2e6))
        * (sim.width * sim.height) / (cfg["sensor"]["width"] * cfg["sensor"]["height"]),
        mask_enabled=mask,
    )
    vib = VibrometerArray.from_config(cfg, sim.width, sim.height,
                                      slice_dt_s=slice_dt_us * 1e-6, scale=scale)

    trace: Dict[str, List[float]] = {k: [] for k in (
        "t_s", "raw_rate", "kept_rate", "cmi", "balance", "coverage",
        "mask_frac", "cmrr_db", "cmrr_raw_db", "irradiance", "truth_tip_px",
        "profile_idx")}
    profiles = {"eclipse": 0, "terminator": 1, "full_sun": 2}
    truth_edges: List[np.ndarray] = []

    # Separate chain on the un-decimated stream, to separate the algorithm's
    # intrinsic rejection from the counting noise the rate cap introduces, and
    # to test how rejection scales with the averaging area of a cell.
    cell_chains = []
    if collect_trace:
        for size in CELL_SWEEP:
            cell_chains.append((
                size,
                EventSliceBinner(sim.width, sim.height, size),
                CommonModeRejector(common_mode if common_mode != "none" else "median"),
            ))
            trace[f"cmrr_cell{size}_db"] = []
    raw_binner = EventSliceBinner(sim.width, sim.height,
                                  int(g.get("cell_size_px", 16)))
    raw_rejector = CommonModeRejector(common_mode if common_mode != "none" else "median")

    def _cmrr_db(counts, per_px) -> float:
        signed = counts.signed_per_px
        before = float(np.mean(np.abs(signed)))
        after = float(np.mean(np.abs(signed - per_px)))
        return 20.0 * math.log10(before / max(after, 1e-12)) if before > 1e-9 else 0.0

    for sl in sim.iterate(duration_s):
        out = guard.process(sl.ts_us, sl.dt_us, sl.events)
        vib.update(sl.ts_us, out.events, out)

        if collect_trace:
            raw_counts = raw_binner.bin(sl.events)
            raw_est = raw_rejector.estimate(raw_counts)
            trace["t_s"].append(sl.truth["t_s"])
            trace["raw_rate"].append(out.stats["raw_rate_evs"])
            trace["kept_rate"].append(out.stats["kept_events"] / (sl.dt_us * 1e-6))
            trace["cmi"].append(out.stats["common_mode_index"])
            trace["balance"].append(out.stats["polarity_balance"])
            trace["coverage"].append(out.stats["active_coverage"])
            trace["mask_frac"].append(out.stats["masked_cell_fraction"])
            trace["cmrr_db"].append(_cmrr_db(out.counts, out.common_mode.per_px))
            trace["cmrr_raw_db"].append(_cmrr_db(raw_counts, raw_est.per_px))
            trace["irradiance"].append(sl.truth["irradiance_w_m2"])
            trace["profile_idx"].append(profiles.get(out.profile, 0))
            for size, binner, rejector in cell_chains:
                counts = binner.bin(sl.events)
                trace[f"cmrr_cell{size}_db"].append(
                    _cmrr_db(counts, rejector.estimate(counts).per_px))

        # Ground-truth edge displacement at each gauge station, in pixels.
        top, bot = sl.truth["edge_top_px"], sl.truth["edge_bottom_px"]
        truth_edges.append(np.array([
            (top if gg.edge == "top" else bot)[min(int(gg.x_px), sim.width - 1)]
            for gg in vib.gauges
        ]))

    truth_arr = np.stack(truth_edges, axis=1)
    truth_arr = truth_arr - truth_arr.mean(axis=1, keepdims=True)

    return {"sim": sim, "guard": guard, "vib": vib,
            "trace": {k: np.array(v) for k, v in trace.items()},
            "truth_px": truth_arr}


# --------------------------------------------------------------------------- #
# Experiment: storm
# --------------------------------------------------------------------------- #

def experiment_storm(cfg: dict, args) -> dict:
    print("\n=== Experiment: storm characterisation ===")
    run_cfg = cfg.get("run", {})
    duration = args.storm_duration or run_cfg.get("storm_duration_s", 30.0)
    slice_us = run_cfg.get("storm_slice_dt_us", 5000)

    t0 = time.perf_counter()
    governed = run_pipeline(cfg, duration_s=duration, slice_dt_us=slice_us,
                            scale=args.scale, collect_trace=True)
    ungoverned = run_pipeline(cfg, duration_s=duration, slice_dt_us=slice_us,
                              scale=args.scale, erc=False, mask=False,
                              common_mode="none", overflow=True,
                              collect_trace=True)
    print(f"  two {duration:.0f} s runs in {time.perf_counter() - t0:.1f} s")

    tr, un = governed["trace"], ungoverned["trace"]
    sim = governed["sim"]
    px_ratio = (1280 * 720) / (sim.width * sim.height)

    # Define the ramp window from the measured rate profile rather than fixed
    # times: the log-rate spike is short and its position depends on the
    # precursor and disk-rise parameters.
    peak_rate = float(tr["raw_rate"].max())
    ramp = tr["raw_rate"] > 0.1 * peak_rate
    quiet = tr["t_s"] > max(20.0, duration * 0.75)
    ramp_window = [round(float(tr["t_s"][ramp].min()), 2),
                   round(float(tr["t_s"][ramp].max()), 2)] if ramp.any() else [0.0, 0.0]

    analytic = sim.sun.event_storm_budget(
        1280 * 720, contrast_threshold=sim.sensor.contrast_threshold_on,
        t_end_s=duration)

    blind = float(np.mean(un["kept_rate"][ramp] == 0.0))
    result = {
        "duration_s": duration,
        "slice_dt_us": slice_us,
        "sensor_px": [sim.width, sim.height],
        "ramp_window_s": ramp_window,
        "irradiance_swing_decades": round(sim.sun.contrast_decades, 3),
        "irradiance_swing_db": round(sim.sun.dynamic_range_db, 1),
        "analytic_budget_full_frame": {
            "crossings_per_pixel": round(analytic["crossings_per_pixel"], 1),
            "total_mev": round(analytic["total_events"] / 1e6, 1),
            "peak_mevs": round(analytic["peak_rate_evs"] / 1e6, 1),
        },
        "simulated_peak_mevs_scaled": round(float(tr["raw_rate"].max()) * px_ratio / 1e6, 1),
        "quiescent_rate_kevs_scaled": round(float(np.mean(tr["raw_rate"][quiet]))
                                            * px_ratio / 1e3, 1),
        "storm_over_quiescent": round(float(np.mean(tr["raw_rate"][ramp]))
                                      / max(float(np.mean(tr["raw_rate"][quiet])), 1e-9), 1),
        "polarity_balance_ramp": round(float(np.mean(tr["balance"][ramp])), 4),
        "polarity_balance_quiet": round(float(np.mean(tr["balance"][quiet])), 4),
        "common_mode_index_ramp": round(float(np.mean(tr["cmi"][ramp])), 4),
        "common_mode_index_quiet": round(float(np.mean(tr["cmi"][quiet])), 4),
        "cmrr_db_ramp_mean": round(float(np.mean(tr["cmrr_db"][ramp])), 1),
        "cmrr_db_ramp_min": round(float(np.min(tr["cmrr_db"][ramp])), 1),
        "cmrr_db_intrinsic_ramp_mean": round(float(np.mean(tr["cmrr_raw_db"][ramp])), 1),
        "cmrr_db_intrinsic_ramp_min": round(float(np.min(tr["cmrr_raw_db"][ramp])), 1),
        "erc_decimation_factor_peak": round(float(np.max(
            tr["raw_rate"] / np.maximum(tr["kept_rate"], 1e-9))), 1),
        "masked_cell_fraction_final": round(float(tr["mask_frac"][-1]), 4),
        "ungoverned_blind_fraction_during_ramp": round(blind, 4),
        "governed_blind_fraction_during_ramp":
            round(float(np.mean(tr["kept_rate"][ramp] == 0.0)), 4),
        "bias_transitions": governed["guard"].scheduler.transitions,
    }

    # Rejection is limited by how many events per pixel the ramp produces in a
    # slice: at the peak that is well under one, so per-pixel counts are binomial
    # and it is averaging over a cell that buys rejection. The prediction is
    # CMRR ~ sqrt(pixels per cell); doubling the cell EDGE quadruples the pixel
    # count, so that is 20*log10(2) = 6.02 dB per doubling of the edge.
    cell_rows = []
    for size in CELL_SWEEP:
        key = f"cmrr_cell{size}_db"
        if key in tr and tr[key].size:
            cell_rows.append({"cell_size_px": size, "px_per_cell": size * size,
                              "cmrr_db": round(float(np.mean(tr[key][ramp])), 1)})
    if len(cell_rows) >= 2:
        span_db = cell_rows[-1]["cmrr_db"] - cell_rows[0]["cmrr_db"]
        doublings = math.log2(cell_rows[-1]["cell_size_px"] / cell_rows[0]["cell_size_px"])
        measured = span_db / doublings
        result["cell_size_sweep"] = {
            "points": cell_rows,
            "db_per_edge_doubling": round(measured, 2),
            "sqrt_law_prediction_db": 6.02,
            "agrees_with_sqrt_law": bool(abs(measured - 6.02) < 1.5),
        }

    print(f"  irradiance swing        : {result['irradiance_swing_decades']} decades "
          f"({result['irradiance_swing_db']} dB)")
    print(f"  ramp window (>10% peak) : {ramp_window[0]:.2f} - {ramp_window[1]:.2f} s")
    print(f"  peak rate (scaled to HD): {result['simulated_peak_mevs_scaled']} Mev/s "
          f"vs analytic {result['analytic_budget_full_frame']['peak_mevs']} Mev/s")
    print(f"  storm / quiescent ratio : {result['storm_over_quiescent']}x")
    print(f"  polarity balance        : ramp {result['polarity_balance_ramp']:+.3f} "
          f"| quiescent {result['polarity_balance_quiet']:+.3f}")
    print(f"  common-mode index       : ramp {result['common_mode_index_ramp']:.3f} "
          f"| quiescent {result['common_mode_index_quiet']:.3f}")
    print(f"  CMRR during ramp        : intrinsic {result['cmrr_db_intrinsic_ramp_mean']} dB "
          f"-> {result['cmrr_db_ramp_mean']} dB after a "
          f"{result['erc_decimation_factor_peak']}x rate cap")
    print(f"  blind time during ramp  : governed "
          f"{result['governed_blind_fraction_during_ramp'] * 100:.1f}% "
          f"| ungoverned {result['ungoverned_blind_fraction_during_ramp'] * 100:.1f}%")
    if "cell_size_sweep" in result:
        sweep = result["cell_size_sweep"]
        pts = " | ".join(f"{p['cell_size_px']}px {p['cmrr_db']}dB"
                         for p in sweep["points"])
        print(f"  CMRR vs cell size       : {pts}")
        print(f"    {sweep['db_per_edge_doubling']} dB per doubling of the cell edge "
              f"(sqrt-averaging law predicts {sweep['sqrt_law_prediction_db']}) "
              f"-> {'confirmed' if sweep['agrees_with_sqrt_law'] else 'DEVIATES'}")

    plt = _matplotlib()
    if plt is not None:
        fig, ax = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
        ax[0].semilogy(tr["t_s"], np.maximum(tr["raw_rate"] * px_ratio, 1e3),
                       label="generated (scaled to 1280x720)", lw=1.2)
        ax[0].semilogy(tr["t_s"], np.maximum(tr["kept_rate"] * px_ratio, 1e3),
                       label="after rate cap", lw=1.2)
        ax[0].semilogy(un["t_s"], np.maximum(un["kept_rate"] * px_ratio, 1e3),
                       label="ungoverned (readout overflow)", lw=1.0, alpha=0.8)
        ax[0].set_ylabel("event rate [ev/s]")
        ax[0].legend(fontsize=8)
        ax[0].set_title("Orbital sunrise event storm")
        ax[0].grid(alpha=0.3)

        ax2 = ax[1].twinx()
        ax[1].plot(tr["t_s"], tr["cmi"], label="common-mode index", color="C3")
        ax[1].plot(tr["t_s"], tr["balance"], label="polarity balance",
                   color="C0", ls="--")
        ax[1].plot(tr["t_s"], tr["coverage"], label="active coverage",
                   color="C2", ls=":")
        ax2.semilogy(tr["t_s"], tr["irradiance"], color="0.6", lw=1.0)
        ax2.set_ylabel("irradiance [W/m^2]", color="0.4")
        ax[1].set_ylabel("index / balance")
        ax[1].legend(fontsize=8, loc="center right")
        ax[1].grid(alpha=0.3)

        ax[2].plot(tr["t_s"], tr["cmrr_raw_db"], color="C4",
                   label="intrinsic (un-decimated)")
        ax[2].plot(tr["t_s"], tr["cmrr_db"], color="C1", ls="--",
                   label="delivered (after rate cap)")
        ax[2].set_ylabel("CMRR [dB]")
        ax[2].set_xlabel("time [s]")
        ax[2].legend(fontsize=8)
        ax[2].grid(alpha=0.3)
        fig.tight_layout()
        path = os.path.join(RESULTS_DIR, "fig_storm.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  figure -> {os.path.relpath(path, HERE)}")

    return result


# --------------------------------------------------------------------------- #
# Experiment: modal
# --------------------------------------------------------------------------- #

def experiment_modal(cfg: dict, args) -> dict:
    print("\n=== Experiment: modal identification, guard ablation ===")
    run_cfg = cfg.get("run", {})
    duration = args.duration or run_cfg.get("duration_s", 1400.0)
    slice_us = run_cfg.get("slice_dt_us", 20000)
    ident = ModalIdentifier.from_config(cfg)
    snap_s = cfg.get("array", {}).get("thermal_snap", {}).get("t_snap_s", 8.0)

    rows = []
    spectra = {}
    series = {}
    for name, kw in ABLATIONS:
        t0 = time.perf_counter()
        run = run_pipeline(
            cfg, duration_s=duration, slice_dt_us=slice_us, scale=args.scale,
            erc=bool(kw["erc"]), mask=bool(kw["mask"]),
            common_mode=str(kw["common_mode"]), overflow=bool(kw.get("overflow", False)),
        )
        vib, sim = run["vib"], run["sim"]
        t_s, disp = vib.series()
        warm = vib.warmup_s
        healthy = vib.healthy(cfg.get("vibrometer", {}).get("max_dropout_fraction", 0.35))

        modes = ident.identify(t_s, disp, warmup_s=warm, healthy=healthy,
                               decay_start_s=max(snap_s, warm))
        truth = sim.array.ground_truth()
        matched = match_modes(modes, truth)

        true_shapes = ground_truth_shapes(sim, vib)
        for row, shape in zip(matched, true_shapes):
            if row.get("found"):
                est = modes[row.pop("_index")].shape_real
                row["mac"] = round(modal_assurance_criterion(est, shape), 4)
            row.pop("_index", None)

        found = [r for r in matched if r.get("found")]
        freq_errs = [r["freq_error_pct"] for r in found]
        damp_errs = [r["damping_error_pct"] for r in found
                     if r.get("damping_error_pct") is not None]
        macs = [r["mac"] for r in found if "mac" in r]

        # Fidelity of the raw displacement trace against ground truth.
        use = t_s >= warm
        ref = run["truth_px"][:, use]
        meas = disp[:, use]
        corr = []
        for i in range(meas.shape[0]):
            if np.std(meas[i]) > 1e-12 and np.std(ref[i]) > 1e-12:
                corr.append(abs(float(np.corrcoef(meas[i], ref[i])[0, 1])))
        summary = {
            "ablation": name,
            "modes_found": len(found),
            "modes_expected": len(truth),
            "freq_error_pct_max": round(max(freq_errs), 3) if freq_errs else None,
            "freq_error_pct_mean": round(float(np.mean(freq_errs)), 3) if freq_errs else None,
            "damping_error_pct_mean": round(float(np.mean(damp_errs)), 2) if damp_errs else None,
            "mac_min": round(min(macs), 4) if macs else None,
            "mac_mean": round(float(np.mean(macs)), 4) if macs else None,
            "displacement_corr_mean": round(float(np.mean(corr)), 4) if corr else None,
            "dropout_fraction_mean": round(float(np.mean(vib.dropout_fraction())), 4),
            "healthy_gauges": int(np.sum(healthy)),
            "total_gauges": len(vib.gauges),
            "runtime_s": round(time.perf_counter() - t0, 1),
            "modes": matched,
        }
        rows.append(summary)

        fs = 1.0 / (slice_us * 1e-6)
        best = int(np.argmax([np.var(d) for d in disp[:, use]]))
        spectra[name] = welch_psd(disp[best, use], fs,
                                  segment_s=ident.segment_s, overlap=ident.overlap)
        series[name] = (t_s, disp[best], run["truth_px"][best])

        print(f"  {name:<16} modes {len(found)}/{len(truth)}  "
              f"f_err {summary['freq_error_pct_max']}%  "
              f"zeta_err {summary['damping_error_pct_mean']}%  "
              f"MAC {summary['mac_min']}  "
              f"corr {summary['displacement_corr_mean']}  "
              f"dropout {summary['dropout_fraction_mean']:.2f}  "
              f"({summary['runtime_s']} s)")

    plt = _matplotlib()
    if plt is not None:
        truth_f = [m["freq_hz"] for m in rows and
                   cfg.get("array", {}).get("modes", [])]
        fig, ax = plt.subplots(2, 1, figsize=(9, 8))
        for name, (f, p) in spectra.items():
            band = (f >= 0.02) & (f <= 3.0)
            ax[0].semilogy(f[band], np.maximum(p[band], 1e-14), lw=1.1, label=name)
        for fk in truth_f:
            ax[0].axvline(fk, color="0.7", ls=":", lw=1.0)
        ax[0].set_xlabel("frequency [Hz]")
        ax[0].set_ylabel("PSD [px^2/Hz]")
        ax[0].set_title("Displacement spectrum by guard configuration "
                        "(dotted = true modal frequencies)")
        ax[0].legend(fontsize=8)
        ax[0].grid(alpha=0.3)

        for name in ("none", "erc+mask+median"):
            if name in series:
                t_s, d, ref = series[name]
                ax[1].plot(t_s, d, lw=0.9, label=f"{name} (measured)")
        t_s, _, ref = series["erc+mask+median"]
        ax[1].plot(t_s, ref, lw=0.9, color="k", ls="--", label="ground truth")
        ax[1].set_xlim(0, min(400.0, float(t_s[-1])))
        ax[1].set_xlabel("time [s]")
        ax[1].set_ylabel("edge displacement [px]")
        ax[1].set_title("Gauge displacement through the terminator crossing")
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        path = os.path.join(RESULTS_DIR, "fig_modal.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  figure -> {os.path.relpath(path, HERE)}")

    return {"duration_s": duration, "slice_dt_us": slice_us, "ablations": rows}


# --------------------------------------------------------------------------- #
# Experiment: ERC sweep
# --------------------------------------------------------------------------- #

def experiment_erc_sweep(cfg: dict, args) -> dict:
    print("\n=== Experiment: rate-cap sweep (decimation bias) ===")
    run_cfg = cfg.get("run", {})
    duration = args.sweep_duration or 700.0
    slice_us = run_cfg.get("slice_dt_us", 20000)
    ident = ModalIdentifier.from_config(cfg)
    snap_s = cfg.get("array", {}).get("thermal_snap", {}).get("t_snap_s", 8.0)
    truth = cfg.get("array", {}).get("modes", [])
    f_true = truth[0]["freq_hz"] if truth else 0.082

    rows = []
    for target_evs in args.sweep_targets:
        run = run_pipeline(cfg, duration_s=duration, slice_dt_us=slice_us,
                           scale=args.scale, erc=True, mask=True,
                           common_mode="median", erc_target_evs=target_evs)
        vib = run["vib"]
        t_s, disp = vib.series()
        modes = ident.identify(t_s, disp, warmup_s=vib.warmup_s,
                               healthy=vib.healthy(),
                               decay_start_s=max(snap_s, vib.warmup_s))
        near = [m for m in modes if abs(m.freq_hz - f_true) / f_true < 0.25]
        est = near[0].freq_hz if near else float("nan")
        decim = run["guard"].governor.decimation_factor
        rows.append({
            "target_evs": target_evs,
            "decimation_factor": round(float(decim), 2),
            "est_freq_hz": None if math.isnan(est) else round(est, 6),
            "freq_error_pct": None if math.isnan(est)
            else round(abs(est - f_true) / f_true * 100.0, 4),
            "modes_found": len(modes),
        })
        print(f"  cap {target_evs / 1e6:>7.3f} Mev/s -> "
              f"f = {rows[-1]['est_freq_hz']} Hz "
              f"(error {rows[-1]['freq_error_pct']}%, {len(modes)} modes)")

    errs = [r["freq_error_pct"] for r in rows if r["freq_error_pct"] is not None]
    verdict = {
        "true_freq_hz": f_true,
        "max_freq_error_pct": round(max(errs), 4) if errs else None,
        "unbiased": bool(errs and max(errs) < 2.0),
        "points": rows,
    }
    print(f"  max frequency error across all rate caps: {verdict['max_freq_error_pct']}% "
          f"-> {'unbiased' if verdict['unbiased'] else 'BIASED'}")

    plt = _matplotlib()
    if plt is not None and errs:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        x = [r["decimation_factor"] for r in rows if r["freq_error_pct"] is not None]
        ax.semilogx(x, errs, "o-")
        ax.axhline(2.0, color="C3", ls="--", lw=1.0, label="2% acceptance")
        ax.set_xlabel("decimation factor")
        ax.set_ylabel("fundamental frequency error [%]")
        ax.set_title("Rate-cap decimation does not bias the frequency estimate")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        path = os.path.join(RESULTS_DIR, "fig_erc_sweep.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  figure -> {os.path.relpath(path, HERE)}")

    return verdict


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate the IAC 2026 result set for event-based solar-array "
                    "vibration monitoring through orbital sunrise.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--experiment", dest="experiment", default="all",
                   choices=["all", "storm", "modal", "erc-sweep"],
                   help="Which experiment to run.")
    p.add_argument("--config", dest="config", default=DEFAULT_CONFIG,
                   help="Path to solar_array_config.json.")
    p.add_argument("--scale", dest="scale", type=int, default=4,
                   help="Sensor downscale factor. 1 = full 1280x720 (slow).")
    p.add_argument("--duration", dest="duration", type=float, default=0.0,
                   help="Modal record length in seconds. 0 = use the config value.")
    p.add_argument("--storm-duration", dest="storm_duration", type=float, default=0.0,
                   help="Storm record length in seconds. 0 = use the config value.")
    p.add_argument("--sweep-duration", dest="sweep_duration", type=float, default=0.0,
                   help="Record length for the rate-cap sweep, in seconds.")
    p.add_argument("--sweep-targets", dest="sweep_targets", type=float, nargs="+",
                   default=[2.0e6, 5.0e5, 1.5e5, 5.0e4, 1.5e4],
                   help="Rate caps to sweep, in events/s at the simulated resolution.")
    p.add_argument("--out", dest="out", default="",
                   help="Output JSON path. Defaults to results/glare_rejection_<ts>.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("IAC 2026 - event-based vibration monitoring of flexible solar arrays")
    print(f"config : {os.path.relpath(args.config, HERE)}")
    print(f"scale  : 1/{args.scale}  "
          f"({cfg['sensor']['width'] // args.scale}x{cfg['sensor']['height'] // args.scale})")

    results: Dict[str, object] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_path": os.path.basename(args.config),
        "scale": args.scale,
    }

    if args.experiment in ("all", "storm"):
        results["storm"] = experiment_storm(cfg, args)
    if args.experiment in ("all", "modal"):
        results["modal"] = experiment_modal(cfg, args)
    if args.experiment in ("all", "erc-sweep"):
        results["erc_sweep"] = experiment_erc_sweep(cfg, args)

    out = args.out or os.path.join(
        RESULTS_DIR, f"glare_rejection_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nresults -> {os.path.relpath(out, HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
