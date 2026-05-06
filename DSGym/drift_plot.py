"""
Plot drift trajectories and divergence stats from drift_analysis.py output.

Reads CSVs under ./drift_analysis/ and writes PNGs under ./drift_analysis/figures/.

Run AFTER drift_analysis.py:
    python drift_analysis.py
    python drift_plots.py

Figures produced:
    fig_drift_trajectory_turn.png       per-turn anchoring (line graph)
    fig_drift_trajectory_tokens.png     anchoring vs cumulative tokens
    fig_drift_trajectory_seconds.png    anchoring vs elapsed seconds
    fig_o_p_disagreement.png            O→P disagreement distribution
    fig_drift_onset_hist.png            distribution of drift onset turns
    fig_drift_onset_tokens_hist.png     drift onset by tokens
    fig_recovery_by_reason.png          recovery rate by trigger reason
    fig_category_breakdown.png          drift category × nudge_mode
    fig_verbal_vs_execution.png         verbal-execution disagreement bars
"""
from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

ANALYSIS_DIR = Path("./drift_analysis")
FIG_DIR = ANALYSIS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (8, 5),
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})


# ---------------------------------------------------------------------------
def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _to_float(x: Any) -> Optional[float]:
    if x is None or x == "" or x == "None":
        return None
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def _to_bool(x: Any) -> Optional[bool]:
    if x is None or x == "" or x == "None":
        return None
    return str(x).strip().lower() in {"true", "1", "yes"}


def _save(fig, name: str) -> None:
    out = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] {out}")


# ===========================================================================
# 1. Drift trajectory line graph (3 versions: turn / tokens / seconds)
# ===========================================================================
def _trajectory_axes(traj: List[Dict[str, Any]],
                     x_key: str) -> Dict[Tuple[str, str], List[Tuple[float, float]]]:
    """Return {(nudge_mode, sample_id|trial): [(x, anchoring), ...]}."""
    series: Dict[Tuple[str, str], List[Tuple[float, float]]] = defaultdict(list)
    for r in traj:
        x = _to_float(r.get(x_key))
        y = _to_float(r.get("task_anchoring"))
        if x is None or y is None:
            continue
        key = (
            r.get("nudge_mode") or "?",
            f"{r.get('sample_id')}::trial{r.get('trial_id')}",
        )
        series[key].append((x, y))
    for k in series:
        series[k].sort(key=lambda t: t[0])
    return series


def _plot_trajectory(x_key: str, x_label: str, fname: str,
                     title: str) -> None:
    traj = _read_csv(ANALYSIS_DIR / "drift_trajectory.csv")
    if not traj:
        print(f"[plots] no trajectory data for {fname}")
        return
    series = _trajectory_axes(traj, x_key)
    if not series:
        print(f"[plots] no usable points for {fname}")
        return

    fig, ax = plt.subplots()

    # Group by nudge_mode → color
    modes = sorted({k[0] for k in series})
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(modes), 1)))
    color_for = {m: colors[i] for i, m in enumerate(modes)}

    # 1. Faded individual sample lines
    for (mode, _sid), pts in series.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=color_for[mode], alpha=0.12, linewidth=0.7)

    # 2. Mean trajectory per mode (binned along x)
    for mode in modes:
        mode_series = [pts for (m, _), pts in series.items() if m == mode]
        if not mode_series:
            continue
        all_x = [p[0] for s in mode_series for p in s]
        if not all_x:
            continue
        x_max = max(all_x)
        n_bins = 12
        bins = np.linspace(0, x_max, n_bins + 1)
        bin_vals: List[List[float]] = [[] for _ in range(n_bins)]
        for s in mode_series:
            for x, y in s:
                if x_max == 0:
                    bin_idx = 0
                else:
                    bin_idx = min(int((x / x_max) * n_bins), n_bins - 1)
                bin_vals[bin_idx].append(y)
        bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(n_bins)]
        bin_means = [statistics.mean(v) if v else None for v in bin_vals]
        bin_stds = [statistics.stdev(v) if len(v) > 1 else 0.0 for v in bin_vals]
        xs = [c for c, m in zip(bin_centers, bin_means) if m is not None]
        ys = [m for m in bin_means if m is not None]
        sds = [s for s, m in zip(bin_stds, bin_means) if m is not None]
        if xs:
            ax.plot(xs, ys, color=color_for[mode], linewidth=2.5,
                    label=f"{mode} (mean, n={len(mode_series)})")
            lo = [y - s for y, s in zip(ys, sds)]
            hi = [y + s for y, s in zip(ys, sds)]
            ax.fill_between(xs, lo, hi, color=color_for[mode], alpha=0.15)

    # 3. Mean drift-onset annotation (per-sample csv)
    onset_rows = _read_csv(ANALYSIS_DIR / "per_sample.csv")
    onset_key = {
        "turn_index": "drift_onset_turn",
        "cumulative_tokens": "drift_onset_tokens",
        "elapsed_sec": "drift_onset_sec",
    }.get(x_key)
    if onset_key:
        for mode in modes:
            vals = [
                _to_float(r.get(onset_key)) for r in onset_rows
                if (r.get("nudge_mode") or "?") == mode
                and _to_float(r.get(onset_key)) is not None
            ]
            if vals:
                m = statistics.mean(vals)
                ax.axvline(m, color=color_for[mode], linestyle="--",
                           linewidth=1.2, alpha=0.7)
                ax.text(m, 0.02, f"  onset μ={m:.1f}",
                        color=color_for[mode], fontsize=8,
                        rotation=90, va="bottom")

    ax.set_xlabel(x_label)
    ax.set_ylabel("Task-anchoring score")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    _save(fig, fname)


# ===========================================================================
# 2. O→P disagreement distribution
# ===========================================================================
def plot_o_p_disagreement() -> None:
    rows = _read_csv(ANALYSIS_DIR / "per_sample.csv")
    if not rows:
        return
    by_mode: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        v = _to_float(r.get("o_p_disagreement"))
        if v is not None:
            by_mode[r.get("nudge_mode") or "?"].append(v)
    if not by_mode:
        print("[plots] no o_p_disagreement values")
        return

    fig, ax = plt.subplots()
    modes = sorted(by_mode.keys())
    data = [by_mode[m] for m in modes]
    bp = ax.boxplot(data, labels=modes, showmeans=True, patch_artist=True)
    for patch, c in zip(bp["boxes"],
                        plt.cm.tab10(np.linspace(0, 1, max(len(modes), 1)))):
        patch.set_facecolor(c)
        patch.set_alpha(0.4)
    ax.set_ylabel("O → P disagreement (1 − plan-code alignment)")
    ax.set_xlabel("Nudge mode")
    ax.set_title("Orchestrator–programmer disagreement by mode")
    ax.set_ylim(-0.02, 1.02)

    # Mean labels
    for i, m in enumerate(modes, start=1):
        mean = statistics.mean(by_mode[m])
        ax.text(i, mean + 0.03, f"μ={mean:.2f}",
                ha="center", fontsize=8, color="black")
    _save(fig, "fig_o_p_disagreement.png")


# ===========================================================================
# 3. Drift onset histograms
# ===========================================================================
def plot_drift_onset(field: str, label: str, fname: str) -> None:
    rows = _read_csv(ANALYSIS_DIR / "per_sample.csv")
    if not rows:
        return
    by_mode: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        v = _to_float(r.get(field))
        if v is not None:
            by_mode[r.get("nudge_mode") or "?"].append(v)
    if not any(by_mode.values()):
        print(f"[plots] no values for {fname}")
        return

    fig, ax = plt.subplots()
    modes = sorted(by_mode.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(modes), 1)))
    all_vals = [v for vs in by_mode.values() for v in vs]
    if not all_vals:
        return
    bins = np.histogram_bin_edges(all_vals, bins=12)
    for m, c in zip(modes, colors):
        if by_mode[m]:
            ax.hist(by_mode[m], bins=bins, alpha=0.55, label=f"{m} (n={len(by_mode[m])})",
                    color=c, edgecolor="white")
    ax.set_xlabel(label)
    ax.set_ylabel("Count of samples")
    ax.set_title(f"Drift onset distribution — {label}")
    ax.legend(fontsize=8)
    _save(fig, fname)


# ===========================================================================
# 4. Recovery rate by reason
# ===========================================================================
def plot_recovery_by_reason() -> None:
    rows = _read_csv(ANALYSIS_DIR / "recovery_rate.csv")
    rows = [r for r in rows if r.get("nudge_reason") != "ALL"]
    if not rows:
        return
    fig, ax = plt.subplots()
    by_reason: Dict[str, Dict[str, Tuple[int, float]]] = defaultdict(dict)
    for r in rows:
        rate = _to_float(r["recovery_rate"]) or 0.0
        n = int(_to_float(r["n_nudged"]) or 0)
        by_reason[r["nudge_reason"]][r["nudge_mode"]] = (n, rate)

    reasons = list(by_reason.keys())
    modes = sorted({m for d in by_reason.values() for m in d})
    width = 0.8 / max(len(modes), 1)
    x = np.arange(len(reasons))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(modes), 1)))
    for i, m in enumerate(modes):
        ys = [by_reason[r].get(m, (0, 0))[1] for r in reasons]
        ns = [by_reason[r].get(m, (0, 0))[0] for r in reasons]
        bars = ax.bar(x + i * width - 0.4 + width / 2, ys,
                      width, label=m, color=colors[i], alpha=0.8)
        for bar, n in zip(bars, ns):
            if n > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        f"n={n}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(reasons, rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Recovery rate")
    ax.set_title("Nudge recovery by trigger reason")
    ax.legend(fontsize=8)
    _save(fig, "fig_recovery_by_reason.png")


# ===========================================================================
# 5. Category breakdown
# ===========================================================================
def plot_category_breakdown() -> None:
    rows = _read_csv(ANALYSIS_DIR / "per_sample.csv")
    if not rows:
        return
    cat_order = ["clean", "early_exit_recovered", "early_exit_failed",
                 "verbal_execution_divergence", "crash", "other_failure", "unknown"]
    by_mode: Dict[str, Dict[str, int]] = defaultdict(lambda: {c: 0 for c in cat_order})
    for r in rows:
        m = r.get("nudge_mode") or "?"
        c = r.get("category") or "unknown"
        if c not in by_mode[m]:
            by_mode[m][c] = 0
        by_mode[m][c] += 1

    modes = sorted(by_mode.keys())
    fig, ax = plt.subplots()
    bottom = np.zeros(len(modes))
    colors = plt.cm.Set2(np.linspace(0, 1, len(cat_order)))
    for i, c in enumerate(cat_order):
        vals = np.array([by_mode[m].get(c, 0) for m in modes], dtype=float)
        if vals.sum() == 0:
            continue
        ax.bar(modes, vals, bottom=bottom, label=c,
               color=colors[i], edgecolor="white")
        bottom = bottom + vals
    ax.set_ylabel("Number of samples")
    ax.set_title("Drift category breakdown by nudge mode")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    _save(fig, "fig_category_breakdown.png")


# ===========================================================================
# 6. Verbal-vs-execution disagreement
# ===========================================================================
def plot_verbal_vs_execution() -> None:
    rows = _read_csv(ANALYSIS_DIR / "divergence.csv")
    if not rows:
        return
    fig, ax = plt.subplots()
    modes = [r["nudge_mode"] for r in rows]
    rates = [_to_float(r["disagreement_rate"]) or 0.0 for r in rows]
    n_with = [int(_to_float(r["n_with_both_scalars"]) or 0) for r in rows]
    bars = ax.bar(modes, rates,
                  color=plt.cm.tab10(np.linspace(0, 1, max(len(modes), 1))),
                  alpha=0.8, edgecolor="white")
    for b, n, r in zip(bars, n_with, rates):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                f"{r:.2f}\n(n={n})", ha="center", fontsize=8)
    ax.set_ylim(0, max(rates + [0.3]) * 1.4)
    ax.set_ylabel("Disagreement rate")
    ax.set_title("Verbal-vs-execution disagreement (samples with both scalars)")
    _save(fig, "fig_verbal_vs_execution.png")


# ===========================================================================
def main() -> None:
    print(f"[plots] writing to {FIG_DIR}/")
    _plot_trajectory("turn_index", "Turn index",
                     "fig_drift_trajectory_turn.png",
                     "Drift trajectory — task anchoring vs turn")
    _plot_trajectory("cumulative_tokens", "Cumulative tokens",
                     "fig_drift_trajectory_tokens.png",
                     "Drift trajectory — task anchoring vs tokens")
    _plot_trajectory("elapsed_sec", "Elapsed seconds",
                     "fig_drift_trajectory_seconds.png",
                     "Drift trajectory — task anchoring vs time")
    plot_o_p_disagreement()
    plot_drift_onset("drift_onset_turn", "Drift onset turn",
                     "fig_drift_onset_hist.png")
    plot_drift_onset("drift_onset_tokens", "Drift onset tokens",
                     "fig_drift_onset_tokens_hist.png")
    plot_recovery_by_reason()
    plot_category_breakdown()
    plot_verbal_vs_execution()
    print(f"\nFigures in {FIG_DIR}/")


if __name__ == "__main__":
    main()