"""Delta bar chart visualization for comparing two experiments.

Computes per-window Δ (experiment A − experiment B) across temporal windows
and renders a bar chart where each bar extends from 0 to Δ, colored green
for positive deltas and red for negative.

Error bars (optional) show one of:
  - ``ci95`` / ``ci90`` / ``ci99``: t-based CI of the mean difference
  - ``std``:  ±1 propagated standard deviation
  - ``2std``: ±2 propagated standard deviations
  - ``sem``:  ±1 standard error of the mean difference

Optional trend overlay:
  - ``gp``: Gaussian-process Bayesian trend with time-correlated credible band

Left y-axis shows absolute Δ values; right y-axis shows percentage change
relative to experiment B.

Reports:
  - mean(Δ)
  - median(Δ)
  - % windows where Δ > 0
  - 95% CI of mean(Δ) via block bootstrap
  - fitted GP kernel (when trend_type='gp')
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.figure import Figure
from scipy import stats as sp_stats

from data_analysis.plot.common import get_output_dir, print_header, run_cli
from util.experiment_data import load_experiment_results, extract_temporal_metrics

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern

    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False

COLOR_POS = "#2ca02c"
COLOR_NEG = "#d62728"

ERRBAR_TYPES = ("ci90", "ci95", "ci99", "std", "2std", "sem")
TREND_TYPES = ("none", "gp")
_CI_LEVELS = {"ci90": 0.90, "ci95": 0.95, "ci99": 0.99}


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_deltas(
    data_a: Dict[str, Any],
    data_b: Dict[str, Any],
    metric: str,
) -> Dict[str, Any]:
    """Compute per-window Δ = mean_A − mean_B for a single metric.

    The propagated standard deviation accounts for covariance between
    experiments when paired run values are available (same number of runs
    per window):

        Var(Δ) = σ_A² + σ_B² − 2·Cov(A, B)

    Falls back to the independence assumption (Cov = 0) when runs cannot
    be paired.

    Returns a dict with keys:
        deltas:     (W,) Δ values per window.
        std_delta:  (W,) propagated std of the difference per window.
        n_runs:     (W,) number of paired runs per window.
        x_vals:     (W,) x-axis positions (end_unit or window number).
        means_b:    (W,) mean of experiment B (used for % axis).
        win_nums:   (W,) sorted window numbers.
    """
    windows_a = data_a["windows"]
    windows_b = data_b["windows"]
    common = sorted(set(windows_a.keys()) & set(windows_b.keys()))
    if not common:
        raise ValueError("No overlapping windows between the two experiments")

    use_temporal = data_a["metadata"]["granularity"] != "unknown"

    deltas, std_deltas, x_vals, means_b, run_counts = [], [], [], [], []
    for w in common:
        wa, wb = windows_a[w], windows_b[w]
        mean_a = wa["mean"][metric]
        mean_b = wb["mean"][metric]
        var_a = wa["std"][metric] ** 2
        var_b = wb["std"][metric] ** 2

        # Compute covariance when runs are paired
        vals_a = wa["values"].get(metric, [])
        vals_b = wb["values"].get(metric, [])
        n = min(len(vals_a), len(vals_b))
        if n > 1 and len(vals_a) == len(vals_b):
            cov_ab = float(np.cov(vals_a, vals_b, ddof=0)[0, 1])
        else:
            cov_ab = 0.0

        var_delta = var_a + var_b - 2 * cov_ab
        std_delta = np.sqrt(max(var_delta, 0.0))

        deltas.append(mean_a - mean_b)
        std_deltas.append(std_delta)
        means_b.append(mean_b)
        run_counts.append(max(n, 1))
        x_vals.append(wa["info"].get("end_unit", w) if use_temporal else w)

    return {
        "deltas": np.array(deltas),
        "std_delta": np.array(std_deltas),
        "n_runs": np.array(run_counts, dtype=int),
        "x_vals": np.array(x_vals),
        "means_b": np.array(means_b),
        "win_nums": np.array(common),
    }


def compute_error_bars(
    std_delta: np.ndarray,
    n_runs: np.ndarray,
    error_bar_type: str = "ci95",
) -> Tuple[np.ndarray, str]:
    """Convert per-window std(Δ) into error-bar half-widths.

    Args:
        std_delta:      (W,) propagated std per window.
        n_runs:         (W,) run count per window.
        error_bar_type: One of ``ERRBAR_TYPES``.

    Returns:
        yerr:  (W,) half-widths.
        label: Human-readable description for the legend.
    """
    if error_bar_type not in ERRBAR_TYPES:
        raise ValueError(
            f"Unknown error_bar_type '{error_bar_type}', "
            f"choose from {ERRBAR_TYPES}"
        )

    n_min = int(n_runs.min())

    if error_bar_type in _CI_LEVELS:
        ci_level = _CI_LEVELS[error_bar_type]
        ci_pct = int(ci_level * 100)
        yerr = np.empty_like(std_delta)
        for i, (sd, n) in enumerate(zip(std_delta, n_runs)):
            if n > 1:
                sem = sd / np.sqrt(n)
                t_crit = sp_stats.t.ppf(1 - (1 - ci_level) / 2, n - 1)
                yerr[i] = t_crit * sem
            else:
                yerr[i] = sd
        label = f"{ci_pct}% CI (t, n={n_min})" if n_min > 1 else "±1 std (n=1)"

    elif error_bar_type == "sem":
        yerr = std_delta / np.sqrt(n_runs.astype(float))
        label = f"±1 SEM (n={n_min})"

    elif error_bar_type == "std":
        yerr = std_delta.copy()
        label = "±1 std"

    elif error_bar_type == "2std":
        yerr = 2.0 * std_delta
        label = "±2 std"

    else:  # pragma: no cover
        raise ValueError(f"Unhandled error_bar_type: {error_bar_type}")

    return yerr, label


# ---------------------------------------------------------------------------
# Block bootstrap 95 % CI for the mean of Δ
# ---------------------------------------------------------------------------

def block_bootstrap_ci(
    deltas: np.ndarray,
    block_size: int = 3,
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Moving block bootstrap for mean(Δ).

    Falls back to ordinary bootstrap behavior when len(deltas) < block_size.

    Returns:
        (lower, upper) bounds of the CI.
    """
    rng = np.random.RandomState(seed)
    n = len(deltas)
    bs = min(block_size, n)
    n_blocks = max(1, int(np.ceil(n / bs)))

    means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        starts = rng.randint(0, n - bs + 1, size=n_blocks)
        sample = np.concatenate([deltas[s : s + bs] for s in starts])[:n]
        means[i] = sample.mean()

    alpha = (1 - ci) / 2
    return (
        float(np.percentile(means, 100 * alpha)),
        float(np.percentile(means, 100 * (1 - alpha))),
    )


# ---------------------------------------------------------------------------
# Bayesian GP trend (time-correlated credible band)
# ---------------------------------------------------------------------------

def compute_gp_trend(
    x_vals: np.ndarray,
    deltas: np.ndarray,
    std_delta: np.ndarray,
    n_runs: np.ndarray,
    ci: float = 0.95,
    grid_size: int = 300,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Fit a GP trend over time and return mean + credible band.

    Uses per-window SEM^2 as observation noise:
        alpha_i = (std_delta_i / sqrt(n_runs_i))^2

    This gives a smooth latent mean trend with time correlation.
    """
    if not _HAS_SKLEARN:
        raise ImportError(
            "scikit-learn is required for trend_type='gp'. "
            "Install it with: pip install scikit-learn"
        )

    x = np.asarray(x_vals, dtype=float).reshape(-1, 1)
    y = np.asarray(deltas, dtype=float)

    if len(x) < 2:
        raise ValueError("Need at least two windows to fit a GP trend")

    x_mean = float(np.mean(x))
    x_std = float(np.std(x))
    if x_std <= 1e-12:
        x_std = 1.0
    x_scaled = (x - x_mean) / x_std

    sem = std_delta / np.sqrt(np.maximum(n_runs.astype(float), 1.0))
    alpha = np.maximum(sem ** 2, 1e-10)

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=1.0,
        length_scale_bounds=(1e-2, 1e2),
        nu=1.5,
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=seed,
    )
    gp.fit(x_scaled, y)

    x_grid = np.linspace(float(np.min(x_vals)), float(np.max(x_vals)), grid_size)
    x_grid_scaled = (x_grid.reshape(-1, 1) - x_mean) / x_std

    mean_pred, std_pred = gp.predict(x_grid_scaled, return_std=True)

    z = sp_stats.norm.ppf(1 - (1 - ci) / 2)
    lower = mean_pred - z * std_pred
    upper = mean_pred + z * std_pred

    return {
        "x_grid": x_grid,
        "mean": mean_pred,
        "lower": lower,
        "upper": upper,
        "std": std_pred,
        "kernel": str(gp.kernel_),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_delta_bars(
    experiment_id_a: str,
    experiment_id_b: str,
    jsonl_path: str = "output/results/experiments.jsonl",
    metrics: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 5),
    font_size: float = 11.0,
    title_size: float = 13.0,
    legend_size: float = 9.0,
    dpi: int = 300,
    output_format: str = "pdf",
    dark_mode: bool = False,
    custom_title: Optional[str] = None,
    no_title: bool = False,
    block_size: int = 3,
    n_bootstrap: int = 10_000,
    seed: int = 42,
    bar_alpha: float = 0.80,
    bar_width: float = 0.8,
    show_grid: bool = True,
    grid_alpha: float = 0.3,
    grid_style: str = "--",
    x_label: Optional[str] = None,
    capsize: float = 3.0,
    show_error_bars: bool = True,
    error_bar_type: str = "ci95",
    trend_type: str = "none",
    trend_ci: float = 0.95,
    trend_grid_size: int = 300,
):
    """Create a per-window Δ bar chart for each requested metric.

    Each metric gets its own figure. Bars extend from 0 to Δ (green if
    positive, red if negative) with optional error bars whose quantity is
    controlled by ``error_bar_type`` (see ``ERRBAR_TYPES``).
    A secondary y-axis displays percentage change relative to experiment B.

    Optionally overlays a Gaussian-process trend line with a time-correlated
    credible band.

    Returns a list of (Figure, summary_dict) tuples, one per metric.
    """
    if trend_type not in TREND_TYPES:
        raise ValueError(
            f"Unknown trend_type '{trend_type}', choose from {TREND_TYPES}"
        )

    data_a = extract_temporal_metrics(
        load_experiment_results(jsonl_path, experiment_id_a), metrics,
    )
    data_b = extract_temporal_metrics(
        load_experiment_results(jsonl_path, experiment_id_b), metrics,
    )

    plot_metrics = data_a["metrics"]
    if not plot_metrics:
        raise ValueError("No metrics found in experiment data")

    if dark_mode:
        plt.style.use("dark_background")

    meta_a = data_a["metadata"]
    meta_b = data_b["metadata"]

    results: List[Tuple[Figure, Dict[str, Any]]] = []

    for metric in plot_metrics:
        delta_data = compute_deltas(data_a, data_b, metric)
        deltas = delta_data["deltas"]
        std_delta = delta_data["std_delta"]
        n_runs_arr = delta_data["n_runs"]
        x_vals = delta_data["x_vals"]
        means_b = delta_data["means_b"]
        n_runs_min = int(n_runs_arr.min())

        if show_error_bars:
            yerr, errbar_label = compute_error_bars(
                std_delta, n_runs_arr, error_bar_type,
            )
        else:
            yerr, errbar_label = None, None

        d_mean = float(np.mean(deltas))
        d_median = float(np.median(deltas))
        pct_positive = float(np.mean(deltas > 0) * 100)
        ci_lo, ci_hi = block_bootstrap_ci(
            deltas, block_size=block_size, n_bootstrap=n_bootstrap, seed=seed,
        )

        summary: Dict[str, Any] = {
            "metric": metric,
            "mean_delta": d_mean,
            "median_delta": d_median,
            "pct_positive": pct_positive,
            "ci_95_lower": ci_lo,
            "ci_95_upper": ci_hi,
            "n_windows": len(deltas),
            "n_runs": n_runs_min,
        }

        trend_data = None
        if trend_type == "gp":
            trend_data = compute_gp_trend(
                x_vals=x_vals,
                deltas=deltas,
                std_delta=std_delta,
                n_runs=n_runs_arr,
                ci=trend_ci,
                grid_size=trend_grid_size,
                seed=seed,
            )
            summary["gp_kernel"] = trend_data["kernel"]

        fig, ax = plt.subplots(figsize=figsize)

        colors = [COLOR_POS if d >= 0 else COLOR_NEG for d in deltas]

        positions = np.arange(len(deltas))
        bar_kw: Dict[str, Any] = dict(
            width=bar_width,
            color=colors,
            alpha=bar_alpha,
            edgecolor="white",
            linewidth=0.5,
            zorder=2,
        )
        if yerr is not None:
            bar_kw.update(
                yerr=yerr,
                capsize=capsize,
                error_kw={"elinewidth": 1.2, "capthick": 1.0, "color": "0.3"},
            )
        ax.bar(positions, deltas, **bar_kw)

        err_handle = None
        if errbar_label is not None:
            err_handle = mlines.Line2D(
                [], [], color="0.3", linewidth=1.2,
                marker="_", markersize=8, label=errbar_label,
            )

        ax.axhline(0, color="black", linewidth=0.8, zorder=1)
        ax.axhline(
            d_mean,
            color="#E74C3C",
            linewidth=1.3,
            linestyle="--",
            label=f"mean(Δ) = {d_mean:.4f}",
            zorder=3,
        )
        ax.axhspan(
            ci_lo,
            ci_hi,
            alpha=0.10,
            color="#E74C3C",
            label=f"95% CI mean(Δ) [{ci_lo:.4f}, {ci_hi:.4f}]",
            zorder=1,
        )

        if trend_data is not None:
            x_numeric = np.asarray(x_vals, dtype=float)
            pos_numeric = positions.astype(float)

            if len(x_numeric) > 1:
                x_trend_pos = np.interp(
                    trend_data["x_grid"],
                    x_numeric,
                    pos_numeric,
                )
            else:
                x_trend_pos = np.full_like(trend_data["x_grid"], pos_numeric[0])

            ax.plot(
                x_trend_pos,
                trend_data["mean"],
                color="#1f77b4",
                linewidth=2.0,
                label=f"GP trend ({int(trend_ci * 100)}% band)",
                zorder=4,
            )
            ax.fill_between(
                x_trend_pos,
                trend_data["lower"],
                trend_data["upper"],
                color="#1f77b4",
                alpha=0.15,
                zorder=3,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(
            [str(int(v)) if float(v).is_integer() else f"{v}" for v in x_vals],
            fontsize=font_size - 1,
            rotation=45 if len(x_vals) > 20 else 0,
            ha="right" if len(x_vals) > 20 else "center",
        )

        granularity = meta_a["granularity"]
        gran_label = granularity.capitalize() if granularity != "unknown" else "Window"
        ax.set_xlabel(x_label or f"Time ({gran_label}s)", fontsize=font_size)
        ax.set_ylabel(f"Δ {metric.upper()} (absolute)", fontsize=font_size)

        if show_grid:
            ax.grid(axis="y", alpha=grid_alpha, linestyle=grid_style)
            ax.set_axisbelow(True)

        mean_b_global = float(np.mean(means_b))
        if abs(mean_b_global) > 1e-12:
            ax_pct = ax.secondary_yaxis(
                "right",
                functions=(
                    lambda v, s=mean_b_global: np.asarray(v, dtype=float) / s * 100,
                    lambda v, s=mean_b_global: np.asarray(v, dtype=float) * s / 100,
                ),
            )
            ax_pct.set_ylabel(f"Δ {metric.upper()} (% change)", fontsize=font_size)
            ax_pct.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.1f%%"))

        handles, labels = ax.get_legend_handles_labels()
        if err_handle is not None:
            handles.append(err_handle)
            labels.append(errbar_label)
        ax.legend(handles, labels, fontsize=legend_size, loc="best")

        if not no_title:
            title = custom_title or (
                f"Δ {metric.upper()}: {meta_a['model']} − {meta_b['model']} "
                f"on {meta_a['dataset']}"
            )
            ax.set_title(title, fontsize=title_size, fontweight="bold")

        fig.tight_layout()

        out_dir = get_output_dir()
        if output_path:
            save_path = Path(output_path)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = f"_{metric}" if len(plot_metrics) > 1 else ""
            trend_suffix = f"_{trend_type}" if trend_type != "none" else ""
            save_path = (
                out_dir
                / f"delta_{experiment_id_a}_vs_{experiment_id_b}"
                  f"{suffix}{trend_suffix}_{timestamp}.{output_format}"
            )

        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format=output_format, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {save_path}")

        results.append((fig, summary))

    _print_report(
        [r[1] for r in results], experiment_id_a, experiment_id_b, meta_a, meta_b,
    )

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_report(
    summaries: List[Dict[str, Any]],
    eid_a: str,
    eid_b: str,
    meta_a: Dict[str, Any],
    meta_b: Dict[str, Any],
) -> None:
    print_header("Δ Bar-Chart Report")
    print(f"  Experiment A : {eid_a}  ({meta_a['model']})")
    print(f"  Experiment B : {eid_b}  ({meta_b['model']})")
    print(f"  Δ = A − B")
    print(f"  Windows      : {summaries[0]['n_windows']}")
    for s in summaries:
        m = s["metric"]
        print(f"\n  {m.upper()}:")
        print(f"    mean(Δ)         = {s['mean_delta']:.6f}")
        print(f"    median(Δ)       = {s['median_delta']:.6f}")
        print(f"    %% windows Δ>0  = {s['pct_positive']:.1f}%")
        print(f"    95%% CI (block) = [{s['ci_95_lower']:.6f}, {s['ci_95_upper']:.6f}]")
        if "gp_kernel" in s:
            print(f"    GP kernel       = {s['gp_kernel']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _run(args: argparse.Namespace) -> None:
    plot_delta_bars(
        experiment_id_a=args.experiment_a,
        experiment_id_b=args.experiment_b,
        jsonl_path=args.jsonl_path,
        metrics=args.metrics,
        output_path=args.output,
        figsize=tuple(args.figsize),
        font_size=args.font_size,
        title_size=args.title_size,
        legend_size=args.legend_size,
        dpi=args.dpi,
        output_format=args.format,
        dark_mode=args.dark_mode,
        custom_title=args.title,
        no_title=args.no_title,
        block_size=args.block_size,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        bar_alpha=args.bar_alpha,
        bar_width=args.bar_width,
        show_grid=not args.no_grid,
        grid_alpha=args.grid_alpha,
        grid_style=args.grid_style,
        x_label=args.x_label,
        capsize=args.capsize,
        show_error_bars=not args.no_error_bars,
        error_bar_type=args.error_bar_type,
        trend_type=args.trend_type,
        trend_ci=args.trend_ci,
        trend_grid_size=args.trend_grid_size,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Δ bar chart comparing two temporal experiments (A − B)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s --experiment-a exp_nrms --experiment-b exp_pop
  %(prog)s --experiment-a exp1 --experiment-b exp2 --metrics ndcg@10 mrr@5
  %(prog)s --experiment-a exp1 --experiment-b exp2 --block-size 5 --n-bootstrap 50000
  %(prog)s --experiment-a exp1 --experiment-b exp2 --trend-type gp
""",
    )

    # --- Data ---
    data_group = parser.add_argument_group("data")
    data_group.add_argument("--experiment-a", required=True, help="Experiment ID for model A")
    data_group.add_argument("--experiment-b", required=True, help="Experiment ID for model B")
    data_group.add_argument("--jsonl-path", default="output/results/experiments.jsonl", help="Path to JSONL results")
    data_group.add_argument("--metrics", nargs="+", help="Metrics to compare (default: all available)")

    # --- Bootstrap ---
    boot_group = parser.add_argument_group("bootstrap")
    boot_group.add_argument("--block-size", type=int, default=3, help="Block size for block bootstrap (default: 3)")
    boot_group.add_argument("--n-bootstrap", type=int, default=10_000, help="Number of bootstrap resamples (default: 10000)")
    boot_group.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    # --- Trend ---
    trend_group = parser.add_argument_group("trend")
    trend_group.add_argument(
        "--trend-type",
        default="none",
        choices=list(TREND_TYPES),
        help="Optional trend overlay: none or gp (default: none)",
    )
    trend_group.add_argument(
        "--trend-ci",
        type=float,
        default=0.95,
        help="Credible interval level for GP band (default: 0.95)",
    )
    trend_group.add_argument(
        "--trend-grid-size",
        type=int,
        default=300,
        help="Number of points in GP trend grid (default: 300)",
    )

    # --- Style ---
    style_group = parser.add_argument_group("style")
    style_group.add_argument("--bar-alpha", type=float, default=0.80, help="Bar opacity (default: 0.80)")
    style_group.add_argument("--bar-width", type=float, default=0.8, help="Bar width (default: 0.8)")
    style_group.add_argument("--capsize", type=float, default=3.0, help="Error-bar cap size (default: 3.0)")
    style_group.add_argument(
        "--no-error-bars", action="store_true",
        help="Disable per-window error bars",
    )
    style_group.add_argument(
        "--error-bar-type", default="ci95",
        choices=list(ERRBAR_TYPES),
        help="Quantity shown by error bars (default: ci95). "
             "ci90/ci95/ci99 = t-based CI; std/2std = ±1/2 std; sem = ±1 SEM",
    )

    # --- Text ---
    text_group = parser.add_argument_group("text")
    text_group.add_argument("--font-size", type=float, default=11.0, help="Axis label font size (default: 11)")
    text_group.add_argument("--title-size", type=float, default=13.0, help="Figure title font size (default: 13)")
    text_group.add_argument("--legend-size", type=float, default=9.0, help="Legend font size (default: 9)")
    text_group.add_argument("--title", type=str, default=None, help="Custom figure title")
    text_group.add_argument("--no-title", action="store_true", help="Hide figure title")
    text_group.add_argument("--x-label", type=str, default=None, help="Custom x-axis label")

    # --- Grid ---
    grid_group = parser.add_argument_group("grid")
    grid_group.add_argument("--no-grid", action="store_true", help="Disable grid lines")
    grid_group.add_argument("--grid-alpha", type=float, default=0.3, help="Grid alpha (default: 0.3)")
    grid_group.add_argument("--grid-style", default="--", choices=["-", "--", "-.", ":"], help="Grid style (default: --)")

    # --- Theme ---
    theme_group = parser.add_argument_group("theme")
    theme_group.add_argument("--dark-mode", action="store_true", help="Dark background theme")

    # --- Output ---
    out_group = parser.add_argument_group("output")
    out_group.add_argument("--output", "-o", help="Output file path")
    out_group.add_argument("--figsize", nargs=2, type=float, default=[14, 5], metavar=("W", "H"), help="Figure size (default: 14 5)")
    out_group.add_argument("--dpi", type=int, default=300, help="DPI (default: 300)")
    out_group.add_argument("--format", default="pdf", choices=["pdf", "png", "svg", "eps"], help="Output format (default: pdf)")

    run_cli(_run, parser)


if __name__ == "__main__":
    main()
