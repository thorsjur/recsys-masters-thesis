"""Shared utilities for plotting modules."""
import argparse
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from util.constants import SEPARATOR
from util.experiment_data import load_experiment_results

DEFAULT_OUTPUT_DIR = Path("data_analysis/output")
AXIS_LABEL_SIZE = 12
PLOT_TITLE_SIZE = 13
LEGEND_FONT_SIZE = 10
ANNOTATION_FONT_SIZE = 10

COLORS = [
    "#2E86AB",
    "#F18F01",
    "#A23B72",
    "#06A77D",
    "#D4AF37",
    "#9B59B6",
    "#E74C3C",
    "#0E6453",
]

SEMANTIC_COLORS = {
    "user_interaction": "#2E86AB",
    "item_interaction": "#D4AF37",
    "item_property": "#06A77D",
}


def style_axis(ax, xlabel: str, ylabel: str, title: str, *, title_pad: float = 12) -> None:
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)
    ax.set_title(title, fontsize=PLOT_TITLE_SIZE, fontweight="bold", pad=title_pad)


def get_output_dir(output_dir: str | None = None) -> Path:
    """Get output directory, creating if needed."""
    path = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig: Figure, path: Path, close: bool = True) -> None:
    """Save figure to PDF and optionally close it."""
    fig.savefig(path, format="pdf", dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
    if close:
        plt.close(fig)


def print_header(title: str) -> None:
    """Print a section header."""
    print(f"\n{SEPARATOR}")
    print(title)
    print(SEPARATOR)


def load_experiment(
    experiment_id: str,
    jsonl_path: str = "output/results/experiments.jsonl",
    require_temporal: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load and validate experiment results.

    Returns:
        Tuple of (results list, first_result dict).

    Raises:
        ValueError: If no results found or missing required temporal info.
    """
    results = load_experiment_results(jsonl_path, experiment_id)
    if not results:
        raise ValueError(f"No results found for experiment_id: {experiment_id}")

    first = results[0]
    if require_temporal and not first.get("window_info"):
        raise ValueError(f"Experiment {experiment_id} has no window_info - not a temporal experiment")

    return results, first


def collect_windows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect and deduplicate windows from results, sorted by window number."""
    by_number: dict[int, dict[str, Any]] = {}
    for result in results:
        w_info = result.get("window_info", {})
        if w_info:
            win_num = w_info.get("window_number")
            if win_num is not None and win_num not in by_number:
                by_number[win_num] = w_info
    return [by_number[k] for k in sorted(by_number.keys())]


def get_time_range(windows: list[dict[str, Any]]) -> tuple[int, int]:
    """Get min/max time units from windows."""
    min_unit = min(w.get("start_unit", 0) for w in windows)
    max_unit = max(w.get("end_unit", 0) for w in windows)
    return min_unit, max_unit


def run_cli(main_func, parser: argparse.ArgumentParser) -> None:
    """Run CLI with standard error handling."""
    args = parser.parse_args()
    try:
        main_func(args)
    except Exception as exc:
        print(f"Unexpected error: {exc}")
        import traceback

        traceback.print_exc()
        exit(1)
