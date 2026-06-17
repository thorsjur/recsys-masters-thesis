from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_analysis.plot.common import dataset_label

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_FIT_ROOT = PROJECT_ROOT / "data_analysis" / "stan" / "output" / "complete_causal"
DEFAULT_LEVELS_PATH = PROJECT_ROOT / "data_analysis" / "dataframes" / "output" / "complete_causal_levels.json"
DEFAULT_OBSERVATIONS_PATH = PROJECT_ROOT / "data_analysis" / "dataframes" / "output" / "complete_causal_observations.csv"


def inv_logit(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def latest_complete_causal_fit(fit_root: Path = DEFAULT_FIT_ROOT) -> Path:
    fit_dirs = sorted(
        path for path in fit_root.iterdir() if path.is_dir() and (path / "draws").is_dir() and (path / "posterior_summary.csv").exists()
    )
    return fit_dirs[-1]


def load_residual_window_data(
    fit_dir: Path | None = None,
    levels_path: Path = DEFAULT_LEVELS_PATH,
    observations_path: Path = DEFAULT_OBSERVATIONS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    levels = _load_levels(levels_path)
    observed_windows = _load_observed_windows(observations_path)
    draws = _load_draws(fit_dir or latest_complete_causal_fit(), levels)
    effects = _summarize_observed_effects(draws, levels, observed_windows)
    draw_values = _make_residual_draws(draws, effects)
    return effects, draw_values


def _load_levels(path: Path) -> dict[str, list[dict[str, object]]]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_observed_windows(path: Path) -> set[tuple[str, int]]:
    observations = pd.read_csv(path, usecols=["dataset", "window_number"])
    return {(str(row.dataset), int(row.window_number)) for row in observations.drop_duplicates().itertuples(index=False)}


def _load_draws(fit_dir: Path, levels: dict[str, list[dict[str, object]]]) -> pd.DataFrame:
    columns = ["alpha", *_dw_columns(levels)]
    frames = [pd.read_csv(path, comment="#", usecols=columns) for path in sorted((fit_dir / "draws").glob("*.csv"))]
    if not frames:
        raise FileNotFoundError(f"No draw CSV files found in {fit_dir / 'draws'}")
    return pd.concat(frames, ignore_index=True)


def _dw_columns(levels: dict[str, list[dict[str, object]]]) -> list[str]:
    return [f"a_DW.{int(dataset['id'])}.{int(window['id'])}" for dataset in levels["D"] for window in levels["W"]]


def _summarize_observed_effects(
    draws: pd.DataFrame,
    levels: dict[str, list[dict[str, object]]],
    observed_windows: set[tuple[str, int]],
) -> pd.DataFrame:
    baseline = inv_logit(draws["alpha"].to_numpy(dtype=float))
    rows = []

    for dataset_level in levels["D"]:
        dataset_id = int(dataset_level["id"])
        dataset = str(dataset_level["label"])
        for window_level in levels["W"]:
            window_id = int(window_level["id"])
            window_number = int(window_level["label"])
            if (dataset, window_number) not in observed_windows:
                continue

            values = draws[f"a_DW.{dataset_id}.{window_id}"].to_numpy(dtype=float)
            metric_delta = inv_logit(draws["alpha"].to_numpy(dtype=float) + values) - baseline
            rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": dataset_label(dataset),
                    "dataset_id": dataset_id,
                    "window_id": window_id,
                    "window_number": window_number,
                    "metric_delta_mean": float(np.mean(metric_delta)),
                    "metric_delta_q05": float(np.quantile(metric_delta, 0.05)),
                    "metric_delta_median": float(np.quantile(metric_delta, 0.50)),
                    "metric_delta_q95": float(np.quantile(metric_delta, 0.95)),
                    "p_gt_0": float(np.mean(metric_delta > 0.0)),
                }
            )

    return pd.DataFrame(rows).sort_values(["dataset", "window_number"]).reset_index(drop=True)


def _make_residual_draws(draws: pd.DataFrame, effects: pd.DataFrame) -> pd.DataFrame:
    baseline = inv_logit(draws["alpha"].to_numpy(dtype=float))
    rows = []

    for row in effects.itertuples(index=False):
        values = draws[f"a_DW.{int(row.dataset_id)}.{int(row.window_id)}"].to_numpy(dtype=float)
        rows.append(
            pd.DataFrame(
                {
                    "dataset": row.dataset,
                    "dataset_label": row.dataset_label,
                    "window_number": int(row.window_number),
                    "metric_delta": inv_logit(draws["alpha"].to_numpy(dtype=float) + values) - baseline,
                }
            )
        )

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
