import json
from pathlib import Path

import pandas as pd


def read_experiments(results_path: Path) -> list[dict]:
    rows = []
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def make_metric_dataframe(results_path: Path, metric: str) -> pd.DataFrame:
    rows = []

    for row in read_experiments(results_path):
        rows.append(
            {
                "experiment_id": row["experiment_id"],
                "description": row["description"],
                "model": row["run_info"]["model"],
                "dataset": row["run_info"]["dataset"],
                "seed": row["run_info"]["seed"],
                "window_number": row["window_info"]["window_number"],
                "y": row["test_results"][metric],
            }
        )

    return pd.DataFrame(rows).sort_values(["experiment_id", "window_number", "seed"]).reset_index(drop=True)
