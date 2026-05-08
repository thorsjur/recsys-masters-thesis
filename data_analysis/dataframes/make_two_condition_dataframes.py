import argparse
import json
from pathlib import Path

import pandas as pd

from common import make_metric_dataframe

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "output/results/experiments.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data_analysis/dataframes/output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--exp-a", required=True)
    parser.add_argument("--exp-b", required=True)
    parser.add_argument("--metric", default="ndcg@5")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def make_observation_dataframe(results_path: Path, exp_a: str, exp_b: str, metric: str) -> pd.DataFrame:
    df = make_metric_dataframe(results_path, metric)
    df = df[df["experiment_id"].isin({exp_a, exp_b})].copy()
    df["condition"] = df["experiment_id"].map({exp_a: "A", exp_b: "B"})
    df = df.sort_values(["window_number", "seed", "condition"]).reset_index(drop=True)

    seed_map = {seed: i + 1 for i, seed in enumerate(sorted(df["seed"].unique()))}
    time_map = {window: i + 1 for i, window in enumerate(sorted(df["window_number"].unique()))}

    df["cond_id"] = df["condition"].map({"A": 1, "B": 2})
    df["seed_id"] = df["seed"].map(seed_map)
    df["time_id"] = df["window_number"].map(time_map)

    return df[
        ["experiment_id", "description", "model", "dataset", "condition", "cond_id", "seed", "seed_id", "window_number", "time_id", "y"]
    ]


def make_stan_dataframe(observations: pd.DataFrame) -> pd.DataFrame:
    return observations[["cond_id", "seed_id", "time_id", "y"]]


def make_stan_data(stan_df: pd.DataFrame) -> dict:
    return {
        "N": len(stan_df),
        "S": int(stan_df["seed_id"].max()),
        "T": int(stan_df["time_id"].max()),
        "cond_id": stan_df["cond_id"].astype(int).tolist(),
        "seed_id": stan_df["seed_id"].astype(int).tolist(),
        "time_id": stan_df["time_id"].astype(int).tolist(),
        "y": stan_df["y"].astype(float).tolist(),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    observations = make_observation_dataframe(args.results_path, args.exp_a, args.exp_b, args.metric)
    stan_df = make_stan_dataframe(observations)
    stan_data = make_stan_data(stan_df)

    observations.to_csv(args.output_dir / "two_condition_observations.csv", index=False)
    stan_df.to_csv(args.output_dir / "two_condition_stan_dataframe.csv", index=False)

    with (args.output_dir / "two_condition_stan_data.json").open("w", encoding="utf-8") as handle:
        json.dump(stan_data, handle, indent=2)


if __name__ == "__main__":
    main()
