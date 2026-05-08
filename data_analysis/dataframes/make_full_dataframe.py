import argparse
from pathlib import Path

import pandas as pd

from common import make_metric_dataframe

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "output/results/experiments.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data_analysis/dataframes/output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--metric", default="ndcg@5")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def make_full_dataframe(results_path: Path, metric: str) -> pd.DataFrame:
    return make_metric_dataframe(results_path, metric)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = make_full_dataframe(args.results_path, args.metric)
    df.to_csv(args.output_dir / "full_dataframe.csv", index=False)


if __name__ == "__main__":
    main()
