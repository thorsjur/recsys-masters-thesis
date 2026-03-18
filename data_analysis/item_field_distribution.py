from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from data_analysis.atomic_file import find_item_file, load_item_dataframe
from data_analysis.plot.field_length_distribution import save_field_length_distribution


def describe(values: np.ndarray) -> dict[str, float]:
    return {
        "count": float(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "median": float(np.median(values)),
        "q1": float(np.percentile(values, 25)),
        "q3": float(np.percentile(values, 75)),
    }


def print_stats(field: str, stats: dict[str, float]) -> None:
    print(f"Field: {field}")
    print(f"Count: {int(stats['count'])}")
    print(f"Min: {stats['min']:.2f}")
    print(f"Max: {stats['max']:.2f}")
    print(f"Mean: {stats['mean']:.2f}")
    print(f"Std: {stats['std']:.2f}")
    print(f"Median: {stats['median']:.2f}")
    print(f"Q1: {stats['q1']:.2f}")
    print(f"Q3: {stats['q3']:.2f}")


def _parse_quantile(value: float | None) -> float | None:
    if value is None:
        return None
    if not 0 < value <= 100:
        raise ValueError("--max-quantile must be in the range (0, 100].")
    return value


def run(
    dataset: str,
    field: str,
    base_path: str,
    output: str | None,
    drop_empty: bool,
    max_quantile: float | None,
    primary_color: str,
) -> None:
    item_file = find_item_file(dataset, base_path)
    df = load_item_dataframe(item_file)

    if field not in df.columns:
        available = ", ".join(df.columns)
        raise ValueError(f"Field '{field}' not found in {item_file}. Available fields: {available}")

    series = df[field].astype(str)
    if drop_empty:
        series = series[series.str.strip() != ""]

    lengths = series.str.split().str.len().astype(int)
    if lengths.empty:
        raise ValueError(f"No non-empty values found for field '{field}' in {item_file}")

    output_path = Path(output) if output else item_file.parent / f"{field}_length_distribution.pdf"
    quantile_value = save_field_length_distribution(
        lengths,
        field,
        output_path,
        max_quantile=max_quantile,
        primary_color=primary_color,
    )

    stats = describe(lengths.to_numpy())
    print(f"Dataset: {dataset}")
    print(f"Item file: {item_file}")
    print(f"Output: {output_path}")
    if quantile_value is not None:
        print(f"Plot trimmed at P{max_quantile:g}: {quantile_value:.2f}")
    print_stats(field, stats)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot and summarize text-length distribution for a field in <dataset>.item")
    parser.add_argument("--dataset", required=True, help="Dataset folder name under data/atomic_files")
    parser.add_argument("--field", required=True, help="Field to analyze (e.g. title, abstract)")
    parser.add_argument("--base-path", default="data/atomic_files", help="Base directory containing dataset folders")
    parser.add_argument("--output", help="Output PDF path (default: <dataset_dir>/<field>_length_distribution.pdf)")
    parser.add_argument("--drop-empty", action="store_true", help="Ignore empty field values before computing lengths")
    parser.add_argument("--primary-color", default="#2E86AB", help="Primary color for histogram bars")
    parser.add_argument(
        "--max-quantile",
        type=float,
        help="Trim the plot to values up to this percentile, e.g. 99 for the 99th quantile",
    )

    args = parser.parse_args()
    run(
        args.dataset,
        args.field,
        args.base_path,
        args.output,
        args.drop_empty,
        _parse_quantile(args.max_quantile),
        args.primary_color,
    )


if __name__ == "__main__":
    main()
