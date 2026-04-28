import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from data_analysis.atomic_file import find_item_file, load_item_dataframe
from data_analysis.plot.common import SEMANTIC_COLORS, get_output_dir
from data_analysis.plot.field_length_distribution import plot_field_length_distribution


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

    output_path = Path(output) if output else get_output_dir() / f"{dataset}_{field}_length_distribution.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    quantile_value = plot_field_length_distribution(
        lengths,
        ax,
        field,
        dataset_name=dataset,
        max_quantile=max_quantile,
        primary_color=primary_color,
    )
    fig.tight_layout()
    fig.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Dataset: {dataset}")
    print(f"Item file: {item_file}")
    print(f"Output: {output_path}")
    if quantile_value is not None:
        print(f"Plot trimmed at P{max_quantile:g}: {quantile_value:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot and summarize text-length distribution for a field in <dataset>.item"
    )
    parser.add_argument("--dataset", required=True, help="Dataset folder name under data/atomic_files")
    parser.add_argument("--field", required=True, help="Field to analyze (e.g. title, abstract)")
    parser.add_argument("--base-path", default="data/atomic_files", help="Base directory containing dataset folders")
    parser.add_argument(
        "--output", help="Output PDF path (default: data_analysis/output/<dataset>_<field>_length_distribution.pdf)"
    )
    parser.add_argument("--drop-empty", action="store_true", help="Ignore empty field values before computing lengths")
    parser.add_argument(
        "--primary-color",
        default=SEMANTIC_COLORS["item_property"],
        help="Primary color for histogram bars",
    )
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
