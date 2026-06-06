from pathlib import Path

import pandas as pd

from atomic_file import load_interaction_dataframe

DATASETS = {"mind": 168, "ebnerd": 336}
MIN_NEGATIVES = 9
COLS = ["neg_item_id_list", "history_item_id_list"]


def test_files(dataset: str, start: int) -> list[Path]:
    base = Path("data/atomic_files") / dataset
    return [base / f"{dataset}.hour_{hour}.inter" for hour in range(start + 36, start + 48)]


def overlap_count(row: pd.Series) -> int:
    history = set(row["history_item_id_list"].split())
    return sum(item in history for item in row["neg_item_id_list"].split())


def dataset_stats(dataset: str, total_units: int) -> pd.Series:
    rows = []
    for start in range(1, total_units - 48 + 2, 12):
        df = load_interaction_dataframe(test_files(dataset, start), COLS).fillna("")
        df["neg_count"] = df["neg_item_id_list"].str.split().str.len()
        df["hist_count"] = df["history_item_id_list"].str.split().str.len()
        df = df[df["neg_count"] >= MIN_NEGATIVES]
        overlap = df.apply(overlap_count, axis=1)
        rows.append(
            {
                "eligible_rows": len(df),
                "history_len": df["hist_count"].mean(),
                "negative_len": df["neg_count"].mean(),
                "overlap_negatives_per_row": overlap.mean(),
                "rows_with_any_overlap": (overlap > 0).mean(),
            }
        )
    df = pd.DataFrame(rows)
    return pd.Series(
        {
            "windows": len(df),
            "eligible_rows": df["eligible_rows"].sum(),
            **df.drop(columns="eligible_rows").mean().to_dict(),
        }
    )


def main() -> None:
    print(pd.DataFrame({ds: dataset_stats(ds, total) for ds, total in DATASETS.items()}).T.round(4).to_string())


if __name__ == "__main__":
    main()
