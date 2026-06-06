import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd

from data_analysis.dataframes.common import make_metric_dataframe, read_experiments
from data_analysis.dataframes.make_complete_causal_dataframe import (
    infer_similarity_function,
    infer_text_scope,
    split_dataset,
)
from data_analysis.plot.common import COLORS, dataset_label, get_output_dir, save_figure, style_axis

RESULTS_PATH = PROJECT_ROOT / "output/results/experiments.jsonl"
METRIC = "ndcg@5"
BASE_MODELS = {
    "random": ("Random", "X"),
    "pop": ("Pop", "P"),
    "tfidf": ("TF-IDF", "o"),
    "glove": ("GloVe", "s"),
    "fasttext": ("FastText", "^"),
    "nrms": ("NRMS", "D"),
    "sbert": ("SBERT", "v"),
    "bert": ("BERT", "*"),
}
TEXT_COLORS = {"title": COLORS[0], "title_abstract": COLORS[1], "none": "#888888"}
SIM_HATCHES = {"cosine": "", "dot": "///", "mlp": "...", "none": "xx"}


def base_model(model: str) -> str:
    model = model.lower()
    return next((label for key, (label, _) in BASE_MODELS.items() if key in model), model.upper())


def metadata(results_path: Path) -> pd.DataFrame:
    rows = []
    seen = set()
    for row in read_experiments(results_path):
        if row["experiment_id"] in seen:
            continue
        seen.add(row["experiment_id"])
        config, model = row.get("full_config") or {}, row["run_info"]["model"]
        model = str(config.get("model") or model)
        base = base_model(model)
        dataset, _ = split_dataset(row["run_info"]["dataset"])
        rows.append(
            {
                "experiment_id": row["experiment_id"],
                "dataset": dataset,
                "base_model": base,
                "text_scope": "none" if base in {"Random", "Pop"} else infer_text_scope(model, row, config),
                "similarity": "none" if base in {"Random", "Pop"} else infer_similarity_function(model, row, config),
            }
        )
    return pd.DataFrame(rows)


def performance_stability(results_path: Path) -> pd.DataFrame:
    df = make_metric_dataframe(results_path, METRIC).drop_duplicates(
        ["experiment_id", "window_number", "seed"],
        keep="last",
    )
    windows = df.groupby(["experiment_id", "window_number"], as_index=False)["y"].mean()
    stats = windows.groupby("experiment_id")["y"].agg(
        mean_ndcg_at_5="mean",
        sd_across_windows=lambda x: x.std(ddof=1) if len(x) > 1 else 0.0,
    )
    df = stats.reset_index().merge(metadata(results_path), on="experiment_id", validate="one_to_one")
    return df[~df["base_model"].isin(["Pop", "Random"])]


def plot_dataset(df: pd.DataFrame, dataset: str, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for _, row in df[df["dataset"].eq(dataset)].sort_values("experiment_id").iterrows():
        ax.scatter(
            row["mean_ndcg_at_5"],
            row["sd_across_windows"],
            s=145,
            marker=dict(BASE_MODELS.values()).get(row["base_model"], "o"),
            facecolor=TEXT_COLORS.get(row["text_scope"], "#BBBBBB"),
            edgecolor="black",
            linewidth=0.7,
            hatch=SIM_HATCHES.get(row["similarity"], ""),
        )

    style_axis(ax, f"Mean {METRIC.upper()}", "SD across windows", f"{dataset_label(dataset)} Performance-Stability")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.margins(0.08)

    color_legend = [
        Patch(facecolor=color, edgecolor="black", label=label.replace("_", " + ").title())
        for label, color in TEXT_COLORS.items()
        if label in set(df.loc[df["dataset"].eq(dataset), "text_scope"])
    ]
    hatch_legend = [
        Patch(facecolor="white", edgecolor="black", hatch=hatch, label=label.title())
        for label, hatch in SIM_HATCHES.items()
        if label in set(df.loc[df["dataset"].eq(dataset), "similarity"])
    ]
    marker_legend = [
        Line2D([0], [0], marker=marker, color="black", linestyle="", markersize=8, label=label)
        for label, marker in dict(BASE_MODELS.values()).items()
        if label in set(df.loc[df["dataset"].eq(dataset), "base_model"])
    ]
    ax.legend(
        handles=color_legend + hatch_legend + marker_legend,
        fontsize=9,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )
    save_figure(fig, output_dir / f"{dataset}_performance_stability.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot mean nDCG@5 against temporal stability.")
    parser.add_argument("--results-path", type=Path, default=RESULTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    df = performance_stability(args.results_path)
    for dataset in ["mind", "ebnerd"]:
        plot_dataset(df, dataset, get_output_dir(args.output_dir))


if __name__ == "__main__":
    main()
