import argparse
import json
import sys
from datetime import timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_analysis.dataset_analysis import load_temporal_interaction_data
from data_analysis.window_validation import compute_all_window_statistics

DEFAULT_RESULTS_PATH = PROJECT_ROOT / "output/results/experiments.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data_analysis/dataframes/output"
DEFAULT_BASE_PATH = PROJECT_ROOT / "data/atomic_files"
DEFAULT_METRIC = "ndcg@5"
DEFAULT_EPSILON = 1e-6
DEFAULT_EXCLUDED_EXPERIMENTS = (
    "E00_pop_eb",
    "E00_pop_mind",
    "E00_random_eb",
    "E00_random_mind",
)

FACTOR_SPECS = [
    ("D", "D_id", "dataset"),
    ("P", "P_id", "preprocessing"),
    ("E", "E_id", "news_encoder"),
    ("L", "L_id", "text_scope"),
    ("U", "U_id", "user_encoder"),
    ("G", "G_id", "similarity_function"),
    ("W", "W_id", "window_number"),
    ("WW", "WW_id", "weekend_weekday"),
    ("ToD", "ToD_id", "time_of_day"),
    ("R", "R_id", "seed"),
]

OBSERVATION_COLUMNS = [
    "experiment_id",
    "description",
    "model",
    "dataset_raw",
    "dataset",
    "preprocessing",
    "news_encoder",
    "text_scope",
    "user_encoder",
    "similarity_function",
    "window_number",
    "weekend_weekday",
    "time_of_day",
    "seed",
    "cold_start_user_ratio",
    "cold_start_item_ratio",
    "user_activity",
    "CS_user",
    "CS_item",
    "UA",
    "n_impressions",
    "Y",
]
CONTEXT_COLUMNS = [
    "dataset",
    "granularity",
    "window_context_key",
    "cold_start_user_ratio",
    "cold_start_item_ratio",
    "user_activity",
    "weekend_weekday",
    "time_of_day",
    "n_impressions",
]
INTERACTION_COLUMNS = ["user_id", "item_id", "timestamp", "impression_id", "label", "time_unit"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare complete causal Stan data from experiment JSONL results.")
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument("--exclude", nargs="+", action="append", default=[])
    parser.add_argument("--base-path", type=Path, default=DEFAULT_BASE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    return parser.parse_args()


def read_experiment_rows(results_path: Path) -> list[dict[str, Any]]:
    with results_path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_exclusions(values: list[Any] | None) -> set[str]:
    flat_values: list[Any] = []
    for value in values or []:
        flat_values.extend(value if isinstance(value, (list, tuple, set)) else [value])
    return {part.strip() for value in flat_values for part in str(value).split(",") if part.strip()}


def make_observation_dataframe(
    results_path: Path,
    metric: str = DEFAULT_METRIC,
    exclude: list[Any] | None = None,
    base_path: Path = DEFAULT_BASE_PATH,
) -> pd.DataFrame:
    metric = metric.lower()
    excluded = normalize_exclusions(exclude)
    rows = []

    for order, result in enumerate(read_experiment_rows(results_path)):
        if result["experiment_id"] in excluded or metric not in result["test_results"]:
            continue
        rows.append(_observation_row(result, metric, order))

    if not rows:
        raise ValueError(f"No observations found in {results_path} for metric={metric!r}")

    observations = (
        pd.DataFrame(rows)
        .sort_values("_order")
        .drop_duplicates(["experiment_id", "window_number", "seed"], keep="last")
        .drop(columns="_order")
    )

    context = build_posthoc_context(observations, base_path)
    observations = observations.merge(
        context,
        on=["dataset", "granularity", "window_context_key"],
        how="left",
        validate="many_to_one",
    )
    if observations[["cold_start_user_ratio", "cold_start_item_ratio", "user_activity", "n_impressions"]].isna().any().any():
        raise ValueError("Missing post-hoc context for at least one observation.")

    observations["CS_user"] = zscore(observations["cold_start_user_ratio"])
    observations["CS_item"] = zscore(observations["cold_start_item_ratio"])
    observations["UA"] = zscore(observations["user_activity"])

    return observations[OBSERVATION_COLUMNS].sort_values(["experiment_id", "window_number", "seed"]).reset_index(drop=True)


def _observation_row(result: dict[str, Any], metric: str, order: int) -> dict[str, Any]:
    run = result["run_info"]
    config = result.get("full_config") or {}
    window = result["window_info"]

    dataset_raw = run["dataset"]
    dataset, preprocessing = split_dataset(dataset_raw)
    model = str(config.get("model") or run["model"])

    return {
        "_order": order,
        "experiment_id": result["experiment_id"],
        "description": result.get("description"),
        "model": model,
        "dataset_raw": dataset_raw,
        "dataset": dataset,
        "preprocessing": preprocessing,
        "news_encoder": infer_news_encoder(model, config),
        "text_scope": infer_text_scope(model, result, config),
        "user_encoder": "nrms" if is_nrms(model, result) else "mean",
        "similarity_function": infer_similarity_function(model, result, config),
        "window_number": int(window["window_number"]),
        "window_context_key": window_context_key(window),
        "window_protocol_key": window_protocol_key(window),
        "seed": int(run["seed"]),
        "granularity": window["granularity"],
        "window_info": window,
        "Y": float(result["test_results"][metric]),
    }


def split_dataset(name: str) -> tuple[str, str]:
    name = name.strip().lower()
    for suffix, preprocessing in [
        ("_cleaned_tokenized", "cleaned_tokenized"),
        ("_cleaned", "cleaned"),
        ("_tokenized", "tokenized"),
    ]:
        if name.endswith(suffix):
            return name.removesuffix(suffix), preprocessing
    return name, "none"


def infer_news_encoder(model: str, config: dict[str, Any]) -> str:
    model_lower = model.lower()
    sentence_model = config.get("sentence_embedding_model")
    sentence_source = config.get("sentence_embedding_source")

    if sentence_model and (
        sentence_source
        or model_lower in {"nrms", "sbert", "sentencetransformer", "sentence_transformer"}
    ):
        underlying = sentence_model
    elif sentence_source:
        underlying = sentence_model or sentence_source
    elif model_lower == "bert":
        underlying = config.get("bert_model_name")
    else:
        underlying = config.get("embedding_source")
    return f"{model}:{underlying}" if underlying else model


def infer_text_scope(model: str, result: dict[str, Any], config: dict[str, Any]) -> str:
    if "use_abstract" in config:
        return "title_abstract" if config["use_abstract"] else "title"
    return "title" if is_nrms(model, result) else "title_abstract"


def infer_similarity_function(model: str, result: dict[str, Any], config: dict[str, Any]) -> str:
    return str(config.get("similarity") or ("dot" if is_nrms(model, result) else "cosine")).lower()


def is_nrms(model: str, result: dict[str, Any]) -> bool:
    run_model = str(result["run_info"].get("model", ""))
    return model.upper() == "NRMS" or run_model.upper().startswith("NRMS")


def build_posthoc_context(observations: pd.DataFrame, base_path: Path) -> pd.DataFrame:
    frames = []
    for (dataset, granularity), group in observations.groupby(["dataset", "granularity"]):
        interactions = load_temporal_interaction_data(
            dataset_path=str(base_path),
            dataset_name=dataset,
            granularity=granularity,
            time_units=requested_time_units(unique_windows(group["window_info"], window_context_key)),
            positive_only=False,
            columns=INTERACTION_COLUMNS,
            strict_columns=False,
        )

        for _, protocol_group in group.groupby("window_protocol_key"):
            windows = unique_windows(protocol_group["window_info"], lambda window: int(window["window_number"]))
            stats = compute_all_window_statistics(interactions, windows, granularity)
            context = stats.merge(window_extras(interactions, windows), on="window_number", validate="one_to_one")

            test_users = context["test_users"].replace(0, np.nan)
            context["dataset"] = dataset
            context["granularity"] = granularity
            context["user_activity"] = (context["test_interactions"] / test_users).fillna(0.0)
            context["n_impressions"] = context["test_impressions"].clip(lower=1).astype(float)
            frames.append(context[CONTEXT_COLUMNS])

    return pd.concat(frames, ignore_index=True)


def unique_windows(windows: pd.Series, key: Callable[[dict[str, Any]], Any]) -> list[dict[str, Any]]:
    by_key = {key(window): window for window in windows}
    return [by_key[value] for value in sorted(by_key)]


def requested_time_units(windows: list[dict[str, Any]]) -> list[int]:
    units: set[int] = set()
    for window in windows:
        start = int(window.get("start_unit") or parse_range(window["train_range"])[0])
        end = int(window.get("end_unit") or parse_range(window["test_range"])[1])
        units.update(range(start, end + 1))
    return sorted(units)


def window_extras(interactions: pd.DataFrame, windows: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for window in windows:
        test_start, test_end = parse_range(window["test_range"])
        test = positive_only(interactions[interactions["time_unit"].between(test_start, test_end)])
        rows.append(
            {
                "window_number": int(window["window_number"]),
                "window_context_key": window_context_key(window),
                "test_impressions": int(test["impression_id"].nunique()) if "impression_id" in test else len(test),
                "weekend_weekday": weekend_weekday(interactions, window),
                "time_of_day": "tod_a" if int(window["window_number"]) % 2 else "tod_b",
            }
        )
    return pd.DataFrame(rows)


def weekend_weekday(interactions: pd.DataFrame, window: dict[str, Any]) -> str:
    end_unit = int(window.get("end_unit") or parse_range(window["test_range"])[1])
    timestamps = pd.to_numeric(interactions.loc[interactions["time_unit"].eq(end_unit), "timestamp"], errors="coerce")
    if timestamps.dropna().empty:
        return "weekday"

    end_time = pd.to_datetime(timestamps.max(), unit="s", utc=True).tz_convert(timezone.utc)
    hours = pd.date_range(end=end_time, periods=12, freq="h")
    return "weekend" if (hours.dayofweek >= 5).mean() > 0.5 else "weekday"


def positive_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["label"].eq(1)] if "label" in df else df


def parse_range(value: Any) -> tuple[int, int]:
    start, end = str(value).split("-", maxsplit=1)
    return int(start), int(end)


def window_context_key(window: dict[str, Any]) -> str:
    return "|".join(str(window.get(key)) for key in ["window_number", "train_range", "test_range", "start_unit", "end_unit"])


def window_protocol_key(window: dict[str, Any]) -> str:
    return "|".join(
        str(window.get(key))
        for key in ["window_size", "window_stride", "window_ratio", "train_units", "valid_units", "test_units", "has_validation"]
    )


def zscore(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values).astype(float)
    return (values - values.mean()) / (values.std(ddof=0) or 1.0)


def make_stan_dataframe(observations: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    stan_df = observations.copy()
    levels: dict[str, list[str]] = {}

    for factor, id_column, source_column in FACTOR_SPECS:
        levels[factor] = factor_levels(stan_df[source_column])
        codes = pd.Categorical(stan_df[source_column].astype(str), categories=levels[factor], ordered=True).codes
        stan_df[id_column] = codes + 1

    stan_columns = [id_column for _, id_column, _ in FACTOR_SPECS] + ["CS_user", "CS_item", "UA", "Y", "n_impressions"]
    return stan_df[stan_columns], levels


def factor_levels(values: pd.Series) -> list[str]:
    values = values.dropna()
    if pd.api.types.is_numeric_dtype(values):
        return [str(value) for value in sorted(values.astype(int).unique())]
    return sorted(values.astype(str).unique())


def make_stan_data(
    stan_df: pd.DataFrame,
    levels: dict[str, list[str]],
    epsilon: float = DEFAULT_EPSILON,
) -> dict[str, Any]:
    stan_data: dict[str, Any] = {"N": int(len(stan_df))}

    for factor, id_column, _ in FACTOR_SPECS:
        stan_data[f"K_{factor}"] = len(levels[factor])
        stan_data[id_column] = stan_df[id_column].astype(int).tolist()

    for column in ["CS_user", "CS_item", "UA", "Y", "n_impressions"]:
        stan_data[column] = stan_df[column].astype(float).tolist()

    stan_data["epsilon"] = float(epsilon)
    validate_stan_data(stan_data)
    return stan_data


def validate_stan_data(stan_data: dict[str, Any]) -> None:
    n = stan_data["N"]
    if not 0.0 < stan_data["epsilon"] < 0.5:
        raise ValueError("epsilon must be in (0, 0.5)")

    for factor, id_column, _ in FACTOR_SPECS:
        ids = np.asarray(stan_data[id_column], dtype=int)
        if len(ids) != n or ids.min() < 1 or ids.max() > stan_data[f"K_{factor}"]:
            raise ValueError(f"{id_column} is not a valid 1-based Stan index.")

    for column in ["CS_user", "CS_item", "UA", "Y", "n_impressions"]:
        values = np.asarray(stan_data[column], dtype=float)
        if len(values) != n or not np.isfinite(values).all():
            raise ValueError(f"{column} has the wrong length or contains non-finite values.")

    y = np.asarray(stan_data["Y"], dtype=float)
    n_impressions = np.asarray(stan_data["n_impressions"], dtype=float)
    if not ((0 <= y).all() and (y <= 1).all() and (n_impressions >= 1).all()):
        raise ValueError("Y must be in [0, 1] and n_impressions must be >= 1.")


def build_complete_causal_data(
    results_path: Path,
    metric: str = DEFAULT_METRIC,
    exclude: list[Any] | None = None,
    base_path: Path = DEFAULT_BASE_PATH,
    epsilon: float = DEFAULT_EPSILON,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, list[str]]]:
    observations = make_observation_dataframe(results_path, metric=metric, exclude=exclude, base_path=base_path)
    stan_df, levels = make_stan_dataframe(observations)
    return observations, stan_df, make_stan_data(stan_df, levels, epsilon=epsilon), levels


def write_complete_causal_outputs(
    observations: pd.DataFrame,
    stan_df: pd.DataFrame,
    stan_data: dict[str, Any],
    levels: dict[str, list[str]],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "observations": output_dir / "complete_causal_observations.csv",
        "stan_dataframe": output_dir / "complete_causal_stan_dataframe.csv",
        "stan_data": output_dir / "complete_causal_stan_data.json",
        "levels": output_dir / "complete_causal_levels.json",
    }

    observations.to_csv(paths["observations"], index=False)
    stan_df.to_csv(paths["stan_dataframe"], index=False)
    paths["stan_data"].write_text(json.dumps(stan_data, indent=2) + "\n", encoding="utf-8")
    paths["levels"].write_text(json.dumps(format_levels(levels), indent=2) + "\n", encoding="utf-8")
    return paths


def format_levels(levels: dict[str, list[str]]) -> dict[str, Any]:
    return {factor: [{"id": index + 1, "label": label} for index, label in enumerate(labels)] for factor, labels in levels.items()}


def main() -> None:
    args = parse_args()
    exclude = [*DEFAULT_EXCLUDED_EXPERIMENTS, *args.exclude]
    observations, stan_df, stan_data, levels = build_complete_causal_data(
        results_path=args.results_path,
        metric=args.metric,
        exclude=exclude,
        base_path=args.base_path,
        epsilon=args.epsilon,
    )
    paths = write_complete_causal_outputs(observations, stan_df, stan_data, levels, args.output_dir)

    print("Prepared complete causal Stan data")
    print(f"  observations = {stan_data['N']}")
    print(f"  metric       = {args.metric}")
    print(f"  excluded     = {len(normalize_exclusions(exclude))}")
    for key, path in paths.items():
        print(f"  {key:14s}= {path}")


if __name__ == "__main__":
    main()
