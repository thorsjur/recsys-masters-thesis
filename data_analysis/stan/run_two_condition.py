import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from cmdstanpy import CmdStanModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a two-condition Bayesian Stan comparison on experiment results stored as JSONL.")
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("output/results/experiments.jsonl"),
        help="Path to the experiment JSONL file.",
    )
    parser.add_argument(
        "--stan-file",
        type=Path,
        default=Path("data_analysis/stan/two_condition.stan"),
        help="Path to the Stan model file.",
    )
    parser.add_argument(
        "--exp-a",
        required=True,
        help="Experiment ID for condition A. This becomes Stan condition 1.",
    )
    parser.add_argument(
        "--exp-b",
        required=True,
        help="Experiment ID for condition B. This becomes Stan condition 2.",
    )
    parser.add_argument(
        "--metric",
        default="ndcg@5",
        help="Metric name inside test_results to model. Default: ndcg@5",
    )
    parser.add_argument(
        "--chains",
        type=int,
        default=4,
        help="Number of MCMC chains.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1500,
        help="Warmup iterations per chain.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Sampling iterations per chain.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for Stan sampling.",
    )
    parser.add_argument(
        "--perf-threshold",
        type=float,
        default=0.0,
        help="Absolute threshold on mean_diff for a practical performance claim. Default: 0.0",
    )
    parser.add_argument(
        "--general-stability-threshold-pct",
        "--resid-stability-threshold-pct",
        dest="general_stability_threshold_pct",
        type=float,
        default=10.0,
        help=("Percentage threshold for relative general-stability gain claims. " "This is based on sigma_y reduction. Default: 10.0"),
    )
    parser.add_argument(
        "--temporal-stability-threshold-pct",
        "--temp-stability-threshold-pct",
        dest="temporal_stability_threshold_pct",
        type=float,
        default=10.0,
        help=("Percentage threshold for relative temporal-stability gain claims. " "This is based on sigma_u reduction. Default: 10.0"),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc

    return rows


def extract_rows(
    raw_rows: list[dict[str, Any]],
    exp_a: str,
    exp_b: str,
    metric: str,
) -> pd.DataFrame:
    selected_ids = {exp_a, exp_b}
    extracted: list[dict[str, Any]] = []

    for row in raw_rows:
        experiment_id = row.get("experiment_id")

        if experiment_id not in selected_ids:
            continue

        test_results = row.get("test_results", {})
        window_info = row.get("window_info", {})
        run_info = row.get("run_info", {})

        if metric not in test_results:
            continue

        extracted.append(
            {
                "experiment_id": experiment_id,
                "seed": int(run_info["seed"]),
                "window_number": int(window_info["window_number"]),
                "metric_value": float(test_results[metric]),
            }
        )

    df = pd.DataFrame(extracted)

    # Keep only seeds and windows that exist in both experiments
    seed_counts = df.groupby("seed")["experiment_id"].nunique()
    common_seeds = seed_counts[seed_counts == 2].index
    df = df[df["seed"].isin(common_seeds)]

    window_counts = df.groupby("window_number")["experiment_id"].nunique()
    common_windows = window_counts[window_counts == 2].index
    df = df[df["window_number"].isin(common_windows)]

    dupes = df.duplicated(subset=["experiment_id", "seed", "window_number"])

    if dupes.any():
        duplicate_rows = df.loc[dupes, ["experiment_id", "seed", "window_number"]]
        raise ValueError("Found duplicate rows for the same experiment_id/seed/window_number " f"combination:\n{duplicate_rows}")

    return df.sort_values(["window_number", "seed", "experiment_id"]).reset_index(drop=True)


def build_stan_data(df: pd.DataFrame, exp_a: str, exp_b: str) -> dict[str, Any]:
    cond_map = {
        exp_a: 1,
        exp_b: 2,
    }

    unique_seeds = sorted(df["seed"].unique().tolist())
    unique_windows = sorted(df["window_number"].unique().tolist())

    seed_map = {seed: i + 1 for i, seed in enumerate(unique_seeds)}
    time_map = {window: i + 1 for i, window in enumerate(unique_windows)}

    stan_df = df.copy()
    stan_df["cond_id"] = stan_df["experiment_id"].map(cond_map)
    stan_df["seed_id"] = stan_df["seed"].map(seed_map)
    stan_df["time_id"] = stan_df["window_number"].map(time_map)

    stan_data = {
        "N": len(stan_df),
        "S": len(unique_seeds),
        "T": len(unique_windows),
        "cond_id": stan_df["cond_id"].astype(int).tolist(),
        "seed_id": stan_df["seed_id"].astype(int).tolist(),
        "time_id": stan_df["time_id"].astype(int).tolist(),
        "y": stan_df["metric_value"].astype(float).tolist(),
    }

    return stan_data


def posterior_interval(draws: pd.DataFrame, name: str) -> tuple[float, float, float, float]:
    q = draws[name].quantile([0.025, 0.5, 0.975])

    return (
        float(draws[name].mean()),
        float(q.loc[0.025]),
        float(q.loc[0.5]),
        float(q.loc[0.975]),
    )


def print_quantity(
    draws: pd.DataFrame,
    name: str,
    base: float | None = None,
) -> None:
    mean_val, lo, med, hi = posterior_interval(draws, name)

    if base is None:
        print(f"{name}: " f"mean={mean_val:.6f}, " f"95% CrI=[{lo:.6f}, {hi:.6f}], " f"median={med:.6f}")
        return

    pct_mean = mean_val / base * 100
    pct_lo = lo / base * 100
    pct_med = med / base * 100
    pct_hi = hi / base * 100

    print(
        f"{name}: "
        f"mean={mean_val:.6f} ({pct_mean:+.2f}%), "
        f"95% CrI=[{lo:.6f}, {hi:.6f}] "
        f"([{pct_lo:+.2f}%, {pct_hi:+.2f}%]), "
        f"median={med:.6f} ({pct_med:+.2f}%)"
    )


def main() -> None:
    args = parse_args()

    raw_rows = read_jsonl(args.results_path)

    df = extract_rows(
        raw_rows=raw_rows,
        exp_a=args.exp_a,
        exp_b=args.exp_b,
        metric=args.metric,
    )

    stan_data = build_stan_data(
        df=df,
        exp_a=args.exp_a,
        exp_b=args.exp_b,
    )

    print("\nData summary:")
    print(f"  observations = {stan_data['N']}")
    print(f"  seeds        = {stan_data['S']}")
    print(f"  windows      = {stan_data['T']}")
    print(f"  condition 1  = {args.exp_a}")
    print(f"  condition 2  = {args.exp_b}")
    print(f"  metric       = {args.metric}")

    model = CmdStanModel(stan_file=str(args.stan_file))

    fit = model.sample(
        data=stan_data,
        chains=args.chains,
        parallel_chains=args.chains,
        iter_warmup=args.warmup,
        iter_sampling=args.samples,
        seed=args.seed,
        adapt_delta=0.99,
        max_treedepth=12,
        show_progress=True,
    )

    draws = fit.draws_pd()

    p_cond2_better = (draws["mean_diff"] > 0).mean()

    p_cond2_more_generally_stable = (draws["general_stability_diff"] < 0).mean()
    p_cond2_more_temporally_stable = (draws["temporal_stability_diff"] < 0).mean()

    p_cond1_better = (draws["mean_diff"] < 0).mean()
    p_cond1_more_generally_stable = (draws["general_stability_diff"] > 0).mean()
    p_cond1_more_temporally_stable = (draws["temporal_stability_diff"] > 0).mean()

    print("\nPosterior probabilities:")
    print(f"P(condition 2 better on average)                  = {p_cond2_better:.3f}")
    print(f"P(condition 2 more generally stable)              " f"= {p_cond2_more_generally_stable:.3f}")
    print(f"P(condition 2 more temporally stable)             " f"= {p_cond2_more_temporally_stable:.3f}")

    print(f"P(condition 1 better on average)                  = {p_cond1_better:.3f}")
    print(f"P(condition 1 more generally stable)              " f"= {p_cond1_more_generally_stable:.3f}")
    print(f"P(condition 1 more temporally stable)             " f"= {p_cond1_more_temporally_stable:.3f}")

    baseline_perf = float(draws["alpha[1]"].mean())
    baseline_general_instability = float(draws["sigma_y[1]"].mean())
    baseline_temporal_instability = float(draws["sigma_u[1]"].mean())

    print(f"\nBaseline performance, condition 1: {baseline_perf:.6f}")
    print(f"Baseline residual sigma_y, condition 1: {baseline_general_instability:.6f}")
    print(f"Baseline temporal sigma_u, condition 1: {baseline_temporal_instability:.6f}")

    print("\nMain posterior quantities:")
    print_quantity(draws, "mean_diff", base=baseline_perf)

    print_quantity(draws, "general_stability_diff", base=baseline_general_instability)
    print_quantity(draws, "temporal_stability_diff", base=baseline_temporal_instability)

    print_quantity(draws, "general_stability_rel_gain", base=1.0)
    print_quantity(draws, "temporal_stability_rel_gain", base=1.0)

    summary = fit.summary()

    wanted = [
        "alpha[1]",
        "alpha[2]",
        "sigma_seed",
        "rho[1]",
        "rho[2]",
        "sigma_u[1]",
        "sigma_u[2]",
        "sigma_y[1]",
        "sigma_y[2]",
        "mean_diff",
        "general_stability_diff",
        "temporal_stability_diff",
        "general_stability_rel_gain",
        "temporal_stability_rel_gain",
    ]

    existing_wanted = [name for name in wanted if name in summary.index]

    print("\nPosterior summary:")
    print(summary.loc[existing_wanted, ["Mean", "5%", "50%", "95%", "R_hat"]])

    print("\nConditions:")
    print(f"  condition 1 = {args.exp_a}")
    print(f"  condition 2 = {args.exp_b}")
    print(f"  metric      = {args.metric}")


if __name__ == "__main__":
    main()
