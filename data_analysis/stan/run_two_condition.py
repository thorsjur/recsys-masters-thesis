import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAN_DIR = PROJECT_ROOT / "data_analysis/stan"
DATAFRAMES_DIR = PROJECT_ROOT / "data_analysis/dataframes"

for import_path in (PROJECT_ROOT, DATAFRAMES_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

RESULTS_PATH = PROJECT_ROOT / "output/results/experiments.jsonl"
DEFAULT_STAN_FILE = "two_condition.stan"
METRIC = "ndcg@5"
CHAINS = 4
WARMUP = 1500
SAMPLES = 1000
SEED = 123


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
        "--model",
        default=DEFAULT_STAN_FILE
    )
    return parser.parse_args()


def resolve_stan_file(model: str) -> Path:
    model_path = Path(model)

    if model_path.is_absolute() or "/" in model or "\\" in model:
        return model_path

    return STAN_DIR / model_path


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

    from data_analysis.dataframes.make_two_condition_dataframes import (
        make_observation_dataframe,
        make_stan_data,
        make_stan_dataframe,
    )

    observations = make_observation_dataframe(RESULTS_PATH, args.exp_a, args.exp_b, METRIC)
    stan_df = make_stan_dataframe(observations)
    stan_data = make_stan_data(stan_df)

    print("\nData summary:")
    print(f"  observations = {stan_data['N']}")
    print(f"  seeds        = {stan_data['S']}")
    print(f"  windows      = {stan_data['T']}")
    print(f"  condition 1  = {args.exp_a}")
    print(f"  condition 2  = {args.exp_b}")
    print(f"  metric       = {METRIC}")
    stan_file = resolve_stan_file(args.model)
    print(f"  model        = {stan_file}")

    from cmdstanpy import CmdStanModel

    model = CmdStanModel(stan_file=str(stan_file))

    fit = model.sample(
        data=stan_data,
        chains=CHAINS,
        parallel_chains=CHAINS,
        iter_warmup=WARMUP,
        iter_sampling=SAMPLES,
        seed=SEED,
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
    print(f"  metric      = {METRIC}")


if __name__ == "__main__":
    main()
