import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAN_DIR = PROJECT_ROOT / "data_analysis/stan"
FIT_OUTPUT_DIR = STAN_DIR / "output/complete_causal"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_analysis.dataframes.make_complete_causal_dataframe import (
    DEFAULT_BASE_PATH,
    DEFAULT_EPSILON,
    DEFAULT_EXCLUDED_EXPERIMENTS,
    DEFAULT_METRIC,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RESULTS_PATH,
    build_complete_causal_data,
    normalize_exclusions,
    write_complete_causal_outputs,
)

DEFAULT_STAN_FILE = "complete_causal.stan"
CHAINS = 4
WARMUP = 1500
SAMPLES = 1000
SEED = 123
ADAPT_DELTA = 0.95
MAX_TREEDEPTH = 12
REFRESH = 100
SAMPLER_METRIC = "dense_e"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the complete causal Stan model.")
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument("--exclude", nargs="+", action="append", default=[])
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exclude = [*DEFAULT_EXCLUDED_EXPERIMENTS, *args.exclude]
    excluded = normalize_exclusions(exclude)

    observations, stan_df, stan_data, levels = build_complete_causal_data(
        results_path=DEFAULT_RESULTS_PATH,
        metric=args.metric,
        exclude=exclude,
        base_path=DEFAULT_BASE_PATH,
        epsilon=DEFAULT_EPSILON,
    )
    output_paths = write_complete_causal_outputs(observations, stan_df, stan_data, levels, DEFAULT_OUTPUT_DIR)
    stan_file = STAN_DIR / DEFAULT_STAN_FILE

    print("\nData summary:")
    print(f"  observations = {stan_data['N']}")
    print(f"  metric       = {args.metric}")
    print(f"  excluded     = {len(excluded)}")
    print(f"  datasets     = {', '.join(levels['D'])}")
    print(f"  encoders     = {len(levels['E'])}")
    print(f"  windows      = {len(levels['W'])}")
    print(f"  seeds        = {len(levels['R'])}")
    print(f"  model        = {stan_file}")
    print("\nPrepared files:")
    for key, path in output_paths.items():
        print(f"  {key:14s}= {path}")

    if args.prepare_only:
        print("\nPrepare-only mode: Stan sampling skipped.")
        return

    from cmdstanpy import CmdStanModel

    model = CmdStanModel(stan_file=str(stan_file))

    print("\nSampling config:")
    print(f"  chains        = {CHAINS}")
    print(f"  parallel      = {CHAINS}")
    print(f"  warmup        = {WARMUP}")
    print(f"  samples       = {SAMPLES}")
    print(f"  adapt_delta   = {ADAPT_DELTA}")
    print(f"  max_treedepth = {MAX_TREEDEPTH}")
    print(f"  metric        = {SAMPLER_METRIC}")

    fit = model.sample(
        data=stan_data,
        chains=CHAINS,
        parallel_chains=CHAINS,
        iter_warmup=WARMUP,
        iter_sampling=SAMPLES,
        seed=SEED,
        adapt_delta=ADAPT_DELTA,
        max_treedepth=MAX_TREEDEPTH,
        metric=SAMPLER_METRIC,
        refresh=REFRESH,
        show_progress=True,
    )

    summary = fit.summary()
    wanted = [
        "alpha",
        "beta_CS_user",
        "beta_CS_item",
        "beta_UA",
        "sigma_obs",
        "sigma_D",
        "sigma_P",
        "sigma_E",
        "sigma_L",
        "sigma_U",
        "sigma_G",
        "sigma_W",
        "sigma_R",
    ]
    existing_wanted = [name for name in wanted if name in summary.index]

    print("\nPosterior summary:")
    columns = [column for column in ["Mean", "5%", "50%", "95%", "R_hat"] if column in summary.columns]
    print(summary.loc[existing_wanted, columns])

    _print_diagnostics(summary)
    _print_sampler_diagnostics(fit, max_treedepth=MAX_TREEDEPTH)

    saved_paths = _save_fit_outputs(
        fit=fit,
        summary=summary,
        metric=args.metric,
        excluded=excluded,
        stan_file=stan_file,
        prepared_paths=output_paths,
    )
    print("\nSaved fit:")
    for key, path in saved_paths.items():
        print(f"  {key:14s}= {path}")


def _save_fit_outputs(
    fit,
    summary,
    metric: str,
    excluded: set[str],
    stan_file: Path,
    prepared_paths: dict[str, Path],
) -> dict[str, Path]:
    run_dir = FIT_OUTPUT_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    draws_dir = run_dir / "draws"
    draws_dir.mkdir(parents=True, exist_ok=False)

    fit.save_csvfiles(dir=str(draws_dir))

    summary_path = run_dir / "posterior_summary.csv"
    summary.to_csv(summary_path)

    metadata_path = run_dir / "metadata.json"
    metadata = {
        "metric": metric,
        "excluded": sorted(excluded),
        "stan_file": str(stan_file),
        "chains": CHAINS,
        "parallel_chains": CHAINS,
        "warmup": WARMUP,
        "samples": SAMPLES,
        "seed": SEED,
        "adapt_delta": ADAPT_DELTA,
        "max_treedepth": MAX_TREEDEPTH,
        "sampler_metric": SAMPLER_METRIC,
        "prepared_files": {key: str(path) for key, path in prepared_paths.items()},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    return {
        "run_dir": run_dir,
        "draws": draws_dir,
        "summary": summary_path,
        "metadata": metadata_path,
    }


def _print_diagnostics(summary) -> None:
    print("\nDiagnostics:")
    if "R_hat" in summary.columns:
        r_hat = summary["R_hat"].dropna()
        if not r_hat.empty:
            print(f"  max R_hat = {float(r_hat.max()):.4f}")

    ess_columns = [column for column in ["ESS_bulk", "Ess_bulk", "ESS_tail", "Ess_tail"] if column in summary.columns]
    for column in ess_columns:
        values = summary[column].dropna()
        if not values.empty:
            print(f"  min {column} = {float(values.min()):.1f}")


def _print_sampler_diagnostics(fit, max_treedepth: int) -> None:
    sampler_vars = fit.method_variables()
    print("\nSampler diagnostics:")

    divergent = sampler_vars.get("divergent__")
    if divergent is not None:
        print(f"  divergences = {int(divergent.sum())}")

    treedepth = sampler_vars.get("treedepth__")
    if treedepth is not None:
        saturated = int((treedepth >= max_treedepth).sum())
        print(f"  max treedepth hits = {saturated}")
        print(f"  max observed treedepth = {int(treedepth.max())}")

    accept_stat = sampler_vars.get("accept_stat__")
    if accept_stat is not None:
        print(f"  mean accept_stat = {float(accept_stat.mean()):.4f}")
        
    print(fit.diagnose())


if __name__ == "__main__":
    main()
