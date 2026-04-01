from pathlib import Path
import pandas as pd


def find_dataset_dir(dataset: str, base_path: str = "data/atomic_files") -> Path:
    dataset_dir = Path(base_path) / dataset
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    return dataset_dir


def find_item_file(dataset: str, base_path: str = "data/atomic_files") -> Path:
    dataset_dir = find_dataset_dir(dataset, base_path)
    path = dataset_dir / f"{dataset}.item"
    if not path.exists():
        raise FileNotFoundError(f"No item file found in {dataset_dir}")
    return path


def find_interaction_files(dataset: str, base_path: str = "data/atomic_files") -> list[Path]:
    dataset_dir = find_dataset_dir(dataset, base_path)
    direct_path = dataset_dir / f"{dataset}.inter"
    if direct_path.exists():
        return [direct_path]

    temporal_files = sorted(
        dataset_dir.glob(f"{dataset}.*.inter"),
        key=lambda path: (_interaction_suffix_rank(path), path.name),
    )
    if temporal_files:
        return temporal_files

    raise FileNotFoundError(f"No interaction files found in {dataset_dir}")


def load_item_dataframe(item_file: Path) -> pd.DataFrame:
    df = pd.read_csv(item_file, sep="\t", dtype=str, keep_default_na=False)
    df.columns = [column.split(":")[0] for column in df.columns]
    return df


def load_interaction_dataframe(interaction_files: list[Path]) -> pd.DataFrame:
    dfs = []
    for path in interaction_files:
        df = pd.read_csv(path, sep="\t")
        df.columns = [column.split(":")[0] for column in df.columns]

        suffix = _parse_interaction_suffix(path)
        if suffix is not None and "time_unit" not in df.columns:
            _, unit = suffix
            df["time_unit"] = unit

        dfs.append(df)

    if not dfs:
        raise ValueError("No interaction data loaded")

    return pd.concat(dfs, ignore_index=True)


def _parse_interaction_suffix(path: Path) -> tuple[str, int] | None:
    parts = path.stem.split(".")
    if len(parts) < 2 or "_" not in parts[-1]:
        return None

    granularity, unit_str = parts[-1].split("_", maxsplit=1)
    if not unit_str.isdigit():
        return None
    return granularity, int(unit_str)


def _interaction_suffix_rank(path: Path) -> tuple[int, str, int]:
    suffix = _parse_interaction_suffix(path)
    if suffix is None:
        return (1, "", 0)

    granularity, unit = suffix
    return (0, granularity, unit)
