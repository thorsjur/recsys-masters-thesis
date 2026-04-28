import logging
import shutil
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class TemporalDatasetBuilder:
    """
    Build train/valid/test splits from hour/day-wise interaction files.

    Helps temporal experiments by combining time based split files (hourly or daily splits) into
    files that RecBole can consume via benchmark_filename (essentially creating a subdataset, with train/(val)/test).
    """

    def __init__(self, data_path: str, dataset_name: str, granularity: str = "day"):
        if granularity not in ("hour", "day"):
            raise ValueError(f"granularity must be 'hour' or 'day', got '{granularity}'")

        self.data_path = Path(data_path)
        self.dataset_name = dataset_name
        self.dataset_dir = self.data_path / dataset_name
        self.granularity = granularity
        self.temp_dir: Optional[Path] = None

        if not self.dataset_dir.exists():
            raise ValueError(f"Dataset directory not found: {self.dataset_dir}")

    def _get_time_file(self, unit: int) -> Path:
        """Get path to time-wise interaction file."""
        return self.dataset_dir / f"{self.dataset_name}.{self.granularity}_{unit}.inter"

    def get_available_time_units(self) -> list:
        """Get sorted list of available time unit numbers."""
        pattern = f"{self.dataset_name}.{self.granularity}_*.inter"
        units = []
        for f in self.dataset_dir.glob(pattern):
            try:
                unit = int(f.stem.split(f"{self.granularity}_")[1])
                units.append(unit)
            except (IndexError, ValueError):
                continue
        return sorted(units)

    def check_time_files_exist(self, start: int, end: int) -> bool:
        """Check if all time files in range exist."""
        return all(self._get_time_file(u).exists() for u in range(start, end + 1))

    def load_time_range(self, start: int, end: int) -> pd.DataFrame:
        """Load and concatenate interactions from a range of time units."""
        if start > end:
            raise ValueError(f"start ({start}) must be <= end ({end})")

        dfs = []
        for unit in range(start, end + 1):
            path = self._get_time_file(unit)
            if not path.exists():
                raise FileNotFoundError(f"Time file not found: {path}")
            dfs.append(pd.read_csv(path, sep="\t"))

        if not dfs:
            raise ValueError(f"No data found for {self.granularity}s {start}-{end}")
        return pd.concat(dfs, ignore_index=True)

    def build_temporal_splits(
        self,
        train_hours: Optional[Tuple[int, int]] = None,
        valid_hours: Optional[Tuple[int, int]] = None,
        test_hours: Optional[Tuple[int, int]] = None,
        train_days: Optional[Tuple[int, int]] = None,
        valid_days: Optional[Tuple[int, int]] = None,
        test_days: Optional[Tuple[int, int]] = None,
        temp_prefix: str = "tmp",
    ) -> Tuple[Path, dict]:
        """Build train/valid/test splits from time ranges. Returns (temp_dir, splits_info)."""
        # Pick parameters based on granularity
        if self.granularity == "hour":
            train_range, valid_range, test_range = train_hours, valid_hours, test_hours
        else:
            train_range, valid_range, test_range = train_days, valid_days, test_days

        if train_range is None or test_range is None:
            raise ValueError(f"train_{self.granularity}s and test_{self.granularity}s are required")

        train_length = train_range[1] - train_range[0] + 1
        val_length = (valid_range[1] - valid_range[0] + 1) if valid_range else 0
        test_length = test_range[1] - test_range[0] + 1

        self.temp_dir = self.dataset_dir
        logger.info(f"Building temporal splits with prefix '{temp_prefix}'")

        splits = {}
        for split_name, time_range in [("train", train_range), ("test", test_range)]:
            start, end = time_range
            logger.info(f"  {split_name}: {self.granularity}s {start}-{end}")

            filename = f"{temp_prefix}_{split_name}_{self.granularity}_{train_length}_{val_length}_{test_length}"
            filepath = self.temp_dir / f"{self.dataset_name}.{filename}.inter"
            if filepath.exists():
                logger.info(
                    f"Using existing files for {temp_prefix}_{split_name} {train_length}:{val_length}:{test_length} split"
                )
                splits[split_name] = filename
                continue

            df = self.load_time_range(start, end)
            df.to_csv(filepath, sep="\t", index=False)
            splits[split_name] = filename
            logger.info(f"    Wrote {len(df)} interactions to {filepath.name}")

        # Handle validation: use provided range or create dummy from test
        if valid_range:
            start, end = valid_range
            logger.info(f"  valid: {self.granularity}s {start}-{end}")
            df = self.load_time_range(start, end)
        else:
            logger.info("  valid: Creating minimal dummy (RecBole compatibility)")
            df = self.load_time_range(*test_range).head(100)

        filename = f"{temp_prefix}_valid"
        filepath = self.temp_dir / f"{self.dataset_name}.{filename}.inter"
        splits["valid"] = filename

        if not filepath.exists():
            df.to_csv(filepath, sep="\t", index=False)
            logger.info(f"    Wrote {len(df)} interactions to {filepath.name}")
        else:
            logger.info(f"Using existing file for {temp_prefix}_valid split")

        # Copy item file if needed
        item_file = self.dataset_dir / f"{self.dataset_name}.item"
        if item_file.exists():
            temp_item = self.temp_dir / f"{self.dataset_name}.item"
            if not temp_item.exists():
                shutil.copy2(item_file, temp_item)
                logger.info(f"  Copied item file: {self.dataset_name}.item")

        splits["benchmark_filename"] = [splits["train"], splits["valid"], splits["test"]]
        splits["temp_dir"] = str(self.temp_dir)
        splits["temp_prefix"] = temp_prefix
        splits["has_valid"] = valid_range is not None

        return self.temp_dir, splits

    def cleanup(self, temp_prefix: Optional[str] = None, cleanup_all: bool = False):
        """Remove temporary split files."""
        if not self.temp_dir or not self.temp_dir.exists():
            return

        if temp_prefix:
            pattern = f"{self.dataset_name}.{temp_prefix}_*.inter"
            files = list(self.temp_dir.glob(pattern))
            if files:
                logger.info(f"Cleaning up {len(files)} files with prefix '{temp_prefix}'")
                for f in files:
                    f.unlink()
                    logger.debug(f"  Deleted {f.name}")
        elif cleanup_all:
            logger.info("Cleaning up entire temp directory")
            shutil.rmtree(self.temp_dir)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
