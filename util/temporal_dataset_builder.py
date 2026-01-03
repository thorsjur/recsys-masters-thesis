import os
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from typing import Tuple, Optional


class TemporalDatasetBuilder:
    """
    Builds temporary train/valid/test splits from hour/day-wise interaction files.
    
    This utility enables temporal stability experiments by combining time-based
    split files into temporary train/valid/test files that can be used with 
    RecBole's benchmark_filename parameter.
    
    Supports both hour-level and day-level granularity, and optional validation sets.
    
    Example usage:
        builder = TemporalDatasetBuilder('datasets/atomic_files', 'mind_small')
        
        # With validation set
        temp_dir, splits = builder.build_temporal_splits(
            train_hours=(1, 120),
            valid_hours=(121, 144),
            test_hours=(145, 168),
            temp_prefix='window1'
        )
        
        # Without validation set (valid_hours=None)
        temp_dir, splits = builder.build_temporal_splits(
            train_hours=(1, 120),
            valid_hours=None,
            test_hours=(121, 144),
            temp_prefix='window1'
        )
        
        # Use splits with RecBole by passing:
        # --params benchmark_filename=['window1_train','window1_test']
        
        # Cleanup when done
        builder.cleanup()
    """
    
    def __init__(self, data_path: str, dataset_name: str, granularity: str = 'day'):
        """
        Initialize builder for a specific dataset.
        
        Args:
            data_path: Path to dataset directory (e.g., 'datasets/atomic_files')
            dataset_name: Name of the dataset (e.g., 'mind_small')
            granularity: Time granularity ('hour' or 'day', default: 'day')
        """
        self.data_path = Path(data_path)
        self.dataset_name = dataset_name
        self.dataset_dir = self.data_path / dataset_name
        self.temp_dir: Optional[Path] = None
        self.granularity = granularity
        
        if granularity not in ['hour', 'day']:
            raise ValueError(f"granularity must be 'hour' or 'day', got '{granularity}'")
        
        if not self.dataset_dir.exists():
            raise ValueError(f"Dataset directory not found: {self.dataset_dir}")
    
    def get_time_file_path(self, time_unit: int) -> Path:
        """Get path to time-wise interaction file (hour or day)."""
        if self.granularity == 'hour':
            return self.dataset_dir / f"{self.dataset_name}.hour_{time_unit}.inter"
        else:
            return self.dataset_dir / f"{self.dataset_name}.day_{time_unit}.inter"
    
    def check_time_files_exist(self, start_unit: int, end_unit: int) -> bool:
        """Check if all time files in range exist."""
        for unit in range(start_unit, end_unit + 1):
            if not self.get_time_file_path(unit).exists():
                return False
        return True
    
    def get_available_time_units(self) -> list:
        """Get list of available time unit numbers from existing files."""
        pattern = f"{self.dataset_name}.{self.granularity}_*.inter"
        time_files = list(self.dataset_dir.glob(pattern))
        units = []
        for f in time_files:
            try:
                unit_num = int(f.stem.split(f'{self.granularity}_')[1])
                units.append(unit_num)
            except (IndexError, ValueError):
                continue
        return sorted(units)
    
    def load_time_range(self, start_unit: int, end_unit: int) -> pd.DataFrame:
        """
        Load and concatenate interactions from a range of time units.
        
        Args:
            start_unit: First time unit (inclusive)
            end_unit: Last time unit (inclusive)
            
        Returns:
            DataFrame with combined interactions
        """
        if start_unit > end_unit:
            raise ValueError(f"start_unit ({start_unit}) must be <= end_unit ({end_unit})")
        
        dfs = []
        for unit in range(start_unit, end_unit + 1):
            time_file = self.get_time_file_path(unit)
            if not time_file.exists():
                raise FileNotFoundError(f"Time file not found: {time_file}")
            
            df = pd.read_csv(time_file, sep='\t')
            dfs.append(df)
        
        if not dfs:
            raise ValueError(f"No data found for {self.granularity}s {start_unit}-{end_unit}")
        
        combined = pd.concat(dfs, ignore_index=True)
        return combined
    
    def build_temporal_splits(
        self,
        train_hours: Optional[Tuple[int, int]] = None,
        valid_hours: Optional[Tuple[int, int]] = None,
        test_hours: Optional[Tuple[int, int]] = None,
        train_days: Optional[Tuple[int, int]] = None,
        valid_days: Optional[Tuple[int, int]] = None,
        test_days: Optional[Tuple[int, int]] = None,
        temp_prefix: str = 'tmp'
    ) -> Tuple[Path, dict]:
        """
        Build temporary train/valid/test splits from time ranges.
        
        Supports both hour-level and day-level granularity. Validation set is optional.
        """
        # Determine which parameters to use based on granularity
        if self.granularity == 'hour':
            train_range = train_hours
            valid_range = valid_hours
            test_range = test_hours
            unit_name = 'hour'
        else:
            train_range = train_days
            valid_range = valid_days
            test_range = test_days
            unit_name = 'day'
        
        if train_range is None or test_range is None:
            raise ValueError(f"train_{unit_name}s and test_{unit_name}s are required")
        
        # Create temporary directory - now using temp_prefix directly in path
        self.temp_dir = self.dataset_dir
        
        print(f"[TemporalDatasetBuilder] Building temporal splits with prefix '{temp_prefix}'")
        
        splits = {}
        split_configs = {
            'train': train_range,
            'test': test_range
        }
        
        # Add validation range if provided, otherwise create dummy from test data
        create_dummy_valid = valid_range is None
        if valid_range is not None:
            split_configs['valid'] = valid_range
        
        for split_name, (start_unit, end_unit) in split_configs.items():
            print(f"  {split_name}: {unit_name}s {start_unit}-{end_unit}")
            
            # Load time range
            df = self.load_time_range(start_unit, end_unit)
            
            # Write to temporary file with consistent naming
            temp_filename = f"{temp_prefix}_{split_name}"
            temp_filepath = self.temp_dir / f"{self.dataset_name}.{temp_filename}.inter"
            df.to_csv(temp_filepath, sep='\t', index=False)
            
            splits[split_name] = temp_filename
            print(f"    Wrote {len(df)} interactions to {temp_filepath.name}")
        
        # Create dummy validation file if not provided (RecBole requires 3 splits)
        if create_dummy_valid:
            print(f"  valid: Creating minimal dummy validation (RecBole compatibility)")
            # Use first 100 interactions from test set as dummy validation
            test_df = self.load_time_range(*test_range)
            dummy_valid_df = test_df.head(min(100, len(test_df)))
            
            temp_filename = f"{temp_prefix}_valid"
            temp_filepath = self.temp_dir / f"{self.dataset_name}.{temp_filename}.inter"
            dummy_valid_df.to_csv(temp_filepath, sep='\t', index=False)
            
            splits['valid'] = temp_filename
            print(f"    Wrote {len(dummy_valid_df)} interactions to {temp_filepath.name} (dummy)")
        
        # Copy item file to temp directory if it exists (needed for RecBole to load item features)
        item_file = self.dataset_dir / f"{self.dataset_name}.item"
        temp_item_file = self.temp_dir / f"{self.dataset_name}.item"
        if item_file.exists():
            if not temp_item_file.exists():
                shutil.copy2(item_file, temp_item_file)
                print(f"  Copied item file to temp directory: {self.dataset_name}.item")
            else:
                print(f"  Using existing item file in temp directory: {self.dataset_name}.item")
        
        # Build benchmark_filename list (always 3 splits for RecBole compatibility)
        benchmark_list = [splits['train'], splits['valid'], splits['test']]
        
        splits['benchmark_filename'] = benchmark_list
        splits['temp_dir'] = str(self.temp_dir)
        splits['temp_prefix'] = temp_prefix
        splits['has_valid'] = not create_dummy_valid  # True if valid was explicitly provided
        
        return self.temp_dir, splits
    
    def cleanup(self, temp_prefix: Optional[str] = None, cleanup_all: bool = False):
        """
        Remove temporary split files.
        
        Args:
            temp_prefix: Prefix of temporary files to delete (e.g., 'window1')
                        If None and cleanup_all is True, removes entire temp directory
            cleanup_all: If True, removes all temporary files including item file
        """
        if self.temp_dir and self.temp_dir.exists():
            if temp_prefix:
                # Delete specific temp files matching the prefix
                pattern = f"{self.dataset_name}.{temp_prefix}_*.inter"
                temp_files = list(self.temp_dir.glob(pattern))
                if temp_files:
                    print(f"[TemporalDatasetBuilder] Cleaning up {len(temp_files)} temporary files with prefix '{temp_prefix}'")
                    for f in temp_files:
                        f.unlink()
                        print(f"  Deleted {f.name}")
            elif cleanup_all:
                # Delete entire temp directory including item file
                print(f"[TemporalDatasetBuilder] Cleaning up entire temp directory")
                shutil.rmtree(self.temp_dir)
                print(f"  Deleted {self.temp_dir}")
            else:
                print(f"[TemporalDatasetBuilder] No cleanup performed (no temp_prefix specified)")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup temporary files."""
        self.cleanup()


def get_temporal_dataset_path(data_path: str, dataset_name: str, temp_prefix: str = 'tmp') -> str:
    """
    Get the data path to use for temporal experiments.
    
    This returns the parent directory that RecBole's create_dataset should use,
    which will contain the temporary split subdirectory.
    
    Args:
        data_path: Base data path (e.g., 'datasets/atomic_files')
        dataset_name: Dataset name (e.g., 'mind_small')
        temp_prefix: Prefix used for temporary directory
        
    Returns:
        Path string that should be passed to RecBole's data_path parameter
    """
    # RecBole expects data_path to be the parent directory
    # It will look for data_path/dataset_name/ for the files
    # Since our temp directory is at data_path/dataset_name/_temp_xxx/
    # We need to return data_path/dataset_name/_temp_xxx as the data_path
    # and still use dataset_name as the dataset parameter
    
    # Actually, RecBole's structure is: data_path / dataset_name / files
    # So we keep data_path the same and just use benchmark_filename
    return data_path
