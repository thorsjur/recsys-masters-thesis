"""JSONL-based experiment results logger."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
from enum import Enum

logger = logging.getLogger(__name__)


def _to_json_safe(obj: Any) -> Any:
    """Recursively convert non-JSON-serializable objects."""
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(item) for item in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    if isinstance(obj, (torch.device, Path)):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "__dict__"):
        return str(obj)
    return obj


class ResultsLogger:
    """Log experiment results to JSONL format."""

    def __init__(self, results_dir: str = "output/results", results_file: str = "experiments.jsonl"):
        self.results_path = Path(results_dir) / results_file
        self.results_path.parent.mkdir(parents=True, exist_ok=True)

    def log_experiment(
        self,
        model: str,
        dataset: str,
        config: Dict[str, Any],
        valid_results: Optional[Dict[str, float]] = None,
        test_results: Optional[Dict[str, float]] = None,
        training_time: Optional[float] = None,
        additional_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Log a single experiment result with structured metadata."""
        info = additional_info or {}

        entry = {
            "experiment_id": info.get("experiment_id"),
            "description": info.get("description"),
            "timestamp": datetime.now().isoformat(),
            "run_info": {
                "model": model,
                "dataset": dataset,
                "seed": config.get("seed"),
            },
            "window_info": _to_json_safe(info.get("window_info")) if info.get("window_info") else None,
        }

        entry["config_summary"] = {
            k: config.get(k)
            for k in ["epochs", "learning_rate", "train_batch_size", "eval_batch_size", "metrics", "topk"]
        }
        
        if test_results:
            entry["test_results"] = _to_json_safe(test_results)
        if valid_results:
            entry["validation_results"] = _to_json_safe(valid_results)
        if training_time is not None:
            entry["training_time_seconds"] = float(training_time)
            
        tmp = {
            "dataset_stats": _to_json_safe(info.get("dataset_stats", {})),
            "runtime": _to_json_safe(info.get("runtime", {})),
            "environment": _to_json_safe(info.get("environment", {})),
            "system_info": {
                "process_stats_pre_train": _to_json_safe(info.get("process_stats_pre_train", {})),
                "process_stats_post_run": _to_json_safe(info.get("process_stats_post_run", {})),
                "gpu_stats": _to_json_safe(info.get("gpu_stats", {})),
            },
        }
        
        entry.update({k: v for k, v in tmp.items() if v})

        entry["full_config"] = self._serialize_config(config)

        # Include additional info not already extracted
        extra = {
            k: v
            for k, v in info.items()
            if k
            not in [
                "experiment_id",
                "description",
                "window_info",
                "dataset_stats",
                "runtime",
                "environment",
                "process_stats_pre_train",
                "process_stats_post_run",
                "gpu_stats",
            ]
        }
        if extra:
            entry["additional_info"] = extra

        with open(self.results_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        return entry

    def _serialize_config(self, config) -> Any:
        """Extract config dict, handling RecBole Config objects."""
        from recbole.config import Config

        config_dict = config.final_config_dict if isinstance(config, Config) else dict(config)
        return _to_json_safe(config_dict)

    def load_results(self) -> list:
        """Load all experiment results from JSONL file."""
        if not self.results_path.exists():
            return []

        results = []
        with open(self.results_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
        return results

    def to_dataframe(self):
        """Convert results to pandas DataFrame."""
        import pandas as pd

        if not self.results_path.exists():
            return pd.DataFrame()
        return pd.read_json(self.results_path, lines=True)

    def get_best_result(self, model: str, dataset: str, metric: str, split: str = "test") -> Optional[Dict]:
        """Get best result for a model on a dataset by metric."""
        results = [
            r
            for r in self.load_results()
            if r.get("run_info", {}).get("model") == model
            and r.get("run_info", {}).get("dataset") == dataset
            and metric in r.get(f"{split}_results", {})
        ]
        if not results:
            return None
        return max(results, key=lambda x: x[f"{split}_results"][metric])

    def get_stability_analysis(self, model: str, dataset: str, split: str = "test", metrics: Optional[list] = None):
        """Analyze stability of a model across multiple runs."""
        from util.stability_metrics import aggregate_runs_stability

        results_key = f"{split}_results"
        filtered = [
            r[results_key]
            for r in self.load_results()
            if r.get("run_info", {}).get("model") == model
            and r.get("run_info", {}).get("dataset") == dataset
            and results_key in r
        ]

        return aggregate_runs_stability(filtered, metrics) if filtered else None
