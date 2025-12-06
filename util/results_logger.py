import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class ResultsLogger:
    """Log experiment results to JSONL format for easy analysis."""
    
    def __init__(self, results_dir: str = "output/results", results_file: str = "experiments.jsonl"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results_path = self.results_dir / results_file
    
    def log_experiment(
        self,
        model: str,
        dataset: str,
        config: Dict[str, Any],
        valid_results: Optional[Dict[str, float]] = None,
        test_results: Optional[Dict[str, float]] = None,
        training_time: Optional[float] = None,
        additional_info: Optional[Dict[str, Any]] = None
    ):
        """
        Log a single experiment result with structured metadata.
        
        Output structure prioritizes experiment identification and context:
        1. Experiment metadata (ID, description, run context)
        2. Dataset information (name, size, sparsity)
        3. Model and configuration
        4. Results (validation, test)
        5. Full config details
        """
        # Extract key information from additional_info
        experiment_id = additional_info.get('experiment_id') if additional_info else None
        description = additional_info.get('description') if additional_info else None
        window_info = additional_info.get('window_info') if additional_info else None
        dataset_stats = additional_info.get('dataset_stats') if additional_info else None
        
        # Build result entry with metadata-first structure
        result_entry = {
            # === EXPERIMENT METADATA (FIRST FOR EASY FILTERING) ===
            "experiment_id": experiment_id,
            "description": description,
            "timestamp": datetime.now().isoformat(),
            
            # === RUN CONTEXT ===
            "run_info": {
                "model": model,
                "dataset": dataset,
                "seed": config.get('seed'),
            },
            
            # === DATASET INFORMATION ===
            "dataset_info": self._make_json_serializable(dataset_stats) if dataset_stats else {},
            
            # === WINDOW/SPLIT INFORMATION (for temporal experiments) ===
            "window_info": self._make_json_serializable(window_info) if window_info else None,
        }
        
        # Add results (convert numpy types to native Python)
        if valid_results is not None:
            result_entry["validation_results"] = self._make_json_serializable(valid_results)
        
        if test_results is not None:
            result_entry["test_results"] = self._make_json_serializable(test_results)
        
        if training_time is not None:
            result_entry["training_time_seconds"] = float(training_time)
        
        # Add simplified config summary (key hyperparameters only)
        result_entry["config_summary"] = {
            "epochs": config.get('epochs'),
            "learning_rate": config.get('learning_rate'),
            "train_batch_size": config.get('train_batch_size'),
            "eval_batch_size": config.get('eval_batch_size'),
            "metrics": config.get('metrics'),
            "topk": config.get('topk'),
        }
        
        # Add full config at the end
        result_entry["full_config"] = self._serialize_config(config)
        
        # Add any remaining additional info
        if additional_info:
            other_info = {k: v for k, v in additional_info.items() 
                         if k not in ['experiment_id', 'description', 'window_info', 'dataset_stats']}
            if other_info:
                result_entry["additional_info"] = other_info
        
        with open(self.results_path, 'a') as f:
            f.write(json.dumps(result_entry) + '\n')
        
        return result_entry
    
    def _serialize_config(self, config) -> Any:
        """Extract relevant config parameters for logging."""
        from recbole.config import Config
        
        if isinstance(config, Config):
            config_dict = config.final_config_dict
        else:
            config_dict = dict(config)
        
        return self._make_json_serializable(config_dict)
    
    def _make_json_serializable(self, obj):
        """Convert non-serializable objects to serializable format."""
        import numpy as np
        import torch
        from enum import Enum
        from pathlib import Path
        
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, torch.device):
            return str(obj)
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj
    
    def load_results(self) -> list:
        """Load all experiment results."""
        if not self.results_path.exists():
            return []
        
        results = []
        with open(self.results_path, 'r') as f:
            for line in f:
                results.append(json.loads(line))
        
        return results
    
    def to_dataframe(self):
        """Convert results to pandas DataFrame for analysis."""
        import pandas as pd
        
        if not self.results_path.exists():
            return pd.DataFrame()
        
        return pd.read_json(self.results_path, lines=True)
    
    def get_best_result(self, model: str, dataset: str, metric: str, split: str = 'test') -> Optional[Dict]:
        """Get best result for a model on a dataset by metric."""
        results = self.load_results()
        
        filtered = [
            r for r in results 
            if r['model'] == model and r['dataset'] == dataset
            and metric in r.get(split, {})
        ]
        
        if not filtered:
            return None
        
        return max(filtered, key=lambda x: x[split][metric])
    
    def compare_models(self, dataset: str, metric: str = 'NDCG@10', split: str = 'test'):
        """Compare all models on a dataset for a specific metric."""
        results = self.load_results()
        
        comparison = {}
        for result in results:
            if result['dataset'] != dataset:
                continue
            
            model = result['model']
            if metric in result.get(split, {}):
                score = result[split][metric]
                
                if model not in comparison or score > comparison[model]['score']:
                    comparison[model] = {
                        'score': score,
                        'timestamp': result['timestamp'],
                        'config': result['config']
                    }
        
        return comparison
    
    def get_stability_analysis(self, model: str, dataset: str, split: str = 'test', metrics: Optional[list] = None):
        """
        Analyze stability of a model across multiple runs.
        
        Args:
            model: Model name
            dataset: Dataset name
            split: 'test' or 'validation'
            metrics: List of metrics to analyze (if None, analyze all)
            
        Returns:
            Dictionary with stability metrics (mean, std, cv, max_drop) for each metric
        """
        from util.stability_metrics import aggregate_runs_stability
        
        results = self.load_results()
        
        # Filter results for this model and dataset
        filtered = [
            r for r in results 
            if r['model'] == model 
            and r['dataset'] == dataset
            and split in r
        ]
        
        if not filtered:
            return None
        
        # Extract the metric results from each run
        run_results = [r[split] for r in filtered]
        
        return aggregate_runs_stability(run_results, metrics)
