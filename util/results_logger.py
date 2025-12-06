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
        """Log a single experiment result."""
        result_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "dataset": dataset,
            "config": self._serialize_config(config),
        }
        
        if valid_results is not None:
            result_entry["validation"] = valid_results
        
        if test_results is not None:
            result_entry["test"] = test_results
        
        if training_time is not None:
            result_entry["training_time_seconds"] = training_time
        
        if additional_info:
            result_entry["additional_info"] = additional_info
        
        with open(self.results_path, 'a') as f:
            f.write(json.dumps(result_entry) + '\n')
        
        return result_entry
    
    def _serialize_config(self, config) -> Dict[str, Any]:
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
