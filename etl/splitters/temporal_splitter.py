import pandas as pd
from typing import Tuple, List
from .base_splitter import BaseSplitter

class GlobalTemporalSplitter(BaseSplitter):
    """Split data by timestamp with configurable ratios.
    
    Not currently used, but I keep it around in case there is a future need for the functionality."""
    
    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = df.sort_values('timestamp').copy()
        
        min_time = df['timestamp'].min()
        max_time = df['timestamp'].max()
        total_duration = max_time - min_time
        
        ratios: List[float] = self.config.get('ratios', [0.8, 0.1, 0.1])
        
        train_end = min_time + (total_duration * ratios[0])
        valid_end = train_end + (total_duration * ratios[1])
        
        train = df[df['timestamp'] <= train_end]
        valid = df[(df['timestamp'] > train_end) & (df['timestamp'] <= valid_end)]
        test = df[df['timestamp'] > valid_end]
        
        print(f"[GlobalTemporalSplitter] Split stats: Train={len(train)}, Valid={len(valid)}, Test={len(test)}")
        
        return train, valid, test