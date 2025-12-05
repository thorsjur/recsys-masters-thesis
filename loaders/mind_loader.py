from pandarallel import pandarallel
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from loaders.base_loader import AbstractDataLoader, DatasetConfig

class MINDDataLoader(AbstractDataLoader):
    def __init__(self, config: DatasetConfig):
        super().__init__(config)

    def _load_raw_data(self):
        print(f"Loading MIND ({self.config.version}) from {self.config.raw_path}...")

        news_path = os.path.join(self.config.raw_path, 'news.tsv')
        behaviors_path = os.path.join(self.config.raw_path, 'behaviors.tsv')

        CHUNK_SIZE = 50000
        all_interactions = []

        print("Parsing news.tsv...")
        self.df_item = pd.read_csv(
            news_path, 
            sep='\t', 
            header=None,
            names=['item_id', 'category', 'sub_category', 'title', 'abstract', 'url', 't_ents', 'a_ents'],
            usecols=['item_id', 'category', 'sub_category', 'title', 'abstract']
        )

        chunk_iterator = pd.read_csv(
            behaviors_path,
            sep='\t',
            header=None,
            names=['impression_id', 'user_id', 'time_str', 'history', 'impressions'],
            usecols=['impression_id', 'user_id', 'time_str', 'impressions'],
            chunksize=CHUNK_SIZE
        )

        print(f"Parsing behaviors.tsv in chunks of {CHUNK_SIZE}...")
        
        for chunk in tqdm(chunk_iterator, desc="Processing chunks"):
            
            # A. Convert time string to unix timestamp
            chunk['timestamp'] = pd.to_datetime(chunk['time_str'], format="%m/%d/%Y %I:%M:%S %p")
            chunk['timestamp'] = chunk['timestamp'].astype(np.int64) // 10**9

            chunk['impressions'] = chunk['impressions'].str.split(' ')
            df_exploded_chunk = chunk.explode('impressions').copy()
            
            # C. Split "N123-1" into "N123" and "1"
            split_data = df_exploded_chunk['impressions'].str.split('-', expand=True)
            
            df_exploded_chunk['item_id'] = split_data[0]
            df_exploded_chunk['label'] = split_data[1].astype(float)
            
            # D. Final selection and append
            df_final_chunk = df_exploded_chunk[['user_id', 'item_id', 'timestamp', 'label', 'impression_id']]
            all_interactions.append(df_final_chunk)
            
        self.df_inter = pd.concat(all_interactions, ignore_index=True)
        
        print(f"Loaded {len(self.df_inter)} atomic interactions.")