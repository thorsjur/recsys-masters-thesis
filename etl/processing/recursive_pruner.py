from loaders.processing.base_preprocessor import BasePreprocessor


class RecursivePruner(BasePreprocessor):
    def __init__(self, min_user_hist=5, min_item_freq=10):
        self.min_user = min_user_hist
        self.min_item = min_item_freq

    def process(self, df_inter, df_item):
        print(f"[{self.__class__.__name__}] Pruning graph (U>{self.min_user}, I>{self.min_item})...")
        
        df = df_inter.copy()
        initial_len = len(df)
        
        while True:
            start_len = len(df)
            
            # Filter items
            item_counts = df['item_id'].value_counts()
            valid_items = item_counts[item_counts >= self.min_item].index
            df = df[df['item_id'].isin(valid_items)]
            
            # Filter users
            user_counts = df['user_id'].value_counts()
            valid_users = user_counts[user_counts >= self.min_user].index
            df = df[df['user_id'].isin(valid_users)]
            
            if len(df) == start_len:
                break
        
        print(f"Dropped {initial_len - len(df)} interactions.")
        
        # Remove items that no longer exist in interactions
        valid_items_final = df['item_id'].unique()
        df_item = df_item[df_item['item_id'].isin(valid_items_final)].copy()
        
        return df, df_item