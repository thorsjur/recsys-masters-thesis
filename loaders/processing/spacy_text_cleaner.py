import spacy
from tqdm import tqdm

from loaders.processing.base_preprocessor import BasePreprocessor

class SpacyTextCleaner(BasePreprocessor):
    def __init__(self, target_col: str, output_col: str, model: str = "en_core_web_sm", batch_size: int = 1000):
        self.target_col = target_col
        self.output_col = output_col
        self.batch_size = batch_size
        self.nlp = spacy.load(model, disable=["tok2vec", "ner", "parser"])
        self.nlp.enable_pipe("senter")

    def process(self, df_inter, df_item):
        if self.target_col not in df_item.columns:
            print(f"Warning: {self.target_col} not in item dataframe. Skipping.")
            return df_inter, df_item

        print(f"[{self.__class__.__name__}] Cleaning '{self.target_col}' -> '{self.output_col}'...")
        
        text_gen = (str(t) for t in df_item[self.target_col].fillna(""))
        results = []

        for doc in tqdm(self.nlp.pipe(text_gen, batch_size=self.batch_size), total=len(df_item), desc=f"Cleaning {self.target_col}"):
            tokens = [t.text.lower() for t in doc if not t.is_stop and not t.is_punct and not t.is_space]
            results.append(" ".join(tokens))
            
        df_item[self.output_col] = results
        return df_inter, df_item