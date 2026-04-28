from logging import getLogger
import pandas as pd
from tqdm import tqdm

from etl.processing.base_preprocessor import BasePreprocessor


class NLTKTokenizer(BasePreprocessor):

    def __init__(
        self,
        item_text_fields: list[str],
        to_lower: bool = True,
        language: str = "english",
    ):
        self.item_text_fields = item_text_fields
        self.to_lower = to_lower
        self.language = language
        self.logger = getLogger(self.__class__.__name__)

        tqdm.pandas()
        self._ensure_nltk()

    @staticmethod
    def _ensure_nltk():
        import nltk

        # Ensure punkt tokenizer models exist (needed by word_tokenize).
        try:
            from nltk.tokenize import word_tokenize as _wt

            _ = _wt("test")
        except LookupError:
            import nltk

            nltk.download("punkt")
            nltk.download("punkt_tab")

    def _tokenize_one(self, text: str) -> str:
        from nltk.tokenize import word_tokenize

        tokens = (
            word_tokenize(text.lower(), language=self.language)
            if self.to_lower
            else word_tokenize(text, language=self.language)
        )
        return " ".join(tokens)

    def process(self, df_inter: pd.DataFrame, df_item: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

        df_inter_out = df_inter.copy()
        df_item_out = df_item.copy()

        for col in self.item_text_fields:
            if col not in df_item_out.columns:
                self.logger.warning(
                    f"Column '{col}' not found in item DataFrame. Skipping tokenization for this column."
                )
                continue

            def _safe_tok(x):
                if pd.isna(x):
                    return ""
                s = str(x).strip()
                if not s:
                    return ""
                return self._tokenize_one(s)

            tqdm.pandas(desc=f"Tokenizing '{col}'")

            df_item_out[col] = df_item_out[col].progress_apply(_safe_tok)

        return df_inter_out, df_item_out
