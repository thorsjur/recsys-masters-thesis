import torch
from recbole.utils import InputType

from models.non_train_base_model import NewsEmbeddingRecommender


class FastText(NewsEmbeddingRecommender):
    """
    FastText-based news recommender.

    Encodes news items using FastText embeddings, builds user representations
    by aggregating their click history, and ranks candidates using similarity.
    """

    input_type = InputType.LISTWISE

    def __init__(self, config, dataset):
        self.fasttext_dim = config.get("fasttext_dim", 300)
        super().__init__(config, dataset)

    def _build_item_embeddings(self, dataset):
        """Build item embeddings by averaging FastText token vectors."""
        self.item_embeddings = self._build_token_provider_item_embeddings(
            dataset,
            dim=self.fasttext_dim,
        )
