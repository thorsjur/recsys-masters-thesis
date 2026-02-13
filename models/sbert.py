import torch
from models.embeddings.sbert_provider import SBERTProvider
from models.non_train_base_model import NewsEmbeddingRecommender
import numpy as np


class SBERT(NewsEmbeddingRecommender):
    """
    SBERT-based non-trainable news recommender.
    """

    def __init__(self, config, dataset):
        self.embedding_provider = SBERTProvider(config=config, dim=384)
        
        super().__init__(config, dataset)

    def _build_item_embeddings(self, dataset):
        self.item_embeddings = self.embedding_provider.encode(
            sentences=self._get_all_sentences(dataset),
            show_progress=True,
            dtype=torch.float32,
        ).to(self.device)
        
        
    def _get_all_sentences(self, dataset):
        item_idxs = dataset.get_item_feature()[dataset.iid_field]
        return [self._get_item_text(dataset, item_idx) for item_idx in item_idxs]
