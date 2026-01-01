import torch
import numpy as np
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType

from models.non_train_base_model import NewsEmbeddingRecommender


class FastText(NewsEmbeddingRecommender):
    """
    FastText-based news recommender.

    Encodes news items using FastText embeddings, builds user representations
    by aggregating their click history, and ranks candidates using similarity.
    """

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        self.fasttext_dim = config["fasttext_dim"] if "fasttext_dim" in config else 300
        
        super().__init__(config, dataset)

    def _build_item_embeddings(self, dataset):
        """Build FastText embeddings for all items."""
        import os
        from gensim.models.fasttext import FastTextKeyedVectors, load_facebook_vectors

        fasttext_path = os.path.expanduser("~/fasttext/cc.en.300.bin")
        fasttext_cache = os.path.expanduser("~/fasttext/cache.kv")

        if not os.path.exists(fasttext_cache):
            self.logger.info(f"Loading FastText model from {fasttext_path}")
            self.fasttext_vectors = load_facebook_vectors(fasttext_path)
            self.logger.info("FastText model loaded")
            
            os.makedirs(os.path.dirname(fasttext_cache), exist_ok=True)
            self.fasttext_vectors.save(fasttext_cache)
            self.logger.info(f"FastText vectors cached at {fasttext_cache}")
        else:
            self.logger.info(f"Loading cached FastText vectors from {fasttext_cache}")
            self.fasttext_vectors = FastTextKeyedVectors.load(fasttext_cache, mmap="r")
            self.logger.info("FastText vectors loaded from cache")
            
        item_embeddings = []

        for idx in range(self.n_items):
            text_tokens = self._get_item_text_tokens(dataset, idx)

            if not text_tokens:
                embedding = np.zeros(self.fasttext_dim)
            else:
                token_embeddings = [
                    self.fasttext_vectors.get_vector(token)
                    for token in text_tokens
                    if token in self.fasttext_vectors
                ]

                if token_embeddings:
                    embedding = np.mean(token_embeddings, axis=0)
                else:
                    embedding = np.zeros(self.fasttext_dim)

            item_embeddings.append(embedding)

        self.item_embeddings = torch.FloatTensor(np.array(item_embeddings)).to(
            self.device
        )

        self.logger.info(f"Built item embeddings: {self.item_embeddings.shape}")
