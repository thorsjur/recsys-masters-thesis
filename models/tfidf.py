import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp

from models.non_train_base_model import NonTrainableNewsEmbeddingRecommender


class TFIDF(NonTrainableNewsEmbeddingRecommender):
    """
    TF-IDF news recommender using "embeddings" derived from item text.:
    """

    def __init__(self, config, dataset):
        self.max_features = (
            config["tfidf_max_features"] if "tfidf_max_features" in config else 5000
        )
        self.min_df = config["tfidf_min_df"] if "tfidf_min_df" in config else 2
        super().__init__(config, dataset)

    def _build_item_embeddings(self, dataset):
        item_texts = [self._get_item_text(dataset, i) for i in range(self.n_items)]

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
        )
        tfidf = self.vectorizer.fit_transform(item_texts)
        
        assert isinstance(tfidf, sp.csr_matrix), "TFIDF output is not a sparse matrix"

        self.item_embeddings = torch.from_numpy(tfidf.toarray()).to(self.device)

        self.logger.info(
            f"TFIDF built item embeddings: shape={tuple(self.item_embeddings.shape)}, "
            f"tfidf_dim={self.item_embeddings.shape[1]}"
        )
