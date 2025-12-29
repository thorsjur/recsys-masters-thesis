import gensim
import torch
import torch.nn.functional as F
import numpy as np
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType
from sklearn.metrics.pairwise import cosine_similarity


class FastText(GeneralRecommender):
    """
    FastText-based news recommender.
    
    Encodes news items using FastText embeddings, builds user representations
    by aggregating their click history, and ranks candidates using similarity.
    
    Config parameters:
        - title_field: Field containing title tokens (default: 'title')
        - abstract_field: Field containing abstract tokens (default: 'abstract')
        - use_abstract: Whether to use abstract in addition to title (default: True)
        - aggregation: User history aggregation method (default: 'mean')
                      Options: 'mean', 'sum', 'max'
        - similarity: Similarity function for scoring (default: 'cosine')
                     Options: 'cosine', 'dot'
        - fasttext_dim: Dimension of FastText embeddings (default: 300)
    """
    
    input_type = InputType.POINTWISE
    
    def __init__(self, config, dataset):
        super(FastText, self).__init__(config, dataset)
        
        self.n_items = dataset.item_num
        self.device = config['device']
        
        self.title_field = config['title_field'] if 'title_field' in config else 'title'
        self.abstract_field = config['abstract_field'] if 'abstract_field' in config else 'abstract'
        self.use_abstract = config['use_abstract'] if 'use_abstract' in config else True
        
        self.aggregation = config['aggregation'] if 'aggregation' in config else 'mean'
        self.similarity = config['similarity'] if 'similarity' in config else 'cosine'
        self.fasttext_dim = config['fasttext_dim'] if 'fasttext_dim' in config else 300
        
        self.logger.info(f"FastText model initializing with aggregation={self.aggregation}, similarity={self.similarity}")
        
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))
        
        self._build_item_embeddings(dataset)
        self._build_user_embeddings(dataset)
    
    def _build_item_embeddings(self, dataset):
        """Build FastText embeddings for all items."""
        import gensim.downloader as api
        import os
        
        fasttext_path = os.path.expanduser('~/fasttext/cc.en.300.bin')
        
        if os.path.exists(fasttext_path):
            from gensim.models.fasttext import load_facebook_vectors
            self.logger.info(f"Loading FastText model from {fasttext_path}")
            self.fasttext_vectors = load_facebook_vectors(fasttext_path)
            self.logger.info("FastText model loaded")
        else:
            from gensim.models import KeyedVectors

            self.logger.info("Custom FastText model not found...")
            self.logger.info("Using fasttext-wiki-news-subwords-300 (650MB download)")

            default_fasttext_path = os.path.expanduser('~/fasttext/default_fasttext_wiki_news_subwords_300')
            kv_path = os.path.join(os.path.dirname(default_fasttext_path), "fasttext.kv.vectors.npy")

            if os.path.exists(kv_path):
                self.logger.info(f"Loading cached FastText vectors from {kv_path}")
                self.fasttext_vectors = KeyedVectors.load(kv_path, mmap="r")
            else:
                self.logger.info("Cached vectors not found; loading via gensim api")
                self.fasttext_vectors = api.load("fasttext-wiki-news-subwords-300")

                assert isinstance(
                    self.fasttext_vectors, gensim.models.keyedvectors.KeyedVectors
                ), f"Loaded vectors must be of type KeyedVectors, got {type(self.fasttext_vectors)}"

                os.makedirs(default_fasttext_path, exist_ok=True)
                self.fasttext_vectors.save(kv_path)
        
        item_embeddings = []

        assert isinstance(self.fasttext_vectors, gensim.models.keyedvectors.KeyedVectors), f"FastText vectors must be KeyedVectors instance, got {type(self.fasttext_vectors)}"
        
        for idx in range(self.n_items):
            text_tokens = self._get_item_text_tokens(dataset, idx)
            
            if not text_tokens:
                embedding = np.zeros(self.fasttext_dim)
            else:
                token_embeddings = [
                    self.fasttext_vectors.get_vector(token)
                    for token in text_tokens
                ]
                
                if token_embeddings:
                    embedding = np.mean(token_embeddings, axis=0)
                else:
                    embedding = np.zeros(self.fasttext_dim)
            
            item_embeddings.append(embedding)
        
        self.item_embeddings = torch.FloatTensor(np.array(item_embeddings)).to(self.device)
        if self.similarity == 'cosine':
            self.item_embeddings = F.normalize(self.item_embeddings, p=2, dim=1)
        self.logger.info(f"Built item embeddings: {self.item_embeddings.shape}")

    def _build_user_embeddings(self, dataset):
        inter = dataset.inter_feat

        user_ids = inter[self.USER_ID].numpy()
        item_ids = inter[self.ITEM_ID].numpy()

        n_users = dataset.user_num

        user_hist_items = [[] for _ in range(n_users)]
        for u, i in zip(user_ids, item_ids):
            user_hist_items[u].append(i)

        user_emb_list = []

        for u in range(n_users):
            hist = user_hist_items[u]
            if not hist:
                user_emb_list.append(np.zeros(self.fasttext_dim, dtype=np.float32))
                continue

            hist_idx = torch.LongTensor(hist)
            hist_emb = self.item_embeddings.cpu()[hist_idx]

            if self.aggregation == 'mean':
                u_emb = hist_emb.mean(dim=0)
            elif self.aggregation == 'sum':
                u_emb = hist_emb.sum(dim=0)
            elif self.aggregation == 'max':
                u_emb = hist_emb.max(dim=0)[0]
            else:
                raise ValueError(f"Unknown aggregation method: {self.aggregation}")

            user_emb_list.append(u_emb.numpy())

        self.user_embeddings = torch.from_numpy(np.stack(user_emb_list, axis=0)).to(self.device)
        if self.similarity == 'cosine':
            self.user_embeddings = F.normalize(self.user_embeddings, p=2, dim=1)
        self.logger.info(f"Built user embeddings: {self.user_embeddings.shape}")


    
    def _get_item_text_tokens(self, dataset, idx):
        """Extract text tokens from item features."""
        tokens = []
        
        if self.title_field in dataset.item_feat:
            title = dataset.item_feat[self.title_field][idx]
            
            if torch.is_tensor(title):
                title = title.cpu().numpy()
            
            for token_id in title:
                token_id = int(token_id)
                if token_id != 0:
                    token = dataset.id2token(self.title_field, [token_id])[0]
                    tokens.append(token)
        
        if self.use_abstract and self.abstract_field in dataset.item_feat:
            abstract = dataset.item_feat[self.abstract_field][idx]
            
            if torch.is_tensor(abstract):
                abstract = abstract.cpu().numpy()
            
            for token_id in abstract:
                token_id = int(token_id)
                if token_id != 0:
                    token = dataset.id2token(self.abstract_field, [token_id])[0]
                    tokens.append(token)
        
        return tokens
    
    def forward(self, interaction):
        """Forward pass for evaluation."""
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        
        user_embeddings = self.user_embeddings[user]
        item_embeddings = self.item_embeddings[item]
        
        return torch.sum(user_embeddings * item_embeddings, dim=1)
    
    def calculate_loss(self, interaction):
        """No training; return zero loss."""
        return self.dummy_param.new_zeros()
    
    @torch.no_grad()
    def predict(self, interaction):
        """Predict scores for user-item pairs."""
        return self.forward(interaction)
    
    @torch.no_grad()
    def full_sort_predict(self, interaction):
        """Predict scores for all items for given users."""
        user = interaction[self.USER_ID]
        user_embeddings = self.user_embeddings[user]
        
        return user_embeddings @ self.item_embeddings.t()
