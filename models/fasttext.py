import gensim
import torch
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
                token_embeddings = []
                for token in text_tokens:
                    if self.fasttext_vectors.has_index_for(token):
                        token_embeddings.append(self.fasttext_vectors.get_vector(token))
                
                if token_embeddings:
                    embedding = np.mean(token_embeddings, axis=0)
                else:
                    embedding = np.zeros(self.fasttext_dim)
            
            item_embeddings.append(embedding)
        
        self.item_embeddings = torch.FloatTensor(np.array(item_embeddings)).to(self.device)
        self.logger.info(f"Built item embeddings: {self.item_embeddings.shape}")
    
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
    
    def _aggregate_user_history(self, user_id, interaction):
        """Aggregate user's historical item embeddings into user representation."""
        # Get user history from the dataset (handles temporal splits correctly)
        hist_items = interaction.dataset.user_history_dict.get(user_id, [])
        hist_items = [int(item) for item in hist_items if item < self.n_items]
        
        if len(hist_items) == 0:
            return torch.zeros(self.fasttext_dim).to(self.device)
        
        hist_embeddings = self.item_embeddings[hist_items]
        
        if self.aggregation == 'mean':
            user_emb = torch.mean(hist_embeddings, dim=0)
        elif self.aggregation == 'sum':
            user_emb = torch.sum(hist_embeddings, dim=0)
        elif self.aggregation == 'max':
            user_emb = torch.max(hist_embeddings, dim=0)[0]
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")
        
        return user_emb
    
    def forward(self, interaction):
        """Forward pass for evaluation."""
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        
        user_embeddings = []
        for u in user.cpu().numpy():
            user_embeddings.append(self._aggregate_user_history(u, interaction))
        user_embeddings = torch.stack(user_embeddings)
        
        item_embeddings = self.item_embeddings[item]
        
        if self.similarity == 'cosine':
            user_norm = torch.nn.functional.normalize(user_embeddings, p=2, dim=1)
            item_norm = torch.nn.functional.normalize(item_embeddings, p=2, dim=1)
            scores = torch.sum(user_norm * item_norm, dim=1)
        elif self.similarity == 'dot':
            scores = torch.sum(user_embeddings * item_embeddings, dim=1)
        else:
            raise ValueError(f"Unknown similarity function: {self.similarity}")
        
        return scores
    
    def calculate_loss(self, interaction):
        """No training; return zero loss."""
        return torch.tensor(0.0).to(self.device)
    
    def predict(self, interaction):
        """Predict scores for user-item pairs."""
        return self.forward(interaction)
    
    def full_sort_predict(self, interaction):
        """Predict scores for all items for given users."""
        user = interaction[self.USER_ID]
        
        user_embeddings = []
        for u in user.cpu().numpy():
            user_embeddings.append(self._aggregate_user_history(u, interaction))
        user_embeddings = torch.stack(user_embeddings)
        
        if self.similarity == 'cosine':
            user_norm = torch.nn.functional.normalize(user_embeddings, p=2, dim=1)
            item_norm = torch.nn.functional.normalize(self.item_embeddings, p=2, dim=1)
            scores = torch.matmul(user_norm, item_norm.t())
        elif self.similarity == 'dot':
            scores = torch.matmul(user_embeddings, self.item_embeddings.t())
        else:
            raise ValueError(f"Unknown similarity function: {self.similarity}")
        
        return scores
