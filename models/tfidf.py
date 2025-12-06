import torch
import torch.nn as nn
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType


class TFIDF(GeneralRecommender):
    """TF-IDF baseline for content-based recommendation."""
    
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(TFIDF, self).__init__(config, dataset)
        
        self.n_items = dataset.item_num
        self.device = config['device']
        
        self.title_field = config['title_field'] if 'title_field' in config else 'title'
        self.abstract_field = config['abstract_field'] if 'abstract_field' in config else 'abstract'
        self.use_abstract = config['use_abstract'] if 'use_abstract' in config else True
        
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))
        
        self._build_tfidf_matrix(dataset)
        
    def _build_tfidf_matrix(self, dataset):
        """Build TF-IDF matrix from item features."""
        item_texts = []
        
        if dataset.item_feat is None:
            raise ValueError("Dataset must have item features for TF-IDF model")
        
        item_feat = dataset.item_feat
        
        for item_idx in range(self.n_items):
            text_parts = []
            
            if self.title_field in item_feat:
                title = item_feat[self.title_field][item_idx]
                if torch.is_tensor(title):
                    title = title.cpu().numpy()
                
                if isinstance(title, np.ndarray) and title.ndim > 0:
                    title_tokens = [dataset.id2token(self.title_field, [int(t)])[0] for t in title if int(t) != 0]
                    title_text = " ".join(title_tokens)
                else:
                    title_val = int(title) if hasattr(title, '__int__') else title
                    title_text = dataset.id2token(self.title_field, [title_val])[0] if title_val != 0 else ""
                text_parts.append(str(title_text))
            
            if self.use_abstract and self.abstract_field in item_feat:
                abstract = item_feat[self.abstract_field][item_idx]
                if torch.is_tensor(abstract):
                    abstract = abstract.cpu().numpy()
                
                if isinstance(abstract, np.ndarray) and abstract.ndim > 0:
                    abstract_tokens = [dataset.id2token(self.abstract_field, [int(t)])[0] for t in abstract if int(t) != 0]
                    abstract_text = " ".join(abstract_tokens)
                else:
                    abstract_val = int(abstract) if hasattr(abstract, '__int__') else abstract
                    abstract_text = dataset.id2token(self.abstract_field, [abstract_val])[0] if abstract_val != 0 else ""
                text_parts.append(str(abstract_text))
            
            item_texts.append(" ".join(text_parts) if text_parts else "")
        
        self.vectorizer = TfidfVectorizer(max_features=5000, min_df=2)
        self.tfidf_matrix = self.vectorizer.fit_transform(item_texts)
        
        self.item_similarity = cosine_similarity(self.tfidf_matrix)
        self.item_similarity_tensor = torch.FloatTensor(self.item_similarity).to(self.device)

    def forward(self):
        pass

    def calculate_loss(self, interaction):
        """TF-IDF doesn't require training, return zero loss."""
        return torch.zeros(1, device=self.device, requires_grad=True)

    def predict(self, interaction):
        """Predict scores based on user history and item similarity."""
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        
        scores = torch.zeros(len(user), device=self.device)
        
        for idx, (u, i) in enumerate(zip(user.cpu().numpy(), item.cpu().numpy())):
            user_history = self._get_user_history(u, interaction)
            if len(user_history) > 0:
                sim_scores = self.item_similarity_tensor[i, user_history]
                scores[idx] = sim_scores.mean()
        
        return scores

    def full_sort_predict(self, interaction):
        """Predict scores for all items based on user history."""
        user = interaction[self.USER_ID]
        
        scores = torch.zeros(len(user), self.n_items, device=self.device)
        
        for idx, u in enumerate(user.cpu().numpy()):
            user_history = self._get_user_history(u, interaction)
            if len(user_history) > 0:
                sim_matrix = self.item_similarity_tensor[:, user_history]
                scores[idx] = sim_matrix.mean(dim=1)
        
        return scores
    
    def _get_user_history(self, user_id, interaction):
        """Get user's historical items from the dataset."""
        try:
            history = interaction.dataset.user_history_dict.get(user_id, [])
            return [int(item) for item in history if item < self.n_items]
        except:
            return []
