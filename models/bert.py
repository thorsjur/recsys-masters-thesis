import torch
from tqdm import tqdm
from models.non_train_base_model import NonTrainableNewsEmbeddingRecommender
from transformers import AutoTokenizer, AutoModel

import os
import json
import hashlib
from pathlib import Path


class BERT(NonTrainableNewsEmbeddingRecommender):
    """
    BERT-based non-trainable news recommender.
    """

    def __init__(self, config, dataset):
        self.bert_model_name = config["bert_model_name"] if "bert_model_name" in config else "bert-base-uncased"
        self.max_length = config["bert_max_length"] if "bert_max_length" in config else 128
        self.pooling = config["bert_pooling"] if "bert_pooling" in config else "cls"
        self.batch_size = config["bert_batch_size"] if "bert_batch_size" in config else 32
        
        self.use_cache = config["bert_use_cache"] if "bert_use_cache" in config else True
        self.cache_dir = config["bert_cache_dir"] if "bert_cache_dir" in config else "~/.cache/news_bert"
        
        super().__init__(config, dataset)

    def _build_item_embeddings(self, dataset):
        cache_path = self._get_bert_cache_path(dataset)

        if self.use_cache and cache_path.exists():
            self.logger.info(f"Loading BERT item embeddings from cache: {cache_path}")
            item_embs = torch.load(cache_path, map_location="cpu")

            if item_embs.shape[0] != self.n_items:
                self.logger.warning(
                    f"Cached embeddings have shape {item_embs.shape}, "
                    f"but n_items={self.n_items}. Recomputing."
                )
            else:
                self.item_embeddings = item_embs.to(self.device)
                self.logger.info(
                    f"Loaded BERT item embeddings from cache: shape={self.item_embeddings.shape}"
                )
                return
            
        self.logger.info(
            f"Initializing BERT encoder: model={self.bert_model_name}, "
            f"max_length={self.max_length}, pooling={self.pooling}, batch_size={self.batch_size}"
        )

        tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
        model = AutoModel.from_pretrained(self.bert_model_name)
        model.to(self.device)
        model.eval()

        all_embs = []

        with torch.no_grad():
            for start in tqdm(
                range(0, self.n_items, self.batch_size),
                desc="Encoding items with BERT",
                unit="item",
            ):
                end = min(start + self.batch_size, self.n_items)

                texts = [self._get_item_text(dataset, idx) or "" for idx in range(start, end)]

                encoded = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                outputs = model(**encoded)
                hidden = outputs.last_hidden_state

                if self.pooling == "cls":
                    emb = hidden[:, 0, :]
                elif self.pooling == "mean":
                    mask = encoded["attention_mask"].unsqueeze(-1)
                    masked_hidden = hidden * mask
                    summed = masked_hidden.sum(dim=1)
                    counts = mask.sum(dim=1).clamp(min=1)
                    emb = summed / counts
                else:
                    raise ValueError(f"Unknown bert_pooling: {self.pooling}")

                all_embs.append(emb.cpu())

        item_embeddings = torch.cat(all_embs, dim=0)
        assert item_embeddings.shape[0] == self.n_items, (
            f"Expected {self.n_items} item embeddings, " f"got {item_embeddings.shape[0]}"
        )

        self.item_embeddings = item_embeddings.to(self.device)

        self.logger.info(f"BERT item embeddings built: shape={self.item_embeddings.shape}")
        
        if self.use_cache:
            try:
                torch.save(self.item_embeddings.cpu(), cache_path)
                self.logger.info(f"Saved BERT item embeddings cache to: {cache_path}")
            except Exception as e:
                self.logger.warning(f"Failed to save BERT cache to {cache_path}: {e}")
        
        


    def _get_bert_cache_path(self, dataset) -> Path:
        cache_root = Path(os.path.expanduser(self.cache_dir))
        cache_root.mkdir(parents=True, exist_ok=True)

        key = {
            "dataset": dataset.dataset_name,
            "n_items": int(self.n_items),
            "bert_model_name": self.bert_model_name,
            "max_length": int(self.max_length),
            "pooling": self.pooling,
        }

        key_str = json.dumps(key, sort_keys=True)
        digest = hashlib.md5(key_str.encode("utf-8")).hexdigest()[:8]

        filename = f"bert_items_{key['dataset']}_{digest}.pt"
        return cache_root / filename

