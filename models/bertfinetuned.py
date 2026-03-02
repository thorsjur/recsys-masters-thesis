import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.utils import InputType
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model, TaskType

logger = logging.getLogger(__name__)

class MeanPoolUserEncoder(nn.Module):
    """Simple mean-pooling user encoder over history item vectors."""

    def __init__(self, news_dim: int):
        super().__init__()
        self.out_dim = news_dim

    def encode(self, seq_vectors: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
        """seq_vectors: (B, T, D), seq_mask: (B, T) bool → (B, D)"""
        mask = seq_mask.unsqueeze(-1).float()  # (B, T, 1)
        return (seq_vectors * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


class BERTFinetuned(SequentialRecommender):
    """BERT-based trainable news recommender.

    Encodes item text through a LoRA-adapted BERT model (parameter-efficient
    fine-tuning), aggregates user history via mean pooling, and trains with
    softmax cross-entropy over impression candidates.
    """

    input_type = InputType.LISTWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # BERT config
        self.bert_model_name = config.get("bert_model_name", "bert-base-uncased")
        self.max_length = int(config.get("bert_max_length", 128))
        self.pooling = config.get("bert_pooling", "cls")
        self.bert_batch_size = int(config.get("bert_batch_size", 64))
        self.grad_batch_size = int(config.get("bert_grad_batch_size", 32))

        # Text fields (for extracting item text from RecBole dataset)
        self.title_field = config.get("title_field", "title")
        self.abstract_field = config.get("abstract_field", "abstract")
        self.use_abstract = config.get("use_abstract", True)

        # Impression fields (same convention as NRMS)
        self.cand_field = config.get("cand_field", "cand_item_id")
        self.pos_index_field = config.get("pos_index_field", "pos_index")

        # Whether to use mixed-precision (AMP) inside the model
        self.use_amp = bool(config.get("enable_amp", False))

        # LoRA config
        lora_r = int(config.get("lora_r", 8))
        lora_alpha = int(config.get("lora_alpha", 16))
        lora_dropout = float(config.get("lora_dropout", 0.1))

        # BERT encoder wrapped with LoRA
        tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
        base_bert = AutoModel.from_pretrained(self.bert_model_name)
        news_dim = base_bert.config.hidden_size

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
            target_modules=["query", "value"],
        )
        self.bert = get_peft_model(base_bert, lora_config)

        self.bert.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # User encoder (mean pooling over history)
        self.user_encoder = MeanPoolUserEncoder(news_dim=news_dim)
        self.hidden_size = self.user_encoder.out_dim

        # Pre-tokenize all items and store as compact buffers
        self._pre_tokenize_items(dataset, tokenizer)
        del tokenizer  # free ~500 MB tokenizer memory; it's not needed again

        # Item embedding cache – avoids running BERT on every history item.
        self._cache_dirty = True
        self._refresh_item_cache()


    def _get_item_text(self, dataset, item_idx: int) -> str:
        """Reconstruct item text from RecBole's tokenised feature fields."""
        item_feat = dataset.get_item_feature()
        tokens = []
        fields = [self.title_field] + ([self.abstract_field] if self.use_abstract else [])
        for field in fields:
            if field not in item_feat:
                continue
            ids = item_feat[field][item_idx]
            if torch.is_tensor(ids):
                ids = ids.cpu().numpy()
            tokens.extend(dataset.id2token(field, [int(t)])[0] for t in ids if int(t) != 0)
        return " ".join(tokens)

    def _pre_tokenize_items(self, dataset, tokenizer) -> None:
        """Tokenize every item with the BERT tokenizer once at init."""
        n_items = dataset.item_num
        texts = [
            self._get_item_text(dataset, i) or ""
            for i in tqdm(range(n_items), desc="Extracting item texts", leave=False)
        ]
        encoded = tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        # Trim trailing pad columns that are all-zero across every item.
        # This reduces the sequence dimension from max_length to the actual
        # longest item, which can be significantly shorter.
        max_real_len = int(attention_mask.sum(dim=1).max().item())
        if max_real_len < input_ids.size(1):
            logger.info(
                "Trimmed item tokenisation from %d to %d tokens",
                input_ids.size(1), max_real_len,
            )
            input_ids = input_ids[:, :max_real_len].contiguous()
            attention_mask = attention_mask[:, :max_real_len].contiguous()

        self.register_buffer("item_input_ids", input_ids.to(torch.int32))
        self.register_buffer("item_attention_mask", attention_mask.to(torch.int32))


    def train(self, mode: bool = True):
        result = super().train(mode)
        # Mark the cache stale whenever we enter training mode
        if mode:
            self._cache_dirty = True
        return result

    def _refresh_item_cache(self) -> None:
        """Recompute embeddings for all items using current LoRA weights.
        """
        vecs = []
        was_training = self.bert.training
        self.bert.eval()
        n_batches = math.ceil(len(self.item_input_ids) / self.bert_batch_size)
        with torch.no_grad():
            for start in tqdm(
                range(0, len(self.item_input_ids), self.bert_batch_size),
                total=n_batches, desc="Refreshing item cache", leave=False,
            ):
                ids = self.item_input_ids[start : start + self.bert_batch_size].long()
                mask = self.item_attention_mask[start : start + self.bert_batch_size].long()
                with torch.autocast("cuda", enabled=self.use_amp):
                    out = self.bert(input_ids=ids, attention_mask=mask)
                    pooled = self._pool(out.last_hidden_state, mask)
                vecs.append(pooled.half().detach())  # store as FP16
        if was_training:
            self.bert.train()
        self._item_cache = torch.cat(vecs, dim=0)
        self._cache_dirty = False


    def _pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return last_hidden_state[:, 0, :]
        # mean pooling
        mask = attention_mask.unsqueeze(-1)
        return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    def encode_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Run items through BERT with LoRA (with gradients).
        item_ids: any shape -> (*shape, D)
        """
        flat = item_ids.reshape(-1)
        uniq, inv = torch.unique(flat, return_inverse=True)

        vecs = []
        n_batches = math.ceil(len(uniq) / self.grad_batch_size)
        for start in tqdm(
            range(0, len(uniq), self.grad_batch_size),
            total=n_batches, desc="Encoding candidates", leave=False,
        ):
            batch = uniq[start : start + self.grad_batch_size]
            input_ids = self.item_input_ids[batch].long()
            mask = self.item_attention_mask[batch].long()
            with torch.autocast("cuda", enabled=self.use_amp):
                out = self.bert(input_ids=input_ids, attention_mask=mask)
                pooled = self._pool(out.last_hidden_state, mask)
            vecs.append(pooled)

        all_vecs = torch.cat(vecs, dim=0)
        return all_vecs[inv].view(*item_ids.shape, -1)

    def encode_items_cached(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Look up pre-computed item embeddings (no BERT forward pass).
        """
        if self._cache_dirty:
            self._refresh_item_cache()
        return self._item_cache.to(item_ids.device, dtype=torch.float32)[item_ids]

    def encode_user(self, item_seq: torch.Tensor) -> torch.Tensor:
        """Encode user history using cached item embeddings (fast)."""
        seq_mask = item_seq != 0
        seq_vecs = self.encode_items_cached(item_seq)
        return self.user_encoder.encode(seq_vecs, seq_mask)


    def forward(self, item_seq: torch.Tensor, item_id: torch.Tensor) -> torch.Tensor:
        u = self.encode_user(item_seq)
        r = self.encode_items_cached(item_id)
        return torch.sum(u * r, dim=-1)

    def calculate_loss(self, interaction) -> torch.Tensor:
        item_seq = interaction[self.ITEM_SEQ]
        cand_item_ids = interaction[self.cand_field]
        pos_index = interaction[self.pos_index_field].long()

        u = self.encode_user(item_seq)
        cand_vecs = self.encode_items(cand_item_ids)
        logits = torch.einsum("bd,bcd->bc", u, cand_vecs)

        loss = F.cross_entropy(logits, pos_index)
        self._cache_dirty = True
        return loss

    def predict(self, interaction) -> torch.Tensor:
        item_seq = interaction[self.ITEM_SEQ]
        item_id = interaction[self.ITEM_ID]
        return self.forward(item_seq, item_id)

    def full_sort_predict(self, interaction) -> torch.Tensor:
        raise NotImplementedError("Not used, so we don't implement full-sort prediction.")
