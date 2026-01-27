import numpy as np
import torch
from logging import getLogger

from recbole.data.dataloader.abstract_dataloader import AbstractDataLoader
from recbole.data.interaction import Interaction


class ImpressionDataLoader(AbstractDataLoader):
    """
    DataLoader following NRMS impression-based training

    Required fields in train inter_feat:
      - impr_neg_field: padded list of impression negatives (B, C), 0 padding
    """

    def __init__(self, config, dataset, sampler=None, shuffle=False):
        self.logger = getLogger()
        if shuffle is False:
            shuffle = True
            self.logger.warning("ImpressionDataLoader should shuffle training data.")

        self.sample_size = len(dataset)

        self.iid_field = dataset.iid_field
        self.neg_prefix = config["NEG_PREFIX"]
        self.neg_item_id_field = self.neg_prefix + self.iid_field

        self.impr_neg_field = config["impr_neg_field"]
        self.padding_idx = int(config.get("padding_idx", 0))
        self.neg_k = int(config.get("neg_sample_num", 4))

        self.cand_field = config.get("cand_field", "cand_item_id")  # (B, 1+K)
        self.pos_index_field = config.get("pos_index_field", "pos_index")  # (B,)
        self.shuffle_within_impression = bool(config.get("shuffle_within_impression", True))

        if self.impr_neg_field not in dataset.inter_feat:
            raise KeyError(f"Missing required field '{self.impr_neg_field}' in train inter_feat.")

        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        batch_size = self.config["train_batch_size"]
        self.step = batch_size
        self.set_batch_size(batch_size)

    @torch.no_grad()
    def _sample_k(self, cand: torch.Tensor, k: int) -> torch.Tensor:
        """
        cand: (B, C) long, padded with padding_idx
        returns: (B, k)
        """
        B, C = cand.shape
        device = cand.device
        valid_mask = cand != self.padding_idx
        valid_counts = valid_mask.sum(dim=1)

        out = torch.empty((B, k), dtype=cand.dtype, device=device)
        for b in range(B):
            cnt = int(valid_counts[b].item())
            if cnt <= 0:
                out[b].fill_(self.padding_idx)
                continue
            pool = cand[b, valid_mask[b]]
            if cnt >= k:
                idx = torch.randperm(cnt, device=device)[:k]
                out[b] = pool[idx]
            else:
                # This is consistent with the code as provided by the authors,
                # see https://github.com/wuch15/EMNLP2019-NRMS/blob/master/Baseline-NRMS.ipynb
                reps = k // cnt + 1
                expanded = pool.repeat(reps)
                idx = torch.randperm(expanded.numel(), device=device)[:k]
                out[b] = expanded[idx]
        return out

    def collate_fn(self, index):
        index = np.array(index)
        data = self._dataset[index]
        transformed = self.transform(self._dataset, data)

        # Sample negatives from impression list
        cand = transformed[self.impr_neg_field]
        if not torch.is_tensor(cand):
            cand = torch.tensor(cand, dtype=torch.long)
        cand = cand.long()
        neg_items = self._sample_k(cand, self.neg_k)  # (B, K)

        # Provide NEG_ITEM_ID for compatibility (optional)
        transformed.update(Interaction({self.neg_item_id_field: neg_items}))

        # Build candidate list (pos + K neg) and optionally shuffle to avoid position bias
        pos = transformed[self.iid_field].view(-1, 1)  # (B,1)
        all_items = torch.cat([pos, neg_items], dim=1)  # (B,1+K)
        B, L = all_items.shape

        if self.shuffle_within_impression:
            perm = torch.rand(B, L, device=all_items.device).argsort(dim=1)
            shuffled = torch.gather(all_items, 1, perm)
            pos_index = (perm == 0).nonzero(as_tuple=False)[:, 1].long()  # where original pos moved to
        else:
            shuffled = all_items
            pos_index = torch.zeros((B,), dtype=torch.long, device=all_items.device)

        transformed.update(
            Interaction({self.cand_field: shuffled, self.pos_index_field: pos_index})  # (B, 1+K)  # (B,)
        )
        return transformed
