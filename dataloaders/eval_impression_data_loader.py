import numpy as np
import torch

from recbole.data.interaction import Interaction

from dataloaders.base_impression_data_loader import ImpressionDataLoader
from datasets.opt_sequential_dataset import OptimizedSequentialDataset


class EvalImpressionDataLoader(ImpressionDataLoader):

    def __init__(self, config, dataset, sampler=None, shuffle=False):
        # Evaluation should be deterministic, and impressions should not be shuffled.
        if shuffle:
            raise ValueError("EvalImpressionDataLoader should not shuffle evaluation data.")

        self.neg_k = int(config.get("eval_neg_sample_num", 4))
        self.shuffle_within_impression = bool(config.get("eval_shuffle_within_impression", False))

        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        batch_size = int(self.config.get("eval_batch_size", self.config.get("test_batch_size", 256)))
        self.step = batch_size
        self.set_batch_size(batch_size)

    @torch.no_grad()
    def collate_fn(self, index):
        index = np.array(index)
        data = self._dataset[index]
        transformed: Interaction = self.transform(self._dataset, data)

        assert isinstance(self._dataset, OptimizedSequentialDataset), \
            "This eval loader expects OptimizedSequentialDataset"

        item_seq, item_len = self._dataset.get_history(torch.tensor(index, dtype=torch.long))

        valid = (item_len > 0)
        if valid.any() and (not valid.all()):
            item_seq = item_seq[valid]
            item_len = item_len[valid]
            transformed = transformed[valid]

        transformed.update(Interaction({
            self._dataset.item_id_list_field: item_seq,
            self._dataset.item_list_length_field: item_len,
        }))

        cand = transformed[self.impr_neg_field]
        assert isinstance(cand, torch.Tensor) and cand.dim() == 2, \
            "Impression negative field must be a (B, M) torch.Tensor"

        neg_items = self._sample_k(cand, self.neg_k)              # (B, K)
        pos = transformed[self.iid_field].view(-1, 1)             # (B, 1)
        cand_items = torch.cat([pos, neg_items], dim=1)           # (B, L) where L=1+K

        B, L = cand_items.shape
        device = cand_items.device

        row_idx = torch.arange(B, device=device).repeat_interleave(L)  # (B*L,)

        positive_u = torch.arange(B, device=device, dtype=torch.long)                 # (B,)
        positive_i = transformed[self.iid_field].to(device=device).long()  # (B,)

        flat_item = cand_items.reshape(-1)  # (B*L,)

        flat_data = {}
        for f in transformed.interaction.keys():
            v = transformed[f]
            if torch.is_tensor(v) and v.dim() >= 1 and v.size(0) == B:
                flat_data[f] = v.repeat_interleave(L, dim=0)
            else:
                flat_data[f] = v

        flat_data[self.iid_field] = flat_item

        interaction_flat = Interaction(flat_data)
        return interaction_flat, row_idx, positive_u, positive_i


