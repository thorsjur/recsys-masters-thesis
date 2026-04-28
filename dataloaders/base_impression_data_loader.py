import torch
from logging import getLogger

from recbole.data.dataloader.abstract_dataloader import AbstractDataLoader


class ImpressionDataLoader(AbstractDataLoader):
    """
    DataLoader following NRMS impression-based training/eval, note that it requires
    a prebuilt impression negative field in the interaction features.
    """

    def __init__(self, config, dataset, sampler=None, shuffle=False):
        self.logger = getLogger()

        self.sample_size = len(dataset)

        self.iid_field = dataset.iid_field

        self.impr_neg_field = config["impr_neg_field"]
        self.padding_idx = int(config.get("padding_idx", 0))

        self.cand_field = config.get("cand_field", "cand_item_id")  # (B, 1+K)
        self.pos_index_field = config.get("pos_index_field", "pos_index")  # (B,)
        
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
            count = int(valid_counts[b].item())
            if count <= 0:
                raise ValueError(f"Cannot sample negatives from empty impression list: row {b}")
            pool = cand[b, valid_mask[b]]
            if count >= k:
                idx = torch.randperm(count, device=device)[:k]
                out[b] = pool[idx]
            else:
                # This is consistent with the code as provided by the NRMS authors,
                # see https://github.com/wuch15/EMNLP2019-NRMS/blob/master/Baseline-NRMS.ipynb
                reps = k // count + 1
                expanded = pool.repeat(reps)
                idx = torch.randperm(expanded.numel(), device=device)[:k]
                out[b] = expanded[idx]
        return out

    
