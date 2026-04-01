from dataloaders.base_impression_data_loader import ImpressionDataLoader
import numpy as np
import torch

from recbole.data.interaction import Interaction

from custom_datasets.opt_sequential_dataset import OptimizedSequentialDataset


class TrainImpressionDataLoader(ImpressionDataLoader):
    
    def __init__(self, config, dataset, sampler=None, shuffle=False):
        if shuffle is False:
            shuffle = True
            self.logger.warning("ImpressionDataLoader should shuffle training data.")

        self.neg_k = int(config.get("neg_sample_num", 4))
        self.shuffle_within_impression = bool(config.get("shuffle_within_impression", True))
        
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        batch_size = self.config["train_batch_size"]
        self.step = batch_size
        self.set_batch_size(batch_size)
        
    def collate_fn(self, index):
        index = np.array(index)
        data = self._dataset[index]
        transformed: Interaction = self.transform(self._dataset, data)
        
        assert isinstance(self._dataset, OptimizedSequentialDataset), "TrainImpressionDataLoader only works with OptimizedSequentialDataset"
        
        item_seq, item_len = self._dataset.get_history(index)
        
        transformed.update(Interaction({
            self._dataset.item_id_list_field: item_seq,
            self._dataset.item_list_length_field: item_len,
        }))

        # Sample negatives from impression list
        cand = transformed[self.impr_neg_field]
        assert isinstance(cand, torch.Tensor), "Impression negative field must be a torch.Tensor"
        
        neg_items = self._sample_k(cand, self.neg_k)  # (B, K)

        # Build candidate list (pos + K neg) and shuffle to avoid position bias
        pos = transformed[self.iid_field].view(-1, 1)  # (B,1)
        all_items = torch.cat([pos, neg_items], dim=1)  # (B,1+K)
        B, L = all_items.shape
        
        if self.shuffle_within_impression:
            perm = torch.rand(B, L, device=all_items.device).argsort(dim=1)
            shuffled = torch.gather(all_items, 1, perm)
            pos_index = (perm == 0).nonzero(as_tuple=False)[:, 1].long()
        else:
            shuffled = all_items
            pos_index = torch.zeros((B,), dtype=torch.long, device=all_items.device)

        transformed.update(
            Interaction({self.cand_field: shuffled, self.pos_index_field: pos_index})
        )
        
        return transformed