import numpy as np
import torch

from recbole.data.dataset import Dataset
from recbole.utils.enum_type import FeatureType, FeatureSource


class OptimizedSequentialDataset(Dataset):
    """
    Avoids eager data_augmentation to prevent OOM issues
    """

    def __init__(self, config):
        self.max_item_list_len = int(config["MAX_ITEM_LIST_LENGTH"])
        self.item_list_length_field = config["ITEM_LIST_LENGTH_FIELD"]
        self.list_suffix = config.get("LIST_SUFFIX", "_list")
        super().__init__(config)

        self._lazy_ready = False
        self._iid_sorted = None
        self._pos_in_sorted = None
        self._user_ptr = None

        assert self.inter_feat is not None, "Interaction features must be loaded"

    def _change_feat_format(self):
        super()._change_feat_format()
        # Register minimal sequential fields (properties only)
        self._sequential_presets_minimal()
        # Build lazy index for history
        self._build_lazy_time_sorted_index()

    def _sequential_presets_minimal(self):
        self._check_field("uid_field", "iid_field", "time_field")

        # This must match SequentialRecommender expectations: ITEM_ID_FIELD + LIST_SUFFIX
        self.item_id_list_field = self.iid_field + self.list_suffix  # usually 'item_id_list'
        setattr(self, f"{self.iid_field}_list_field", self.item_id_list_field)

        self.set_field_property(
            self.item_id_list_field,
            FeatureType.TOKEN_SEQ,
            FeatureSource.INTERACTION,
            self.max_item_list_len,
        )
        self.set_field_property(
            self.item_list_length_field,
            FeatureType.TOKEN,
            FeatureSource.INTERACTION,
            1,
        )

    def _build_lazy_time_sorted_index(self):
        assert self.inter_feat is not None, "Interaction features must be loaded"
        
        uid = self.inter_feat[self.uid_field].cpu().numpy()
        iid = self.inter_feat[self.iid_field].cpu().numpy()
        ts = self.inter_feat[self.time_field].cpu().numpy()

        order = np.lexsort((ts, uid))  # uid then time
        uid_s = uid[order]
        iid_s = iid[order]

        pos_in_sorted = np.empty_like(order)
        pos_in_sorted[order] = np.arange(order.shape[0], dtype=order.dtype)

        n_users = self.user_num
        counts = np.bincount(uid_s, minlength=n_users)
        user_ptr = np.zeros(n_users + 1, dtype=np.int64)
        user_ptr[1:] = np.cumsum(counts)

        self._iid_sorted = torch.from_numpy(iid_s).long()
        self._pos_in_sorted = torch.from_numpy(pos_in_sorted).long()
        self._user_ptr = torch.from_numpy(user_ptr).long()
        self._lazy_ready = True

    def get_history(self, inter_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        inter_index: (B,) indices into self.inter_feat
        returns:
            item_id_list: (B, max_len) right-aligned padded with 0
            item_length:  (B,)
        """
        if not self._lazy_ready:
            self._build_lazy_time_sorted_index()

        assert (
            self._pos_in_sorted is not None and self._iid_sorted is not None and self._user_ptr is not None
        ), "Lazy indices not built"
        assert self.inter_feat is not None, "Interaction features must be loaded"

        if not torch.is_tensor(inter_index):
            inter_index = torch.tensor(inter_index, dtype=torch.long)
        inter_index = inter_index.long().cpu()

        B = inter_index.numel()
        L = self.max_item_list_len
        item_seq = torch.zeros((B, L), dtype=torch.long)
        item_len = torch.zeros((B,), dtype=torch.long)

        pos_sorted = self._pos_in_sorted[inter_index]
        uid = self.inter_feat[self.uid_field][inter_index].cpu().long()

        for b in range(B):
            u = int(uid[b].item())
            p = int(pos_sorted[b].item())
            start = int(self._user_ptr[u].item())
            end = p
            hist_len = max(0, end - start)
            take = min(L, hist_len)
            item_len[b] = take
            if take > 0:
                hist = self._iid_sorted[end - take : end]
                item_seq[b, L - take : L] = hist

        return item_seq, item_len

    def build(self):
        ordering_args = self.config["eval_args"]["order"]
        if ordering_args != "TO":
            raise ValueError("Sequential recommendation requires eval_args.order == 'TO'")
        return super().build()
    
    def sort(self, *args, **kwargs):
        out = super().sort(*args, **kwargs)

        # Sort can invalidate lazy indices, rebuild on next use
        self._lazy_ready = False
        self._iid_sorted = None
        self._pos_in_sorted = None
        self._user_ptr = None

        return out
        
