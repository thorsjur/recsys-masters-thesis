import numpy as np
import torch
import pandas as pd

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

        self.impr_hist_field = config["impr_hist_field"]

        super().__init__(config)

        assert self.inter_feat is not None, "Interaction features must be loaded"
        self.inter_feat: pd.DataFrame
        

    # A kinda hacky way to re-parse only the impression history field with newest-truncation,
    # since we want to keep the last L items, not the first L items (history is time-ordered).
    # Adapted from Dataset._load_feat
    def _load_feat(self, filepath, source):
        df = super()._load_feat(filepath, source)
        if df is None:
            return None
        
        field = self.impr_hist_field
        if field not in df.columns or self.field2type.get(field) != FeatureType.TOKEN_SEQ:
            self.logger.warning(f"Field '{field}' not found or not TOKEN_SEQ, skipping re-parse.")
            return df

        field_separator = self.config["field_separator"]
        encoding = self.config["encoding"]
        seq_separator = self.config["seq_separator"]

        # Re-read only this column as raw string, then parse
        raw = pd.read_csv(
            filepath,
            delimiter=field_separator,
            usecols=[f"{field}:token_seq"],
            dtype={f"{field}:token_seq": str},
            encoding=encoding,
            engine="python",
        )
        raw.columns = [field]
        raw[field].fillna(value="", inplace=True)
        parsed = [
            np.array(list(filter(None, s.split(seq_separator))))
            for s in raw[field].values
        ]

        # Apply newest truncation if seq_len configured
        L = None
        if self.config["seq_len"] and field in self.config["seq_len"]:
            L = int(self.config["seq_len"][field])
            parsed = [seq[-L:] if len(seq) > L else seq for seq in parsed]

        df[field] = parsed

        max_seq_len = max(map(len, df[field].values)) if len(df[field]) else 0
        if L is not None:
            self.field2seqlen[field] = min(L, max_seq_len)
        else:
            self.field2seqlen[field] = max_seq_len

        return df

    def _right_align_token_seq(self, field: str, padding_value: int = 0):
        """
        Right-align the item_seq field
        """
        hist = self.inter_feat[field]  # (N, L)
        if not isinstance(hist, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor for field '{field}', got {type(hist)}")
        
        device = hist.device
        N, L = hist.shape
        

        # Lengths per row (assumes padding_value is pad)
        if padding_value == 0:
            lens = torch.count_nonzero(hist, dim=1)
        else:
            lens = (hist != padding_value).sum(dim=1)
        lens = lens.to(torch.long) # type: ignore

        # Build gather indices
        ar = torch.arange(L, device=device).unsqueeze(0)   # (1, L)
        pos = lens.unsqueeze(1) - L + ar                   # (N, L)
        mask = pos >= 0
        pos = pos.clamp(min=0)

        gathered = hist.gather(1, pos)
        out = gathered.masked_fill(~mask, padding_value)

        self.inter_feat[field] = out

    def _change_feat_format(self):
        super()._change_feat_format()
        # Register minimal sequential fields (properties only)
        self._sequential_presets_minimal()
        
        self._right_align_token_seq(self.impr_hist_field)

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

    @torch.no_grad()
    def get_history(self, inter_index) -> tuple[torch.Tensor, torch.Tensor]:
        """
        inter_index: (B,) indices into self.inter_feat
        returns:
            item_id_list: (B, L) right-aligned padded with 0
            item_length:  (B,)
        """
        assert self.inter_feat is not None, "Interaction features must be loaded"
        assert self.impr_hist_field in self.inter_feat, (
            f"'{self.impr_hist_field}' not found in inter_feat. "
        )
        
        # Ensure LongTensor index on same device as inter_feat storage
        if not torch.is_tensor(inter_index):
            inter_index = torch.as_tensor(inter_index, dtype=torch.long)
        else:
            inter_index = inter_index.to(dtype=torch.long)

        hist_all: torch.Tensor = self.inter_feat[self.impr_hist_field]
        device = hist_all.device
        inter_index = inter_index.to(device=device)

        # Fetch histories in one shot
        hist = torch.index_select(hist_all, dim=0, index=inter_index)  # (B, M)

        L = self.max_item_list_len
        M = hist.size(1)

        item_len = torch.count_nonzero(hist, dim=1).to(torch.long)

        # Take at most L (should be aligned with the already set seq_len)
        take = torch.minimum(item_len, torch.tensor(L, device=device, dtype=torch.long))

        # Assumes right-aligned sequences
        if M == L:
            out = hist
        elif M > L:
            out = hist[:, -L:]
        else:
            # M < L: left-pad with zeros to make (B, L)
            out = hist.new_zeros((hist.size(0), L))
            out[:, -M:] = hist

        # If M > L, lengths need clamping to L
        item_len = take
        return out.to(torch.long), item_len


    def build(self):
        ordering_args = self.config["eval_args"]["order"]
        if ordering_args != "TO":
            raise ValueError("Sequential recommendation requires eval_args.order == 'TO'")
        return super().build()
