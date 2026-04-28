from typing import Optional
import numpy as np
import torch
import pandas as pd
from logging import getLogger

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
        parsed = [np.array(list(filter(None, s.split(seq_separator)))) for s in raw[field].values]

        # Apply newest truncation if seq_len configured
        L = None
        if self.config["seq_len"] and field in self.config["seq_len"]:
            L = int(self.config["seq_len"][field])
            parsed = [seq[-L:] if len(seq) > L else seq for seq in parsed]

        df[field] = parsed # type: ignore

        max_seq_len = max(map(len, df[field].values)) if len(df[field]) else 0
        if L is not None:
            self.field2seqlen[field] = min(L, max_seq_len)
        else:
            self.field2seqlen[field] = max_seq_len

        return df

    def _right_align_token_seq(self, field: str, padding_value: int = 0, len_field: Optional[str] = None):
        """
        Right-align a TOKEN_SEQ field and (optionally) store per-row lengths in len_field.
        """
        hist = self.inter_feat[field]
        if not isinstance(hist, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor for field '{field}', got {type(hist)}")

        device = hist.device
        N, L = hist.shape

        # Lengths per row
        if padding_value == 0:
            lens = torch.count_nonzero(hist, dim=1)
        else:
            lens = (hist != padding_value).sum(dim=1)
        lens = lens.to(torch.long)  # type: ignore

        # Build gather indices for right-alignment
        ar = torch.arange(L, device=device).unsqueeze(0)  # (1, L)
        pos = lens.unsqueeze(1) - L + ar  # (N, L)
        mask = pos >= 0
        pos = pos.clamp(min=0)

        gathered = hist.gather(1, pos)
        out = gathered.masked_fill(~mask, padding_value)

        self.inter_feat[field] = out # type: ignore

        if len_field is not None:
            self.inter_feat[len_field] = lens.clamp_max(self.max_item_list_len) # type: ignore

    def _change_feat_format(self):
        super()._change_feat_format()
        # Register minimal sequential fields (properties only)
        self._sequential_presets_minimal()

        self.impr_hist_len_field = f"{self.impr_hist_field}_len"
        self._right_align_token_seq(self.impr_hist_field, padding_value=0, len_field=self.impr_hist_len_field)

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
        assert self.inter_feat is not None
        assert (
            self.impr_hist_field in self.inter_feat and self.impr_hist_len_field in self.inter_feat
        ), f"Impression history fields '{self.impr_hist_field}' and '{self.impr_hist_len_field}' must be present in inter_feat."

        hist_all: torch.Tensor = self.inter_feat[self.impr_hist_field]
        len_all: torch.Tensor = self.inter_feat[self.impr_hist_len_field]  # type: ignore
        device = hist_all.device

        if not torch.is_tensor(inter_index):
            inter_index = torch.as_tensor(inter_index, dtype=torch.long, device=device)
        else:
            inter_index = inter_index.to(device=device, dtype=torch.long, non_blocking=True)

        hist = hist_all.index_select(0, inter_index)  # (B, M)
        item_len = len_all.index_select(0, inter_index)  # (B,)

        B, M = hist.shape
        L = self.max_item_list_len

        if M == L:
            out = hist
        elif M > L:
            out = hist[:, -L:]
            item_len = item_len.clamp_max(L)
        else:
            out = hist.new_zeros((B, L))
            out[:, -M:] = hist

        if out.dtype != torch.long:
            out = out.long()

        return out, item_len

    def build(self):
        ordering_args = self.config["eval_args"]["order"]
        if ordering_args != "TO":
            raise ValueError("Sequential recommendation requires eval_args.order == 'TO'")

        datasets = super().build()

        # Apply per-phase impression-negatives filtering on the split datasets.
        phase_names = ["train", "val", "test"]
        for ds, phase in zip(datasets, phase_names):
            self._filter_by_impression_negatives(ds, phase)

        return datasets

    def _filter_by_impression_negatives(self, dataset, phase: str):
        """Drop interactions whose impression-negative count falls outside the
        configured ``impression_negatives_num_interval``.

        Called after ``build()`` splits the data, so each phase can have its
        own interval.
        """
        logger = getLogger()
        raw = self.config.get("impression_negatives_num_interval", None)
        if raw is None:
            return

        if isinstance(raw, dict):
            interval_str = raw.get(phase, None)
        else:
            interval_str = raw

        interval = self._parse_intervals_str(interval_str)
        if interval is None:
            return

        impr_neg_field = self.config["impr_neg_field"]
        padding_idx = int(self.config.get("padding_idx", 0))

        neg_tensor = dataset.inter_feat[impr_neg_field]
        neg_counts = (neg_tensor != padding_idx).sum(dim=1)  # (N,)

        keep_mask = torch.tensor(
            [self._within_intervals(float(c), interval) for c in neg_counts],
            dtype=torch.bool,
        )
        n_before = len(dataset)
        n_drop = int((~keep_mask).sum().item())

        if n_drop == 0:
            logger.info(
                f"{phase.upper()}: impression_negatives_num_interval={interval_str} "
                f"\u2013 all {n_before} impressions pass, nothing filtered."
            )
            return

        keep_idx = keep_mask.nonzero(as_tuple=False).squeeze(1).tolist()
        dataset.inter_feat = dataset.inter_feat[keep_idx]

        logger.info(
            f"{phase.upper()}: impression_negatives_num_interval={interval_str} "
            f"\u2013 dropped {n_drop}/{n_before} impressions, {len(dataset)} remaining."
        )
