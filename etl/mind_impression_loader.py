from typing import List, Tuple
import numpy as np
import pandas as pd

from etl.mind_base_loader import MINDBaseDataLoader


class MINDImpressionDataLoader(MINDBaseDataLoader):
    """
    Keeps one row per positive click within an impression.

    Output columns: user_id, item_id, timestamp, impression_id, neg_item_id_list
    """

    @staticmethod
    def _parse_impression_token(token: str, impression_id: int) -> Tuple[str, int]:
        """
        Parse a token like 'N123-1' into (item_id, label_int).
        """
        try:
            item, lab = token.rsplit("-", 1)
        except ValueError as e:
            raise ValueError(f"Malformed impression token '{token}' for impression_id={impression_id}.") from e

        if lab not in ("0", "1"):
            raise ValueError(f"Unexpected label '{lab}' in token '{token}' for impression_id={impression_id}.")

        return item, (1 if lab == "1" else 0)

    def _process_behaviors_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        timestamps = pd.to_datetime(chunk["time_str"], format="%m/%d/%Y %I:%M:%S %p").astype(np.int64) // 10**9

        impressions_split = chunk["impressions"].fillna("").str.split(" ")

        user_ids = chunk["user_id"].to_numpy()
        impression_ids = chunk["impression_id"].to_numpy()
        hists = chunk["history"].to_numpy()
        ts_vals = timestamps.to_numpy()

        out_user: List[str] = []
        out_pos_item: List[str] = []
        out_ts: List[int] = []
        out_imp: List[int] = []
        out_negs: List[str] = []
        out_hist: List[str] = []

        for i, tokens in enumerate(impressions_split.tolist()):
            imp_id = int(impression_ids[i])
            u = user_ids[i]
            ts = int(ts_vals[i])
            h = hists[i]

            tokens = [t for t in tokens if t]
            if not tokens:
                raise ValueError(f"Empty impressions for impression_id={imp_id} (user_id={u}).")

            positives: List[str] = []
            negatives: List[str] = []

            for t in tokens:
                item, lab = self._parse_impression_token(t, imp_id)
                if lab == 1:
                    positives.append(item)
                else:
                    negatives.append(item)

            if len(positives) == 0:
                raise AssertionError(
                    f"Expected at least 1 positive in impression_id={imp_id}, found 0. Tokens={tokens}"
                )

            for pos_item in positives:
                out_user.append(u)
                out_pos_item.append(pos_item)
                out_ts.append(ts)
                out_imp.append(imp_id)
                out_negs.append(" ".join(negatives))
                out_hist.append(h)

        out = pd.DataFrame(
            {
                "user_id": np.asarray(out_user, dtype=object),
                "item_id": np.asarray(out_pos_item, dtype=object),
                "timestamp": np.asarray(out_ts, dtype=np.int64),
                "impression_id": np.asarray(out_imp, dtype=np.int64),
                "neg_item_id_list": out_negs,
                "history_item_id_list": out_hist,
            }
        )

        return out[["user_id", "item_id", "timestamp", "impression_id", "neg_item_id_list", "history_item_id_list"]]
    
    def _finalize_interactions_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return super()._finalize_interactions_df(df)
