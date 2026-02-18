from typing import List
import numpy as np
import pandas as pd

from etl.ebnerd_base_loader import EBNeRDBaseDataLoader


class EBNeRDImpressionDataLoader(EBNeRDBaseDataLoader):
    """
    Keeps one row per positive click within an impression.

    Output columns: user_id, item_id, timestamp, impression_id, neg_item_id_list, history_item_id_list
    """

    def _process_behaviors_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        timestamps = chunk["impression_time"].astype("datetime64[ns]").astype(np.int64) // 10**9

        impression_ids = chunk["impression_id"].to_numpy()
        user_ids = chunk["user_id"].to_numpy()
        ts_vals = timestamps.to_numpy()
        inview_lists = chunk["article_ids_inview"].tolist()
        clicked_lists = chunk["article_ids_clicked"].tolist()

        has_history = "article_id_fixed" in chunk.columns
        if has_history:
            histories = chunk["article_id_fixed"].tolist()

        out_user: List = []
        out_pos_item: List = []
        out_ts: List[int] = []
        out_imp: List[int] = []
        out_negs: List[str] = []
        out_hist: List[str] = []

        for i in range(len(chunk)):
            imp_id = int(impression_ids[i])
            u = user_ids[i]
            ts = int(ts_vals[i])

            inview = inview_lists[i]
            clicked = clicked_lists[i]

            if clicked is None or (hasattr(clicked, '__len__') and len(clicked) == 0):
                continue

            clicked_set = set(clicked)
            negatives = (
                [str(a) for a in inview if a not in clicked_set]
                if inview is not None and hasattr(inview, '__len__') and len(inview) > 0
                else []
            )

            # Build history string from the separate history file (merged in base loader)
            hist_str = ""
            if has_history:
                h = histories[i]
                if h is not None and not (isinstance(h, float) and np.isnan(h)):
                    hist_str = " ".join(str(a) for a in h) # type: ignore

            for pos_item in clicked:
                out_user.append(u)
                out_pos_item.append(pos_item)
                out_ts.append(ts)
                out_imp.append(imp_id)
                out_negs.append(" ".join(negatives))
                out_hist.append(hist_str)

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
