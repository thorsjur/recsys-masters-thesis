import numpy as np
import pandas as pd

from etl.ebnerd_base_loader import EBNeRDBaseDataLoader


class EBNeRDDataLoader(EBNeRDBaseDataLoader):
    """
    Explode impressions into atomic interactions with labels.
    Output columns: user_id, item_id, timestamp, label, impression_id
    """

    def _process_behaviors_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        timestamps = chunk["impression_time"].astype("datetime64[ns]").astype(np.int64) // 10**9

        impression_ids = chunk["impression_id"].to_numpy()
        user_ids = chunk["user_id"].to_numpy()
        ts_vals = timestamps.to_numpy()
        inview_lists = chunk["article_ids_inview"].tolist()
        clicked_lists = chunk["article_ids_clicked"].tolist()

        out_user = []
        out_item = []
        out_ts = []
        out_label = []
        out_imp = []

        for i in range(len(chunk)):
            imp_id = int(impression_ids[i])
            u = user_ids[i]
            ts = int(ts_vals[i])

            inview = inview_lists[i]
            clicked = clicked_lists[i]

            if inview is None or (hasattr(inview, "__len__") and len(inview) == 0):
                continue

            clicked_set = (
                set(clicked) if clicked is not None and hasattr(clicked, "__len__") and len(clicked) > 0 else set()
            )

            for article_id in inview:
                out_user.append(u)
                out_item.append(article_id)
                out_ts.append(ts)
                out_label.append(1.0 if article_id in clicked_set else 0.0)
                out_imp.append(imp_id)

        return pd.DataFrame(
            {
                "user_id": np.asarray(out_user, dtype=object),
                "item_id": np.asarray(out_item, dtype=object),
                "timestamp": np.asarray(out_ts, dtype=np.int64),
                "label": np.asarray(out_label, dtype=np.float32),
                "impression_id": np.asarray(out_imp, dtype=np.int64),
            }
        )
