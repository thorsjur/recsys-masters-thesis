import pandas as pd
import numpy as np
from etl.mind_base_loader import MINDBaseDataLoader


class MINDDataLoader(MINDBaseDataLoader):
    """
    Explode impressions into atomic interactions with labels.
    Output columns: user_id, item_id, timestamp, label, impression_id
    """

    def _process_behaviors_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        timestamps = pd.to_datetime(chunk["time_str"], format="%m/%d/%Y %I:%M:%S %p").astype(np.int64) // 10**9

        impressions_split = chunk["impressions"].fillna("").str.split(" ")
        impression_lengths = impressions_split.str.len().astype(np.int64)

        df_exploded = pd.DataFrame(
            {
                "user_id": chunk["user_id"].values.repeat(impression_lengths),
                "timestamp": timestamps.values.repeat(impression_lengths),
                "impression_id": chunk["impression_id"].values.repeat(impression_lengths),
                "impressions": np.concatenate(impressions_split.values),
            }
        )

        split_data = df_exploded["impressions"].str.split("-", n=1, expand=True)
        split_data.columns = ["item_id", "label"]
        df_exploded["item_id"] = split_data["item_id"]
        df_exploded["label"] = split_data["label"].astype(np.float32)

        df_exploded.drop(columns=["impressions"], inplace=True)
        return df_exploded[["user_id", "item_id", "timestamp", "label", "impression_id"]]
