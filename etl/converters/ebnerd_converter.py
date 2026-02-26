from typing import Dict
from etl.converters.base_converter import BaseAtomicConverter


class EBNeRDAtomicConverter(BaseAtomicConverter):

    def __init__(self, config, df_inter_loaded=None, df_item_loaded=None):
        super().__init__(config, df_inter_loaded, df_item_loaded)

    @property
    def inter_fields(self) -> Dict[str, str]:
        return {
            'user_id': 'user_id:token',
            'item_id': 'item_id:token',
            'label': 'label:float',
            'timestamp': 'timestamp:float',
            'impression_id': 'impression_id:token',
        }

    @property
    def item_fields(self) -> Dict[str, str]:
        return {
            'item_id': 'item_id:token',
            'category_str': 'category:token',
            'title': 'title:token_seq',
            'subtitle': 'abstract:token_seq',
        }

    def load_inter_df(self):
        if self._df_inter is not None:
            return self._df_inter
        raise ValueError("Interaction DataFrame was not provided.")

    def load_item_df(self):
        if self._df_item is not None:
            return self._df_item
        raise ValueError("Item DataFrame was not provided.")
