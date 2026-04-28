from typing import Dict
from etl.converters.ebnerd_converter import EBNeRDAtomicConverter


class EBNeRDImpressionAtomicConverter(EBNeRDAtomicConverter):
    """
    Only difference from EBNeRDAtomicConverter is the addition of 'neg_item_id_list' field and removal of 'label' field.
    """

    def __init__(self, config, df_inter_loaded=None, df_item_loaded=None):
        super().__init__(config, df_inter_loaded, df_item_loaded)

    @property
    def inter_fields(self) -> Dict[str, str]:
        return {
            "user_id": "user_id:token",
            "item_id": "item_id:token",
            "timestamp": "timestamp:float",
            "impression_id": "impression_id:token",
            "neg_item_id_list": "neg_item_id_list:token_seq",
            "history_item_id_list": "history_item_id_list:token_seq",
        }
