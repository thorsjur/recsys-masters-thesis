from typing import Dict
from etl.converters.mind_converter import MINDAtomicConverter

class MINDImpressionAtomicConverter(MINDAtomicConverter):
    """
    Only difference from MINDAtomicConverter is the addition of 'negatives' field and removal of 'label' field.
    """
    
    def __init__(self, config, df_inter_loaded=None, df_item_loaded=None):
        super().__init__(config, df_inter_loaded, df_item_loaded)

    @property
    def inter_fields(self) -> Dict[str, str]:
        return {
            'user_id': 'user_id:token',
            'item_id': 'item_id:token',
            'timestamp': 'timestamp:float',
            'impression_id': 'impression_id:token',
            'negatives': 'negatives:token_seq'
        }
