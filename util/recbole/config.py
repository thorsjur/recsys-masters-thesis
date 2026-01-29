
from recbole.config import Config

class CConfig(Config):
  
  def __init__(self, **kwargs):
      super().__init__(**kwargs)
      
  def get(self, key, default=None):
      return self[key] if key in self else default