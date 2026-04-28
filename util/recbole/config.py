from recbole.config import Config


class CConfig(Config):
    """Simple wrapper around RecBole's Config to allow for dict-like access with defaults."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get(self, key, default=None):
        return self[key] if key in self else default
