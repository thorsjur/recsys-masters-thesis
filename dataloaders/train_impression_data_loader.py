from dataloaders.base_impression_data_loader import ImpressionDataLoader


class TrainImpressionDataLoader(ImpressionDataLoader):
    
    def __init__(self, config, dataset, sampler=None, shuffle=False):
        if shuffle is False:
            shuffle = True
            self.logger.warning("ImpressionDataLoader should shuffle training data.")

        self.neg_k = int(config.get("neg_sample_num", 4))
        self.shuffle_within_impression = bool(config.get("shuffle_within_impression", True))
        
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        batch_size = self.config["train_batch_size"]
        self.step = batch_size
        self.set_batch_size(batch_size)