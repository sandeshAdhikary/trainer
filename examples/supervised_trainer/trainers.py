from typing import Dict
from trainer import MLTrainer
from src.tests.ml_trainer.data import SimpleDataset
from torch.utils.data import DataLoader
import numpy as np

class SimpleTrainer(MLTrainer):
    """
    Define the training data and any additional log_epoch
    """

    def setup_data(self, config):
        self.train_data = DataLoader(SimpleDataset(mode='train'), batch_size=self.config['batch_size'])
