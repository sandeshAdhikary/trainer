from abc import ABC, abstractmethod
from typing import Dict, Union
from types import SimpleNamespace
import numpy as np
import torch

class Model(ABC):

    def __init__(self, config):
        self.config = self.parse_config(config)

    def parse_config(self, config: Dict) -> Union[SimpleNamespace, Dict]:
        """
        Parse the config dictionary into input config for the model
        """
        if config.get('device') is None:
            config['device'] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        return config

    def training_step(self):
        raise NotImplementedError

    def evaluation_step(self):
        raise NotImplementedError

    def save_model(self, filename):
        """
        Save model 
        """
        raise NotImplementedError

    def load_model(self, filename):
        """
        Load model
        """
        raise NotImplementedError

    def train(self):
        """
        Set model to training mode
        """
        raise NotImplementedError

    def eval(self):
        """
        Set model to evaluation mode
        """
        raise NotImplementedError

    def zero_grad(self):
        raise NotImplementedError
    
