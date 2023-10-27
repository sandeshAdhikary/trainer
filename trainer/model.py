from abc import ABC
from typing import Dict, Union
from types import SimpleNamespace
import torch
# from trainer.utils import register_class

class Model(ABC):

    def __init__(self, config):
        self.config = self.parse_config(config)
        # self._register_model()

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
    
    def state_dict(self):
        raise NotImplementedError

    def save_model(self, filename, save_optimizers=True):
        """
        Save model 
        """
        raise NotImplementedError

    def load_model(self, state_dict=None, model_file=None, model_dir=None):
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
    
    @property
    def module_path(self):
        return None

    # def _register_model(self):
    #     if self.module_path is not None:
    #         register_class('model', self.__class__.__name__, self.module_path)