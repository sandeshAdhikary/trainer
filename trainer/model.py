from abc import ABC, abstractmethod, abstractproperty
from typing import Dict, Union
from types import SimpleNamespace
import torch
from copy import deepcopy
# from trainer.utils import register_class

class Model(ABC):

    def __init__(self, config, model, optimizer, loss_fn):
        self.config = self.parse_config(config)
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    @abstractmethod
    def training_step(self):
        raise NotImplementedError
    
    @abstractmethod
    def evaluation_step(self):
        raise NotImplementedError

    def parse_config(self, config: Dict) -> Union[SimpleNamespace, Dict]:
        """
        Parse the config dictionary into input config for the model
        """
        if config.get('device') is None:
            config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        return config
    

    def forward(self, x):
        return self.model(x)

    def state_dict(self, include_optimizers=False):
        return {'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict() if include_optimizers else None
                }

    def load_model(self, state_dict):
        """
        Load model
        """
        self.model.load_state_dict(state_dict=state_dict['model'])


    def train(self):
        """
        Set model to training mode
        """
        return self.model.train()

    def eval(self):
        """
        Set model to evaluation mode
        """
        return self.model.train()

    def zero_grad(self):
        return self.model.zero_grad()
    
    @property
    def module_path(self):
        return None
    
    def __call__(self, x):
        return self.forward(x)


class RegressionModel(Model):
    """
    Base class for regression models.
    Assumes:
        1. batch = (x, y) where x is data, and y is the regression target
        2. model(x) returns a scalar regression prediction
    
    Assumes self.model(x) returns a scalar regression prediction
    """

    def __init__(self, config, model, optimizer, loss_fn):
        super().__init__(config, model, optimizer, loss_fn)

    def training_step(self, batch, batch_idx):
        """
        Compute single training_step for a given batch, and update model parameters
        returns the loss
        """
        x, y = batch
        pred = self.model(x)
        self.zero_grad()
        loss = self.loss_fn(pred.view(-1), y.view(-1))
        loss.backward()
        self.optimizer.step()
    
        return {
            'x': x.detach().cpu().numpy(),
            'y': y.detach().cpu().numpy(),
            'preds': pred.view(-1).detach().cpu().numpy(),
            'loss': loss.view(-1).detach().cpu().numpy(),
            'model': self.state_dict(include_optimizers=False)
        }

    def evaluation_step(self, batch, batch_idx):
        """
        Compute single evaluation step for a given batch
        returns: output dict with keys x, y, preds, loss
        """

        x, y = batch
        pred = self.model(x)
        loss = self.loss_fn(pred.view(-1), y.view(-1))
        return {
            'x': x.detach().cpu().numpy(),
            'y': y.detach().cpu().numpy(),
            'preds': pred.view(-1).detach().cpu().numpy(),
            'loss': loss.view(-1).detach().cpu().numpy()
        }
    