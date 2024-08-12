from abc import ABC, abstractmethod, abstractproperty
from typing import Dict, Union
from types import SimpleNamespace
import torch
from copy import deepcopy
# from trainer.utils import register_class

class Model(ABC):

    def __init__(self, config, model, optimizer, loss_fn, scheduler=None):
        self.config = self.parse_config(config)
        self.device = torch.device(self.config.get('device'))
        if self.device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler


    @abstractmethod
    def training_step(self, batch, batch_idx):
        raise NotImplementedError
    
    @abstractmethod
    def evaluation_step(self, batch, batch_idx):
        raise NotImplementedError

    def parse_config(self, config: Dict) -> Union[SimpleNamespace, Dict]:
        """
        Parse the config dictionary into input config for the model
        """
        if config.get('device') is None:
            config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        return config
    

    def forward(self, x):
        if x.device != self.device:
            x = x.to(self.device)
        return self.model(x)

    def loss(self, *args, **kwargs):
        
        # Move all tensors args to device
        new_args = []
        for a in args:
            if torch.is_tensor(a) and (a.device != self.device):
                a = a.to(self.device)
            new_args.append(a)
        
        # Move all tensor kwargs to device
        new_kwargs = {}
        for k, v in kwargs.items():
            if torch.is_tensor(kwargs[k]) and kwargs[k].device != self.device:
                v = v.to(self.device)
            new_kwargs[k] = v

        return self.loss_fn(*new_args, **new_kwargs)

    def state_dict(self, include_optimizers=False):
        return {'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict() if include_optimizers else None,
                'schduler': self.scheduler.state_dict() if (include_optimizers and self.scheduler is not None) else None
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
        self.optimizer.zero_grad()
    
    @property
    def module_path(self):
        return None
    
    def __call__(self, x):
        return self.forward(x)

    @property
    def training(self):
        return self.model.training

class NullModel(Model):
    """
    A null model class. Used when initializing trainer/evaluator without a model (e.g. with random agent)
    """

    def __init__(self, *args, **kwargs):
        pass

    def training_step(self, batch, batch_idx):
        pass
    
    def evaluation_step(self, batch, batch_idx):
        pass

    def forward(self, x):
        """
        The NullModel's forward model should not be called
        """
        raise RuntimeError('NullModel forward should not be called')
    
    def loss(self, *args, **kwargs):
        raise RuntimeError('NullModel loss should not be called')

    def state_dict(self, include_optimizers=False):
        raise RuntimeError('NullModel state_dict should not be called')

    def train(self):
        """
        Set model to training mode
        """
        pass

    def eval(self):
        """
        Set model to evaluation mode
        """
        pass

    def zero_grad(self):
        pass
    
    @property
    def module_path(self):
        raise RuntimeError('NullModel module_path should not be called')
    
    def __call__(self, x):
        pass

    @property
    def training(self):
        pass


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
        pred = self.forward(x)
        self.zero_grad()
        loss = self.loss(pred.view(-1), y.view(-1))
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
        pred = self.forward(x)
        loss = self.loss(pred.view(-1), y.view(-1))
        return {
            'x': x.detach().cpu().numpy(),
            'y': y.detach().cpu().numpy(),
            'preds': pred.view(-1).detach().cpu().numpy(),
            'loss': loss.view(-1).detach().cpu().numpy()
        }
    

class EncoderModel(Model):
    """
    Base class for regression models.
    Assumes:
        1. batch = (x, y) or just (x)
        2. model(x) returns a vecror output (encoding)
    """

    def __init__(self, config, model, optimizer, loss_fn, scheduler=None):
        super().__init__(config, model, optimizer, loss_fn, scheduler=scheduler)

    def training_step(self, batch, batch_idx):
        """
        Compute single training_step for a given batch, and update model parameters
        returns the loss
        """
        if len(batch) == 2:
            # Target provided
            x, y = batch
        else:
            # Targett may not be provided
            x = batch
            y = None

        pred = self.forward(x)
        self.zero_grad()
        loss = self.loss(x, pred, y)
        loss.backward()
        self.optimizer.step()
    
        return {
            'x': x.detach().cpu().numpy(),
            'y': y.detach().cpu().numpy() if y is not None else y,
            'preds': pred.detach().cpu().numpy(),
            'loss': loss.view(-1).detach().cpu().numpy(),
            'model': self.state_dict(include_optimizers=False)
        }

    def evaluation_step(self, batch, batch_idx):
        """
        Compute single evaluation step for a given batch
        returns: output dict with keys x, y, preds, loss
        """

        if len(batch) == 2:
            # Target provided
            x, y = batch
        else:
            # Targett may not be provided
            x = batch
            y = None

        pred = self.forward(x)
        loss = self.loss(x, pred, y)
        return {
            'x': x.detach().cpu().numpy(),
            'y': y.detach().cpu().numpy() if y is not None else y,
            'preds': pred.detach().cpu().numpy(),
            'loss': loss.view(-1).detach().cpu().numpy()
        }