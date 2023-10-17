from types import SimpleNamespace
from typing import Dict, Union
import yaml
from trainer.trainer import Trainer
from trainer.model import Model
from trainer.logger import Logger
import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import DataLoader
from contextlib import nullcontext

class SimpleMLModel(Model):
    def __init__(self, config):
        super().__init__(config)
        self.model = nn.Sequential(
            nn.Linear(10, self.config['latent_dim']),
            nn.ReLU(),
            nn.Linear(self.config['latent_dim'], 1)
        )

        self.optimzier = torch.optim.Adam(self.model.parameters(), 
                                          lr=self.config['lr'])
        self.loss = nn.MSELoss()

    def parse_config(self, config: Dict) -> Union[SimpleNamespace, Dict]:
        # Set latent dimension        
        latent_dims = {'small': 8, 'medium': 16, 'large': 32}
        config['latent_dim'] = latent_dims[config['model_size']]
        return config

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = self.loss(pred.view(-1), y.view(-1))
        self.model.zero_grad()
        loss.backward()
        self.optimzier.step()

        step_log =  {'loss': loss.item(), 
                     'num_updates': 1
                     }

        return step_log

    def evaluation_step(self, eval_data):
        eval_losses = []
        for idx, batch in enumerate(eval_data):
            x, y = batch
            pred = self.model(x)
            loss = self.loss(pred.view(-1), y.view(-1))
            eval_losses.append(loss.item())
        step_log = {'loss': np.mean(eval_losses)}
        return step_log
    
    def save_model(self, filename):
        """
        Save model constructors as a single file
        i.e. everything needed to re-instantiate the model
        """
        model_constructor = {'model_state_dict': self.model.state_dict(),
                             'optimizer_state_dict': self.optimzier.state_dict(),
                             }
        torch.save(model_constructor, filename)

    def load_model(self, filename):
        restored_model = torch.load(filename)
        self.model.load_state_dict(restored_model['model_state_dict'])
        self.optimzier.load_state_dict(restored_model['optimizer_state_dict'])
        
    def train(self):
        self.model.train()
    
    def eval(self):
        self.model.eval()

    def zero_grad(self):
        self.model.zero_grad()


class MLTrainer(Trainer):
    
    def setup_data(self, config):
        # Training data
        train_x = np.random.rand(100, 10).astype(np.float32)
        train_y = np.linalg.norm(train_x, axis=1).astype(np.float32)
        self.train_data = DataLoader(list(zip(train_x, train_y)),
                                     batch_size=self.config['train_batch_size'], 
                                     shuffle=True)
        # Test data
        eval_x = np.random.rand(100, 10).astype(np.float32)
        eval_y =  np.linalg.norm(eval_x, axis=1).astype(np.float32)
        self.eval_data = DataLoader(list(zip(eval_x, eval_y)),
                                    batch_size=self.config['eval_batch_size'], 
                                    shuffle=False)
        
        

    def log_step(self, info=None):
        # Log train metrics
        if len(self.train_log) > 0:
            train_log_step = self.train_log[-1]['step']
            train_loss = self.train_log[-1]['log']['loss']
            self.logger.log(log_dict={'trainer_step': train_log_step, 'train/loss': train_loss})

        # Log eval metrics
        if len(self.eval_log) > 0:
            eval_log_step = self.eval_log[-1]['step']
            eval_loss = self.eval_log[-1]['log']['loss']
            self.logger.log(log_dict={'trainer_step': eval_log_step, 'eval/loss': eval_loss})

if __name__ == "__main__":
        
    with open('trainer/tests/MLTrainer_test_config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    
    import wandb
    run_context = nullcontext()
    run_context = wandb.init(project='test', id='uw55kkff', resume='must', dir=config['logger']['dir'])
    with run_context:
        model = SimpleMLModel(config['model'])
        logger = Logger(config['logger'])
        trainer = MLTrainer(config['trainer'], model, logger)
        trainer.fit()
    
    