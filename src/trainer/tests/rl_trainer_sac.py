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

class SimpleRLModel(Model):
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
        pass

    def evaluation_step(self, eval_data):
        pass
    
    def save_model(self, filename):
        """
        Save model constructors as a single file
        i.e. everything needed to re-instantiate the model
        """
        model_constructor = {'model_state_dict': self.model.state_dict(),
                             'optimizer_state_dict': self.optimzier.state_dict(),
                             }
        torch.save(model_constructor, filename)

    def load_model(self, filename, zip=False):
        restored_model = torch.load(filename)
        self.model.load_state_dict(restored_model['model_state_dict'])
        self.optimzier.load_state_dict(restored_model['optimizer_state_dict'])
        
    def train(self):
        self.model.train()
    
    def eval(self):
        self.model.eval()

    def zero_grad(self):
        self.model.zero_grad()


class RLTrainer(Trainer):
    
    def setup_data(self, config):
        pass
        # Set up the environment

        # Set up the replay buffer

        # # Training data
        # train_x = np.random.rand(100, 10).astype(np.float32)
        # train_y = np.linalg.norm(train_x, axis=1).astype(np.float32)
        # self.train_data = DataLoader(list(zip(train_x, train_y)),
        #                              batch_size=self.config['train_batch_size'], 
        #                              shuffle=True)
        # # Test data
        # eval_x = np.random.rand(100, 10).astype(np.float32)
        # eval_y =  np.linalg.norm(eval_x, axis=1).astype(np.float32)
        # self.eval_data = DataLoader(list(zip(eval_x, eval_y)),
        #                             batch_size=self.config['eval_batch_size'], 
        #                             shuffle=False)
        
        

    def log_step(self):
        pass

if __name__ == "__main__":
        
    with open('trainer/tests/RLTrainer_test_config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    
    import wandb
    run_context = nullcontext()
    # run_context = wandb.init(project='test', id='uw55kkff', resume='must', dir=config['logger']['dir'])
    with run_context:
        model = SimpleRLModel(config['model'])
        logger = Logger(config['logger'])
        trainer = RLTrainer(config['trainer'], model, logger)
        trainer.fit()
    
    