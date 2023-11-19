import numpy as np
from torch.utils.data import DataLoader, Dataset
from typing import Dict
from torch.utils.data import DataLoader
from typing import Dict
from torch import nn
from torch.optim import Adam
from trainer import (
    SupervisedTrainer, RegressionModel, SupervisedEvaluator
)
from trainer.metrics import StaticScalarMetric

class SimpleDataset(Dataset):
    """
    Dataset object for training, validation, and testing and data
    """
    
    DATA_PATHS = {
        'train':'examples/supervised_trainer/data/valid_data.npy',
        'valid':'examples/supervised_trainer/data//valid_data.npy',
        'test':'examples/supervised_trainer/data/test_data.npy'
    }

    def __init__(self, mode='train'):
        self.data = np.load(self.DATA_PATHS[mode])

    def __getitem__(self, index):
        """
        First 6 elements are the data, last element is the label
        """
        return self.data[index][0:6], self.data[index][7]

    def __len__(self):
        return len(self.data)

class PredictionMetric(StaticScalarMetric):
    def log(self, data):
        return {
            'max': np.max(data['preds']),
            'min': np.min(data['preds']),
        }
    
class ModelWeightsMetric(StaticScalarMetric):
    def log(self, data):
        """
        return a dict of form:
         {param_name: {'avg': avg_weight, 'std': std_weight}
         for all params in data['model']
        """
        
        # Compute average weights
        model_weights = [x['model'] for x in data['model']]
        output = {k: {'avg': [], 'std': []} for k in model_weights[0].keys()}
        for key in output.keys():
            param_weights = []
            for idx in range(len(model_weights)):
                param_weights.append(model_weights[idx][key].detach().cpu().numpy())
            param_weights = np.concatenate(param_weights)
            output[key]['avg'] = np.mean(param_weights)
            output[key]['std'] = np.std(param_weights)
        
        return output
    
class SimpleSupervisedTrainer(SupervisedTrainer):
    """
    Define the training data and any additional log_epoch
    """

    def setup_data(self, config):
        self.train_data = DataLoader(SimpleDataset(mode='train'), batch_size=self.config['batch_size'])

    def log_epoch(self, info=None):
        """
        Log values from self.train_log and self.eval_log
        Optionally, additional info may be provided via `info`
        """

        # Train log
        train_data = [log['log'] for log in self.train_log]
        train_metrics = self._get_metrics(train_data, self.metrics, 
                                          storage=self.output_storage)
        self.logger.log(log_dict={
            'trainer_step': self.step,
            'train/loss_avg': train_metrics['prediction_loss']['avg'],
            'train/loss_std': train_metrics['prediction_loss']['std'],
        })

        # Log model weights
        train_model_weights = train_metrics['model_weights']
        for param_name in  train_metrics['model_weights'].keys():
            self.logger.log(log_dict={
                f'train/weights_{param_name}_avg': train_model_weights[param_name]['avg'],
                f'train/weights_{param_name}_std': train_model_weights[param_name]['std'],
            })


        # Eval log
        eval_data = [log['log'] for log in self.eval_log]
        eval_metrics = self._get_metrics(eval_data, self.evaluator.metrics, 
                                         storage=self.evaluator.output_storage)
        self.logger.log(log_dict={
            'eval_step': self.step,
            'eval/loss_avg': eval_metrics['prediction_loss']['avg'],
            'eval/loss_std': eval_metrics['prediction_loss']['std'],
        })

class SimpleSupervisedEvaluator(SupervisedEvaluator):
    """
    Simple evaluator
    """

    def setup_data(self):
        self.dataloader = DataLoader(SimpleDataset(mode='valid'), batch_size=self.config['batch_size'])

class SimpleRegressionModel(RegressionModel):
    """
    A simple single layer NN regression model
    """

    def __init__(self, config: Dict):
        """
        Define the model, optimizer, and loss_fn
        """
        # Define the model
        model = nn.Sequential(
            nn.Linear(config['input_dim'], config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['hidden_dim'], 1)
        )
        # Define the optimizer
        optimizer = Adam(model.parameters(), 
                            **config['optimizer']['optimizer_kwargs'])
        # Define the loss
        loss_fn = nn.MSELoss()

        super().__init__(config, model, optimizer, loss_fn)