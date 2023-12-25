import numpy as np
from torch.utils.data import DataLoader, Dataset
from typing import Dict
from torch.utils.data import DataLoader
from typing import Dict
from torch import nn
from torch.optim import Adam
from trainer import (
    SupervisedTrainer, EncoderModel, SupervisedEvaluator
)
from einops import rearrange
import os
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import trange

class EncoderDataset(Dataset):
    """
    Dataset object for training, validation, and testing and data
    """
    
    DATA_PATHS = {
        'train':'examples/encoder/data/data.npy',
    }

    def __init__(self, mode='train'):
        self.data = np.load(self.DATA_PATHS[mode]).astype(np.float64)

    def __getitem__(self, index):
        """
        First 6 elements are the data, last element is the label
        """
        return self.data[index]

    def __len__(self):
        return len(self.data)
    

    
class SimpleEncoderTrainer(SupervisedTrainer):
    """
    Define the training data and any additional log_epoch
    """

    def setup_data(self, config):
        self.train_data = DataLoader(EncoderDataset(mode='train'), 
                                     batch_size=self.config['batch_size'], 
                                     shuffle=False)

    def log_epoch(self, info=None):
        """
        Log values from self.train_log and self.eval_log
        Optionally, additional info may be provided via `info`
        """

        if self.epoch % self.logger.log_freq == 0:

            # Train log
            train_data = [log['log'] for log in self.train_log]
            train_metrics = self._get_metrics(train_data, self.metrics, 
                                            storage=self.output_storage)
            self.logger.log(log_dict={
                'trainer_step': self.step,
                'train/loss_avg': train_metrics['loss']['avg'],
                'train/loss_std': train_metrics['loss']['std'],
            })

            # Log model weights
            train_model_weights = train_metrics['model_weights']
            for param_name in  train_metrics['model_weights'].keys():
                self.logger.log(log_dict={
                    f'model_weights/weights_{param_name}_avg': train_model_weights[param_name]['avg'],
                    f'model_weights/weights_{param_name}_std': train_model_weights[param_name]['std'],
                })


            # Eval log
            eval_data = [log['log'] for log in self.eval_log]
            eval_metrics = self._get_metrics(eval_data, self.evaluator.metrics, 
                                            storage=self.evaluator.output_storage)
            self.logger.log(log_dict={
                'eval_step': self.step,
                'eval/loss_avg': eval_metrics['loss']['avg'],
                'eval/loss_std': eval_metrics['loss']['std'],
            })

            # Log model outputs versus true eigenvexctors
            inputs = eval_data[-1]['x']
            encodings = eval_data[-1]['preds']
            # encodings = rearrange(encodings, '(r d) -> r d', r=self.model.grid_size**2)
            encodings = rearrange(encodings, '(r1 c1) d -> r1 c1 d', r1=self.model.grid_size)
            
            true_eigvecs = self.model.true_eigvec_fn(inputs.reshape(-1,2))
            true_eigvecs = rearrange(true_eigvecs, '(r1 c1) d -> r1 c1 d', r1=self.model.grid_size)

            # get plots of the top-n eigenvectors vs encodings
            max_viz_dim = min(encodings.shape[-1], 10)
            for idx in range(max_viz_dim):
                encoding = self.make_heatmap(encodings[:,:,idx])
                eigvec = self.make_heatmap(true_eigvecs[:,:,idx].numpy())
                img = make_grid([torch.from_numpy(encoding), torch.from_numpy(eigvec)], nrow=1)
                self.logger.log_image(f'eval/encodigs_{idx}', img, image_mode='chw')


        if self.epoch == 1:
            # Plot the kernel/weights
            weights = rearrange(self.model.precomp_kernel, 'r1 c1 r2 c2 -> (r1 c1) (r2 c2)')
            weights = self.make_heatmap(weights)
            self.logger.log_image(f'target/weights', weights, image_mode='chw')

    def make_heatmap(self, array, cmap='viridis', img_mode='chw'):
        """
        """
        # Normalize the input array to be in the range [0, 1]
        normalized_array = (array - np.min(array)) / (np.max(array) - np.min(array))

        # Choose the colormap
        colormap = plt.get_cmap(cmap)

        # Map the normalized values to RGBA values using the chosen colormap
        img = (colormap(normalized_array) * 255).astype(np.uint8)

        img = Image.fromarray(img).resize((84, 84)).convert('RGB')
        img = np.array(img)
        
        if img_mode == 'chw':
            img = rearrange(img, 'h w c -> c h w')
        return img


class SimpleSupervisedEvaluator(SupervisedEvaluator):
    """
    Simple evaluator
    """

    def setup_data(self, config):
        self.dataloader = DataLoader(EncoderDataset(mode='train'), batch_size=self.config['batch_size'], shuffle=False)

    

class SimpleEncoderModel(EncoderModel):
    """
    A simple NN regression model
    """

    def __init__(self, config: Dict):
        """
        Define the model, optimizer, and loss_fn
        """
        # Define the model
        model = nn.Sequential(
            nn.Linear(config['input_dim'], config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['hidden_dim'], config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['hidden_dim'], config['encoder_dim'])
        )
        model = model.double()
        # Define the optimizer
        optimizer = Adam(model.parameters(), 
                            **config['optimizer']['optimizer_kwargs'])
        # Define the loss
        super().__init__(config, model, optimizer, self.loss_fn)

        # kernel specifics
        self.grid_size = config['grid_size']
        self.ortho_loss_reg = config['ortho_loss_reg']
        self.loss_type = config['loss_type']
        self.bandwidth = config['bandwidth']
        self.precomp_kernel = self.get_precomp_kernel()
        self.precomp_eigvecs = self.get_precomp_eigvecs()


    def forward(self, x):
        """
        Forward pass of the model
        """
        if x.device != self.device:
            x = x.to(self.device)
        h = self.model(x) # (B, d)
        # L2-batch normalization
        # sigma = torch.norm(h, p=2, dim=0, keepdim=True) / np.sqrt(h.shape[0])
        sigma = torch.einsum('bd,bd->d', h, h)
        h = h / sigma
        return h

        
    def get_precomp_eigvecs(self):
        W = torch.from_numpy(self.get_precomp_kernel())
        W = rearrange(W, 'r1 c1 r2 c2 -> (r1 c1) (r2 c2)')
        # Build Laplacian
        D = torch.diag(W.sum(dim=1))
        L = D - W
        # Get eigenecs
        # eigvals, eigvecs = torch.linalg.eigh(L)
        eigvals, eigvecs = torch.lobpcg(L, k=self.config['encoder_dim'], B = D, largest=False, )
        # Solves Ax=\lambda Bx,
        # We want to solve L x = \lambda D x
        eigvecs = rearrange(eigvecs, '(r1 c1) d -> r1 c1 d', r1=self.grid_size)
        return eigvecs

    def loss_fn(self, inputs, encodings, targets=None):
        """
        Compute reconstruction loss between encodings and the true eigenvectors
        """
        if self.loss_type == 'reconstruction':
            ## Simple Reconstruction
            true_eigvecs = self.true_eigvec_fn(inputs.view(-1,2))
            # Reconstruction loss
            loss = nn.MSELoss()(encodings, true_eigvecs[:,:encodings.shape[1]].to(encodings.device).to(torch.float32))
        elif self.loss_type == 'spectralnet':
            W = rearrange(self.precomp_kernel, 'r1 c1 r2 c2 -> (r1 c1) (r2 c2)')
            W = torch.from_numpy(W).to(encodings.device).to(encodings.dtype)
            batch_size = W.shape[0]
            
            D = torch.sum(W, dim=1)
            h = encodings/D[:, None]
                # h = encodings

            Dh = torch.cdist(h,h,p=2)
            dist_loss = torch.sum(W * Dh.pow(2)) / (batch_size**2)
            
            D = torch.diag(torch.sum(W, dim=1))
            est = (h.T @ D @ h)
            ortho_loss = (est - torch.eye(h.shape[1]).to(h.device))**2
            ortho_loss = ortho_loss.sum()/(h.shape[1])
            
            loss = dist_loss + self.ortho_loss_reg*ortho_loss

        elif self.loss_type == 'neuralef':
            # NeuralEF
            A = rearrange(self.precomp_kernel, 'r1 c1 r2 c2 -> (r1 c1) (r2 c2)')
            A = torch.from_numpy(A).to(encodings.device).to(encodings.dtype)
            D_sqrt = torch.diag(torch.diagonal(A)**-0.5)
            kernel = D_sqrt @ A @ D_sqrt

            batch_size = kernel.shape[0]

            # Distance loss
            R = encodings.T @ kernel @ encodings
            R /= batch_size**2
            dist_loss = torch.trace(R)

            # Orthogonality loss
            with torch.no_grad():
                encodings_stopgrad = self.model(inputs)
            R_hat = encodings_stopgrad.T @ kernel @ encodings
            R_hat /= batch_size**2

            ortho_loss = torch.triu(R_hat**2, diagonal=1).sum(dim=1) # column sums of (squared) upper off-diagonal elements
            ortho_loss = ortho_loss.sum() # Sum over rows

            loss = dist_loss - self.ortho_loss_reg*ortho_loss
            loss = -loss # NeuralEF loss is an argmax, but we're minimizing
        else:
            raise ValueError(f'loss_type {self.loss_type} not recognized')
        return loss

    def true_eigvec_fn(self, inputs):
        return torch.stack([self.precomp_eigvecs[int(x[0]), int(x[1])] for x in inputs])

    def kernel_fn(self, x, y):
        return self.precomp_kernel[int(x[0].item()),
                                   int(x[1].item()),
                                   int(y[0].item()),
                                   int(y[1].item())
                                   ]
    
    def get_precomp_kernel(self):
        # obstacles = np.concatenate(
        #     [
        #         np.array([(int(self.grid_size/2), x) for x in range(self.grid_size) if x not in [4,5,6,7]]),
        #         np.array([(x, int(self.grid_size/2)) for x in range(self.grid_size) if x not in [4,5,6,7]]),
        #     ]
        #     )
        obstacles = None

        self.neighbor_eps = 20
        dist_mat = np.zeros((self.grid_size, self.grid_size, self.grid_size, self.grid_size))
        for r1 in trange(self.grid_size):
            for c1 in range(self.grid_size):
                for r2 in range(self.grid_size):
                    for c2 in range(self.grid_size):

                        if r1 == r2 and c1 == c2:
                            dist_mat[r1,c1,r2,c2] = 0.0
                        else:
                            if obstacles is not None:
                                if [r1,c1] in obstacles.tolist() or [r2,c2] in obstacles.tolist():
                                    dist_mat[r1,c1,r2,c2] = np.inf
                                    continue

                            if (abs((r2 - r1)) <= self.neighbor_eps) and (abs((c2 - c1)) <= self.neighbor_eps):
                                dist_mat[r1,c1,r2,c2] = np.sqrt((r1-r2)**2 + (c1-c2)**2)
        kernel = np.exp(-(dist_mat**2)/(2*self.bandwidth**2))
                            
        return kernel