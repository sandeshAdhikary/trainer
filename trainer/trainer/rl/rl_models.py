from trainer.model import Model
from torch.optim import Adam
from torch import nn

class SimpleRLModel(Model):
    def __init__(self, config):
        model = NNPolicy()
        optimizer = Adam(model.parameters(), lr=config['lr'])
        loss_fn = self.loss_fn
        super().__init__(config, model, optimizer, loss_fn)
    
    def loss_fn(self):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def evaluation_step(self, batch, batch_idx):
        pass


class NNPolicy(nn.Module):
    pass