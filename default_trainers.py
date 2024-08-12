from trainer.rl import RLTrainer

class DefaultRLTrainer(RLTrainer):

    def train_step(self, batch, batch_idx):
        pass

    def log_epoch(self, info):
        pass