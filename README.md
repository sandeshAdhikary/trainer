# trainer
Trainer class for general ML and RL experiments. Currently, focusing on building out the RL components first.

# API

## Model
A light wrapper for torch models to allow a consistent API for model training, loading, evaluation, etc.

## Trainer
Trainer class used to train models intended to be used as follows:
```
  trainer = Trainer(config)
  model = Model(model_config)
  logger = Logger(logg_config)
  trainer.set_model(model)
  trainer.set_logger(logger)
  trainer.fit(max_steps=1_000)
```
The ``.fit()`` method is structured as follows:
```
    def fit(self, num_train_steps: int=None, trainer_state=None) -> None:
        self.num_train_steps = num_train_steps or self.config['num_train_steps']
    
        self.before_train(trainer_state) # will load trainer checkpoint if needed
        with self.terminal_display:
            while not self.train_end:
                self.before_epoch()
                # Run training epoch
                for batch_idx, batch in enumerate(self.train_data):
                    self.before_step()
                    self.train_step(batch, batch_idx)
                    self.after_step()
                self.evaluate(self.eval_data, training_mode=True)
                self.after_epoch()
                
        self.after_train()
```
The before/after callback functions (e.g. ``self.after_epoch()``) can be customized e.g. to record statistics, save models, etc. Currently, the general Trainer class needs more work -- the specific RLTrainer class should be functional for training RL models.

### RL Trainer
A specific instantiation of Trainer for training RL models. The ``.fit()`` function is strucuted as follows
```
    def fit(self, num_train_steps: int=None, trainer_state=None) -> None:
        self.num_train_steps = num_train_steps or self.config['num_train_steps']
    
        self.before_train(trainer_state) # will load trainer checkpoint if needed
        with self.terminal_display:
            while not self.train_end:
                self.before_epoch()
                # Collect rollout with policy
                rollout_data = self.collect_rollouts(self.obs, add_to_buffer=True)
                # Update the agent
                num_updates = self.get_num_updates()
                for batch_idx in range(num_updates):
                    self.before_step()
                    batch = self.replay_buffer.sample()
                    self.train_step(batch, batch_idx)

                self.after_step()
                if (self.epoch % self.config['eval_freq'] == 0) and (self.epoch > 0):
                    self.evaluate(async_eval=self.async_eval)
                self.after_epoch({'rollout': rollout_data, 'num_model_updates': num_updates})
                
        self.after_train()
```

## Evaluator

### Async Evaluator

## Sweeper

## Logger

## Storage







