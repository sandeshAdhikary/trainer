from typing import Any, Dict
from stable_baselines3.common.callbacks import BaseCallback
from abc import ABC, abstractmethod
from stable_baselines3.common.buffers import ReplayBuffer
import numpy as np
import torch
from abc import abstractmethod

class TrainerSB3Callback(BaseCallback):
    """
    Base callback class to link SB3 with Trainer.
    Provides a set_trainer() function to link with trainer object
    The trainer object's methods (including model, logger, etc.) can then be used
    with SB3 calls.
    """

    def __init__(self, config, trainer=None):
        self.config = config
        verbose = config.get('verbose', 0)
        super().__init__(verbose)
        if trainer is not None:
            self.set_trainer(trainer)

    def set_trainer(self, trainer):
        self.trainer = trainer

    def _on_step(self) -> bool:
        return True
    

class SB3TrainingCallback(TrainerSB3Callback):
    """
    A callback to run trainer's training sets.
    Since we use SB3's learn() step tha
    """
    def on_rollout_start(self) -> None:
        """
        No self.after_step() calls since SB3 onpolicy algo
        does a full epoch on train() before callbacks are called
        So, we do all after_step() updates within after_epoch()
        """
        self.before_epoch()
        if self.trainer.step > 0:
            self.after_epoch()

    def before_epoch(self):
        """

        """
        self.trainer.before_epoch() # Run any before_epoch() steps
        self.trainer.step = self.model.num_timesteps

    def after_epoch(self):
        # Update trainer state
        self.trainer.update_trainer_state() # update train_end bool
        if self.trainer.progress is not None:
            self.trainer.progress.update(self.trainer.progress_train, completed=self.trainer.step)
            # update progress
            # completed = (1.0*self.trainer.step)

        # Compute num_updates
        num_sb3_epochs = self.trainer.model.model.n_epochs
        num_sb3_batches = self.trainer.model.model.rollout_buffer.buffer_size // self.trainer.model.model.batch_size
        num_updates = num_sb3_epochs*num_sb3_batches
        rollout_data = {}
        self.trainer.after_epoch({'rollout': rollout_data,
                                    'num_model_updates': num_updates})


class SB3EvalCallback(TrainerSB3Callback):
    """
    Run trainer's evaluation steps as an SB3 callback
    """

    def on_rollout_end(self) -> None:
        """
        Run trainer's evaluation steps as an SB3 callback
        """
        if (self.trainer.epoch % self.trainer.config['eval_freq'] == 0) and (self.trainer.epoch > 0):
            self.trainer.evaluate(async_eval=self.trainer.async_eval, suffix=f"epoch_{self.trainer.epoch}")

class SB3ModelUpdateCallback(TrainerSB3Callback, ABC):
    """
    Callback to interleave SB3 algorithms' normal training steps with 
    any other training steps (defined by update_model() method).
    The update_model() method gives access to the trainer object, 
    which, in turn, gives access to the model, buffer, etc.
    """

    @abstractmethod
    def update_model(self, trainer):
        pass

    def on_rollout_end(self) -> None:
        assert hasattr(self, 'trainer'), "Trainer not set!"

        self.trainer.model.train() # Set training mode
        self.update_model(self.trainer)
        # TODO: Reset to original train or eval mode
    
    def on_training_start(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        self._setup_model()
        return super().on_training_start(locals_, globals_)

    def _setup_model(self):
        """
        Set up the model for training
        """
        self.model = self.trainer.model.model

class SB3InternalLoggerCallback(TrainerSB3Callback):

    # def __init__(self, config):
    #     self.logger_mode = config.get('logger_mode', 'train')
    #     assert self.logger_mode in ['train', 'eval'], f"logger_mode {self.logger_mode} not recognized"

    def on_rollout_end(self) -> None:
        """
        Add data from SB3's internal logger to trainer's logger
        """
        pass


class SB3EncoderUpdateCallback(SB3ModelUpdateCallback):
    """
    A callback to update the encoder model
    """

    def __init__(self, config, trainer):
        self.config = config
        super().__init__(config)
        assert trainer is not None, "Trainer not set!"
        self.set_trainer(trainer)
        # Set up a replay buffer
        buffer_kwargs = {
            'buffer_size': config.get('buffer_size', 10_000),
            'observation_space': self.trainer.env.observation_space,
            'action_space': self.trainer.env.action_space,
            'device': self.trainer.device,
            'n_envs': self.trainer.env.num_envs,
            'optimize_memory_usage': config.get('optimize_memory_usage', False),
            'handle_timeout_termination': config.get('handle_timeout_termination', True),
        }
        self.replay_buffer = ReplayBuffer(**buffer_kwargs)
        self.batch_size = self.config.get('batch_size', 64)
        self.num_updates = self.config.get('num_updates', 3)

    @abstractmethod
    def update_model(self, data):
        """
        Update the encoder model
        """
        raise NotImplementedError


    def _setup_model(self):
        """
        Set up the model for training
        """
        super()._setup_model()
        self.encoder = self.model.policy.features_extractor
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), 
                                          lr=self.config.get('learning_rate', 1e-3))

    def on_rollout_end(self) -> None:
        assert hasattr(self, 'trainer'), "Trainer not set!"

        # Add rollouts to replay buffer
        self._add_to_replay_buffer(self.trainer)
        # Run train step
        self.trainer.model.train() # Set training mode
        if self.replay_buffer.pos >= self.batch_size:
            for idx in range(self.num_updates):
                self.update_model(self.replay_buffer.sample(self.batch_size))
        # TODO: Reset to original train or eval mode
        return True
    
    
    def _add_to_replay_buffer(self, trainer):
        rollout_buffer = trainer.model.model.rollout_buffer
        obses = rollout_buffer.observations # (B, num_Envs, H, W)
        next_obses = obses[1:]
        obses = obses[:-1]
        rewards = rollout_buffer.rewards
        actions = rollout_buffer.actions
        rewards = rollout_buffer.rewards
        dones = np.zeros_like(rewards)
        infos = [[{}]*rewards.shape[1]] * rewards.shape[0]
        for (obs,next_obs,action,reward,done,info) in zip(
            obses,next_obses,actions,rewards,dones,infos
        ):
            self.replay_buffer.add(
                obs, next_obs, action, reward, done, info
            )