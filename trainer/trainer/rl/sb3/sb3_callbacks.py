from typing import Any, Dict
from stable_baselines3.common.callbacks import BaseCallback
from abc import ABC, abstractmethod
from stable_baselines3.common.buffers import ReplayBuffer
import numpy as np
import torch
from abc import abstractmethod
from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first
from einops import rearrange
from gymnasium.spaces import Box
from trainer.rl.sb3.sb3_utils import evaluate_policy


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

    def __init__(self, config, trainer=None):
        super().__init__(config, trainer)
        self.num_update_decay_rate = config.get('num_update_decay_rate', None)
        self.min_num_updates = int(config.get('min_num_updates', 1))

    @abstractmethod
    def update_model(self, trainer):
        pass

    def on_rollout_end(self) -> None:
        assert hasattr(self, 'trainer'), "Trainer not set!"
        orig_in_train_mode = self.trainer.model.training
        self.trainer.model.train() # Set training mode
        self.update_model(self.trainer)
        if not orig_in_train_mode:
            self.trainer.model.eval()
    
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
            'buffer_size': config.get('buffer_size', 100_000),
            'observation_space': self.trainer.env.observation_space,
            'action_space': self.trainer.env.action_space,
            'device': self.trainer.device,
            'n_envs': self.trainer.env.num_envs,
            'optimize_memory_usage': config.get('optimize_memory_usage', False),
            'handle_timeout_termination': config.get('handle_timeout_termination', True),
        }

        # If image observations, check if they need to be transposed
        # Note: SB3's buffers transpose images to channels first
        if self._check_img_transpose(trainer.env):
            old_obs_space = buffer_kwargs['observation_space']
            buffer_kwargs['observation_space'] = Box(
                low=rearrange(old_obs_space.low, 'h w c -> c h w'),
                high=rearrange(old_obs_space.high, 'h w c -> c h w'),
                dtype=old_obs_space.dtype,
                shape=(old_obs_space.shape[-1], *old_obs_space.shape[:2])
            )

        self.replay_buffer = ReplayBuffer(**buffer_kwargs)
        self.batch_size = self.config.get('batch_size', 64)
        self.num_updates = self.config.get('num_updates', 3)
        self.num_rollouts_collected = 0
        self.buffer_rollouts_freq = self.config.get('buffer_rollouts_freq', 5)

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
        self.num_rollouts_collected += 1
        if self.num_rollouts_collected % self.buffer_rollouts_freq == 0:
            # Collect rollouts and add to buffer
            self._add_to_replay_buffer(self.trainer)
        if self.replay_buffer.pos >= self.batch_size:
            # Run train step
            orig_in_train_mode = self.trainer.model.training
            self.trainer.model.train() # Set training mode
            for idx in range(self.num_updates):
                self.update_model(self.replay_buffer.sample(self.batch_size))
            self._decay_num_updates()
            if not orig_in_train_mode:
                self.trainer.model.eval()
        return True

    def _decay_num_updates(self):
        """
        Decay the number of updates
        """
        if hasattr(self, 'num_update_decay_rate') and self.num_update_decay_rate is not None:
            # self.num_updates = int(self.num_updates*self.num_update_decay_rate))
            step_frac = self.trainer.step/self.trainer.num_train_steps
            decay = 1.0 - self.num_update_decay_rate
            self.num_updates = int(self.num_updates * np.exp(-decay*step_frac))
            self.num_updates = max(self.min_num_updates, self.num_updates)

    def _add_to_replay_buffer(self, trainer):
        obses, next_obses, actions, rewards, dones, infos = evaluate_policy(
            trainer.model.model,
            trainer.env,
            n_eval_episodes=3,
            deterministic=False
            )

        for (obs,next_obs,action,reward,done,info) in zip(
            obses,next_obses,actions,rewards,dones,infos
        ):
            self.replay_buffer.add(
                obs, next_obs, action, reward, done, info
            )

    def _check_img_transpose(self, env):
        """
        Check if the image observations need to be transposed
        """
        return is_image_space(env.observation_space) and not is_image_space_channels_first(
            env.observation_space  # type: ignore[arg-type]
        )


class RolloutToReplayBufferCallback(TrainerSB3Callback):
    def __init__(self, config, trainer):
        """
        At the end of every rollout, add experiences to a replay buffer.
        This is useful when you want to save experiences for SB3 models
        that use a rollout buffer (which is reset before every rollout)
        """
        self.config = config
        assert trainer is not None, "Trainer not set!"
        super().__init__(config,trainer=trainer)
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

        # If image observations, check if they need to be transposed
        # Note: SB3's buffers transpose images to channels first
        if self._check_img_transpose(trainer.env):
            old_obs_space = buffer_kwargs['observation_space']
            buffer_kwargs['observation_space'] = Box(
                low=rearrange(old_obs_space.low, 'h w c -> c h w'),
                high=rearrange(old_obs_space.high, 'h w c -> c h w'),
                dtype=old_obs_space.dtype,
                shape=(old_obs_space.shape[-1], *old_obs_space.shape[:2])
            )

        self.replay_buffer = ReplayBuffer(**buffer_kwargs)
        self.num_rollouts_collected = 0
        self.buffer_rollouts_freq = self.config.get('buffer_rollouts_freq', 5)

    def on_rollout_end(self) -> None:
        self.num_rollouts_collected += 1
        assert hasattr(self, 'trainer'), "Trainer not set!"
        if self.num_rollouts_collected % self.config.get('buffer_rollouts_freq', 1) == 0:
            # Add rollouts to replay buffer
            self._add_to_replay_buffer(self.trainer)
        return True
    
    def _check_img_transpose(self, env):
        """
        Check if the image observations need to be transposed
        """
        return is_image_space(env.observation_space) and not is_image_space_channels_first(
            env.observation_space  # type: ignore[arg-type]
        )

    def _add_to_replay_buffer(self, trainer):
        obses, next_obses, actions, rewards, dones, infos = evaluate_policy(
            trainer.model.model,
            trainer.env,
            n_eval_episodes=3,
            deterministic=False
            )

        for (obs,next_obs,action,reward,done,info) in zip(
            obses,next_obses,actions,rewards,dones,infos
        ):
            self.replay_buffer.add(
                obs, next_obs, action, reward, done, info
            )

    # def _add_to_replay_buffer(self, trainer):
    #     rollout_buffer = trainer.model.model.rollout_buffer
    #     obses = rollout_buffer.observations # (B, num_Envs, H, W)
    #     next_obses = obses[1:]
    #     obses = obses[:-1]
    #     rewards = rollout_buffer.rewards
    #     actions = rollout_buffer.actions
    #     rewards = rollout_buffer.rewards
    #     dones = np.zeros_like(rewards)
    #     infos = [[{}]*rewards.shape[1]] * rewards.shape[0]

    #     for (obs,next_obs,action,reward,done,info) in zip(
    #         obses,next_obses,actions,rewards,dones,infos
    #     ):
    #         self.replay_buffer.add(
    #             obs, next_obs, action, reward, done, info
    #         )

class TorchDebugCallback(BaseCallback):

    def __init__(self, config, trainer=None):
        super().__init__(verbose=0)

    def _on_step(self) -> bool:
        return True
    
    def on_rollout_start(self) -> None:
        pass

    def on_rollout_end(self) -> None:
        pass
     

class DummyCallback(BaseCallback):
    """
    Dummy callback that does nothing
    """
    def _on_step(self) -> bool:
        return True
    