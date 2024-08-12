from trainer.rl.rl_trainer import RLTrainer
from trainer import Model
from typing import Dict
from functools import partial
from stable_baselines3.common.env_util import make_vec_env as sb3_make_env
from stable_baselines3.common.logger import Logger
import tempfile
import torch
import os
from copy import deepcopy
from trainer.rl.rl_evaluator import TrainingRLEvaluator
from trainer.metrics import Metric
from trainer.utils import import_module_attr
from trainer.rl.sb3.sb3_callbacks import SB3TrainingCallback, SB3EvalCallback
from trainer.rl.sb3.sb3_utils import SB3InternalLogger
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from trainer.rl.sb3.sb3_utils import lr_schedule
from stable_baselines3 import PPO

class SB3RLModel(Model):

    def __init__(self, config, trainer):
        self.config = config
        self.seed = trainer.config.get('seed')
        env = trainer.env
        self.model = self.get_sb3_model(env, config)
        # Maintain an internal SB3 logger
        # we'll extract metrics from here when we need to log
        self.internal_logger = SB3InternalLogger(trainer=trainer)
        self.model.set_logger(self.internal_logger)
        # TODO: Allow saving buffer after train for off-policy models
        self.save_buffer_after_train = False


    def get_sb3_model(self, env, config):
        """
        SB3 models mix trainer+model, so we need to extract
        some parameters from the trainer
        """
        model_cls = import_module_attr(config['sb3_algo_module_path'])
        model_kwargs = deepcopy(config['model_kwargs'])

        # Use an LR schedule if needed
        model_kwargs['learning_rate'] = self.get_learning_rate(model_kwargs)
        model_kwargs['seed'] = self.seed

        # Define features extractor if provided
        features_extractor_path = config.get('features_extractor_module_path', None)
        if features_extractor_path is not None:
            model_kwargs['policy_kwargs'] = config.get('policy_kwargs', {})
            model_kwargs['policy_kwargs']['features_extractor_class'] = import_module_attr(features_extractor_path)
            model_kwargs['policy_kwargs']['features_extractor_kwargs'] = config.get('features_extractor_kwargs', {})
        return model_cls(env=env, **model_kwargs)

    def get_learning_rate(self, model_kwargs):
        """
        Return fixed learning rate or a schedule
        """
        lr_config_or_value = model_kwargs.get('learning_rate')
        if isinstance(lr_config_or_value, dict):
            schedule = lr_schedule(lr_config_or_value)
            return schedule(schedule.init_lr)
        return lr_config_or_value

    def train(self):
        """
        Set the model to training mode.
        Note: This would generally just be self.model.train
              But SB3's model.train() actually does the training
        """
        self.model.policy.set_training_mode(True)

    def eval(self):
        """
        Set the model to eval mode.
        """
        self.model.policy.set_training_mode(False)

    def training_step(self, batch, batch_idx):
        """
        SB3 model's learn() method is called in the trainer's fit() method
        So, training_step() is not called.
        """
        raise NotImplementedError
    
    def evaluation_step(self, batch, batch_idx):
        """
        SB3 model's learn() method is called in the trainer's fit() method
        So, evaluation_step() is not called.
        """
        raise NotImplementedError
    
    def select_action(self, obs, batched=True):
        action, _ = self.model.predict(obs, deterministic=True)
        return action
        
    def sample_action(self, obs, batched=True):
        action, _ = self.model.predict(obs, deterministic=False)
        return action
    
    @property
    def training(self):
        """
        Return true if the SB3 policy is in training mode
        """
        return self.model.policy.training

class SB3RLTrainer(RLTrainer):
    def __init__(self, config: Dict, model: Model = None, logger: Logger = None) -> None:
        """
        model: An SB3 model
        """
        super().__init__(config, model, logger)
        self._setup_callbacks(config)

    def _setup_callbacks(self, config):
        self.callbacks = []
        ## Default Callbacks: Training, Evaluation, internal logging
        self.callbacks.append(SB3TrainingCallback(config, trainer=self))
        self.callbacks.append(SB3EvalCallback(config, trainer=self))
        
        # Add other optional callbacks
        if config.get('sb3') and config['sb3'].get('callbacks'):
            for cbk_name,cbk_config in config['sb3']['callbacks'].items():
                callback_cls = import_module_attr(cbk_config['module_path'])
                callback = callback_cls(cbk_config, trainer=self)
                self.callbacks.append(callback)
            

    def fit(self, num_train_steps: int=None, trainer_state=None) -> None:
        self.num_train_steps = num_train_steps or self.config['num_train_steps']
        self.before_train(trainer_state)

        with self.terminal_display:
            self.model.model.learn(total_timesteps=self.num_train_steps,
                                callback=self.callbacks,
                                reset_num_timesteps=False)
        self.after_train(save_buffer_after_train=self.save_buffer_after_train) 
        
    def setup_data(self, config: Dict):
        # Setup the environment
        self._setup_env(config['env'])

        # Set up other config
        self.num_eval_episodes = self.config.get('num_eval_episodes', 3)

        # Queues needed for async evaluation
        self.eval_job = None
        self.eval_job_step = None
        self.eval_job_log_file = None
        self.eval_job_output_file = None

    def _setup_env(self, env_config):
        """
        Don't use TrainerEnv, instead use SB3's VecEnv
        """
        self.env = sb3_make_env(env_id= partial(self.make_env, env_config), 
                                n_envs=env_config['num_envs'])

    def after_epoch(self, epoch_info=None):
        self.epoch += 1

        if (self.save_checkpoint_freq is not None) and (self.epoch % int(self.save_checkpoint_freq) == 0) and (self.step > 0):
            self._save_checkpoint(ckpt_state_args={
                'save_buffer':True,
                'save_optimizers':True,
                'save_logs': False
            }, 
            # TODO: Currently checkpoint is not logged (e.g. to wandb) for speed
            #       create flag for this
            log_checkpoint=False 
            )
            self.num_checkpoint_saves += 1
            self.logger.log(log_dict={'trainer_step': self.step, 'checkpoint/num_checkpoint_saves': self.num_checkpoint_saves})

        if (self.log_epoch_freq is not None) and (self.epoch % self.log_epoch_freq == 0) and (self.step > 0):
            self.log_epoch(epoch_info)

        if self.queue is not None:
            self.queue.after_epoch(self)

    def log_epoch(self, info=None):
        # Log episode metrics from the internal SB3 logger

        if self.async_eval:
            raise NotImplementedError
        else:
            if self.progress_eval is not None:
                self.progress_eval.update(0, completed=0)

        # Log train metrics
        sb3_log_dict = self.model.internal_logger.name_to_value
        sb3_log_dict['trainer_step'] = self.step
        self.logger.log(log_dict=sb3_log_dict)

        # Log eval metrics
        if len(self.eval_log) > 0:
            last_item = self.eval_log[-1]
            log_dict={'eval_step': int(last_item['step']),
                                      'eval/episode_reward_avg': float(last_item['log']['avg_episode_reward']['avg']),
                                      'eval/episode_reward_std': float(last_item['log']['avg_episode_reward']['std'])}
            # Set the current score
            if self.score_mode == 'max_eval_reward':
                self.score = self.eval_log[-1]['log']['avg_episode_reward']['avg']
            if self.score is not None:
                log_dict.update({'score':self.score})
            self.logger.log(log_dict=log_dict)

    
    def _get_checkpoint_state(self, save_optimizers, **kwargs):
        """
        Note: OnPolicyAlgorithms use a ``rollout_buffer`` instead of a ``replay_buffer``. 
              The ``rollout_buffer`` is reset each time ``collect_rollouts`` is called at every iteration. 
              So there is no need to save the buffer
        """
        trainer_state = self._get_trainer_state()
        sb3_model_state = self.model.state_dict(include_optimizers=save_optimizers)
        trainer_state['sb3_trainer'] = sb3_model_state.pop('sb3_trainer')
        ckpt_state = {
            'trainer': trainer_state,
            'model': sb3_model_state,
        }
        return ckpt_state

    def _load_checkpoint(self, chkpt_name='ckpt', log_checkpoint=True, load_buffer=False):
        """
        Same as super(), but don't load the buffer by default
        """
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Download ckpt.zip and extract its files = [trainer_ckpt.pt, sb3_ckpt.zip]
                self.input_storage.download('ckpt.zip', tmp_dir, extract_archives=True)
                # Load SB3 model from checkpoint
                env = self.model.model.get_env() # Get current env
                internal_logger = self.model.internal_logger
                self.model.model = self.model.model.load(os.path.join(tmp_dir, 'sb3_ckpt.zip'))
                self.model.model.env = env # Assign env since model.load() makes model from scratch
                self.model.model.set_logger(internal_logger)
                # Load trainer state
                self.init_trainer_state(torch.load(os.path.join(tmp_dir, 'trainer_ckpt.pt')))
                # Note: On policy SB3 algo doesn't have a saved replay buffer
        except Exception as e:
            if isinstance(e, FileNotFoundError):
                # Checkpoint may not have been created yet; so don't raise error
                return e
            else:
                # Something went wrong with loading existing checkpoint; raise error
                self.logger.log(log_dict={'trainer_step': self.step, 'checkpoint_load_error': 1})
                self.logger.tag('ckpt_load_err')
                raise e
        
        self.num_checkpoint_loads += 1
        if log_checkpoint:
            # Log the checkpoint load event
            self.logger.log(key='checkpoint/num_checkpoint_loads', value=self.num_checkpoint_loads, step=self.logger._sw.step)


    def _create_checkpoint_files(self, temporary_dir=True, archive=True, ckpt_state_args=None, **kwargs):
        """
        Save checkpoint files to storage
        """
        archive_file_names = []

        if ckpt_state_args is None:
            ckpt_state_args = {}
        ckpt_name = kwargs.get('ckpt_name', 'ckpt')
        storage = self.tmp_storage if temporary_dir else self.output_storage
        # Save sb3's checkpoint zip file
        self.model.model.save(path=f"{storage.dir}/sb3_ckpt.zip")
        archive_file_names.append('sb3_ckpt.zip')

        # Save trainer's checkpoint 
        trainer_state = self._get_trainer_state()
        storage.save(f"trainer_ckpt.pt", trainer_state, 'torch')
        archive_file_names.append('trainer_ckpt.pt')

        # Save the model's replay buffer
        model = self.model.model # The SB3 model
        if hasattr(model, 'replay_buffer'):
            storage.save('sb3_buffer.pt', model.replay_buffer, 'torch')
            archive_file_names.append('sb3_buffer.pt')
        
        # Some callbacks may have their own replay buffers
        for callback in self.callbacks:
            if hasattr(callback, 'replay_buffer'):
                cbk_name = type(callback).__name__
                storage.save(f'sb3_buffer_callback_{cbk_name}.pt', callback.replay_buffer, 'torch')
                archive_file_names.append(f'sb3_buffer_callback_{cbk_name}.pt')

        # Create a single zip with [sb3_ckpt.zip, trainer_ckpt.pt]
        storage.make_archive(archive_file_names, zipfile_name=f'{ckpt_name}.zip')
        # Delete original files that have been archived
        storage.delete(files=archive_file_names)


    def setup_evaluator(self):

        eval_env_config = deepcopy(self.config['env'])
        if self.config.get('eval_env') is not None:
            eval_env_config.update(self.config.get('eval_env'))
        # Evaluator's input storage need to load checkpoints
        # So, it's set to trainer's output storage
        eval_storage_config = {}
        eval_storage_config['input'] = deepcopy(self.config['storage']['output'])
        # Evaluator will store results in trainer's output storage
        eval_storage_config['output'] = deepcopy(self.config['storage']['output'])
        # Get eval metrics
        
        evaluator_config = {
            'project': self.project,
            'run': self.run,
            'num_envs': eval_env_config['num_envs'],
            'max_eval_jobs': 1,
            'async_eval': False, # Evalutor does not run async; even if trainer runs async evals
            'model_name': 'ckpt.zip',   
            'envs': {'eval_env': eval_env_config},
            'storage' : eval_storage_config,
        }

        eval_metrics = None
        if self.config.get('eval_metrics') is not None:
            # Set up eval meitrcs (in addition to the default metrics)
            eval_metrics = {}
            for metric_name, metric_config in self.config['eval_metrics'].items():
                eval_metrics[metric_name] = Metric(metric_config)


        self.evaluator = TrainingRLEvaluator(evaluator_config, self, metrics=eval_metrics)
        self.evaluator.set_model(self.model)

    def evaluate(self, async_eval=False, **kwargs):
        """"
        Evaluation outputs are written onto a global dict
        to allow for asynchronous evaluation
        """
        suffix = kwargs.get('suffix', None)
        assert hasattr(self, 'evaluator') and (self.evaluator is not None), 'Evaluator is not set!'
        if not async_eval:
            if self.progress_eval is not None:
                self.progress_eval.update(0, description='[yellow] Running Evaluation...', completed=0)
            # Update the evaluator's model
            self.evaluator.set_model(self.model)
            # self.evaluator.load_model(state_dict=self.model.state_dict())
            eval_output = self.evaluator.run_eval(suffix=suffix)
            self.eval_log.append({'step': self.step, 'log': eval_output['eval_env']})
            if self.progress_eval is not None:
                self.progress_eval.update(0, description='[green] Complete!', completed=0)
            # self.progress_eval.update(0, completed=0)
        else:
            self._wait_for_eval_job() # If previous eval job has not finished, wait for it
            self._start_async_eval_job() # Start a new eval job


class CustomCNNFeaturesExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        interim_kernel_size = 4 if n_input_channels == 3 else 1
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=4, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=interim_kernel_size, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
    

class CustomMLPFeaturesExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = th.as_tensor(observation_space.sample()[None]).float().shape[1]

        self.mlp = nn.Sequential(
            nn.Linear(n_flatten, features_dim), 
            nn.ReLU(),
            nn.Linear(features_dim, features_dim), 
            # nn.ReLU()
            )
    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.mlp(observations)