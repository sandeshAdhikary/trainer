from trainer.trainer import Trainer
from trainer.rl.envs import TrainerEnv
from typing import Dict
from abc import ABC, abstractmethod
from trainer.rl.replay_buffers import ReplayBuffer
import numpy as np
from rich.panel import Panel
from rich.columns import Columns
from rich.layout import Layout
from rich.live import Live
from warnings import warn
from copy import deepcopy
from functools import partial
from trainer.rl.rl_evaluator import TrainingRLEvaluator
import subprocess
import tempfile
from glob import glob
import os
import torch
class RLTrainer(Trainer, ABC):

    def setup_data(self, config: Dict):
        # Setup the environment
        self._setup_env(config['env'])

        # Setup the replay buffer
        env_shapes = self.env.get_env_shapes()
        self.replay_buffer = ReplayBuffer(
            obs_shape=env_shapes['obs_shape'],
            action_shape=env_shapes['action_shape'],
            capacity=config['replay_buffer']['replay_buffer_capacity'],
            batch_size=config['batch_size'],
            device=self.device,
        )
        # Max size of chunks when saving replay buffer
        self.replay_buffer_chunk_len = 10_000

        # Set up other config
        self.num_eval_episodes = self.config.get('num_eval_episodes', 3)

        # Queues needed for async evaluation
        self.eval_job = None
        self.eval_job_step = None
        self.eval_job_log_file = None
        self.eval_job_output_file = None
        
    
    def _setup_terminal_display(self, config: Dict) -> None:
        super()._setup_terminal_display(config)
        if self.progress is not None:
            # Get the evaluation progress bar from the evaluator
            self.progress_eval = self.evaluator.progress_bar
            self._term_eval_panel = Panel.fit(Columns([self.progress_eval]), title="Evaluation", border_style="steel_blue3")

            orig_panels = self._term_layout.children
            self._term_layout = Layout()
            self._term_layout.split(
                *orig_panels,
                Layout(self._term_eval_panel, size=5, name="evaluation"),
            )
            self.terminal_display = Live(self._term_layout, 
                                        screen=False, 
                                        refresh_per_second=config.get('terminal_refresh_rate', 1))


    @abstractmethod
    def make_env(self, args):
        raise NotImplementedError

    def env_fns(self, config):
        """
        config: config dict defining the envs
        return: env_fn, eval_env_fn i.e. env generating functions that can be executed as env_fn()
        """
        if config['num_envs'] > 1:
            env_fns = [partial(self.make_env, config) for _ in range(config['num_envs'])]
        else:
            env_fns = partial(self.make_env, config)
        
        return env_fns

    def _setup_env(self, env_config):
        env_fns = self.env_fns(env_config)
        self.env = TrainerEnv(env_fns)

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

    def _get_checkpoint_state(self, save_optimizers=True, save_buffer=True):
        # Get ckpt state for trainer and model
        ckpt_state = super()._get_checkpoint_state(save_optimizers=save_optimizers)
        # Add buffer state to the checkpoint
        if save_buffer:
            buffer_state = self.replay_buffer.state_dict()
            buffer_len = len(buffer_state['obses'])
            buffer_no = 0
            for idx in range(0, buffer_len, self.replay_buffer_chunk_len):
                buffer_chunk = {k: v[idx:idx+self.replay_buffer_chunk_len] for k,v in buffer_state.items() if k != 'idx'}
                ckpt_state.update({
                    f'replay_buffer_{buffer_no}': buffer_chunk
                })
                buffer_no += 1
            
            # TODO: Split up replay buffer into chunks here?
            # if 'buffer_' in self.replay_buffer.keys():
            #   # Save buffer in chunks
            # else:
            # Just save the single buffer
            # ckpt_state.update({
            #     'replay_buffer': self.replay_buffer.state_dict()
            # })
        return ckpt_state

    def setup_evaluator(self):

        eval_env_config = deepcopy(self.config['env'])
        eval_env_config.update(self.config['eval_env'])
        # Evaluator's input storage need to load checkpoints
        # So, it's set to trainer's output storage
        eval_storage_config = {}
        eval_storage_config['input'] = deepcopy(self.config['storage']['output'])
        # Evaluator will store results in trainer's output storage
        # but inside a separate eval folder
        eval_storage_config['output'] = deepcopy(self.config['storage']['output'])
        evaluator_config = {
            'project': self.project,
            'run': self.run,
            'num_envs': self.config['eval_env']['num_envs'],
            'max_eval_jobs': 1,
            'async_eval': False, # Evalutor does not run async; even if trainer runs async evals
            'model_name': 'ckpt.zip',   
            'envs': {'eval_env': eval_env_config},
            'storage' : eval_storage_config
        }

        self.evaluator = TrainingRLEvaluator(evaluator_config, self)
        self.evaluator.set_model(self.model)

    def evaluate(self, async_eval=False):
        """"
        Evaluation outputs are written onto a global dict
        to allow for asynchronous evaluation
        """
        assert hasattr(self, 'evaluator') and (self.evaluator is not None), 'Evaluator is not set!'
        if not async_eval:
            # Update the evaluator's model
            self.evaluator.load_model(state_dict=self.model.state_dict())
            eval_output = self.evaluator.run_eval()
            self.eval_log.append({'step': self.step, 'log': eval_output['eval_env']})
        else:
            self._wait_for_eval_job() # If previous eval job has not finished, wait for it
            self._start_async_eval_job() # Start a new eval job


    def _wait_for_eval_job(self):
        """
        If an eval job is already running, wait for it to finish
        Record the stderr output of job, in case it is not successful
        """
        if self.eval_job is not None:
            
            # Wait until eval job is done
            job_status = None
            while job_status is None:
                job_status = self.eval_job.poll()
                try:
                    eval_job_log = self.output_storage.load(self.eval_job_log_file, filetype='text')
                    step = eval_job_log.split('\n')[-2].split('step:')[-1]
                    self.progress_eval.update(0, completed=float(step))
                except FileNotFoundError:
                    # File may not have been populated yet
                    pass

            err = None if job_status == 0 else self.eval_job.stderr.readlines()
            if err is None:
                # Add evaluation results to self.eval_log
                eval_output = self.output_storage.load(self.eval_job_output_file, filetype='torch')
                self.eval_log.append({'step': self.eval_job_step, 'log': eval_output})
            else:
                self.output_storage.save(f"eval_job_errs.txt", '\n'.join(err), filetype='text', write_mode='w')
                raise Exception(f"Eval job failed with error. Check error log at {self.output_storage.storage_path('eval_job_errs.txt')}")
            
            # Reset eval_job and tracked files
            self.eval_job = None 
            self.eval_job_log_file = None
            self.eval_job_output_file = None
            self.eval_job_step = None

    def _start_async_eval_job(self):
        
        # Create an eval packet so the async_evaluator can recreate evaluator and model
        tmp_evaluator_config = deepcopy(self.evaluator.config)
        tmp_evaluator_config['envs'] = {'eval_env': self.evaluator.config['envs']['eval_env']}
        tmp_evaluator_config['async_eval'] = False
        tmp_evaluator_config['save_output'] = False
        tmp_trainer_config = deepcopy(self.config)
        # Make replay buffer tiny to save memory
        tmp_trainer_config['replay_buffer']['replay_buffer_capacity'] = 1
        self.eval_job_log_file = "eval_log_eval_env.txt"
        self.eval_job_output_file = "eval_output_eval_env.pt"
        eval_packet = {
            'evaluator': {
                'module': self.evaluator.module_path, 
                'class': self.evaluator.__class__.__name__,
                'config': tmp_evaluator_config
                },
            'trainer': {
                'module': self.module_path,
                'class': self.__class__.__name__,
                'config': tmp_trainer_config
                },
            'model': {
                'module': self.model.module_path,
                'class': self.model.__class__.__name__,
                'config': self.model.config,
                'state_dict': self.model.state_dict()
            },
            'eval_storage': 'output'
            }

        eval_packet_name = "training_eval_packet.pkl"
        self.tmp_storage.save(eval_packet_name, eval_packet, filetype='pickle')
        command = f"""python -m trainer.evaluator_async \
            --eval_packet {self.tmp_storage.storage_path(eval_packet_name)}
            """
        self.eval_job = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True  # Enable text mode (for string output)
        )
        self.eval_job_step = self.step

    def after_epoch(self, epoch_info=None):
        # Update state from rollout data
        rollout_data = epoch_info['rollout']
        # Update trainer state
        self.reward = rollout_data['reward']
        self.obs = rollout_data['next_obs']
        self.done = rollout_data['terminated'] | rollout_data['truncated']
        self.step += rollout_data['num_steps']
        self.epoch += 1
        self.num_model_updates += epoch_info['num_model_updates']
        self.current_episode_reward += self.reward
        
        if not self.env.is_vec_env:
            # Check if episodes are done
            if self.done:
                # Reset environment
                self.obs = self.env.reset()
                # Increment episode counter
                self.num_episodes += self.done
                # Update episode rewards
                self.episode_reward_list.append(self.current_episode_reward)
                # Clear current episode reward tracker once done
                self.current_episode_reward = 0
        else:
            if any(self.done):
                # Note: No need to reset since VecEnv resets automatically after an episode is done
                # Increment episode counter
                self.num_episodes += self.done
                # Update episode rewards
                for idx in range(self.env.num_envs):
                    if self.done[idx]:
                        # Record episode reward in list
                        self.episode_reward_list[idx].append(self.current_episode_reward[idx])
                        # Clear current episode reward tracker if done
                        self.current_episode_reward[idx] = 0

        # Log replay buffer size
        self.logger.log(log_dict={'trainer_step': self.step, 
                                  'train/replay_buffer_idx': self.replay_buffer.idx})

        # Save checkpoint
        if (self.save_checkpoint_freq is not None) and (self.epoch % self.save_checkpoint_freq == 0) and (self.step > 0):
            self._save_checkpoint(ckpt_state_args={
                'save_buffer':True,
                'save_optimizers':True
            })
            self.num_checkpoint_saves += 1
            self.logger.log(log_dict={'trainer_step': self.step, 'checkpoint/num_checkpoint_saves': self.num_checkpoint_saves})


        self.log_epoch(epoch_info)

    def after_train(self, info=None):
        """
        Callbacks after training ends
        """
        self.log_train(info)
        self._save_checkpoint(ckpt_state_args={
            'save_buffer':False,
            'save_optimizers':False
        })
        self.logger.finish()


    def _load_checkpoint(self, chkpt_name='ckpt', log_checkpoint=True):
        try:
            # Restore checkpoint: model,trainer,replay_buffer
            # ckpt = super()._load_checkpoint_dict(chkpt_name=chkpt_name,
            #                                      filenames=['model_ckpt.pt', 'trainer_ckpt.pt', 'replay_buffer_ckpt.pt'],
            #                                      filetypes=['torch', 'torch', 'torch']
            #                                      )
            # Download the checkpooint
            ckpt = {}
            with tempfile.TemporaryDirectory() as tmp_dir:
                self.input_storage.download('ckpt.zip', tmp_dir, extract_archives=True)
                # Load train_ckpt
                ckpt['trainer'] = torch.load(os.path.join(tmp_dir, 'trainer_ckpt.pt'))
                ckpt['model'] = torch.load(os.path.join(tmp_dir, 'model_ckpt.pt'))
                try:
                    # Single replay buffer file
                    ckpt['replay_buffer'] = torch.load(os.path.join(tmp_dir, 'replay_buffer_ckpt.pt'))
                except FileNotFoundError:
                    # Replay buffer is split into chunks
                    buffer_files = sorted(glob(os.path.join(tmp_dir, 'replay_buffer_*_ckpt.pt')))
                    buffer_ckpt = torch.load(buffer_files[0])
                    for buffer_file in buffer_files[1:]:
                        buffer_chunk = torch.load(buffer_file)
                        for k in buffer_ckpt.keys():
                                if isinstance(buffer_ckpt[k], torch.Tensor):
                                    buffer_ckpt[k] = torch.cat([buffer_ckpt[k], buffer_chunk[k]], dim=0)
                                elif isinstance(buffer_ckpt[k], np.ndarray):
                                    buffer_ckpt[k] = np.concatenate([buffer_ckpt[k], buffer_chunk[k]], axis=0)
                                else:
                                    raise ValueError('Unknown buffer type')
                    buffer_ckpt['idx'] = buffer_ckpt['obses'].shape[0]
                    ckpt['replay_buffer'] = buffer_ckpt
            # if filenames is None:
            #     # Assume only model and trainer checkpoints
            #     filenames = ['model_ckpt.pt', 'trainer_ckpt.pt']
            # if filetypes is None:
            #     # Assume all files are torch-loadable
            #     filetypes = ['torch']*len(filenames)
            # # Download checkpoint from logger
            # if self.config['load_checkpoint_type'] == 'torch':
            #     ckpt = self.input_storage.load(f'model_{chkpt_name}.pt')
            # elif self.config['load_checkpoint_type'] == 'zip':
            #     self.input_storage.download(archive_name, tmp_dir, extract_archives=True)
            #     ckpt = self.input_storage.load_from_archive("ckpt.zip", 
            #                                                 filenames=filenames,
            #                                                 filetypes=filetypes)

        # return ckpt
        except (UserWarning, Exception) as e:
            self.logger.log(log_dict={'trainer_step': self.step, 'checkpoint_load_error': 1})
            warn(f"Checkpoint load error: Could not load state dict. Error: {e.args}")
            return e
        # Update model, trainer and buffer
        try:
            self.model.load_model(state_dict=ckpt['model'])
            self.init_trainer_state(ckpt['trainer'])
            self.replay_buffer.load_state_dict(ckpt['replay_buffer'])
        except (UserWarning, Exception) as e:
            self.logger.log(log_dict={'trainer_step': self.step, 'checkpoint_load_error': 1})
            warn(f"Checkpoint load error: State dicts loaded, but could not update model, trainer or buffer. Error: {e.args}")
            return e

        if log_checkpoint:
            # Log the checkpoint load event
            self.num_checkpoint_loads += 1
            self.logger.log(key='checkpoint/num_checkpoint_loads', value=self.num_checkpoint_loads, step=self.logger._sw.step)

    def log_step(self, info=None):
        pass

    def log_epoch(self, info=None):
        # Log episode metrics when an episode is done
        ep_done = any(self.done) if self.env.is_vec_env else self.done
        if ep_done:
            # Log episode rewards (if all envs have completed at least one episode)
            if (self.step > 0) and (all([x >=1 for x in self.num_episodes])):
                avg_ep_reward = np.mean([x[-1] for x in self.episode_reward_list])
                self.logger.log(log_dict={'trainer_step': self.step,
                                 'train/episode_reward': avg_ep_reward})
        if self.epoch % self.logger.log_freq == 0:
            self.logger.log(log_dict={'trainer_step': self.step,
                            'train/episode': sum(self.num_episodes)})
            self.logger.log(log_dict={'trainer_step': self.step,
                            'train/num_model_updates': self.num_model_updates})


        # If async eval, check log files for progress

        if self.eval_job_log_file is not None:
            try:
                eval_job_log = self.output_storage.load(self.eval_job_log_file, filetype='text')
                # Log progress
                step = eval_job_log.split('\n')[-2].split('step:')[-1]
                self.progress_eval.update(0, completed=float(step))
            except FileNotFoundError as e:
                pass
                    
        # Log eval metrics
        if len(self.eval_log) > 0:
            last_item = self.eval_log[-1]
            self.logger.log(log_dict={'eval_step': int(last_item['step']),
                                      'eval/episode_reward_avg': float(last_item['log']['episode_rewards_avg']),
                                      'eval/episode_reward_std': float(last_item['log']['episode_rewards_std'])}
                                      )

    def log_train(self, info=None):
        pass

    def collect_rollouts(self, obs, add_to_buffer=False):
        # Get Action
        if self.step < self.config['init_steps']:
            action = self.env.action_space.sample()
        else:
            action = self.model.sample_action(obs, batched=self.env.is_vec_env)
        
        # Env Step
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        if add_to_buffer:
            # TODO: Move this to after_epoch()
            # Add to buffer: allow infinite bootstrap: don't store truncated as done
            curr_reward = self.reward
            self.replay_buffer.add(obs, action, curr_reward, reward, next_obs, terminated, 
                                   batched=self.env.is_vec_env)

        num_steps = self.env.num_envs

        return {
            'action': action,
            'next_obs': next_obs,
            'reward': reward,
            'terminated': terminated,
            'truncated': truncated,
            'info': info,
            'num_steps': num_steps
        }

    def get_num_updates(self):
        """
        No updates until init_steps have been reached.
        At init_steps-th step, update init_steps times
        After that update as many times as there are environments
        """
        if self.step >= self.config['init_steps']:
            # Update only after init_steps
            return self.config['init_steps'] if self.step == 0 else self.env.num_envs
        return 0

    def init_trainer_state(self, state_dict=None):
        state_dict = state_dict or {}
        # Initialize/update step, train/eval logs, etc.
        super().init_trainer_state(state_dict)
        # Update RL specific params
        self.obs = state_dict.get('obs', self.env.reset()[0])
        self.done = state_dict.get('done', [False]*self.env.num_envs)
        self.reward = state_dict.get('reward', [0]*self.env.num_envs)
        # Set up counters
        self.num_episodes = state_dict.get('num_episodes', np.zeros(self.env.num_envs))
        self.episode = state_dict.get('episode', np.zeros(self.env.num_envs))
        self.current_episode_reward = state_dict.get('current_episode_reward', np.zeros(self.env.num_envs))
        self.episode_reward_list = state_dict.get('episode_reward_list', [[] for _ in range(self.env.num_envs)])
        self.num_model_updates = state_dict.get('num_model_updates', 0)
        self.num_checkpoint_saves = state_dict.get('num_checkpoint_saves', 0)
        self.num_checkpoint_loads = state_dict.get('num_checkpoint_loads', 0)


        
        


