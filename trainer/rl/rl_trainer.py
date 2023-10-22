from trainer.trainer import Trainer
from trainer.rl.envs import TrainerEnv
from typing import Dict
from abc import ABC, abstractmethod
from trainer.rl.replay_buffers import ReplayBuffer
import numpy as np
from einops import rearrange
import os
import pickle
import subprocess
import re
import torch
from rich.panel import Panel
from rich.columns import Columns
from rich.layout import Layout
from rich.live import Live
from rich.progress import Progress, BarColumn, TextColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from warnings import warn
from copy import deepcopy
from functools import partial

class RLTrainer(Trainer, ABC):

    def setup_data(self, config: Dict):
        # Setup the environment
        self._setup_env(config['env'], config['eval_env'])

        # Setup the replay buffer
        env_shapes = self.env.get_env_shapes()
        self.replay_buffer = ReplayBuffer(
            obs_shape=env_shapes['obs_shape'],
            action_shape=env_shapes['action_shape'],
            capacity=config['replay_buffer']['replay_buffer_capacity'],
            batch_size=config['batch_size'],
            device=self.device,
        )

        # Set up other config
        self.num_eval_episodes = self.config.get('num_eval_episodes', 3)
        
    
    def _setup_terminal_display(self, config: Dict) -> None:
        super()._setup_terminal_display(config)
        if self.progress is not None:

            # Add evaluation progress bar
            self.eval_progress_bar = Progress(
                TextColumn("Evaluation Progress"),
                BarColumn(complete_style="steel_blue3"),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                redirect_stdout=False
                )
            self.progress_eval = self.eval_progress_bar.add_task("[steel_blue3] Evaluating...", 
                                                             total=self.eval_env.max_episode_steps * self.num_eval_episodes)
            
            orig_panels = self._term_layout.children
            self._term_eval_panel = Panel.fit(Columns([self.eval_progress_bar]), title="Evaluation", border_style="steel_blue3")
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

    def _setup_env(self, env_config, eval_env_config_input=None):
        # Set up env function
        env_fns = self.env_fns(env_config)

        # Set up eval env function
        if eval_env_config_input is None:
            eval_env_config = deepcopy(env_config)
            # Create another copy of the training env with a different seed
            eval_env_config.update({'seed': env_config['seed'] + 1})
            eval_env_fns = self.env_fns(eval_env_config)
        else:
            num_envs = eval_env_config_input.get('num_envs', 1)
            if num_envs < 2:
                # Create eval_env config and make an env function (with different seed)
                eval_env_config = deepcopy(env_config)
                eval_env_config.update(eval_env_config_input)
                eval_env_config.update({'seed': env_config['seed'] + 1})
                eval_env_fns = self.env_fns(eval_env_config)
            else:
                # Create multiple eval_env_configs, each with different seed
                eval_env_fns = []
                for idx in range(num_envs):
                    eval_env_config = deepcopy(env_config)
                    eval_env_config.update(eval_env_config_input)
                    eval_env_config.update({'seed': env_config['seed'] + idx, 
                                            'num_envs': 1})
                    eval_env_fns.append(self.env_fns(eval_env_config))

        self.env = TrainerEnv(env_fns)
        self.eval_env = TrainerEnv(eval_env_fns)

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
                self.evaluate(training_mode=True, async_eval=self.async_eval)
                self.after_epoch({'rollout': rollout_data, 'num_model_updates': num_updates})
                
        self.after_train() 

    def _create_checkpoint_files(self, chkpt_name='checkpoint', chkpt_dir=None, save_optimizers=True, save_buffer=True, **kwargs):
        """
        """
        # Create a (temporary) checkpoint directory
        chkpt_dir = chkpt_dir or os.path.join(self.logger.logdir, chkpt_name)

        # Save trainer state
        trainer_state = {'step': self.step,
                         'epoch': self.epoch,
                         'train_log': self.train_log, 
                         'eval_log': self.eval_log,
                         'train_end': self.train_end,
                         'num_checkpoint_loads': self.num_checkpoint_loads,
                         'obs': self.obs,
                         'done': self.done,
                         'reward': self.reward,
                         'num_episodes': self.num_episodes,
                         'episode': self.episode,
                         'current_episode_reward': self.current_episode_reward,
                         'episode_reward_list': self.episode_reward_list,
                         'num_model_updates': self.num_model_updates,
                         'epoch': self.epoch
                        }

        torch.save(trainer_state, os.path.join(chkpt_dir, f'trainer_{chkpt_name}.pt'))
        # Save model state
        self.model.save_model(os.path.join(chkpt_dir, f'model_{chkpt_name}.pt'), save_optimizers=save_optimizers)
        
        # Save replay buffer 
        if save_buffer:
            # Create buffer to store replay buffer files
            buffer_save_folder = os.path.join(chkpt_dir, 'replay_buffer')
            os.makedirs(buffer_save_folder, exist_ok=True)
            self.replay_buffer.save(buffer_save_folder)
                                                


    def evaluate(self, max_ep_steps=None, training_mode=False, async_eval=False, eval_log_file=None, eval_output_file=None):
        """"
        Evaluation outputs are written onto a global dict
        to allow for asynchronous evaluation
        """
    
        run_eval = True
        if training_mode:
            run_eval = (self.step > self.config['init_steps']) and (self.epoch % self.eval_freq == 0) 

        if run_eval:
            self.model.eval()
            evaluation_fn = self.evaluate_vec if self.env.is_vec_env else self.evaluate_nonvec
            if async_eval:
                
                # Check if there is an existing evaluation in progress
                if hasattr(self, 'eval_job') and self.eval_job is not None:
                    eval_job_status = self.eval_job.poll()
                    if eval_job_status is None:
                        # Wait for the previous job to finish
                        # print("Waiting for previous evaluation...")
                        eval_job_status = self.eval_job.poll()
                        while eval_job_status is None:
                            if os.path.exists(self.eval_log_file):
                                with open(self.eval_log_file, 'r') as f:
                                    try:
                                        step = f.readlines()[-1].strip("step:")
                                        step = float(re.sub(r'[\t\n]', '', step))
                                        if hasattr(self, 'progress_eval'):
                                            self.eval_progress_bar.update(self.progress_eval, completed=step)
                                    except IndexError:
                                        pass 
                            eval_job_status = self.eval_job.poll()

                    if eval_job_status == 0:
                        # Evaluation ended without errors
                        # self.eval_output_file = self.eval_job.args.split("--eval_output_file ")[-1].split(" ")[0]
                        eval_step = self.eval_job.args.split("--step ")[-1].split(" ")[0]
                        eval_step = re.sub(r'[\t\n]', '', eval_step)
                        with open(self.eval_output_file, 'rb') as f:
                            eval_output = pickle.load(f)

                        self.eval_log.append({'step': eval_step, 'log': eval_output})
                        # print("Evaluation complete")
                        self.eval_job = None
                        
                    else:
                            err_file = os.path.join(self.logger.logdir, 'error_log.txt')
                            with open(err_file, 'w') as f:
                                for line in self.eval_job.stderr:
                                    f.writelines(line)
                            raise SystemError(f"Evaluation ended with errors. See {err_file} for details")
                    
                # Create a dir to save eval_checkpoints
                eval_chkpt_dir = os.path.join(self.logger.logdir, 'eval_checkpoint')
                # os.makedirs(eval_chkpt_dir, exist_ok=True)
                
                # Save state dicts
                self._save_checkpoint(chkpt_name='eval_checkpoint', log_checkpoint=False, save_buffer=False)
                # Save trainer and model info so we can import them again
                trainer_info = {'trainer_class': self.__class__, 
                                'model_class': self.model.__class__,
                                'trainer_config': self.config,
                                'model_config': self.model.config}
                with open(os.path.join(eval_chkpt_dir, "eval_trainer_info.pkl"), 'wb') as f:
                    pickle.dump(trainer_info, f)
                # Launch asynchronous processs to evaluate the model
                self.eval_output_file = os.path.join(eval_chkpt_dir, 'eval_out')
                self.eval_log_file = os.path.join(eval_chkpt_dir, 'eval_log')
                command = f"""python -m trainer.rl.rl_evaluator \
                    --trainer_info {os.path.join(eval_chkpt_dir, 'eval_trainer_info.pkl')} \
                    --eval_checkpoint {os.path.join(eval_chkpt_dir, 'model_eval_checkpoint.pt')} \
                    --eval_log_file {self.eval_log_file} \
                    --eval_output_file {self.eval_output_file} \
                    --step {self.step}
                    """
                self.eval_job = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True  # Enable text mode (for string output)
                )

            else:
                eval_step_output = evaluation_fn(eval_log_file=eval_log_file, max_ep_steps=max_ep_steps, eval_output_file=eval_output_file)
                if hasattr(self, 'eval_log'):
                    self.eval_log.append({'step': self.step, 'log': eval_step_output})
                if hasattr(self, 'progress_eval'):
                    self.eval_progress_bar.update(self.progress_eval, completed=0)

    def evaluate_nonvec(self, max_ep_steps=None, eval_log_file=None):
        """
        Evaluation step for non-vectorized environments
        """
        raise NotImplementedError

    def evaluate_vec(self, max_ep_steps=None, eval_log_file=None, eval_output_file=None):
        """
        Evaluation step for vectorized environments
        if eval_log_file is provided, write evaluation metrics to file
        """
        eval_info = {}

        eval_env = self.eval_env

        max_ep_steps = max_ep_steps or eval_env.max_episode_steps
        num_frames = eval_env.frames

        obses = []       

        # eval_env.max_episode_steps * 

        obs, info = eval_env.reset() # obs: (num_env, *obs_shape)
        dones = [False] * eval_env.num_envs
        episode_reward_list = []
        steps = 0
        steps_to_keep = [None]*eval_env.num_envs
        num_eps = [0]*eval_env.num_envs

        # Create a tmp file to write progress to
        max_steps = self.num_eval_episodes*max_ep_steps

        # Eval log context
        for i in range(max_steps):
            if eval_log_file is not None:
                with open(eval_log_file, 'a') as f:
                    f.write(f"step:{i}\n")

            if hasattr(self, 'progress_eval'):
                self.eval_progress_bar.update(self.progress_eval, completed=i)

            # No  need to reset again since vec_env resets individual envs automatically
            if None not in steps_to_keep:
                # All envs have completed max_eps
                if eval_log_file is not None:
                    with open(eval_log_file, 'a') as f:
                        f.write(f"step:{max_steps}\n")
                break
            steps += 1
            # Get actions for all envs
            action = self.model.select_action(obs, batched=True)

            obses.append(obs)

            obs, reward, terminated, truncated, info = eval_env.step(action)
            dones = [x or y for (x,y) in zip(terminated, truncated)]
    
            for ide in range(eval_env.num_envs):
                if dones[ide]:
                    num_eps[ide] += 1

                if num_eps[ide] >= self.num_eval_episodes:
                    steps_to_keep[ide] = steps

            episode_reward_list.append(reward)

        episode_reward_list = np.array(episode_reward_list)

        max_steps = episode_reward_list.shape[0]
        mask = [np.pad(np.ones(n), (0, max_steps-n),mode='constant') for n in steps_to_keep]
        mask = np.stack(mask, axis=1)
        episode_reward_list *= mask

        # Log video of evaluation observations
        obses = np.stack(obses) # (steps, num_envs, *obs_shape)
        # Stack frames horizontally; Stack episodes along batches
        obses = rearrange(obses, 'b n (f c) h w -> b c (n h) (f w)', f=len(num_frames)) 

        # Get average episode rewards across all environments
        eval_info['episode_rewards_avg'] = episode_reward_list.sum(axis=0).mean()
        eval_info['episode_rewards_std'] = episode_reward_list.sum(axis=0).std()


        # L.log('eval/episode_reward', eval_episode_reward, step)
        if eval_output_file is not None:
            # Write the eval results to file
            with open(eval_output_file, 'wb') as f:
                pickle.dump(eval_info, f)


        return eval_info


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

        # Save checkpoint
        if (self.save_checkpoint_freq is not None) and (self.epoch % self.save_checkpoint_freq == 0) and (self.step > 0):
            self._save_checkpoint()
            os.path.exists('/project/logdir/wandb/run-20231022_181630-q8ey8o5h/files/checkpoint.zip')

            self.num_checkpoint_saves += 1
            self.logger.log(log_dict={'trainer_step': self.step, 'checkpoint/num_checkpoint_saves': self.num_checkpoint_saves})


        self.log_epoch(epoch_info)

    def after_train(self, info=None):
        """
        Callbacks after training ends
        """
        self.log_train(info)
        # Replace existing checkpoint with final checkpoint (without buffer and optimizers)
        self._save_checkpoint(chkpt_name='checkpoint', 
                              log_checkpoint=True, save_buffer=False, save_optimizers=False,
                              overwrite=True)
        self.logger.finish()


    def _load_checkpoint(self, chkpt_name='checkpoint', log_checkpoint=True):
        # Restore model and trainer state to the checkpoint
        load_exception = super()._load_checkpoint(chkpt_name=chkpt_name, log_checkpoint=log_checkpoint)
        # Restore the replay buffer
        if load_exception is None:
            try:
                self.replay_buffer.load(f'{self.logger.logdir}/checkpoint/replay_buffer')
            except (UserWarning, Exception) as e:
                warn("Could not restore replay buffer.")


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
        
        self.logger.log(log_dict={'trainer_step': self.step,
                         'train/episode': sum(self.num_episodes)})
        self.logger.log(log_dict={'trainer_step': self.step,
                         'train/num_model_updates': self.num_model_updates})

        if hasattr(self, 'eval_job') and (self.eval_job is not None):
            if os.path.exists(self.eval_log_file):
                with open(self.eval_log_file, 'r') as f:
                    step = f.readlines()[-1].strip("step:")
                    step = float(re.sub(r'[\t\n]', '', step))
                if hasattr(self, 'progress_eval'):
                    self.eval_progress_bar.update(self.progress_eval, completed=step)

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


