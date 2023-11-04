from abc import ABC, abstractmethod, abstractproperty
from trainer.evaluator import Evaluator
from functools import partial
import subprocess
import queue
from trainer.rl.envs import TrainerEnv
from copy import deepcopy
import numpy as np
from einops import rearrange
from rich.progress import Progress, BarColumn, TextColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
import time
from contextlib import nullcontext

class RLEvaluator(Evaluator, ABC):
    
    def __init__(self, config):
        self.num_eval_episodes = config.get('num_eval_episodes', 1)
        self.display_progress = config.get('display_progress', False)
        super().__init__(config)

    @abstractmethod
    def make_env(self, config):
        raise NotImplementedError

    @abstractproperty
    def module_path(self):
        return None

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

    def setup_data(self):
        self.eval_envs = {}
        for name,config in self.config['envs'].items():
            self.eval_envs[name] = self.setup_env(config)
            # TrainerEnv(self.env_fns(config))


    def setup_env(self, config):
        """
        If using vectorized env with multiple envs, 
        potentially instantiate each eith a different seed
        """
        num_envs = config['num_envs']
        if num_envs > 1:
            env_fns = []
            for idx in range(num_envs):
                env_config = deepcopy(config)
                env_config.update({'seed': config['seed'] + idx, 'num_envs': 1})
                env_fns.append(self.env_fns(env_config))
            eval_env = TrainerEnv(env_fns, vec_env=True)
        else:
            eval_env = TrainerEnv(self.env_fns(config), vec_env=True)
        return eval_env


    def evaluate(self, max_ep_steps=None, async_eval=False, logger=None, progress=None, storage='output'):
        """
        Evaluate the model. 
        Optionally write log and outputs to eval_log_file and eval_output_file
        if async_eval: Run evaluation asynchronously for each environment in self.eval_envs.
                       A separate sub-process is launched for each eval_env.
                       If max_eval_jobs is reached, we wait for the first job in the jobs queue to finish before launching another.
                       All sub-processes write their logs and outputs to the output_storage
                       Once all sub-processes have completed, we collect and combine outputs from all sub-processes
                       If any of the sub-processes returns an error, the error is logged to output_storage
        """
        
        self.eval_jobs = queue.Queue() if async_eval else None
        self.eval_job_errs = [] # To record evaluation errors
        eval_output = {} # Store outputs for each env
        for env_name, env in self.eval_envs.items():
            if async_eval:
                # Start a new async job; if max jobs reached, wait for first job to end
                self._start_async_job(env_name)
            else:
                try:
                    # Pick function for vectorized or non-vectorized environments
                    evaluation_fn = self.vec_eval_step if env.is_vec_env else self.nonvec_eval_step
                    env_output = evaluation_fn(eval_env=self.eval_envs[env_name],
                                               env_name=env_name,
                                               max_ep_steps=max_ep_steps, 
                                               storage=storage
                                               )
                    eval_output[env_name] = env_output
                    self.eval_job_errs.append(None)
                except Exception as e:
                    self.eval_job_errs.append(e.args[0])



        
        # If async, wait until all eval jobs have completed
        if async_eval:
            self._wait_for_all_eval_jobs()
            # If eval failed, log error and break
            self._maybe_log_eval_error()
            eval_output = self._collect_async_eval_results()
        else:
            # If eval failed, log error and break
            self._maybe_log_eval_error()

        return eval_output
    
    def _setup_terminal_display(self):
        # Create a progress bar
        self.progress_bar = Progress(
            TextColumn("Evaluation Progress"),
            BarColumn(complete_style="steel_blue3"),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            redirect_stdout=False
            )
        self.progress_tasks = {}
        for env_name, env in self.eval_envs.items():
            eval_task = self.progress_bar.add_task(f"[steel_blue3] Evaluating {env_name}",
                                                   total=env.max_episode_steps*self.num_eval_episodes
                                                   )
            self.progress_tasks[env_name] = eval_task


    def _maybe_log_eval_error(self):
        if not all([x is None for x in self.eval_job_errs]):
            # One of the eval jobs did not finish
            for err in self.eval_job_errs:
                if err is not None:
                    self.output_storage.save(f"eval_job_errs.txt", '\n'.join(err), filetype='text', write_mode='a')
                    self.output_storage.save(f"eval_job_errs.txt", '\n ____ \n', filetype='text', write_mode='a')
            raise ValueError(f"Evaluation failed. Check error log in output storage at {self.output_storage.storage_path('eval_job_errs.txt')}")

    def _maybe_max_jobs_wait(self):
        if self.eval_jobs.qsize() >= self.config['max_eval_jobs']:
            # Wait until the first eval job has ended before launching another
            first_job = self.eval_jobs.get()
            job_status = None
            while job_status is None:
                job_status = first_job.poll()
            job_err = None if job_status == 0 else first_job.stderr.readlines()
            self.eval_job_errs.append(job_err)

    def _wait_for_all_eval_jobs(self):
        # Wait until all eval jobs have completed
        while self.eval_jobs.qsize() > 0:
            next_job = self.eval_jobs.get()
            job_status = None
            while job_status is None:
                job_status = next_job.poll()
            job_err = None if job_status == 0 else next_job.stderr.readlines()
            self.eval_job_errs.append(job_err)


    def _start_async_job(self, env_name):

        self._maybe_max_jobs_wait()

        # Create a temp config for this env
        tmp_evaluator_config = deepcopy(self.config)
        tmp_evaluator_config['envs'] = {env_name: self.config['envs'][env_name]}
        tmp_evaluator_config['async_eval'] = False
        tmp_evaluator_config['save_output'] = False

        eval_packet = {'evaluator_config': tmp_evaluator_config,
                        'evaluator_class': self.__class__,
                        'model_class': self.model.__class__,
                        'model_config': self.model.config,
                        }

        self.tmp_storage.save(f"{env_name}_eval_packet.pkl", eval_packet, filetype='pickle')

        command = f"""python -m trainer.evaluator_async \
            --eval_packet {self.tmp_storage.storage_path(f'{env_name}_eval_packet.pkl')}
            """
        eval_job = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True  # Enable text mode (for string output)
        )

        self.eval_jobs.put(eval_job)

    def _collect_async_eval_results(self):
        # Collect result
        eval_output = {}
        # All jobs successful. Collect results
        for k in self.eval_envs.keys():
            eval_output[k] = self.output_storage.load(f"{k}_eval_output", 
                                                        filetype='torch')
        return eval_output


    def nonvec_eval_step(self, eval_env=None, max_ep_steps=None, env_name=None, storage=None):
        """
        Evaluation step for non-vectorized environments
        """
        raise NotImplementedError

    def vec_eval_step(self, eval_env=None, max_ep_steps=None, env_name=None, storage=None):
        """
        Evaluation step for vectorized environments
        If storage is provided, write logs and outputs to storage
        """

        # Set up storage to write logs and outputs
        eval_storage = None
        if storage is not None:
            eval_log_file = f'eval_log_{env_name}.txt'
            eval_output_file = f'eval_output_{env_name}.pt'
            if storage == 'output':
                eval_storage = self.output_storage
            elif storage == 'tmp':
                eval_storage = self.tmp_storage
    

        max_ep_steps = max_ep_steps or eval_env.max_episode_steps
        num_frames = eval_env.frames

        obses = []       
        obs, info = eval_env.reset() # obs: (num_env, *obs_shape)
        dones = [False] * eval_env.num_envs
        episode_reward_list = []
        steps = 0
        steps_to_keep = [None]*eval_env.num_envs
        num_eps = [0]*eval_env.num_envs

        # Create a tmp file to write progress to
        max_steps = self.num_eval_episodes*max_ep_steps
    

        eval_info = {}
        progress_bar = self.progress_bar if self.display_progress else nullcontext()
        with progress_bar:
            for i in range(max_steps):
                # print(f"Step {i}/{max_steps}")
                if eval_storage is not None:
                    eval_storage.save(eval_log_file, f"step:{i}\n", filetype='text', write_mode='a')

                if hasattr(self, 'progress_bar') and (env_name is not None):
                    self.progress_bar.update(self.progress_tasks[env_name], completed=i)

                # No  need to reset again since vec_env resets individual envs automatically
                if None not in steps_to_keep:
                    # All envs have completed max_eps
                    if eval_storage is not None:
                        eval_storage.save(eval_log_file, f"step:{max_steps}\n", filetype='text', write_mode='a')
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
            eval_info['episode_rewards'] = episode_reward_list
            eval_info['episode_obs'] = obses        

            if eval_storage is not None:
                eval_storage.save(eval_output_file, eval_info, filetype='torch')

            if hasattr(self, 'progress_bar') and (env_name is not None):
                self.progress_bar.update(self.progress_tasks[env_name], completed=max_steps)
                time.sleep(1)

        return eval_info
    
    def after_eval(self, info=None):
        # Save evaluation output
        if self.config.get('save_output'):
            self.output_storage.save('eval_output.pt', info, filetype='torch')

        # Delete the temporary storage directory
        self.tmp_root_dir.cleanup()


class TrainingRLEvaluator(RLEvaluator):
    """
    Evaluator class to be used during training
    """
    def __init__(self, config, trainer):
        self.trainer = trainer
        super().__init__(config)

    def make_env(self, config):
        return self.trainer.make_env(config)

    @property
    def module_path(self):
        return 'trainer.rl.rl_evaluator'