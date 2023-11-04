from abc import ABC, abstractmethod
from functools import partial
import numpy as np
import subprocess
import queue
from trainer.rl.envs import TrainerEnv
from einops import rearrange
from trainer.storage import Storage
import tempfile
from copy import deepcopy
from abc import abstractproperty

class Evaluator():
    
    def __init__(self, config):
        self.config = config
        self.project = self.config['project']
        self.run = self.config['run']
        self.sweep = self.config.get('sweep')
        self.model_name = self.config.get('model_name', 'model_checkpoint.pt')
        self.saved_model_type = self.config.get('saved_model_type', 'torch')
        self._set_storage()
        self.setup_data()
        # self._register_evaluator()
        self._setup_terminal_display()

    @abstractproperty
    def module_path(self):
        return None

    def _setup_terminal_display():
        pass

    # def _register_evaluator(self, overwrite=False):
    #     if self.module_path is not None:
    #         register_class('evaluator', self.__class__.__name__, self.module_path)

    def _set_storage(self):
        # Set up model storage: model will be loaded from here
        self.input_storage = Storage(self.config['storage']['input'])
        # Set up output storage: evaluation outputs will be saved here
        self.output_storage = Storage(self.config['storage']['output'])
        # Create a temporary directory for temp storage
        self.tmp_root_dir = tempfile.TemporaryDirectory(prefix='trainer_')
        self.tmp_storage = Storage({
            'type': 'local',
            'root_dir': self.tmp_root_dir.name,
            'project': self.project,
            'run': self.run
        })

    def set_model(self, model):
        self.model = model

    def set_logger(self, logger):
        self.logger = logger

    def run_eval(self, **kwargs):
        self.before_eval(**kwargs)
        eval_output = self.evaluate(async_eval=self.config['async_eval'], **kwargs)
        self.after_eval(eval_output)
        return eval_output
    
    def before_eval(self, info=None):
        """
        Set up the evaluation dataset/environment
        """
        assert self.model is not None, "Model not set"
        if (info is not None) and info.get('load_checkpoint'):
            self.load_model()
        # Set model to eval mode
        self.model.eval()

    def load_model(self, state_dict=None):

        if state_dict is not None:
            model_ckpt = state_dict
        else:
            if self.saved_model_type == 'torch':
                model_ckpt = self.input_storage.load(self.model_name, filetype='torch')
            elif self.saved_model_type == 'zip':
                # basename = os.path.splitext(self.model_name)[0]
                # TODO: Make model names consistent when saving
                model_ckpt = self.input_storage.load_from_archive(self.model_name, 
                                                                filenames='model_checkpoint.pt',
                                                                filetypes='torch')
            else:
                raise ValueError(f"Invalid input model format {self.input_model_format}")
        self.model.load_model(model_ckpt)

    def evaluate(self):
        """
        Evaluate model and store evaluation results
        """
        raise NotImplementedError

    def after_eval(self, info=None):
        """
        Process and save evaluation results
        """
        pass
        
    @abstractmethod
    def setup_data(self):
        raise NotImplementedError
    
class RLEvaluator(Evaluator):
    
    def __init__(self, config):
        super().__init__(config)
        self.num_eval_episodes = config.get('num_eval_episodes', 1)

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
            self.eval_envs[name] = TrainerEnv(self.env_fns(config))


    def evaluate(self, max_ep_steps=None, async_eval=False, eval_log_file=None, eval_output_file=None):
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
                    env_output = evaluation_fn(env=self.eval_envs[env_name],
                                               max_ep_steps=max_ep_steps, 
                                               eval_log_file=self.output_storage.storage_path(f"{env_name}_eval_log"), 
                                               eval_output_file=self.output_storage.storage_path(f"{env_name}_eval_output")
                                               )
                except Exception as e:
                    self.eval_job_errs.append(e.args[0])

                eval_output[env_name] = env_output
                self.eval_job_errs.append(None)

        # If async, wait until all eval jobs have completed
        if async_eval:
            self._wait_for_all_eval_jobs()
            eval_output = self._collect_async_eval_results()


        # If eval failed, log error and break
        self._maybe_log_eval_error()

        return eval_output


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
                        'eval_log_file': f'{env_name}_eval_log_file',
                        'eval_output_file': f'{env_name}_eval_output_file',
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


    def nonvec_eval_step(self, env_name, max_ep_steps=None, eval_log_file=None, **kwargs):
        """
        Evaluation step for non-vectorized environments
        """
        raise NotImplementedError

    def vec_eval_step(self, env=None, max_ep_steps=None, eval_log_file=None, eval_output_file=None):
        """
        Evaluation step for vectorized environments
        if eval_log_file is provided, write evaluation metrics to file
        """
        eval_info = {}

        # If no env is passed, assume there is a single self.eval_env
        eval_env = env or self.eval_env

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

        for i in range(max_steps):
            print(f"Step {i}/{max_steps}")
            if eval_log_file is not None:
                self.output_storage.save(eval_log_file, f"step:{i}\n", filetype='text', write_mode='a')

            if hasattr(self, 'progress_eval'):
                self.eval_progress_bar.update(self.progress_eval, completed=i)

            # No  need to reset again since vec_env resets individual envs automatically
            if None not in steps_to_keep:
                # All envs have completed max_eps
                if eval_log_file is not None:
                    self.output_storage.save(eval_log_file, f"step:{max_steps}\n", filetype='text', write_mode='a')
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
        eval_info['episode_obs'] = obses


        if eval_output_file is not None:
            self.output_storage.save(eval_output_file, eval_info, filetype='torch')


        return eval_info
    
    def after_eval(self, info=None):
        # Save evaluation output
        if self.config['save_output']:
            self.output_storage.save('eval_output.pt', info, filetype='torch')

        # Delete the temporary storage directory
        self.tmp_root_dir.cleanup()