from trainer.trainer import Trainer
from trainer.rl.envs import TrainerEnv
from typing import Dict
from abc import ABC, abstractmethod
from trainer.rl.replay_buffers import ReplayBuffer

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
    
    @abstractmethod
    def env_fns(self, config):
        """
        config: config dict defining the envs
        return: env_fn, eval_env_fn i.e. env generating functions that can be executed as env_fn()
        """
        raise NotImplementedError

    def _setup_env(self, env_config):
        env_fn, eval_env_fn = self.env_fns(env_config)
        self.env = TrainerEnv(env_fn, vec_env=False)
        self.eval_env = TrainerEnv(eval_env_fn)

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
                if num_updates > 0:
                    for batch_idx in range(num_updates):
                        self.before_step()
                        batch = self.replay_buffer.sample()
                        self.train_step(batch, batch_idx)
                        self.after_step()
                else:
                    # update trainer state
                    self.after_step()
                self.after_epoch({'rollout': rollout_data})
                # self.evaluate(self.eval_data, training_mode=True)
        self.after_train() 

    def after_epoch(self, epoch_info=None):
        # Update state from rollout data
        rollout_data = epoch_info.get('rollout_data')
        if rollout_data:
            self.reward = rollout_data['reward']
            self.obs = rollout_data['next_obs']
            self.done = rollout_data['terminated'] | rollout_data['truncated']
            self.step += rollout_data['num_steps']

    def log_step(self, info=None):
        pass
    def log_epoch(self, info=None):
        pass
    def log_train(self, info=None):
        pass

    def collect_rollouts(self, obs, add_to_buffer=False):
        # Get Action
        if self.step < self.config['init_steps']:
            action = self.env.action_space.sample()
        else:
            action = self.model.model.sample_action(obs, batched=self.env.is_vec_env)
        
        # Env Step
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        if add_to_buffer:
            # Add to buffer: allow infinite bootstrap: don't store truncated as done
            curr_reward = self.reward
            self.replay_buffer.add(obs, action, curr_reward, reward, next_obs, terminated, info, 
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

    def parse_config(self, config: Dict):
        # Convert eval_img_sources from a single list to a list of strings
        if isinstance(config['env'].get('eval_img_sources'), str):
            config['env']['eval_img_sources'] = config['env']['eval_img_sources'].strip("(')").replace("'", "")
            config['env']['eval_img_sources'] = config['env']['eval_img_sources'].replace("[", "")
            config['env']['eval_img_sources'] = config['env']['eval_img_sources'].replace("]", "")
            config['env']['eval_img_sources'] = [item.strip() for item in config['env']['eval_img_sources'].split(',')]


        if config.get('img_source') == 'none':
            config['env']['img_source'] = None

        return config

    def init_trainer_state(self, state_dict=None):
        # Initialize/update step, train/eval logs, etc.
        super().init_trainer_state(state_dict)
        # Update RL specific params
        self.obs, _ = self.env.reset()
        self.done = [False]*self.env.num_envs
        self.reward = [0]*self.env.num_envs
        # Set up counters
        self.num_episodes = np.zeros(self.env.num_envs)
        # self.episode_reward = np.zeros(self.env.num_envs)
        # self.episode_reward_list = [[] for _ in range(self.env.num_envs)]
        self.num_model_updates = 0