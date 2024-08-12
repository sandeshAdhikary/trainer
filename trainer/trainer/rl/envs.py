import gymnasium as gym
import numpy as np
from einops import rearrange
from gymnasium.wrappers import FrameStack


class VecEnvWrapper(gym.Wrapper):
    """
    A wrapper to vectorize environments
    Requires a list of environment initialization functions
    """

    def __init__(self, env_fns):
        """
        env_fns: 
        """
        
        # self.env = gym.vector.AsyncVectorEnv([lambda: e for e in envs])
        self.env = gym.vector.SyncVectorEnv(env_fns)
        self.envs = self.env.envs
        try:
            self._max_episode_steps = max([e._max_episode_steps for e in self.envs])
        except AttributeError as e:
            try:
                self._max_episode_steps = self.envs[0].unwrapped.max_steps
            except:
                if str(e).startswith("accessing private attribute"):
                    self._max_episode_steps = max([e.spec.max_episode_steps for e in self.envs])
        self.max_episode_steps = self._max_episode_steps

        if hasattr(self.envs[0], '_frames'):
            self._frames = [e._frames for e in self.envs]
            assert all([e._frames == self._frames[0] for e in self.envs])
            self._frames = self._frames[0]

    
    def render(self):
        imgs = []
        for e in self.envs:
            imgs.append(e.render())
        imgs = np.stack(imgs) # (n, h, w, c) or (n ,c, h, w)
        # # Stack renders from envs vertically
        if imgs.shape[-1] in [1,3]:
            imgs = rearrange(imgs, 'n h w c -> n c h w')
        return imgs
    
    @property
    def base_observation_space(self):
        return self.observation_space.shape[1:] # Skip num_envs dimension
     
    @property
    def base_action_space(self):
        return self.action_space.shape[1:] # Skip num_envs dimension
            


class TrainerEnv(gym.Wrapper):
    VEC_ENV_TYPES = (VecEnvWrapper, gym.vector.AsyncVectorEnv, gym.vector.SyncVectorEnv, gym.vector.VectorEnv)

    def __init__(self, env_fns, vec_env=True):
        """
        If vec_env is true, the env is wrapped as VecEnv even if there is only one env
        If vec_env is false:
                if len(env_fns) > 1: the env is wrapped as VecEnv
                if len(env_fns) == 1: the env is not wrapped as VecEnv (it's just env_fns())
        """
        self.env = self._maybe_vec_wrap_envs(env_fns, vec_env)

        if not isinstance(self.env, self.VEC_ENV_TYPES):
            self.num_envs = 1
            self.envs = [self.env]
        self.set_base_attributes()
        
        
    def set_base_attributes(self):
        base_env = self.env.envs[0] if isinstance(self.env, self.VEC_ENV_TYPES) else self.env

        attributes = ['max_episode_steps']
        if hasattr(self.envs[0], '_frames'): 
            attributes.append('frames')

        for attr in attributes:
            try:
                attr_value = getattr(base_env, attr) or getattr(base_env, f"_{attr}")
                if attr_value is None:
                    raise AttributeError
                setattr(self, attr, attr_value)
            except (AttributeError,AssertionError):
                try:
                    attr_value = getattr(base_env.unwrapped, attr) or getattr(base_env.unwrapped, f"_{attr}")
                    setattr(self, attr, attr_value)
                except  (AttributeError,AssertionError):
                    setattr(self, attr, getattr(base_env.spec, attr))
            
            assert getattr(self, attr) is not None
                


    @property
    def is_vec_env(self):
        return isinstance(self.env, self.VEC_ENV_TYPES)

    def _maybe_vec_wrap_envs(self, env_fns, vec_env=True):
        """
        env_fns: A list of environment constructor functions. Can be single env too
        vec_env: 
                If true/false and len(env_fns) > 1: returns a vectorized env
                If true and len(env_fns) == 1: returns a vectorized env
                If false and len(env_fns) == 1: returns a single env
                
        """
        if isinstance(env_fns, list) and len(env_fns) > 1:
            # Always vectorize if multiple envs are given
            return VecEnvWrapper(env_fns)

        if vec_env:
            # Vectorize even if single env is given
            env_fns = env_fns if isinstance(env_fns, list) else [env_fns]
            return VecEnvWrapper(env_fns)
        
        return env_fns() # Return single env


    def get_env_shapes(self):
        # Model needs env information to set up obs_space, action_space etc
        obs_shape = self.env.observation_space.shape
        action_shape = self.env.action_space.shape
        if isinstance(self.env, VecEnvWrapper):
            obs_shape = self.env.base_observation_space
            action_shape = self.env.base_action_space
        elif isinstance(self.env, (gym.vector.AsyncVectorEnv, gym.vector.SyncVectorEnv, gym.vector.VectorEnv)):
            obs_shape = obs_shape[1:] # skip the num_envs dimension
            action_shape = action_shape[1:] # skip the num_envs dimension
        else:
            ValueError("The env must be wrapped in a VecEnvWrapper")
        return {'obs_shape': obs_shape, 'action_shape': action_shape}


