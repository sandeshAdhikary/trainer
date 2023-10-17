import gym
import numpy as np
from einops import rearrange


class VecEnvWrapper(gym.Wrapper):
    """
    A wrapper to vectorize environments
    Requires a list of environment initialization functions
    """

    def __init__(self, env_fns):
        
        # self.env = gym.vector.AsyncVectorEnv([lambda: e for e in envs])
        self.env = gym.vector.SyncVectorEnv(env_fns)
        self.envs = self.env.envs

        self._max_episode_steps = max([e._max_episode_steps for e in self.envs])
        self._frames = [e._frames for e in self.envs]
        assert all([e._frames == self._frames[0] for e in self.envs])
        self._frames = self._frames[0]

    
    def render(self):
        imgs = []
        for e in self.envs:
            imgs.append(e.render())
        imgs = np.stack(imgs)
        # Stack renders from envs vertically
        if imgs.shape[-1] in [1,3]:
            imgs = rearrange(imgs, 'n h w c -> (n h) w c')
        else:
            imgs = rearrange(imgs, 'n c h w -> (n h) w c')
        return imgs
    
    @property
    def base_observation_space(self):
        return self.observation_space.shape[1:] # Skip num_envs dimension
     
    @property
    def base_action_space(self):
        return self.action_space.shape[1:] # Skip num_envs dimension
            