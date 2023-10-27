import os
import numpy as np
import torch
from copy import copy


class ReplayBuffer(object):
    """
    Buffer to store environment transitions.
    Class copied from https://github.com/facebookresearch/deep_bisim4control
    """
    def __init__(self, obs_shape, action_shape, capacity, batch_size, device, store_infos=False):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.obs_shape = obs_shape
        self.action_shape = action_shape 
        self.store_infos = store_infos
        self.init_empty_buffer()

    def init_empty_buffer(self):
        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(self.obs_shape) == 1 else np.uint8

        self.obses = np.empty((self.capacity, *self.obs_shape), dtype=obs_dtype)
        self.k_obses = np.empty((self.capacity, *self.obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((self.capacity, *self.obs_shape), dtype=obs_dtype)
        self.actions = np.empty((self.capacity, *self.action_shape), dtype=np.float32)
        self.curr_rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self.rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((self.capacity, 1), dtype=np.float32)
        self.infos = None
        if self.store_infos:
            self.infos = [None for _ in range(self.capacity)]

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, action, curr_reward, reward, next_obs, done, info=None, batched=False):
        if batched:
            for idx in range(obs.shape[0]):
                # Add individual experiences separately
                single_info = info[idx,:] if info else None
                self._add_single(obs[idx,:], action[idx,:], curr_reward[idx],
                                  reward[idx], next_obs[idx,:], done[idx], single_info)
        else:
            self._add_single(obs, action, curr_reward, reward, next_obs, done, info)



    def _add_single(self, obs, action, curr_reward, reward, next_obs, done, info=None):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.curr_rewards[self.idx], curr_reward)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        if (self.infos is not None) and (info is not None):
            self.infos[self.idx] = copy(info)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, k=False):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        curr_rewards = torch.as_tensor(self.curr_rewards[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            self.next_obses[idxs], device=self.device
        ).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)


        if k: 
            sample_outputs = obses, actions, rewards, next_obses, not_dones, torch.as_tensor(self.k_obses[idxs], device=self.device), infos
        else:
            sample_outputs = obses, actions, curr_rewards, rewards, next_obses, not_dones


        if self.infos is not None:
            # Return infos as well
            infos = [self.infos[idx] for idx in idxs]
            sample_outputs = (*sample_outputs, infos)
        
        return sample_outputs

    def state_dict(self):
        state_dict = {
            'obses': self.obses[:self.idx],
            'next_obses': self.next_obses[:self.idx],
            'actions': self.actions[:self.idx],
            'rewards': self.rewards[:self.idx],
            'curr_rewards': self.curr_rewards[:self.idx],
            'not_dones': self.not_dones[:self.idx],
            'idx': self.idx
        }
        if self.infos is not None:
            state_dict.update({
                'infos': self.infos[:self.idx]
            })
        return state_dict


    def load_state_dict(self, state_dict):
        # Initialize buffer with zeros
        self.init_empty_buffer()
        # Load saved buffer
        self.idx = state_dict['idx']
        self.obses[:self.idx] = state_dict['obses']
        self.next_obses[:self.idx] = state_dict['next_obses']
        self.actions[:self.idx] = state_dict['actions']
        self.rewards[:self.idx] = state_dict['rewards']
        self.curr_rewards[:self.idx] = state_dict['curr_rewards']
        self.not_dones[:self.idx] = state_dict['not_dones']
        if self.infos is not None:
            self.infos[:self.idx] = state_dict['infos']

    def save(self, save_dir, save_chunks=False):
        if save_chunks:
            raise NotImplementedError
            # self._save_chunks(save_dir)
        else:
            torch.save(self.state_dict(), path)

    def load(self, save_dir, load_chunks=False):
        if load_chunks:
            raise NotImplementedError
            # self._load_chunks(save_dir)
        else:
            raise NotImplementedError
            # # Initialize buffer with zeros
            # self.init_empty_buffer()
            # # Load saved buffer
            # payload = torch.load(os.path.join(save_dir, "buffer.pt"))
            # # Infer the size of saved buffer; set counter to it
            # self.idx = payload[0].shape[0]
            # # Fill up the loaded buffer values
            # self.obses[:self.idx] = payload[0]
            # self.next_obses[:self.idx] = payload[1]
            # self.actions[:self.idx] = payload[2]
            # self.rewards[:self.idx] = payload[3]
            # self.curr_rewards[:self.idx] = payload[4]
            # self.not_dones[:self.idx] = payload[5]
            # if self.infos is not None:
            #     self.infos[:self.idx] = payload[6]
                
    def _save_chunks(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.curr_rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        if self.infos is not None:
            # Add infos to payload
            payload.append(self.infos[self.last_save:self.idx])

        self.last_save = self.idx
        torch.save(payload, path)

    def _load_chunks(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        self.last_save = int(chucks[-1].split('_')[1].split('.')[0]) # The last chunk's end index
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.curr_rewards[start:end] = payload[4]
            self.not_dones[start:end] = payload[5]
            if self.infos is not None:
                self.infos[start:end] = payload[6]
            self.idx = end
