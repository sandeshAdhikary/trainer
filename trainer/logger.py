from abc import ABC, abstractmethod
from typing import Dict, Union
import torch
import torchvision
from einops import rearrange
import wandb
import os
import json
from collections import defaultdict
from termcolor import colored
# from trainer.defaults import DEFAULTS
import zipfile
from warnings import warn
import trainer.utils as utils
import shutil

class Logger(ABC):
    def __init__(self, config: dict) -> None:
        config = self.parse_config(config)
        self.project = config.get('project', 'misc')
        self.dir =  os.path.abspath(config.get('dir', './logdir'))
        self.video_log_freq = config['video_log_freq']
        # Make sure self.dir is writable
        if not utils.is_directory_writable(self.dir):
            raise ValueError("Directory %s is not writable!" % self.dir)

        # Set image downscaling factor to prevent huge file sizes. 1 means no downscaling
        self.img_downscale_factor = config.get('img_downscale_factor', 3)
        self.img_downscale_factor = max(int(self.img_downscale_factor), 1)

        self.sw_type = config.get('sw', None) # Summary writer
        # Create a summary writer if not provided
        if self.sw_type == 'tensorboard':
            raise NotImplementedError
        elif self.sw_type == 'wandb':
            tracked_params = config.get('tracked_params', {})
            tags = config.get('tags', None)
            if tags and not isinstance(tags, list):
                tags = list(tags)
            if wandb.run is None:
                # Initialize a new summary-writer / wandb-run
                wandb_api = config.get('wandb_api')
                if wandb_api is None:
                    # Try to get it from the environment
                    wandb_api = os.environ.get('WANDB_API_KEY')
                # Try interactive login (unless already logged in)
                wandb.login()
                project = config.get('project', self.project)
                self._sw = wandb.init(project=project, dir=self.dir, config=tracked_params, tags=tags)
            else:
                # The summary-writer/wandb-run has already been initialized
                self._sw = wandb.run
                project = wandb.run.project
                self.dir = wandb.run.dir
            
            self.run_id = self._sw.id
            self.run_name = self._sw.name
        
        self.logdir = self._sw.dir
    

    def log(self, **kwargs):
        log_dict = kwargs.get('log_dict')
        if log_dict is not None:
            self._try_sw_log(log_dict=log_dict)
        else:
            key, value, step = kwargs.get('key'), kwargs.get('value'), kwargs.get('step')
            assert not any([x is None for x in [key, value, step]]), \
            "Must either provide a log_dict or a (key, value and step) tuple!"
            self._try_sw_log(key=key, value=value, step=step)

    def log_checkpoint(self):
        self._try_sw_log_checkpoint()

    def restore_checkpoint(self):
        """
        Restore checkpoint from the summary writer onto a dir
        """
        return self._try_sw_restore_checkpoint()
    
    def finish(self):
        if self.sw_type == 'wandb':
            self._sw.finish()

    def parse_config(self, config):
        if isinstance(config.get('tags'), str):
            # if tags = '[this, that]' then make ['this', 'that']
            config['tags'] = config['tags'] .strip("(')").replace("'", "")
            config['tags']  = config['tags'] .replace("[", "")
            config['tags']  = config['tags'] .replace("]", "")
            config['tags']  = [item.strip() for item in config['tags'] .split(',')]

        # Set logger_video_log_freq
        if config.get('video_log_freq') in [None, 'none', 'None']:
            # Set logger_video_log_freq so we get max num_video_logs videos per run
            num_video_logs = config.get('num_video_logs', 5)
            num_evals = int(config['num_train_steps'] // config['eval_freq'])
            config['video_log_freq'] = max(int(num_evals / num_video_logs), 1)

        return config

    def log_video(self, key, frames, step, image_mode='hwc'):
        self._try_sw_log_video(key, frames, step, image_mode)

    ########################################
    # Summary writer specific functions ####
    ########################################

    def _try_sw_log(self, **kwargs):
        log_dict = kwargs.get('log_dict')
        if self.sw_type == 'wandb':
            if log_dict is not None:
                self._sw.log(log_dict)
            else:
                self._sw.log({kwargs['key']: kwargs['value']}, step=kwargs['step'])
        else:
            raise ValueError("Invalid summary writer type: %s" % self.sw_type)


    def _try_sw_log_checkpoint(self):
        if self.sw_type == 'wandb':
            wandb.save(os.path.join(self.logdir, 'checkpoint.zip'), policy='now', base_path=self.logdir)
            # Sync wandb
            command = f'wandb sync {self.logdir} --id {self.run_id} -p {self.project}'
            os.system(command)

    def _try_sw_restore_checkpoint(self):
        if self.sw_type == 'wandb':
            restore_error = False
            try:
                # Download checkpoint zip file
                wandb.restore("checkpoint.zip", run_path=self._sw.path, replace=True)
                chkpt_zip = f"{self.dir}/checkpoint.zip"
                if os.path.exists(chkpt_zip):
                    # Create a temp folder to extract zipped files
                    tmp_dir = f"{self.logdir}/restored_files"
                    os.makedirs(tmp_dir, exist_ok=True)
                    # Extract the restored files ono tmp dir
                    with zipfile.ZipFile(chkpt_zip, 'r') as zip_ref:
                        zip_ref.extractall(tmp_dir)
                    # Delete the downloaded checkpoint zip file
                    os.remove(chkpt_zip)
                    # Copy files from the tmp dir to the files dir   
                    os.makedirs(f"{self.logdir}/checkpoint", exist_ok=True)
                    for root, _, files in os.walk(tmp_dir):
                        for file in files:
                            # if file != 'checkpoint.zip':
                            source_path = os.path.join(root, file)
                            dest_path = os.path.join(f"{self.logdir}", os.path.relpath(source_path, tmp_dir))
                            # print(dest_path)
                            # os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                            shutil.copy2(source_path, dest_path)
                    # Delete the tmp dir
                    shutil.rmtree(tmp_dir)
                else:
                    warn("No checkpoint found!")
                    restore_error = True
            except ValueError as e:
                warn(e.args[0])
                restore_error = True
        return restore_error

    def _try_sw_log_video(self, key, frames, step, image_mode='hwc'):
        if self.sw_type == 'tensorboard':
            raise NotImplementedError("Tensorboard logger not implemented")
        elif self.sw_type == 'wandb':
            import numpy as np
            frames = np.array(frames)
            if image_mode == 'hwc':
                frames = rearrange(frames, 't h w c -> t c h w')
            frames = frames[:,:,::self.img_downscale_factor,::self.img_downscale_factor]
            self._sw.log({key: wandb.Video(frames, fps=1)}, step=step)


    

    # def _try_sw_log_image(self, key, image, step, image_mode='hwc'):
    #     if self.sw_type == 'tensorboard':
    #         if not torch.is_tensor(image):
    #             image = torch.from_numpy(image)
    #         assert image.dim() == 3
    #         grid = torchvision.utils.make_grid(image.unsqueeze(0))
    #         self._sw.add_image(key, grid, step)
    #     elif self.sw_type == 'wandb':
    #         if image_mode == 'chw':
    #             image = rearrange(image, 'c h w -> h w c')
    #         if torch.is_tensor(image):
    #             image = image.detach().cpu().numpy()
    #         image = image[:,::self.img_downscale_factor,::self.img_downscale_factor]
    #         self._sw.log({key: [wandb.Image(image)]}, step=step)



    # def _try_sw_log_histogram(self, key, histogram, step):
    #     if self.sw_type == 'tensorboard':
    #         self._sw.add_histogram(key, histogram, step)
    #     elif self.sw_type == 'wandb':
    #         histogram_np = histogram
    #         if isinstance(histogram, torch.Tensor):
    #             histogram_np = histogram.detach().cpu().numpy()
    #         self._sw.log({key: wandb.Histogram(histogram_np)}, step=step)

    # def _try_sw_log_table(self, key, data, step):
    #     if self.sw_type == 'wandb':
    #         data = data.reshape(data.shape[0], -1)
    #         table = wandb.Table(data=list(data), columns=list(range(data.shape[1])))
    #         self._sw.log({key: table}, step=step)

    # def _try_sw_log_agent(self, key, agent, step):
    #     model_dir = os.path.join(self.dir, 'models')
    #     if not os.path.exists(model_dir):
    #         os.makedirs(model_dir)
    #     agent.save(self.model_dir, step)


    # def log(self, key, value, step, n=1):
    #     assert key.startswith('train') or key.startswith('eval')
    #     if type(value) == torch.Tensor:
    #         value = value.item()
    #     self._try_sw_log(key, value / n, step)
    #     mg = self._train_mg if key.startswith('train') else self._eval_mg
    #     mg.log(key, value, n)



    # def log_param(self, key, param, step):
    #     self.log_histogram(key + '_w', param.weight.data, step)
    #     if hasattr(param.weight, 'grad') and param.weight.grad is not None:
    #         self.log_histogram(key + '_w_g', param.weight.grad.data, step)
    #     if hasattr(param, 'bias'):
    #         self.log_histogram(key + '_b', param.bias.data, step)
    #         if hasattr(param.bias, 'grad') and param.bias.grad is not None:
    #             self.log_histogram(key + '_b_g', param.bias.grad.data, step)

    # def log_image(self, key, image, step, image_mode='hwc'):
    #     assert key.startswith('train') or key.startswith('eval')
    #     self._try_sw_log_image(key, image, step, image_mode)



    # def log_histogram(self, key, histogram, step):
    #     assert key.startswith('train') or key.startswith('eval')
    #     self._try_sw_log_histogram(key, histogram, step)

    # def log_table(self, key, data, step):
    #     assert key.startswith('train') or key.startswith('eval')
    #     self._try_sw_log_table(key, data, step)
    
    # def log_agent(self, key, model, step):
    #     self._try_sw_log_agent(key, model, step)


    


    # def dump(self, step):
    #     self._train_mg.dump(step, 'train')
    #     self._eval_mg.dump(step, 'eval')

