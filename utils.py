import random
import numpy as np
import torch
import os
import yaml
import importlib
from importlib.resources import files
from warnings import warn
import collections
from importlib import import_module
import time
import hashlib
from dotenv import load_dotenv
from omegaconf.dictconfig import DictConfig
from torch.autograd import set_detect_anomaly
from contextlib import nullcontext
from copy import deepcopy
from omegaconf import DictConfig, OmegaConf

DEFAULT_ENV_PATH = os.path.join(os.path.dirname(__file__), '.env')

CLASS_TYPES = ['trainer', 'model', 'evaluator']
COLORS = {
   'tableau10': ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", 
              "#EDC949", "#AF7AA1", "#FF9DA7", "#9C755F", "#BAB0AB"],
    'tableau20': ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", 
              "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF", 
              "#AEC7E8", "#FFBC79", "#98DF8A", "#FF9896", "#C5B0D5", 
              "#C49C94", "#F7B6D2", "#C7C7C7", "#DBDB8D", "#9EDAE5"]
}

class eval_mode(object):
    """
    Context manager that sets models to eval mode
    and then returns them back to original mode
    Code from https://github.com/facebookresearch/deep_bisim4control/tree/main
    """
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        # Store previous train/eval modes for models
        # set all models to eval mode
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.eval()

    def __exit__(self, *args):
        # Return models to original train/eval modes
        for model, train_mode in zip(self.models, self.prev_states):
            if train_mode:
                model.train()
            else:
                model.eval()
        return False

def pretty_title(x):
    x = x.replace('_', ' ').title()
    return x.lstrip(' ').rstrip(' ')


def flatten_dict(d, parent_key='', separator='.'):
    flattened = {}
    for key, value in d.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key
        if isinstance(value, (dict, DictConfig)):
            sub_keys = list(value.keys())
            if (len(sub_keys) == 1) and (sub_keys[0] == 'values'):
                flattened[new_key] = value
            else:
                flattened.update(flatten_dict(value, new_key, separator=separator))
        else:
            flattened[new_key] = value
            # raise ValueError(f"Invalid value type {type(value)} for key {key}")
    return flattened

def unflatten_dict(d, separator='.'):
    unflattened = {}
    for key, value in d.items():
        keys = key.split(separator)
        current_level = unflattened
        for k in keys[:-1]:
            current_level = current_level.setdefault(k, {})
        current_level[keys[-1]] = value
    return unflattened

def nested_dict_update(d, u):
    # https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = nested_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def check_class_registration(pkg_config, class_type, name, module):
    """
    Check if a class with name has already been registered under class_type
    """
    if pkg_config is not None:
        class_dict = pkg_config.get(class_type + 's')
        if (class_dict is not None) and (class_dict.get(name)==module):
            return True
    
    return False

# def register_class(class_type, name, module, overwrite=False):
#     """
#     Register the class 
#     """
#     assert class_type in CLASS_TYPES
#     with open(files('trainer').joinpath('config.yaml'), 'r') as f:
#         pkg_config = yaml.safe_load(f)
#     pkg_config = pkg_config or {}
#     # Check if class has already been registered
#     already_registered = check_class_registration(pkg_config, class_type, name, module)
#     # Potentially register class in config
#     if (not already_registered) or (already_registered and overwrite):
#         with open(files('trainer').joinpath('config.yaml'), 'w') as f:
#             pkg_config[class_type] = pkg_config.get(class_type) or {}
#             pkg_config[class_type][name] = module
#             yaml.safe_dump(pkg_config, f)

def import_registered_classes(globals):
    with open(files('trainer').joinpath('config.yaml')) as f:
        pkg_config = yaml.load(f, Loader=yaml.FullLoader)
    if pkg_config is not None:
        # Import registered models
        for module_type in CLASS_TYPES:
            if pkg_config.get(module_type) is not None:
                for name, module in pkg_config[module_type].items():
                    try:
                        module_obj = importlib.import_module(module)
                        globals.update({name: getattr(module_obj, name)})    
                    except ModuleNotFoundError:
                        warn(f"Could not import module {module}", stacklevel=2)

def set_seed_everywhere(seed):
    """
    Set all seeds. Function copied from SB3
    """
    # Seed python RNG
    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed)
    # seed the RNG for all devices (both CPU and CUDA)
    torch.manual_seed(seed)

    # Deterministic operations for CuDNN, it may impact performances
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def is_directory_writable(directory_path):
    """
    Try to write a file to the directory to check if it is writable
    """
    try:
        # Generate a temporary file name and try to create it in the directory
        tmp_file = os.path.join(directory_path, 'test_file.txt')
        with open(tmp_file, 'w') as f:
            f.write('Test')
        # If the file was created successfully, it means the directory is writable
        return True
    except (OSError, IOError):
        # If there was an exception, the directory is not writable
        return False
    finally:
        # Clean up the temporary file if it was created
        if os.path.exists(tmp_file):
            os.remove(tmp_file)

def import_module_attr(module_path):
    """
    e.g. if module_path is 'src.trainer.Trainer'
         executes 'from src.trainer import Trainer'
         and returns Trainer object
    """
    cls_name = module_path.rsplit('.')[-1]
    module_name = '.'.join(module_path.rsplit('.')[:-1])
    return getattr(import_module(module_name), cls_name) 

def generate_unique_hash(string, max_len=10, include_time=True):
    if string is not None:
        if include_time:
            # Get current time in milliseconds
            current_time = str(int(time.time() * 1000))
        
            # Concatenate string and current time
            string = string + current_time
        
        # Calculate SHA-256 hash
        sha256_hash = hashlib.sha256(string.encode()).hexdigest()
        
        return sha256_hash[-max_len:]
    
def load_env(env_file_path=None):
    """
    Load environment variables from a .env file
    """
    if env_file_path is not None:
        load_dotenv(env_file_path, override=True)
    else:
        load_dotenv(DEFAULT_ENV_PATH, override=True)


def make_configs(cfg):
    # Resolve the config
    cfg = DictConfig(OmegaConf.to_container(cfg, resolve=False))
    # Get the composed config
    study_config = deepcopy(cfg)

    exp_config = {}
    project_overrides = {}
    if cfg.get('project'):
        project_overrides = deepcopy(cfg['project']['overrides'])
    if cfg.get('exp'):
        exp_config = deepcopy(cfg['exp'])
        study_config.__delattr__('exp')

    # Project overrides
    study_config = OmegaConf.merge(study_config, project_overrides)    

    return study_config, exp_config

def trainer_context(config):
    
    context = nullcontext()
    if config['trainer'].get('debug', False):
        # Torch's anomaly detection context for debuggig
        context = set_detect_anomaly(True)    
    return context

def convert_string(value):
    """
    Read in a string value as int/float if possible, otherwise return as string
    """
    try:
        # Try to convert to integer
        return int(value)
    except ValueError:
        try:
            # Try to convert to float
            return float(value)
        except ValueError:
            # If both conversions fail, return as string
            return value

def remove_protected_keys(config):
    """
    Remove any protected (e.g. passwords) from a given FLAT config
    Useful when logging a config while not revealing sensitive info
    """
    new_config = deepcopy(config)
    protected_keywords = ['password', 'storage', 'username', 'host']
    for proc_kwrd in protected_keywords:
        for key in list(new_config.keys()):
            if proc_kwrd.lower() in key.lower():
                new_config[key] = 'REDACTED'
        # protected_keys = [x for x in list(config.keys()) if 'storage' in x]
        # [new_config.pop(x) for x in protected_keys]
    return new_config