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

CLASS_TYPES = ['trainer', 'model', 'evaluator']
COLORS = {
   'tableau10': ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", 
              "#EDC949", "#AF7AA1", "#FF9DA7", "#9C755F", "#BAB0AB"],
    'tableau20': ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", 
              "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF", 
              "#AEC7E8", "#FFBC79", "#98DF8A", "#FF9896", "#C5B0D5", 
              "#C49C94", "#F7B6D2", "#C7C7C7", "#DBDB8D", "#9EDAE5"]
}


def pretty_title(x):
    x = x.replace('_', ' ').title()
    return x.lstrip(' ').rstrip(' ')


def flatten_dict(d, parent_key='', separator='.'):
    flattened = {}
    for key, value in d.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key
        if isinstance(value, dict):
            sub_keys = list(value.keys())
            if (len(sub_keys) == 1) and (sub_keys[0] == 'values'):
                flattened[new_key] = value
            else:
                flattened.update(flatten_dict(value, new_key, separator=separator))
        else:
            raise ValueError(f"Invalid value type {type(value)} for key {key}")
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
         exeuctures 'from src.trainer import Trainer'
         and returns Trainer object
    """
    cls_name = module_path.rsplit('.')[-1]
    module_name = '.'.join(module_path.rsplit('.')[:-1])
    return getattr(import_module(module_name), cls_name) 