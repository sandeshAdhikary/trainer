import random
import numpy as np
import torch
import os
import yaml
import importlib
from importlib.resources import files
from warnings import warn

def register_class(class_type, name, module):
    assert class_type in ['trainer', 'model']
    with open('trainer/config.yaml', 'r') as f:
        yaml_dict = yaml.load(f, Loader=yaml.SafeLoader)
    with open('trainer/config.yaml', 'w') as f:
        yaml_dict[class_type + 's'][name] = module
        yaml.dump(yaml_dict, f)

def import_registered_classes(globals):
    with open(files('trainer').joinpath('config.yaml')) as f:
        pkg_config = yaml.load(f, Loader=yaml.FullLoader)

    # Import registered models
    for module_type in ['models', 'trainers']:
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