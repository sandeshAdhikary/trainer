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
    
    def __init__(self, config, metrics=None):
        self.config = config
        self.project = self.config['project']
        self.run = self.config['run']
        self.sweep = self.config.get('sweep')
        self.model_name = self.config.get('model_name', 'model_checkpoint.pt')
        self.saved_model_type = self.config.get('saved_model_type', 'torch')
        self._set_storage()
        self.setup_data()
        self._setup_metric_loggers(metrics)
        self._setup_terminal_display()

    @abstractproperty
    def module_path(self):
        return None

    def _setup_terminal_display():
        pass


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
    
    def _setup_metric_loggers(self, metrics=None):
        pass