from abc import ABC, abstractmethod
from trainer.storage import Storage
import tempfile
from abc import abstractproperty
from trainer.metrics import PredictionErrorMetric
import numpy as np
from trainer.metrics import Metric

class Evaluator():
    
    def __init__(self, config, metrics=None):
        self.config = config
        self.project = self.config['project']
        self.run = self.config['run']
        self.sweep = self.config.get('sweep')
        self.model_name = self.config.get('model_name', 'model_checkpoint.pt')
        self.saved_model_type = self.config.get('saved_model_type', 'torch')
        self._set_storage()
        self.setup_data(config)
        self._setup_metric_loggers(self.config, metrics)
        self._setup_terminal_display(self.config)

    @abstractproperty
    def module_path(self):
        return None

    def _setup_terminal_display(self, config):
        # TODO: Set up terminal display
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
                model_ckpt = self.input_storage.load_from_archive(self.model_name, 
                                                                filenames='model_checkpoint.pt',
                                                                filetypes='torch')
            else:
                raise ValueError(f"Invalid input model format {self.input_model_format}")
        self.model.load_model(model_ckpt)

    def evaluate(self, async_eval=False):
        """
        Evaluate model and store evaluation results
        """
        raise NotImplementedError

    def after_eval(self, info=None):
        """
        Process and save evaluation results
        """
        # Save evaluation output
        if self.config.get('save_output'):
            self.output_storage.save('eval_output.pt', info, filetype='torch')

        # Delete the temporary storage directory
        self.tmp_root_dir.cleanup()
        
    @abstractmethod
    def setup_data(self, config=None):
        raise NotImplementedError
    
    def _setup_metric_loggers(self, config, metrics=None):
        pass


class SupervisedEvaluator(Evaluator):
    """
    Evaluator class for supervised learning
    """

    def evaluate(self, async_eval=False, storage=None):
        """
        Evaluate model and store evaluation results
        return eval_outputs dict
        """
        # Set model to evaluation mode
        self.model.eval()

        # Loop through evaluation data and get evaluation outputs
        tracked_data = {}
        for batch_idx, batch in enumerate(self.dataloader):
            # Collect output from evaluation step
            eval_step_output = self.model.evaluation_step(batch, batch_idx)
            # Track the evaluation step outputs
            if batch_idx == 0:
                for k, v in eval_step_output.items():
                    tracked_data[k] = []
            
            for k, v in eval_step_output.items():
                tracked_data[k].append(v)
    
        # Concatenate arrays in tracked data
        for k, v in tracked_data.items():
            if isinstance(v[0], np.ndarray):
                tracked_data[k] = np.concatenate(v, axis=0)

        return tracked_data

    def _setup_metric_loggers(self, config, metrics=None):
        self.metrics = {}
        config_metrics = config.get('metrics') or self._default_metric_config
        for metric_name, metric_config in config_metrics.items():
            self.metrics[metric_name] = Metric(metric_config)

    @property
    def _default_metric_config(self):
        return {
            'prediction_loss': {
                'name': 'prediction_loss',
                'type': 'scalar',
                'temporal': 'true',
                'table_spec': 'null'
                }
                }