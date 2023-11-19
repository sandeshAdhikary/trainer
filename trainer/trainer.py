import torch
from abc import ABC, abstractmethod
from typing import Dict
from trainer.model import Model
from trainer.evaluator import Evaluator
from trainer import utils
from trainer.logger import Logger
from rich.layout import Layout
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress, MofNCompleteColumn, BarColumn, TextColumn,
    TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn)
from rich.columns import Columns
from rich.live import Live
from collections import deque
from warnings import warn
from contextlib import nullcontext
from trainer.storage import Storage
import tempfile
from copy import deepcopy
from trainer.utils import import_module_attr
import numpy as np
from trainer.metrics import Metric

class Trainer(ABC):

    def __init__(self, config: Dict, model: Model = None, logger: Logger = None) -> None:
        self.config = self.parse_config(config)
        self._setup(config, model, logger)

    @abstractmethod
    def setup_data(self, config: Dict) -> None:
        """
        """
        raise NotImplementedError
    
    @abstractmethod
    def fit(self, num_train_steps: int=None, num_epochs: int=None, trainer_state=None) -> None:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError
    
    @abstractmethod
    def _setup_metric_loggers(self, config, metrics=None):
        """
        Define metrics that need to be logged from training data
        """
        raise NotImplementedError

    def train_step(self, batch, batch_idx=None):
        train_step_output = self.model.training_step(batch, batch_idx)
        self.train_log.append({'step': self.step, 'log': train_step_output})
    
    def set_model(self, model: Model) -> None:
        self.model = model

    def set_logger(self, logger: Logger) -> None:
        self.logger = logger

    def before_train(self, trainer_state: Dict =None, info=None):
        """
        Callbacks before training starts
        """
        self.init_trainer_state(trainer_state)
        
        # Make sure the model and logger are set
        assert self.model is not None, "Model not set. Use trainer.set_model(model) to do so."
        assert self.logger is not None, "Logger not set. Use trainer.set_logger(logger) to do so"
        # TODO: Should project/runs be defined before logger?
        self.project = self.logger.project
        self.run = self.logger.run_id
        self._set_storage()
        self.setup_evaluator()

        self._setup_terminal_display(self.config)
        self._set_seeds(self.config)
        # Optionally load from checkpoint
        if self.load_from_checkpoint:
            self._load_checkpoint()

        self.max_iter_type = 'epoch' if self.num_epochs else 'step'

        if self.progress is not None:
            # Initialize progress
            current_iter = self.epoch if self.max_iter_type=='epoch' else self.step
            self.progress.update(self.progress_train, completed=current_iter)
        self.logger.start()
        # Set model to train mode
        self.model.train()

        # Upload configs to storage
        self.output_storage.save(filename='trainer_config.yaml', 
                                 data=self.config,
                                 filetype='yaml')
        self.output_storage.save(filename='model_config.yaml', 
                                 data=self.model.config,
                                 filetype='yaml')
        self.output_storage.save(filename='logger_config.yaml', 
                                 data=self.logger.config,
                                 filetype='yaml')


        if self.num_epochs is not None:
            self.current_iter = self.epoch
        elif self.num_train_steps is not None:
            self.current_iter = self.step
        else:
            raise ValueError("Either num_epochs or num_train_steps must be defined")

    def before_epoch(self, info=None):
        pass

    def before_step(self, info=None):
        self.model.train() # Set model to training mode

    def after_step(self, info=None):
        self.step += 1
        # Update trainer state
        self.update_trainer_state()
        # Logging
        self.log_step(info)
        # Update progress
        if self.max_iter_type=='step' and (self.progress is not None):
            self.progress.update(self.progress_train, completed=self.step)

        self.progress.update(self.progress_within_epoch, completed=(1.0*self.step%self.steps_per_epoch)/self.steps_per_epoch)

    def after_epoch(self, info=None):
        self.epoch += 1
        self.log_epoch(info)
        
        if self.max_iter_type=='epoch' and (self.progress is not None):
            self.progress.update(self.progress_train, completed=self.epoch)

        # Save checkpoint
        if (self.save_checkpoint_freq is not None) and (self.epoch % self.save_checkpoint_freq == 0) and (self.step > 0):
            self._save_checkpoint(ckpt_state_args={'save_optimizers': True})
            self.num_checkpoint_saves += 1
            self.logger.log(log_dict={'trainer_step': self.step, 'checkpoint/num_checkpoint_saves': self.num_checkpoint_saves})

    def after_train(self, info=None):
        """
        Callbacks after training ends
        """
        # Close logger
        # Close any open connections
        self.log_train(info)
        self.logger.finish()


    def log_step(self, info=None):
        """
        Log using self.logger after step
        """
        pass

    def log_epoch(self, info=None):
        """
        Log using self.logger after epoch
        """
        pass

    def log_train(self, info=None):
        """
        Log using self.logger after train
        """
        pass


    def update_trainer_state(self):
        if self.num_epochs is not None:
            self.train_end = self.epoch >= self.num_epochs
        elif self.num_train_steps is not None:
            self.train_end = self.step >= self.num_train_steps

    #############################

    
    def _set_storage(self):
        # Update project and run_id for stroage defs
        for name in self.config['storage'].keys():
            self.config['storage'][name].update({
                'project': self.project,
                'run': self.run
            })

        # Set up model storage: model will be loaded from here
        self.input_storage = Storage(self.config['storage']['input'])
        # Set up output storage: evaluation outputs will be saved here
        self.output_storage = Storage(self.config['storage']['output'])
        # Create a temporary directory for temp storage
        self.tmp_root_dir = tempfile.TemporaryDirectory(prefix='trainer')
        self.tmp_storage = Storage({
            'type': 'local',
            'root_dir': self.tmp_root_dir.name,
            'project': self.project,
            'run': self.run
        })

    def parse_config(self, config: Dict) -> Dict:
        return config

    def init_trainer_state(self, state_dict=None):
        state_dict = state_dict or {}
        # Load from state_dict or initialize for start of training
        self.step = state_dict.get('step', 0)
        self.epoch = state_dict.get('epoch', 0)
        self.train_log = state_dict.get('train_log', deque(maxlen=self.log_length_train))
        self.eval_log = state_dict.get('eval_log', deque(maxlen=self.log_length_eval))
        self.train_end = state_dict.get('train_end', False)
        self.num_checkpoint_loads = state_dict.get('num_checkpoint_loads', 0)
        self.num_checkpoint_saves = state_dict.get('num_checkpoint_saves', 0)

    def _setup(self, config: Dict = None, model: Model = None, logger: Logger =None) -> None:

        self.device = torch.device(config.get('device', 
                                              'cuda' if torch.cuda.is_available() else 'cpu')
                                              )

        self.load_from_checkpoint = config.get('load_from_checkpoint', False)
        self.save_checkpoints = config.get('save_checkpoints', True)
        self.save_checkpoint_freq = config.get('save_checkpoint_freq', 1)
        self.eval_freq = config.get('eval_freq', 2)
        self.async_eval = config.get('async_eval', False)
        self.log_length_train = config.get('log_length_train', 100)
        self.log_length_eval = config.get('log_length_eval', 100)

        self.model = model     
        self.logger = logger
        self._set_seeds(config)
        self.setup_data(config)
        self._setup_metric_loggers(config)
        

    def setup_evaluator(self):
        evaluator_config = self.config['evaluator']
        eval_storage_config = {}
        # Evaluator's input storage need to load checkpoints
        # So, it's set to trainer's output storage
        eval_storage_config['input'] = deepcopy(self.config['storage']['output'])
        # Evaluator will store results in trainer's output storage
        eval_storage_config['output'] = deepcopy(self.config['storage']['output'])
        evaluator_config['storage'] = eval_storage_config

        # Add project and run info
        evaluator_config.update({
            'project': self.project,
            'run': self.run,
        })

        # evaluator_cls = self.config.get('module_path', Evaluator)
        evaluator_cls = self.config['evaluator'].get('module_path')
        if evaluator_cls is not None:
            self.evaluator = import_module_attr(evaluator_cls)(evaluator_config)
        else:
            self.evaluator = Evaluator(evaluator_config)

        self.evaluator.set_model(self.model)

    def _setup_terminal_display(self, config: Dict) -> None:
            
        if config.get('terminal_display') in ['rich', 'rich_minimal']:

            self._term_console = Console()
            self._term_layout = Layout()
            # Header: Run info panel
            self._term_logger_panel = Panel.fit(f"""Project: [orange4] {self.logger.project} [/] \nRun: [orange4] {self.logger.run_name}({self.logger.run_id})[/]\
                                                \nResumed: [orange4] {self.logger.resumed_run}[/] \nLogdir: [orange4] {self.logger.logdir}[/]""",
                                    title='Logger', border_style='orange4')
            
            # Body: Progress panel
            self.progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(complete_style="dark_sea_green4"),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                MofNCompleteColumn(),
                redirect_stdout=False,
                )
            
            max_iters = self.num_epochs or self.num_train_steps # use epochs if both given

            self.progress_train = self.progress.add_task("[dark_sea_green4] Training Progress", total=max_iters)
            self.progress_within_epoch = self.progress.add_task("[dark_sea_green4]\u00A0 \u00A0 Steps", total=1)

            if config.get('terminal_display') == 'rich_minimal':
                # Only a progress bar
                self.terminal_display = self.progress
            else:
                # Full terminal display
                self._term_progress_panel = Panel.fit(Columns([self.progress]), title="Training", border_style="dark_sea_green4")
                # Footer: Misc info
                self._term_footer_text = f"""[bold]Device[/]: {self.device}"""
                self._term_footer_panel = Panel(self._term_footer_text, title="Info", border_style='misty_rose1')
                # Add panels to layout
                self._term_layout.split(
                    Layout(self._term_logger_panel, size=6, name="logger"),
                    Layout(self._term_progress_panel, size=5, name="progress"),
                    Layout(self._term_footer_panel, size=3, name="footer"),
                )
                self.terminal_display = Live(self._term_layout, 
                                            screen=False, 
                                            refresh_per_second=config.get('terminal_refresh_rate', 1))

        else:
            self.terminal_display = nullcontext()
            self.progress = None

    def _set_seeds(self, config: Dict) -> None:
        """
        Set seeds for random, numpy, cuda, torch
        """
        seed = config.get('seed')
        if seed:
            utils.set_seed_everywhere(seed)

    def _load_checkpoint(self, chkpt_name='checkpoint', log_checkpoint=True):
        try:
            # Restore checkpoint: model,trainer,replay_buffer
            ckpt = super()._load_checkpoint_dict(chkpt_name=chkpt_name,
                                                 filenames=['model_ckpt.pt', 'trainer_ckpt.pt'],
                                                 filetypes=['torch', 'torch']
                                                 )
        except (UserWarning, Exception) as e:
            self.logger.log(log_dict={'trainer_step': self.step, 'checkpoint_load_error': 1})
            warn(f"Checkpoint load error: Could not load state dict. Error: {e.args}")
            return e

        # Update model and trainer states
        try:
            self.model.load_model(state_dict=ckpt['model_ckpt.pt'])
            self.init_trainer_state(ckpt['trainer_ckpt.pt'])
        except (UserWarning, Exception) as e:
            self.logger.log(log_dict={'trainer_step': self.step, 'checkpoint_load_error': 1})
            warn(f"Checkpoint load error: State dicts loaded, but could not update model or trainer. Error: {e.args}")
            return e

        if log_checkpoint:
            # Log the checkpoint load event
            self.num_checkpoint_loads += 1
            self.logger.log(key='checkpoint/num_checkpoint_loads', value=self.num_checkpoint_loads, step=self.logger._sw.step)

    def _load_checkpoint_dict(self, chkpt_name='checkpoint', filenames=None, filetypes=None):
        if filenames is None:
            # Assume only model and trainer checkpoints
            filenames = ['model_ckpt.pt', 'trainer_ckpt.pt']
        if filetypes is None:
            # Assume all files are torch-loadable
            filetypes = ['torch']*len(filenames)
        # Download checkpoint from logger
        if self.config['load_checkpoint_type'] == 'torch':
            ckpt = self.input_storage.load(f'model_{chkpt_name}.pt')
        elif self.config['load_checkpoint_type'] == 'zip':

            ckpt = self.input_storage.load_from_archive("ckpt.zip", 
                                                        filenames=filenames,
                                                        filetypes=filetypes)

        return ckpt
        

    def _save_checkpoint(self, log_checkpoint=True, save_to_output_storage=True, ckpt_state_args=None, ckpt_name=None):
        """
        Zip the checkpoint folder and log it
        """

        if ckpt_name is None:
            ckpt_name = 'ckpt'

        # Create an archive of the checkpoint files
        self._create_checkpoint_files(archive=True, ckpt_state_args=ckpt_state_args, ckpt_name=ckpt_name)

        # Check if zip file was created properly
        if self.tmp_storage.archive_filenames(f'{ckpt_name}.zip') is None:
            warn("Checkpoint zip file was not created properly (in tmp storage). Checkpoint not saved in output storage.")
            self.logger.log(log_dict={'trainer_step': self.step, 'checkpoint_save_error': 1})
            return None

        if save_to_output_storage:
            # First make a backup copy of the checkpoint
            # if 'ckpt.zip' in self.output_storage.get_filenames():
            #     self.output_storage.copy('ckpt.zip', 'ckpt_backup.zip')

            # upload checkpoint to output storage as a temp file
            self.output_storage.upload(files=self.tmp_storage.storage_path(f'{ckpt_name}.zip'), new_dir=f'{ckpt_name}_temp')
            # Check if the new temp ckpt was created successfully
            if self.output_storage.archive_filenames(f'{ckpt_name}_temp/{ckpt_name}.zip') is None:
                # warn("The ckpt_temp.zip file was not uploaded properly (to output storage)")
                # Replace 'ckpt.zip' with the backup
                # self.output_storage.copy('ckpt_backup.zip', 'ckpt.zip')
                self.logger.log(log_dict={'trainer_step': self.step, 'checkpoint_save_error': 1})
                raise Exception(f"The {ckpt_name}_temp.zip file was not uploaded properly (to output storage)")
            else:
                # No errors in the new checkpoint, replace existing backup
                self.output_storage.copy(f'{ckpt_name}_temp/{ckpt_name}.zip', f'{ckpt_name}.zip')
                # Delete the temp file
                self.output_storage.delete(directory=f"{ckpt_name}_temp")

        if log_checkpoint:
            # Save checkpoint to logger (e.g. wandb)
            self.logger.log_checkpoint(filepath=self.tmp_storage.storage_path(f'{ckpt_name}.zip'))

    def _create_checkpoint_files(self, temporary_dir=True, archive=True, ckpt_state_args=None, **kwargs):
        """
        Save checkpoint files to storage
        """
        ckpt_name = kwargs.get('ckpt_name', 'ckpt')

        storage = self.tmp_storage if temporary_dir else self.output_storage
        # Get checkpoint state
        ckpt_state_args = ckpt_state_args or {}
        ckpt_state = self._get_checkpoint_state(**ckpt_state_args)

        for name,state in ckpt_state.items():
            storage.save(f"{ckpt_name}/{name}_{ckpt_name}.pt", state, 'torch')

        if archive:
            files=[f"{ckpt_name}/{name}_{ckpt_name}.pt" for name in ckpt_state.keys()]
            # Make the archive
            storage.make_archive(files, zipfile_name=f'{ckpt_name}.zip')
            # Delete the files once they've been archived
            storage.delete(directory=ckpt_name)

    def _get_trainer_state(self):
        """
        The current trainer state
        """
        return {'step': self.step,
                'epoch': self.epoch,
                'train_log': self.train_log, 
                'eval_log': self.eval_log,
                'train_end': self.train_end,
                'num_checkpoint_saves': self.num_checkpoint_saves,
                'num_checkpoint_loads': self.num_checkpoint_loads,
            }

    def _get_checkpoint_state(self, save_optimizers, **kwargs):
        trainer_state = self._get_trainer_state()
        ckpt_state = {
            'trainer': trainer_state,
            'model': self.model.state_dict(include_optimizers=save_optimizers),
        }
        return ckpt_state


class SupervisedTrainer(Trainer):
    """
    Trainer class for supervised learning
    """
    
    def fit(self, num_epochs: int=None, trainer_state=None) -> None:
        assert hasattr(self, 'train_data') and (self.train_data is not None)

        # Use num_epochs instead of num_train_steps
        self.num_epochs = num_epochs or self.config.get('num_epochs')
        self.num_train_steps = None
        self.max_iters = self.num_epochs
        self.steps_per_epoch = len(self.train_data)
    
        self.before_train(trainer_state) # will load trainer checkpoint if needed
        with self.terminal_display:
            while not self.train_end:
                self.before_epoch()
                # Run training epoch
                for batch_idx, batch in enumerate(self.train_data):
                    print(batch_idx/len(self.train_data))
                    print(f"Epoch : {self.epoch}")
                    self.before_step()
                    self.train_step(batch, batch_idx)
                    self.after_step()
                    if self.train_end:
                        break
                self.evaluate()
                self.after_epoch()     
        self.after_train() 


    def evaluate(self, async_eval=False):
        """"
        Evaluation outputs are written onto a global dict
        to allow for asynchronous evaluation
        """
        assert hasattr(self, 'evaluator') and (self.evaluator is not None), 'Evaluator is not set!'
        if not async_eval:
            # Update the evaluator's model
            self.evaluator.load_model(state_dict=self.model.state_dict())
            eval_output = self.evaluator.run_eval()
            self.eval_log.append({'step': self.step, 'log': eval_output})
        else:
            raise NotImplementedError

    def _setup_metric_loggers(self, config, metrics=None):
        """
        Trainer metric is prediction error
        """
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

    def log_epoch(self, info=None):
        """
        Log values from self.train_log and self.eval_log
        Optionally, additional info may be provided via `info`
        """

        # Train log
        train_data = [log['log'] for log in self.train_log]
        train_metrics = self._get_metrics(train_data, self.metrics, 
                                          storage=self.output_storage)
        self.logger.log(log_dict={
            'trainer_step': self.step,
            'train/loss_avg': train_metrics['prediction_loss']['avg'],
            'train/loss_std': train_metrics['prediction_loss']['std'],
        })

        # Eval log
        eval_data = [log['log'] for log in self.eval_log]
        eval_metrics = self._get_metrics(eval_data, self.evaluator.metrics, 
                                         storage=self.evaluator.output_storage)
        self.logger.log(log_dict={
            'eval_step': self.step,
            'eval/loss_avg': eval_metrics['prediction_loss']['avg'],
            'eval/loss_std': eval_metrics['prediction_loss']['std'],
        })


    def _get_metrics(self, input_data, metrics, storage=None):

        # train_storage = self.output_storage

        data = {}
        for key in input_data[0].keys():
            data[key] = [x[key] for x in input_data]
            if isinstance(data[key][0], np.ndarray):
                data[key] = np.concatenate(data[key])

        metrics_dict = {}
        for metric_name, metric in metrics.items():
            if metric.type in ['image', 'video']:
                # If image or video, need storage so files can be saved
                metrics_dict[metric_name] = metric.log(data, storage)
            else:
                metrics_dict[metric_name] = metric.log(data)
        
        return metrics_dict