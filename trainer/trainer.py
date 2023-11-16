import torch
from abc import ABC, abstractmethod
from typing import Dict
from trainer.model import Model
import trainer.utils as utils
from trainer.logger import Logger
from rich.layout import Layout
from rich.console import Console
from rich.panel import Panel
from rich.progress import (Progress, MofNCompleteColumn, BarColumn, TextColumn, 
                           TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn)
from rich.columns import Columns
from rich.live import Live
from collections import deque
from warnings import warn
from contextlib import nullcontext
# from trainer.utils import register_class
from trainer.storage import Storage
import tempfile

class Trainer(ABC):

    def __init__(self, config: Dict, model: Model = None, logger: Logger = None) -> None:
        self.config = self.parse_config(config)
        self._setup(config, model, logger)
        # self._register_trainer()
    
    def fit(self, num_train_steps: int=None, trainer_state=None) -> None:
        self.num_train_steps = num_train_steps or self.config['num_train_steps']
    
        self.before_train(trainer_state) # will load trainer checkpoint if needed
        with self.terminal_display:
            while not self.train_end:
                self.before_epoch()
                # Run training epoch
                for batch_idx, batch in enumerate(self.train_data):
                    self.before_step()
                    self.train_step(batch, batch_idx)
                    self.after_step()
                self.evaluate(self.eval_data, training_mode=True)
                self.after_epoch()
                
        self.after_train() 
    

    def train_step(self, batch, batch_idx=None):
        train_step_output = self.model.training_step(batch, batch_idx)
        self.train_log.append({'step': self.step, 'log': train_step_output})

    def evaluate(self, eval_data=None, trainer_mode=False):
        raise NotImplementedError
    
    def set_model(self, model: Model) -> None:
        self.model = model

    def set_logger(self, logger: Logger) -> None:
        self.logger = logger

    @abstractmethod
    def setup_data(self, config: Dict) -> None:
        """
        Setup the dataset
        e.g. 
        self.data = {
        'train': train_dataset/dataloader,
        'eval': eval_dataset/dataloader,
        'replay_buffer': replay_buffer,
        }
        """
        self.data = None

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
        if self.progress is not None:
            # Initialize progress
            self.progress.update(self.progress_train, completed=self.step)
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



    def before_epoch(self, info=None):
        pass
    
    def before_step(self, info=None):
        self.model.train() # Set model to training mode

    def after_train(self, info=None):
        """
        Callbacks after training ends
        """
        # Close logger
        # Close any open connections
        self.log_train(info)
        self.logger.finish()

    def after_epoch(self, info=None):
        self.epoch += 1
        self.log_epoch(info)
        
        # Save checkpoint
        if (self.save_checkpoint_freq is not None) and (self.epoch % self.save_checkpoint_freq == 0) and (self.step > 0):
            self._save_checkpoint(ckpt_state_args={'save_optimizers': True})
            self.num_checkpoint_saves += 1
            self.logger.log(log_dict={'trainer_step': self.step, 'checkpoint/num_checkpoint_saves': self.num_checkpoint_saves})


    def after_step(self, info=None):
        # Update trainer state
        self.update_trainer_state()
        # Logging
        self.log_step(info)
        # Update progress
        if self.progress is not None:
            self.progress.update(self.progress_train, completed=self.step)


    def log_step(self, info=None):
        """
        Log using self.train_log and self.eval_log
        self.*_log = deque({'step': int, 'output': Dict})
        """
        pass

    def log_epoch(self, info=None):
        pass

    def log_train(self, info=None):
        pass



    def update_trainer_state(self):
        # Check if training is complete
        if self.step >= self.num_train_steps:
            self.train_end = True


    #############################
    
    # @property
    # def module_path(self):
    #     return None
    
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

    # def _register_trainer(self, overwrite=False):
    #     if self.module_path is not None:
    #         register_class('trainer', self.__class__.__name__, self.module_path)

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

    def setup_evaluator(self):
        raise NotImplementedError

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
                TextColumn("Training Progress"),
                BarColumn(complete_style="dark_sea_green4"),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                MofNCompleteColumn(),
                redirect_stdout=False
                )
            self.progress_train = self.progress.add_task("[dark_sea_green4] Training...", total=self.num_train_steps)

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

    def _get_checkpoint_state(self, save_optimizers, **kwargs):
        # # Save trainer state
        trainer_state = {'step': self.step,
                         'epoch': self.epoch,
                         'train_log': self.train_log, 
                         'eval_log': self.eval_log,
                         'train_end': self.train_end,
                         'num_checkpoint_saves': self.num_checkpoint_saves,
                         'num_checkpoint_loads': self.num_checkpoint_loads,
                         'obs': self.obs,
                         'done': self.done,
                         'reward': self.reward,
                         'num_episodes': self.num_episodes,
                         'episode': self.episode,
                         'current_episode_reward': self.current_episode_reward,
                         'episode_reward_list': self.episode_reward_list,
                         'num_model_updates': self.num_model_updates,
                         'epoch': self.epoch
                        }
        ckpt_state = {
            'trainer': trainer_state,
            'model': self.model.state_dict(save_optimizers=save_optimizers),
        }
        return ckpt_state

if __name__ == "__main__":
    import yaml
    with open('trainer/test_config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    model = Model(config['model'])
    logger = Logger(config['logger'])
    trainer = Trainer(config['trainer'], model, logger)

    trainer.train(config['trainer']['num_train_steps'])