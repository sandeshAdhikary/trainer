import torch
from abc import ABC, abstractmethod
from typing import Dict
from trainer.model import Model
import trainer.utils as utils
import os
from trainer.logger import Logger
import multiprocessing as mp
import numpy as np
import rich
from rich.layout import Layout
from rich.console import Console
from rich.panel import Panel
from rich.progress import (Progress, MofNCompleteColumn, BarColumn, TextColumn, 
                           TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn)
from rich.columns import Columns
from rich.live import Live
from collections import deque
import shutil
from warnings import warn
from contextlib import nullcontext
from datetime import datetime

class Trainer(ABC):

    def __init__(self, config: Dict, model: Model = None, logger: Logger = None) -> None:
        config = self.parse_config(config)
        self._setup(config, model, logger)
    
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

    def evaluate(self, eval_data=None, training_mode=False):
        """"
        Evaluation outputs are written onto a global dict
        to allow for asynchronous evaluation
        """
        run_eval = True
        if training_mode:
            run_eval = (self.step > 0) and (self.step % self.eval_freq == 0) 

        if run_eval:
            self.model.eval()
            if self.async_eval:
                # Run evaluation asynchronously, so training can continue
                if hasattr(self, 'eval_process') and self.eval_process is not None:
                    # Get results from last eval_process
                    # If last eval_process has not completed, wait for it
                    eval_step_output = self.eval_process.join()
                    self.eval_log.append({'step': self.step, 'log': eval_step_output})
                self.eval_process = mp.Process(target=self.model.evaluation_step, args=(eval_data,))
                self.eval_process.start()
            else:
                eval_step_output = self.model.evaluation_step(eval_data)
                self.eval_log.append({'step': self.step, 'log': eval_step_output})
    
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

        self._setup_terminal_display(self.config)
        self._set_seeds(self.config)
        # Optionally load from checkpoint
        if self.load_from_checkpoint:
            self._load_checkpoint()
        if self.progress is not None:
            # Initialize progress
            self.progress.update(self.progress_train, completed=self.step)
        # Set model to train mode
        self.model.train()

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

    def after_epoch(self, info=None):
        self.log_epoch(info)


    def after_step(self, info=None):
        # Update trainer state
        self.update_trainer_state()
        # Logging
        self.log_step(info)
        # Update progress
        if self.progress is not None:
            self.progress.update(self.progress_train, completed=self.step)
        # Save checkpoint
        self._save_checkpoint()


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
        self.step += 1
        # Check if training is complete
        if self.step >= self.num_train_steps:
            self.train_end = True


    #############################

    def parse_config(self, config: Dict) -> Dict:
        return config

    def init_trainer_state(self, state_dict=None):
        state_dict = state_dict or {}
        # Load from state_dict or initialize for start of training
        self.step = state_dict.get('step', 0)
        self.train_log = state_dict.get('train_log', deque(maxlen=self.log_length_train))
        self.eval_log = state_dict.get('eval_log', deque(maxlen=self.log_length_eval))
        self.train_end = state_dict.get('train_end', False)
        self.num_checkpoint_loads = state_dict.get('num_checkpoint_loads', 0)

    def _setup(self, config: Dict = None, model: Model = None, logger: Logger =None) -> None:
        self.config = config

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

    def _setup_terminal_display(self, config: Dict) -> None:
            
        if config.get('terminal_display') in ['rich', 'rich_minimal']:

            self._term_console = Console()
            self._term_layout = Layout()
            # Header: Run info panel
            self._term_header_panel = Panel.fit(f""" Project: [orange4] {self.logger.project} [/]\
                                    Run: [orange4] {self.logger.run_name}({self.logger.run_id})[/]""",
                                    title='Run Info', border_style='orange4')
            # Body: Progress panel
            self.progress = Progress(
                TextColumn("Overall Progress"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                MofNCompleteColumn(),
                redirect_stdout=False
                )
            self.progress_train = self.progress.add_task("[red] Training...", total=self.num_train_steps)

            if config.get('terminal_display') == 'rich_minimal':
                # Only a progress bar
                self.terminal_display = self.progress
            else:
                # Full terminal display
                self._term_progress_panel = Panel.fit(Columns([self.progress]), title="Progress", border_style="dark_sea_green4")
                # Footer: Misc info
                self._term_footer_text = f"""[bold]Device[/]: {self.device}"""
                self._term_footer_panel = Panel(self._term_footer_text, title="Misc", border_style='red')
                # Add panels to layout
                self._term_layout.split(
                    Layout(self._term_header_panel, size=3, name="header"),
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

    def _load_checkpoint(self):
        # Download checkpoint from logger
        try:
            self.logger.restore_checkpoint()
            # Load the trainer state
            trainer_state_dict = torch.load(f'{self.logger.logdir}/checkpoint/trainer_chkpt.pt')
            self.init_trainer_state(trainer_state_dict)
            # Load the model state
            self.model.load_model(f'{self.logger.logdir}/checkpoint/model_chkpt.pt')
            # Log the checkpoint load event
            self.num_checkpoint_loads += 1
            self.logger.log(key='train/num_checkpoint_loads', value=self.num_checkpoint_loads, step=self.logger._sw.step)
        except (UserWarning, Exception) as e:
            warn("Could not restore checkpoint.")
        

    def _save_checkpoint(self):
        """
        Zip the checkpoint folder and log it
        """
        if (self.save_checkpoint_freq is not None) and (self.step % self.save_checkpoint_freq == 0) and (self.step > 0):
            self._create_checkpoint_files()
            # Compress checkpoint folder to a single file. Then delete the chkpt directory
            chkpt_dir = os.path.join(self.logger.logdir,'checkpoint')
            shutil.make_archive(base_name=chkpt_dir,
                                format='zip', 
                                base_dir='checkpoint',
                                root_dir=self.logger.logdir)
            # Delete the checkpoint dir once it has been zipped
            shutil.rmtree(chkpt_dir)
            # Log the checkpoint files to the logger
            self.logger.log_checkpoint()


    def _create_checkpoint_files(self):
        """
        Custom Trainer class can add other checkpoint files here.
        The checkpoint files should have format {name}_chkpt.pt
        """

        # Create a (temporary) checkpoint directory
        chkpt_dir = os.path.join(self.logger.logdir,'checkpoint')
        if os.path.exists(chkpt_dir):
            # Delete the old checkpoint
            shutil.rmtree(chkpt_dir)
        os.makedirs(chkpt_dir, exist_ok=False)

        # Save trainer state
        trainer_state = {'step': self.step,
                         'train_log': self.train_log, 
                         'eval_log': self.eval_log,
                         'train_end': self.train_end,
                         'num_checkpoint_loads': self.num_checkpoint_loads
                        }
        torch.save(trainer_state, os.path.join(chkpt_dir, 'trainer_chkpt.pt'))
        with open(os.path.join(chkpt_dir, 'test.txt'), 'w') as f:
            f.write(f"time: {datetime.now()}\n trainer_step: {self.step}")
        # Save model state
        self.model.save_model(os.path.join(chkpt_dir, 'model_chkpt.pt'))



if __name__ == "__main__":
    import yaml
    with open('trainer/test_config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    model = Model(config['model'])
    logger = Logger(config['logger'])
    trainer = Trainer(config['trainer'], model, logger)

    trainer.train(config['trainer']['num_train_steps'])