import wandb
from functools import partial
from copy import deepcopy
from datetime import datetime
from time import sleep
from trainer.utils import flatten_dict, unflatten_dict, nested_dict_update

class Sweeper:

    def __new__(cls, config, trainer_cls, model_cls, logger_cls):
        sweeper_type = config['sweeper'].get("sweeper_type")

        if sweeper_type == "wandb":
            return WandBSweeper(config, trainer_cls, model_cls, logger_cls)
        else:
            raise ValueError("Invalid sweeper_type in the config")

class WandBSweeper():

    def __init__(self, config, trainer_cls, model_cls, logger_cls):
        self.config = config['sweeper']
        self.trainer_cls = trainer_cls
        self.model_cls = model_cls
        self.logger_cls = logger_cls

        self.project = self.config['project']
        self.sweep_name = self.config.get('name')
        # A run is considered timed-out after this many seconds
        # This determines whether that run should be picked up by another process
        self.heartbeat_timeout = self.config.get('heartbeat_timeout', 30)
        # Default params: everything but the 'sweeper' config
        default_params = deepcopy(config)
        default_params.pop('sweeper')
        self.default_params = default_params
    
    def make_wandb_sweep(self):
        parameters = self.config['parameters']
        method = self.config.get('method', 'random') # dafault to random
        goal = self.config.get('goal', 'maximize') # default to maximize
    
        wandb_sweep_config = {
            'name': self.sweep_name,
            'method': method,
            'metric': {"goal": goal, "name": "sweep_score"},
            'parameters': flatten_dict(parameters),    
        }
        
        # Create sweep and set up sweep if
        return wandb.sweep(wandb_sweep_config, project=self.project)
    

    def run_from_queue(self):
        # Query the api for sweep info

        api = wandb.Api()
        
        queue_empty = False
        while not queue_empty:
            sweep_query = api.sweep(f"{self.project}/{self.sweep_id}")
            run_queue = []
            for run in sweep_query.runs:
                time_since_heartbeat = (datetime.utcnow() - datetime.strptime(run._attrs['heartbeatAt'], "%Y-%m-%dT%H:%M:%S")).seconds
                if ('InProgress' in run.tags) and (time_since_heartbeat >= self.heartbeat_timeout):
                    run_queue.append(run.id)
                
            if len(run_queue) > 0:
                # Execute first run in the queue
                self.objective(run_queue[0])
                # Allow enough time so run can be picked up
                print(f"Waiting for heartbeat timeout for {self.heartbeat_timeout} seconds...")
                sleep(self.heartbeat_timeout)
            else:
                # No runs in the queue
                queue_empty = True        


    def sweep(self, count):
        """
        Sets up a wandb.sweep() and calls wandb.agent()
        """

        assert self.default_params is not None, "Default params not set. Run set_default_params() first."
        
        if self.sweep_exists():
            print(f"Loading sweep {self.sweep_name} from project {self.project}")
            print(f"Waiting for heartbeat timeout for {self.heartbeat_timeout} seconds...")
            sleep(self.heartbeat_timeout)
            self.sweep_id = self.project_sweeps_dict()[self.sweep_name]['id']
            if self.config['load_runs_from_queue']:
                self.run_from_queue()                    
        else:
            print(f"Creating new sweep {self.sweep_name} in project {self.project}")
            self.sweep_id = self.make_wandb_sweep()
        
        wandb.agent(self.sweep_id, 
                    function=partial(self.objective), 
                    count=count, 
                    project=self.project)


    def objective(self, run_id=None):

        wandb_kwargs = {
            'project': self.project,
            'dir': self.default_params['logger']['dir'],
        }
        if run_id is not None:
            wandb_kwargs.update({
                'id': run_id, 
                'resume': 'must'
            })
    
        # Get trial_params from wandb
        run = wandb.init(**wandb_kwargs)
        trail_params = unflatten_dict(dict(wandb.config))
        
        # Create a config for the objective run
        obj_config = deepcopy(self.default_params)
        obj_config = nested_dict_update(obj_config, trail_params) # Update with trial params from wandb sweeper

            
        # Instantiate trainer
        trainer = self.trainer_cls(obj_config['trainer'])
        # Set up model
        obj_config['model'].update(trainer.env.get_env_shapes())
        model = self.model_cls(obj_config['model'])
        trainer.set_model(model)
        # Set up logger
        logger = self.logger_cls(obj_config['logger'], run=run)
        trainer.set_logger(logger)
        # Train
        trainer.fit()

    def project_sweeps_dict(self):
        project = wandb.Api().project(self.project)
        try:
            project_sweeps = project.sweeps()
            # wandb.errors.CommError
            if len(project_sweeps) > 0:
                project_sweeps = {x.name: {'id': x.id} for x in project_sweeps}
            else:
                project_sweeps = {}
            return project_sweeps
        except wandb.errors.CommError:
            return {}


    def sweep_exists(self):
        return self.sweep_name in self.project_sweeps_dict().keys()