from abc import ABC
from omegaconf import DictConfig, OmegaConf
import os
from trainer.storage import Storage
from hydra import compose, initialize_config_dir
from trainer.utils import make_configs
from trainer.utils import import_module_attr
from omegaconf import OmegaConf, DictConfig
from datetime import datetime, timezone, timedelta
import json
from trainer.utils import unflatten_dict
from multiprocessing import cpu_count, Process
from tqdm import tqdm
from hydra.core.global_hydra import GlobalHydra
from ast import literal_eval

class TrainingQueue(ABC):

    def __init__(self, cfg):
        self.config = cfg
        self.storage = Storage(cfg['storage'])
        # Heartbeat 
        self.heartbeat_timeout = cfg.get('heartbeat_timeout', 20)
        self.last_heartbeat = None
        self.complete = False
        self.multi_proc_heartbeat = cfg.get('multi_proc_heartbeat', True) and cpu_count() > 1
        self.config_names = self._get_config_names()

    def run_from_queue(self, queue_name, default_config_dir=None, default_config_file=None):
        default_config_dir = default_config_dir or os.environ['DEFAULT_CONFIG_DIR']
        default_config_file = default_config_file or os.environ['DEFAULT_CONFIG_FILE']

        self.run = self.storage.dequeue_run(queue_name, self.heartbeat_timeout)
        if self.run is None:
            print("No runs in queue!")
            return None
        
        heartbeat_process = None
        if self.multi_proc_heartbeat:
            # Send heartbeats in a separate process
            heartbeat_process = Process(target=self._update_heartbeat)
            heartbeat_process.start()
        
        # The run task
        if self.run['status'] == 'InProgress':
            # Resume existing run
            self._resume_existing_run(default_config_dir)
        else:
            # Start new run
            self._start_new_run(default_config_dir, default_config_file)
    
        if heartbeat_process is not None:
            heartbeat_process.join()


    def _get_config_names(self):
        config_storage_type = os.environ['DEFAULT_CONFIG_STORAGE_TYPE']
        if config_storage_type == 'ssh':
            storage_params = {
                'type': 'ssh',
                'host': os.environ['SSH_HOST'],
                'username': os.environ['SSH_USERNAME'],
                'password': os.environ['SSH_PASSWORD'],
                'root_dir': os.path.join(os.environ['SSH_DIR'], os.environ['DEFAULT_CONFIG_DIR'])
            }
        elif config_storage_type == 'local':
            storage_params = {
                'type': 'local',
                'root_dir': os.environ['DEFAULT_CONFIG_DIR']
            }
        else:
            raise ValueError(f"Invalid config storage type: {config_storage_type}")
        cfg_storage = Storage(storage_params)
        project_names = cfg_storage.get_filenames(dir=cfg_storage.dir + '/project')
        project_names = [x.split('.yaml')[0] for x in project_names if x.endswith('.yaml')]
        exp_names = cfg_storage.get_filenames(dir=cfg_storage.dir + '/exp')
        exp_names = [x.split('.yaml')[0] for x in exp_names if x.endswith('.yaml')]
        cfg_storage.close_connection()
        return {
            'projects': project_names,
            'exps': exp_names
        }

    def _start_new_run(self, default_config_dir, default_config_file):
        GlobalHydra().clear()
        with initialize_config_dir(config_dir=default_config_dir, version_base=None):
            run_cmd = [
                f"""+project={self.run['project']}""",
                f"""+exp={self.run['exp']}""",
                """++exp.exp_mode=train""",
            ]   
            cfg = compose(config_name=default_config_file, overrides=run_cmd)
            self.study_config, self.exp_config = make_configs(cfg)
            overrides = {}
            if self.run.get('overrides'):
                overrides = json.loads(self.run['overrides'].replace('\\n', '').replace('\\t', ''))
                if not isinstance(overrides, dict):
                    overrides = json.loads(overrides)
                overrides = unflatten_dict(overrides)

            # Append tags to logger
            logger_cfg = self.exp_config['overrides'].get('logger', {})
            logger_cfg['tags'] = logger_cfg.get('tags', [])
            logger_cfg['tags'] = list(logger_cfg['tags']) 
            queue_tags = self.run.get('tags')
            if queue_tags: 
                new_tags = list(literal_eval(queue_tags))
                logger_cfg['tags'] += new_tags
            
            self.exp_config['overrides']['logger'] = logger_cfg

            self.exp_config['overrides'] = OmegaConf.merge(self.exp_config['overrides'], 
                                                           DictConfig(overrides))
            self.exp_config['overrides'] = OmegaConf.to_container(self.exp_config['overrides'])

            study_module_path = self.study_config['study'].get('study_module_path')
            assert study_module_path is not None
            study_class = import_module_attr(study_module_path)
            study = study_class(self.study_config)
            study.train(self.exp_config, 
                        pre_train_func=self._init_training_queue)
        self.complete = True

    def _resume_existing_run(self, default_config_dir):
        GlobalHydra().clear()
        with initialize_config_dir(config_dir=default_config_dir, version_base=None):
            self.study_config = OmegaConf.create(json.loads(self.run['study_config']))
            self.exp_config = OmegaConf.create(json.loads(self.run['exp_config']))

            study_module_path = self.study_config['study'].get('study_module_path')
            assert study_module_path is not None
            study_class = import_module_attr(study_module_path)
            study = study_class(self.study_config)
            logger_sw = study.config['logger']['sw']
            if logger_sw == 'wandb':
                # Resume a wandb run
                from wandb import init as wandb_init
                run = wandb_init(id=self.run['run_id'], 
                                 project=study.config['project']['name'],
                                 resume=True)
                study.train(self.exp_config,
                            pre_init_run=run,
                            pre_train_func=self._init_training_queue)
            else:
                raise NotImplementedError(f"Queuing with logger {logger_sw} is not implemented")

        self.complete = True

    def _init_training_queue(self, trainer):
        """
        Called by trainer after _make_trainer() and before trainer.fit()
        """
        if trainer.queue is None:
            trainer.set_queue(self)

        self.storage.update_run(self.run, 
                                new_run_info = {'exp_config': json.dumps(OmegaConf.to_container(self.exp_config)),
                                                'study_config': json.dumps(OmegaConf.to_container(self.study_config)),
                                                'status': 'InProgress'} 
        )

    def add_runs_to_queue(self, runs_config=None):
        runs = runs_config or self.config['runs']
        GlobalHydra().clear()
        with initialize_config_dir(config_dir=self.default_config_dir, version_base=None):
            for exp_config in tqdm(runs):
                run_info = {
                    'project': exp_config.get('project'),
                    'exp': exp_config.get('exp'),
                    'overrides': exp_config['overrides'],
                    'run_id': None,
                    # 'study_config': None,
                    'exp_config': None,
                    'priority': exp_config.get('priority')
                    }
                self.storage.enqueue_run(run_info)


    def before_train(self, trainer):
        if trainer.logger.sw_type == 'wandb':
            new_run_info = self.storage.update_run(self.run, 
                                    new_run_info = {'run_id': trainer.logger._sw.id,
                                                    'status': 'InProgress'}
                                    )
            self.run.update(new_run_info)
        else:
            raise NotImplementedError(f"Logging via {trainer.logger.sw_type} is not implemented")

    def after_epoch(self, trainer):
        """
        Update heartbeat
        """
        new_run_info = self.storage.update_run(self.run, 
                                new_run_info = {
                                    'heartbeat_at': datetime.now(timezone.utc),
                                    'progress': round(trainer.step/trainer.num_train_steps, 3),
                                    }
                                )
        self.run.update(new_run_info)


    def _update_heartbeat(self):
        while not self.complete:
            current_time = datetime.now(timezone.utc)
            since_heartbeat = current_time - self.last_heartbeat if self.last_heartbeat else timedelta(seconds=self.heartbeat_timeout+1)
            
            if since_heartbeat > timedelta(seconds=self.heartbeat_timeout):
                storage = Storage(self.config['storage'])
                storage.update_run(self.run, 
                                new_run_info = {'heartbeat_at': current_time,}
                                )
                self.last_heartbeat = current_time
                storage.close_connection()

    def after_train(self, trainer):
        """
        Change status to "Completed"
        """
        new_run_info = self.storage.update_run(self.run, 
                                new_run_info = {'status': "Completed",}
                                )
        self.run.update(new_run_info)