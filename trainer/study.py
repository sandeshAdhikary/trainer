from omegaconf import OmegaConf, DictConfig
from trainer.storage import Storage
from copy import deepcopy
from trainer.utils import import_module_attr
from trainer import Sweeper
from trainer.rl.rl_evaluator import StudyRLEvaluator
import os
from trainer.metrics import Metric
from abc import ABC
from trainer.model import NullModel

class Study(ABC):
    
    def __init__(self, cfg):
        self.config = cfg
        self.name = self.config['study']['name']
        self.storage = Storage(cfg['study']['storage'])
        self._setup_metrics(cfg['study'].get('metrics'))
        database = cfg['study'].get('database')
        if database:
            self._setup_db(cfg['study']['database'])

    def train(self, config):
        trainer = self._make_trainer(config['overrides'])
        trainer.fit()

    def sweep(self, config, num_runs=1):
        sweeper = self._make_sweeper(config)
        sweeper.sweep(count=num_runs, make_trainer_fn=self._make_trainer)

    def evaluate(self, config):

        evaluator_config = self._merge_configs(self.config['evaluator'], config['evaluator'])

        project_name = evaluator_config['project']
        sweep_names = config['evaluator'].get('sweeps')
        run_names = config['evaluator'].get('runs')
        
        if config['evaluator'].get('random_agent'):
            self._evaluate_random_agent(project_name, evaluator_config)
        elif run_names is not None:
            self._evaluate_runs(project_name, run_names, evaluator_config)
        elif sweep_names is not None:
            self._evaluate_sweeps(project_name, sweep_names, evaluator_config)
        else:
            raise ValueError("Either evaluator.runs or evaluator.sweeps must be provided")


    def _evaluate_random_agent(self, project_name, evaluator_config):
        run_eval_config = deepcopy(evaluator_config)
        run_eval_config['sweep'] = 'random_agent'
        run_eval_config['run'] = 'random_agent'
        run_eval_config['random_agent'] = True
        self._run_evaluation(run_eval_config)

    def _evaluate_sweeps(self, project_name, sweep_names, evaluator_config):
        for sweep_name in sweep_names:
            root_dir = os.path.join(self.storage.dir, project_name, f"sweep_{sweep_name}")
            run_folders = self.storage.get_filenames(dir=os.path.join(project_name, root_dir))
            for run_name in run_folders:
                run_eval_config = deepcopy(evaluator_config)
                run_eval_config['run'] = run_name
                run_eval_config['sweep'] = sweep_name
                try:
                    self._run_evaluation(run_eval_config)
                except Exception as e:
                    print(f"Error running {run_name} in sweep {sweep_name}. Error: {e}")
                    continue

    def _evaluate_runs(self, project_name, run_names, evaluator_config):
        # Check storage for runs in the sweep
        for run_name in run_names:
            sweep_name = None
            if len(run_name.split('/')) > 1:
                sweep_name, run_name = run_name.split('/')
            
            run_eval_config = deepcopy(evaluator_config)
            run_eval_config['run'] = run_name
            run_eval_config['sweep'] = sweep_name
            try:
                self._run_evaluation(run_eval_config)
            except Exception as e:
                print(f"Error running {run_name} in sweep {sweep_name or '[No sweep]'}. Error: {e}")
                continue


    def show_runs(self, run_names=None, limit=None, output_format='pandas'):
        """
        Return a dictionary of studies in the run
        """
        return self.db.show_runs(run_names, limit, output_format)
    

    def run_info(self, project, run_id, sweep=None):
        """
        """
        run_path = f"{project}/{run_id}/train"
        if sweep is not None:
            run_path = f"{project}/sweep_{sweep}/{run_id}/train"
        model_config = self.storage.load(f"{run_path}/model_config.yaml", filetype='yaml')
        trainer_config = self.storage.load(f"{run_path}/trainer_config.yaml", filetype='yaml')

        return {'model': model_config, 'trainer': trainer_config}

    def metric_table(self, metric_name, limit=None):
        """
        """
        return self.db.show_metric_table(metric_name, limit)

    def _setup_metrics(self, config):
        self.metrics = {}
        if config is not None:
            for metric, metric_config in config.items():
                metric_config.update({'name': metric})
                self.metrics[metric] = Metric(metric_config)

    def _setup_db(self, config):
        self.db = Storage(config)
        for metric_name, metric in self.metrics.items():
            self.db.add_metric_table(metric_name, metric)
        
    def _make_trainer(self, config, pre_init_run=None):
        """
        config: config dict with trainer, model, and logger configs
        pre_init_runs: Optional pre-initialized run object (e.g. wandb.run)
                       This is needed to initialize the logger for resumed runs
        """
        # Set up the trainer
        train_config = self._merge_configs(self.config['trainer'], config.get('trainer', {}))
        train_config['storage']['input']['project'] = train_config['project']
        train_config['storage']['output']['project'] = train_config['project']
        trainer_cls = import_module_attr(train_config['module_path'])
        trainer = trainer_cls(dict(train_config))

        # Set up model
        model = self._make_model(config, trainer=trainer)
        trainer.set_model(model)
        # Set up logger
        logger = self._make_logger(config, pre_init_run=pre_init_run)
        trainer.set_logger(logger)
        return trainer
    
    def _make_logger(self, config, pre_init_run=None):
        logger_config = self._merge_configs(self.config['logger'], config.get('logger', {}))
        logger_cls = import_module_attr(logger_config['module_path'])
        return logger_cls(dict(logger_config), run=pre_init_run)

    def _make_model(self, config, trainer):
        model_config = self._merge_configs(self.config['model'], config.get('model', {}))
        model_cls = import_module_attr(model_config['module_path'])
        return model_cls(dict(model_config))
    
    def _load_model(self, config, state_dict):
        model_config = self._merge_configs(self.config['model'], config)
        model_cls = import_module_attr(model_config['module_path'])
        model = model_cls(model_config)
        model.load_model(state_dict=state_dict)
        return model

    def _make_sweeper(self, config):
        sweeper_config = {'project': self.config['project']['name']}
        sweeper_config['sweeper'] = self._merge_configs(self.config.get('sweeper', {}), 
                                             config.get('sweeper', {}))
        
        # Apply trainer and model overrides
        exp_overrides = config.get('overrides', {}) or {}
        
        sweeper_config['trainer'] = self._merge_configs(self.config['trainer'], 
                                                        exp_overrides.get('trainer', {}))
        sweeper_config['model'] = self._merge_configs(self.config['model'], 
                                                        exp_overrides.get('model', {}))
        sweeper_config['logger'] = self._merge_configs(self.config['logger'], 
                                                        exp_overrides.get('logger', {}))
        

        sweeper_config['trainer']['storage']['input'].update({
                'project': sweeper_config['project'],
                'sweep': sweeper_config['sweeper']['name']
        })
        sweeper_config['trainer']['storage']['output'].update({
                'project': sweeper_config['project'],
                'sweep': sweeper_config['sweeper']['name']
        })
 
        return Sweeper(sweeper_config)
    
    def _merge_configs(self, orig_cfg, new_cfg, to_dict=True):
        output_cfg = OmegaConf.merge(orig_cfg, new_cfg)
        if to_dict:
            # Convert to a standard (resolved) dictionary object
            output_cfg = OmegaConf.to_container(output_cfg, resolve=True)
        return output_cfg

    def _run_evaluation(self, config):
        evaluator = self._make_evaluator(config)
        if not config['random_agent']:
            # If not a random agent, load a model
            model_config = evaluator.input_storage.load('model_config.yaml', filetype='yaml')
            ckpt_name = 'best_ckpt' if config.get('use_best_ckpt') else 'ckpt'
            model_state_dict = evaluator.input_storage.load_from_archive(f'{ckpt_name}.zip', 
                                                                filenames=[f'model_{ckpt_name}.pt'],
                                                                filetypes=['torch']
                                                                )
            model = self._load_model(model_config, model_state_dict[f'model_{ckpt_name}.pt'])
            
        else:
            model = NullModel()
        evaluator.set_model(model)
        evaluator.run_eval()

    def _make_evaluator(self, config):
        raise NotImplementedError


class RLStudy(Study):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _make_model(self, config, trainer):
        model_config = self._merge_configs(self.config['model'], config.get('model', {}))
        model_config['obs_shape'] = trainer.env.observation_space.shape[1:]
        model_config['action_shape'] = trainer.env.action_space.shape
        if len(model_config['action_shape']) > 1:
            model_config['action_shape'] = model_config['action_shape'][1:]
        model_cls = import_module_attr(model_config['module_path'])
        return model_cls(dict(model_config))


    def _make_evaluator(self, config):
        evaluator_config = deepcopy(config)
        # evaluator_config = self._merge_configs(self.config['evaluator'], config)
        
        # Make trainer config to get make_env function
        train_config = self._merge_configs(self.config['trainer'], config.get('trainer', {}))
        evaluator_config['make_env_module_path'] = train_config['make_env_module_path']
        
        # Assign run directories
        evaluator_config['storage']['input']['run'] = config['run']
        evaluator_config['storage']['input']['project'] = config['project']
        evaluator_config['storage']['output']['run'] = config['run']
        evaluator_config['storage']['output']['project'] = config['project']
        
        # (Optionally) Assign sweep directories
        if evaluator_config.get('sweep') is not None:
            evaluator_config['storage']['input']['sweep'] = evaluator_config['sweep']
            evaluator_config['storage']['output']['sweep'] = evaluator_config['sweep']

        # Set up evaluation metrics
        metrics_dict = {name: Metric(cfg) for (name, cfg) in evaluator_config['metrics'].items()}
        # for metric_name, metric in evaluator_config['metrics'].items():
        #     metric_dict.update({metric_name: Metric(metric)})
        # evaluator_config['metrics'][metric_name] = metric

        if evaluator_config.get('module_path') is not None:
            evaluator_cls = import_module_attr(evaluator_config['module_path'])
        else:
            evaluator_cls = StudyRLEvaluator

        return evaluator_cls(evaluator_config, db=self.db, metrics=metrics_dict)

class RLStudySB3(RLStudy):

    def _make_model(self, config, trainer):
        """
        SB3 models need environment to initialize. 
        So, we'll just pass the trainer object
        """
        model_config = self._merge_configs(self.config['model'], config.get('model', {}))
        model_config['obs_shape'] = trainer.env.observation_space.shape[1:]
        model_config['action_shape'] = trainer.env.action_space.shape
        if len(model_config['action_shape']) > 1:
            model_config['action_shape'] = model_config['action_shape'][1:]
        model_cls = import_module_attr(model_config['module_path'])
        return model_cls(dict(model_config), trainer=trainer)


if __name__ == '__main__':

    study_config_path = '/project/src/study/configs/bisim_study_defaults.yaml'
    run_config_path = '/project/src/studies/configs/bisim_sample.yaml'
    # run_config_path = '/project/src/studies/configs/bisim_study.yaml'

    study_cfg = OmegaConf.load(study_config_path)
    run_cfg = OmegaConf.load(run_config_path)
    
    study = Study(study_cfg)

    study.sweep(run_cfg)
    # study.train(run_cfg)
    # study.evaluate(run_cfg)