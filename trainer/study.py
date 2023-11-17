from omegaconf import OmegaConf, DictConfig
from trainer.storage import Storage
from copy import deepcopy
from trainer.utils import import_module_attr
from trainer import Sweeper
from trainer.rl.rl_evaluator import StudyRLEvaluator
import os
from trainer.metrics import Metric

class Study:
    
    def __init__(self, cfg):
        self.config = cfg
        self.name = self.config['study']['name']
        self.storage = Storage(cfg['study']['storage'])
        self._setup_metrics(cfg['study']['metrics'])
        self._setup_db(cfg['study']['database'])

    def train(self, config):
        trainer = self._make_trainer(config)
        trainer.fit()

    def sweep(self, config, num_runs=1):
        sweeper = self._make_sweeper(config)
        sweeper.sweep(count=num_runs, make_trainer_fn=self._make_trainer)

    def evaluate(self, config):

        evaluator_config = self._merge_configs(self.config['evaluator'], config['evaluator'])

        project_name = config['evaluator']['project']
        sweep_name = config['evaluator'].get('sweep')
        run_name = config['evaluator'].get('run')
        
        # Check storage for runs in the sweep
        root_dir = os.path.join(self.storage.dir, project_name)
        if sweep_name is not None:
            root_dir = os.path.join(root_dir, f"sweep_{sweep_name}")

        project_folders = self.storage.get_filenames(dir=os.path.join(self.config['name'], root_dir))

        assert len(project_folders) > 0, f'No runs found in {root_dir}'

        if run_name is not None:
            # Run the single run provided
            assert run_name in project_folders, f'Run {run_name} not found in {root_dir}'
            self._run_evaluation(evaluator_config)
        else:
            # Loop through all runs in the sweep folder
            for run_name in project_folders:
                if run_name.startswith('sweep'):
                    sweep_name = run_name.split('sweep_')[-1]
                    sweep_folders = self.storage.get_filenames(dir=os.path.join(self.config['name'], root_dir, run_name))
                    for idr, sweep_run_name in enumerate(sweep_folders):
                        print(f"Running {sweep_run_name} in sweep {sweep_name} ({idr}/{len(sweep_folders)})")
                        run_eval_config = deepcopy(evaluator_config)
                        run_eval_config['run'] = sweep_run_name
                        run_eval_config['sweep'] = sweep_name
                        self._run_evaluation(run_eval_config)
                else:
                    run_eval_config = deepcopy(evaluator_config)
                    run_eval_config['run'] = run_name
                    self._run_evaluation(run_eval_config)


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
        # TODO: Avoid having to get env_shapes from trainer
        model_config.update(trainer.env.get_env_shapes()) # Need env shapes from trainer's env
        model_cls = import_module_attr(model_config['module_path'])
        return model_cls(dict(model_config))
    
    def _load_model(self, config, state_dict):
        model_config = self._merge_configs(self.config['model'], config)
        model_cls = import_module_attr(model_config['module_path'])
        model = model_cls(model_config)
        model.load_model(state_dict=state_dict)
        return model

    def _make_sweeper(self, config):
        sweeper_config = {'project': self.config['project']['project']}
        sweeper_config['sweeper'] = self._merge_configs(self.config.get('sweeper', {}), 
                                             config.get('sweeper', {}))
        sweeper_config['trainer'] = self._merge_configs(self.config['trainer'], 
                                                        config.get('trainer', {}))
        # Make sure input/output storages use appropriate root_dir/project/sweep folders
        sweeper_config['trainer']['storage']['input'].update({
                'project': sweeper_config['project'],
                'sweep': sweeper_config['sweeper']['name']
        })
        sweeper_config['trainer']['storage']['output'].update({
                'project': sweeper_config['project'],
                'sweep': sweeper_config['sweeper']['name']
        })
        sweeper_config['logger'] = self._merge_configs(self.config['logger'], 
                                                        config.get('logger', {}))
        sweeper_config['model'] = self._merge_configs(self.config['model'], 
                                                        config.get('model', {}))
        return Sweeper(sweeper_config)
    

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

        # Set environment domain and task names
        for env_name in evaluator_config['envs'].keys():
            evaluator_config['envs'][env_name]['domain_name'] = evaluator_config['domain_name']
            evaluator_config['envs'][env_name]['task_name'] = evaluator_config['task_name']

        return StudyRLEvaluator(evaluator_config, db=self.db)

    def _merge_configs(self, orig_cfg, new_cfg, to_dict=True):
        output_cfg = OmegaConf.merge(orig_cfg, new_cfg)
        if to_dict:
            # Convert to a standard (resolved) dictionary object
            output_cfg = OmegaConf.to_container(output_cfg, resolve=True)
        return output_cfg

    def _run_evaluation(self, config):
        evaluator = self._make_evaluator(config)
        model_config = evaluator.input_storage.load('model_config.yaml', filetype='yaml')
        model_state_dict = evaluator.input_storage.load_from_archive('ckpt.zip', 
                                                                        filenames=['model_ckpt.pt'],
                                                                        filetypes=['torch'])
        model = self._load_model(model_config, model_state_dict['model_ckpt.pt'])
        evaluator.set_model(model)
        evaluator.run_eval()


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