from trainer.study import RLStudy
from omegaconf import DictConfig, OmegaConf
import hydra
from copy import deepcopy


@hydra.main(version_base=None, config_path="configs", config_name='default_config')
def main(cfg: DictConfig) -> (DictConfig, DictConfig):

    # Resolve the config
    cfg = DictConfig(OmegaConf.to_container(cfg, resolve=False))

    # Get the composed config
    study_config = deepcopy(cfg)
    project_overrides = deepcopy(cfg['project']['overrides'])
    exp_config = deepcopy(cfg['exp'])
    study_config.__delattr__('exp')

    # Project overrides
    study_config = OmegaConf.merge(study_config, project_overrides)

    # Define study
    study = RLStudy(study_config)
    
    # Get the experiment mode: train/sweep/evaluate
    exp_mode = exp_config['exp_mode']
    exp_config.__delattr__('exp_mode')

    # Run experiment
    if exp_mode == 'train':
        study.train(exp_config)
    elif exp_mode == 'sweep':
        study.sweep(exp_config, num_runs=exp_config['sweeper']['num_runs'])
    elif exp_mode == 'evaluate':
        study.evaluate(exp_config)
    else:
        raise ValueError(f'exp_mode {exp_mode} not recognized')


if __name__ == '__main__':
    main()
