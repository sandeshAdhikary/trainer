import numpy as np
import os
import tempfile
import importlib
import yaml
import argparse
import pickle
import zipfile

if __name__ == "__main__":

    # Load all models and 
    # This is needed to load model/trainer classes from pickles
    with open('trainer/.config.yaml') as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
    # Import registered models
    for module_type in ['models', 'trainers']:
        module_dict = yaml_dict[module_type]
        for name, module in module_dict.items():
            module_obj = importlib.import_module(module)
            globals().update({name: getattr(module_obj, name)})    

    args = argparse.ArgumentParser()
    # logdir = "logdir/wandb/run-20231019_012741-fy9ilbkb/files"
    # logdir/wandb/run-20231018_230036-j4smoh3p/files/eval_checkpoint.zip
    args.add_argument("--trainer_info", type=str, required=True)
    args.add_argument("--eval_checkpoint", type=str, required=True)
    # args.add_argument("--num_eval_episodes", type=int, default=10)
    args.add_argument("--eval_log_file", type=str, default=None)
    args.add_argument("--eval_output_file", type=str, default=None)
    args.add_argument("--step", type=int, default=None)
    args = args.parse_args()

    # Load trainer info and configs
    with open(args.trainer_info, 'rb') as f:
        trainer_info = pickle.load(f)
    model_config = trainer_info['model_config']
    trainer_config = trainer_info['trainer_config']

    # Load model
    model = trainer_info['model_class'](model_config)
    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(args.eval_checkpoint, 'r') as f:
            f.extractall(tmp_dir)
        model.load_model(os.path.join(tmp_dir), 'state_dicts')

    # Load trainer
    trainer = trainer_info['trainer_class'](trainer_config, model=model)

    # Evaluate
    trainer.evaluate(training_mode=False, 
                     async_eval=False,
                     eval_log_file=args.eval_log_file,
                     eval_output_file=args.eval_output_file
                     )
