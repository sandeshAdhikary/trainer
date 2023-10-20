import numpy as np
import os
import tempfile
import importlib
import yaml
import argparse
import pickle
import zipfile
from warnings import warn
# Load all registered models and trainers
# This is needed to load model/trainer classes from pickles
from trainer.utils import import_registered_classes
import_registered_classes(globals())

if __name__ == "__main__":

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
    # with tempfile.TemporaryDirectory() as tmp_dir:
    #     with zipfile.ZipFile(args.eval_checkpoint, 'r') as f:
    #         f.extractall(tmp_dir)
    model.load_model(model_file=args.eval_checkpoint)

    # Load trainer
    trainer = trainer_info['trainer_class'](trainer_config, model=model)

    # Evaluate
    trainer.evaluate(training_mode=False, 
                     async_eval=False,
                     eval_log_file=args.eval_log_file,
                     eval_output_file=args.eval_output_file
                     )
