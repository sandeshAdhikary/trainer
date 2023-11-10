import builtins
import argparse
import pickle
from trainer.rl.rl_evaluator import TrainingRLEvaluator
from importlib import import_module

def import_class(module_path, class_name):
    return getattr(import_module(module_path),class_name)

if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--eval_packet", type=str, required=True)
    args = args.parse_args()

    with open(args.eval_packet, 'rb') as f:
        eval_packet = pickle.load(f)

    # Set up evaluator
    evaluator_packet = eval_packet['evaluator']
    evaluator_class = import_class(evaluator_packet['module'], evaluator_packet['class'])
    if evaluator_class==TrainingRLEvaluator:
        # This class needs a trainer as input
        trainer_packet = eval_packet['trainer']
        trainer_class = import_class(trainer_packet['module'], trainer_packet['class'])
        trainer = trainer_class(trainer_packet['config'])
        evaluator = evaluator_class(evaluator_packet['config'], trainer)
    else:
        evaluator = evaluator_class(evaluator_packet['config'])


    # Set up model
    model_packet = eval_packet['model']
    model_class = import_class(model_packet['module'], model_packet['class'])
    model = model_class(model_packet['config'])
    if model_packet.get('state_dict') is not None:
        # Load model state dict
        model.load_model(state_dict=model_packet['state_dict'])    
    evaluator.set_model(model)

    # Run evaluation
    evaluator.evaluate(storage=eval_packet['eval_storage'])