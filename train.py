import os
import json
import torch
import numpy as np
import importlib
import sys
from core.trainer import trainer
from utils.networks import DNN
import utils.createdata

def training(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    layer = config.get('layer', [128, 128, 128])
    learning_rate = config.get('lr', 1e-3)
    epochs = config.get('epoch', 10000)
    batch_size = config.get('batch_size', 500)
    output_dir = config.get('output_dir', os.path.dirname(config_path))
    model_filename = config.get('model_filename', 'parameters/simple2d.pth')

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, model_filename)
    
    function_name = config.get('function_name')
    region = np.array(config.get('region', []))
    num_points = config.get('num_points', 5000)
    dataset_type = config.get('dataset_type', 'random')
    seed = config.get('seed', 42)

    config_dir = os.path.dirname(config_path)
    
    function_module_path_config = config.get('function_module_path', 'functions.py')
    if not os.path.isabs(function_module_path_config):
        function_module_path = os.path.join(config_dir, function_module_path_config)
    else:
        function_module_path = function_module_path_config
    
    if os.path.exists(function_module_path):
        function_name = config.get('function_name', 'Potential')
        module_dir = os.path.dirname(function_module_path)
        module_name = os.path.splitext(os.path.basename(function_module_path))[0]
        if module_dir not in sys.path:
            sys.path.append(module_dir)

        functions_module = importlib.import_module(module_name)

        potential_function = getattr(functions_module, function_name, None)
        if potential_function:
            print(f"Generating data using {dataset_type} dataset with {function_name} function...")
            if dataset_type == 'random':
                data, target = utils.createdata.create_random_dataset(potential_function, region, num_points, seed)
            elif dataset_type == 'random_with_noise':
                proportion = config.get('proportion', 1.0)
                std_dev = config.get('std_dev', 0.1)
                data, target = utils.createdata.create_random_dataset_Noise(potential_function, region, num_points, proportion, std_dev, seed)
            elif dataset_type == 'random_with_grad':
                potential_grad = getattr(functions_module, f"{function_name}_grad", None)
                if potential_grad:
                    proportion = config.get('proportion', 1.0)
                    data, target = utils.createdata.create_random_dataset_grad(potential_function, potential_grad, region, num_points, proportion=proportion, seed=seed)
                else:
                    raise ValueError(f"Gradient function {function_name}_grad not found in {function_module_path}")
            elif dataset_type == 'regular':
                data, target = utils.createdata.create_regular_dataset(potential_function, region, num_points)
            elif dataset_type == 'parametric':
                data, target = utils.createdata.create_parametric_dataset(potential_function, region, num_points)
            else:
                raise ValueError(f"Unknown dataset type: {dataset_type}")
            
            print(f"Data generated: {len(data)} points")
        else:
            raise ValueError(f"Function {function_name} not found in {function_module_path}")
    else:
        raise ValueError(f"Function module not found at {function_module_path}")
    
    TrainingLoss = trainer(
        seed=seed,
        data=data,
        target=target,
        model_path=model_path,
        layer=layer,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        report=config.get('report', True),
        report_interval=config.get('report_interval', 100),
        grad_output = config.get('grad_output', False),
        grad_data=(dataset_type== 'random_with_grad')
    )
    
    print(f"Training completed, model saved to: {model_path}")
    return TrainingLoss

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        print("Usage: python script.py config.json")
        sys.exit(1)
    
    training(config_file)