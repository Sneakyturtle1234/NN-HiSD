import os
import sys
import torch
import numpy as np
import json
import subprocess
import importlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core import hisdnn
from utils.networks import DNN

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, 'rosen_num.json')
train_script_path = os.path.join(current_dir, '../../train.py')

params_dir = os.path.join(current_dir, 'parameters')
os.makedirs(params_dir, exist_ok=True)

for num_points in range(10000, 55000, 10000):
    with open(config_path, 'r') as f:
        config = json.load(f)

    config['num_points'] = num_points
    config['model_filename'] = f'parameters/rosen_num.pth'

    temp_config_path = os.path.join(current_dir, f'rosen_num.json')
    with open(temp_config_path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"Train the surrogate model. Use points number = {num_points} ...")
    result = subprocess.run([sys.executable, train_script_path, temp_config_path], 
                                cwd=os.path.dirname(train_script_path), text=True)

    model_path = os.path.join(current_dir, config['model_filename'])
    model_info = torch.load(model_path)
    net = DNN(input_size=model_info.get('input_size'), 
                layer_sizes=model_info.get('layer_sizes'), 
                output_size=model_info.get('output_size')).double()
    net.load_state_dict(model_info['state_dict'])
    print(f"Successfully load the model: {model_path}")

    initial_points = [np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])] 
    dt = 1e-4

    for i, w0 in enumerate(initial_points):
        saddle_point, record = hisdnn(model=net, w0=w0, v0=None, dt=dt, ds=dt,
                                        nesterov=True, restart=500, max_iter=2000, k=3, sub_iter=1,
                                        momentum=0.0, step_interval=100)
        
        print(saddle_point)

    if os.path.exists(model_path):
        os.remove(model_path)
