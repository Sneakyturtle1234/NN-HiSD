import numpy as np
import torch
import os
import sys

sys.path.append(".../..")
import core, utils

w0 = np.array([6.61, 3.26, 5.92])

k = 0
steps = 5000
dt = 1e-4
alpha = 6

dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(dir, 'parameters/parametric3d.pth')
model_info = torch.load(model_path)
net = utils.networks.DNN(input_size=model_info.get('input_size'), layer_sizes=model_info.get('layer_sizes'), 
                         output_size=model_info.get('output_size')).double()
net.load_state_dict(model_info['state_dict'])
print(f"Loaded pre-trained model with structure: {model_info.get('model_type')}")
print(f"Model configuration - Input size: {model_info.get('input_size')}, "
        f"Layers: {model_info.get('layer_sizes')}, "
        f"Output size: {model_info.get('output_size')}")


alpha0 = torch.tensor(alpha, dtype=torch.float64)
alpha0 = alpha0.unsqueeze(-1)
model1 = lambda w: net(torch.cat((w, alpha0)))
for i in range(20):
    addition = np.random.normal(0, 0.05, 3)
    w = w0 + addition
    w_saddle, _ = core.hisdnn(model=model1, w0=w, v0=None, method='sirqit',
                         dt=dt, ds=dt, max_iter=steps, k=k,
                         report=True, initial_hessian='full_hessian',
                         sub_iter=1, momentum=0.0, step_interval=10000)
    if (-1 < w_saddle[0] < 7) & (-1 < w_saddle[1] < 7) & (-1 < w_saddle[2] < 7):
        print(f"[{w_saddle[0]:.2f}, {w_saddle[1]:.2f}, {w_saddle[2]:.2f}]")
