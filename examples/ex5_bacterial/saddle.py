import os
import time
import sys
import torch
import numpy as np
import torch.nn as nn

sys.path.append(".../..")
import utils, core

class DNN(nn.Module):
    def __init__(self, in_dim, layer, out_dim):
        super(DNN, self).__init__()
        self.layer = layer
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mean = torch.tensor([-6.579694, 0.6807031], dtype=torch.float64)
        self.std = torch.tensor([77.77191, 90.70427], dtype=torch.float64)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, layer[0]))
        for i in range(1, len(layer)):
            self.layers.append(nn.Linear(layer[i-1], layer[i]))
        self.layers.append(nn.Linear(layer[-1], out_dim))
        self.activation = nn.Tanh()
              
    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = self.activation(self.layers[i](x))
        x = self.layers[-1](x)
        return x

dt = 5e-4
w0 = np.array([69, -105])
k = 2

Region = np.array([[-150, 150], [-200, 150]])
[in_dim, out_dim] = [2, 1]
layer = [128, 128, 128]
model = DNN(in_dim=in_dim, layer=layer, out_dim=out_dim)
model_filename = 'bacterial.pth'
dir = os.path.dirname(os.path.abspath(__file__))
model.load_state_dict(torch.load(os.path.join(dir, model_filename)))
model.double()

for i in range(10):
    addition = np.random.normal(0, 1, 2)
    w = (w0 + addition - model.mean.numpy())/model.std.numpy()
    start_time = time.time()
    w_saddle, _ = core.hisdnn(model=model, w0=w, v0=None, method='sirqit',
                         dt=dt, ds=dt, max_iter=5000, k=k,
                         report=True, initial_hessian='full_hessian',
                         sub_iter=1, momentum=0.5, step_interval=500)
    end_time = time.time()
    print(f"Running Time: {end_time - start_time:.4f} s")
    
    w_saddle = w_saddle * model.std.numpy() + model.mean.numpy()
    print(f"[{w_saddle[0]:.0f}, {w_saddle[1]:.0f}]")