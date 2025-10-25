import numpy as np
import torch
import os
import time
import sys

sys.path.append(".../..")
import utils, core

steps = 1000
dt = 0.01

net = utils.networks.DNN(input_size=2, layer_sizes=[128, 128, 128, 128, 128], output_size=1)
net.double()
dir = os.path.dirname(os.path.abspath(__file__))
model_filename = os.path.join(dir, 'nanma.pth')
net.load_state_dict(torch.load(model_filename))

w0 = np.array([69, -105])
start_time = time.time() 
w = w0 / 180
w_saddle, _ = core.hisdnn(model=net, w0=w, v0=None, method='sirqit',
                            dt=dt, ds=dt, max_iter=steps, k=1,
                            report=True, initial_hessian='full_hessian',
                            sub_iter=1, momentum=0.2, step_interval=10)
end_time = time.time()
print(f"Running Time: {end_time - start_time:.4f} s")

w_saddle = w_saddle * 180
print(w_saddle)