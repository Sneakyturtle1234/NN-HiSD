import os
import torch
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append(".../..")
import utils, core

layer = [128, 128, 128]

w0 = np.array([0.15, 1.5])
steps = 500
k = 1
dt = 3e-4

Saddle_error = []

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'parameters/mb.pth')
model_info = torch.load(model_path)
net = utils.networks.DNN(input_size=model_info.get('input_size'), layer_sizes=model_info.get('layer_sizes'), 
                         output_size=model_info.get('output_size')).double()
net.load_state_dict(model_info['state_dict'])
print(f"Loaded pre-trained model with structure: {model_info.get('model_type')}")
print(f"Model configuration - Input size: {model_info.get('input_size')}, "
        f"Layers: {model_info.get('layer_sizes')}, "
        f"Output size: {model_info.get('output_size')}")

saddle, record = core.hisdnn(model=net, w0=w0, v0=None, dt=dt, ds=dt,
                         max_iter=steps, k=k, sub_iter=1,
                         momentum=0.0, step_interval=100)
saddle0, record0 = core.hisdnn(model=net, w0=w0, v0=None, dt=dt, ds=dt,
                         nesterov=True, max_iter=steps, k=k, sub_iter=1, restart=5,
                         momentum=0.0, step_interval=100)
saddle1, record1 = core.hisdnn(model=net, w0=w0, v0=None, dt=dt, ds=dt,
                         nesterov=True, max_iter=steps, k=k, sub_iter=1, method='lobpcg', restart=5,
                         momentum=0.0, step_interval=100)
saddle2, record2 = core.hisdnn(model=net, w0=w0, v0=None, dt=dt, ds=dt,
                         max_iter=steps, k=k, sub_iter=1,
                         momentum=0.8, step_interval=100)
saddle4, record4 = core.hisdnn(model=net, w0=w0, v0=None, dt=dt, ds=dt,
                         max_iter=steps, k=k, sub_iter=1,
                         momentum=0.6, step_interval=100)
saddle_point = saddle0

dist = np.linalg.norm(np.array(record) - np.array(saddle_point), axis=1)
dist2 = np.linalg.norm(np.array(record2) - np.array(saddle_point), axis=1)
dist0 = np.linalg.norm(np.array(record0) - np.array(saddle_point), axis=1)
dist1 = np.linalg.norm(np.array(record1) - np.array(saddle_point), axis=1)
dist4 = np.linalg.norm(np.array(record4) - np.array(saddle_point), axis=1)

plt.semilogy(dist, label='$\mathrm{NN{-}HiSD}$', color='brown', marker='s', markersize=6, markevery=20)
plt.semilogy(dist4, label='$\mathrm{NN{-}HiSD}_\mathrm{Hb}$, $\gamma$=0.6', color='gray')
plt.semilogy(dist2, label='$\mathrm{NN{-}HiSD}_\mathrm{Hb}$, $\gamma$=0.8', color='blue')
plt.semilogy(dist1, label='$\mathrm{NN{-}HiSD}_\mathrm{NA}$, LOBPCG', color='green', marker='o', markersize=6, markevery=20)
plt.semilogy(dist0, label='$\mathrm{NN{-}HiSD}_\mathrm{NA}$, SIRQIT', color='red', marker='o', markersize=6, markevery=20)
plt.legend(fontsize=15)
plt.title('MB',fontsize=20)
plt.xlabel('Iteration', fontsize=15)
plt.ylabel('$||x^{(n)}-x_{NN}^*||_2$', fontsize=13, labelpad=-3)
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=13)
# plt.savefig('MB.png', dpi=300)
plt.show()