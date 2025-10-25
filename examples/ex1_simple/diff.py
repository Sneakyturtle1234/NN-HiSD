import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(".../..")
import utils, core
from functions import Potential_2d

layer = [128, 128, 128]

fontsize = 25
alpha = 5
w0 = np.array([0.7, 0.7])
dt = 5e-2
k = 1
steps = 1000
plt.figure(figsize=(8, 6))
plt.title('Dimer method', fontsize=fontsize)

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'parameters/simple2d.pth')
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
                             momentum=0.0, step_interval=100, ADAD=False)
saddle0, record0 = core.hisdnn(model=net, w0=w0, v0=None, dt=dt, ds=dt,
                               max_iter=steps, k=k, sub_iter=1,
                         momentum=0.7, step_interval=100, ADAD=False)
saddle1, record1 = core.hisdnn(model=net, w0=w0, v0=None, dt=dt, ds=dt,
                               max_iter=steps, k=k, sub_iter=1,
                               momentum=0.8, step_interval=100, ADAD=False)
saddle2, record2 = core.hisdnn(model=net, w0=w0, v0=None, dt=dt, ds=dt,
                               nesterov=True, max_iter=steps, k=k, sub_iter=1, restart=20,
                               momentum=0.0, step_interval=100, ADAD=False)
saddle_point = saddle2
print(saddle_point)

dist = np.linalg.norm(np.array(record) - np.array(saddle_point), axis=1)
dist0 = np.linalg.norm(np.array(record0) - np.array(saddle_point), axis=1)
dist1 = np.linalg.norm(np.array(record1) - np.array(saddle_point), axis=1)
dist2 = np.linalg.norm(np.array(record2) - np.array(saddle_point), axis=1)

plt.semilogy(dist, label='$\mathrm{NN{-}HiSD}$', color='brown', marker='s', markevery=40)
plt.semilogy(dist0, label='$\mathrm{NN{-}HiSD}_\mathrm{Hb}$, $\gamma$=0.7', color='red')
plt.semilogy(dist1, label='$\mathrm{NN{-}HiSD}_\mathrm{Hb}$, $\gamma$=0.8', color='grey')
plt.semilogy(dist2, label='$\mathrm{NN{-}HiSD}_\mathrm{NA}$', color='blue', marker='o', markevery=20)
plt.legend(fontsize=15)
plt.xlabel('Iteration', fontsize=fontsize)
plt.ylabel('$||x^{(n)}-x_{NN}^*||_2$', fontsize=20, labelpad=-5)
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=15)
# plt.savefig('Dimer.png', dpi=300)
plt.show()


plt.figure(figsize=(8, 6))
plt.title('ADAD', fontsize=fontsize)

saddle, record = core.hisdnn(model=net, w0=w0, v0=None, dt=dt, ds=dt,
                         max_iter=steps, k=k, sub_iter=1,
                         momentum=0.0, step_interval=100, ADAD=True)
saddle0, record0 = core.hisdnn(model=net, w0=w0, v0=None, dt=dt, ds=dt,
                         max_iter=steps, k=k, sub_iter=1,
                         momentum=0.7, step_interval=100, ADAD=True)
saddle1, record1 = core.hisdnn(model=net, w0=w0, v0=None, dt=dt, ds=dt,
                         max_iter=steps, k=k, sub_iter=1,
                         momentum=0.8, step_interval=100, ADAD=True)
saddle2, record2 = core.hisdnn(model=net, w0=w0, v0=None, dt=dt, ds=dt,
                         nesterov=True, max_iter=steps, k=k, sub_iter=1, restart=20,
                         momentum=0.0, step_interval=100, ADAD=True)
saddle_point = saddle2
print(saddle_point)

dist = np.linalg.norm(np.array(record) - np.array(saddle_point), axis=1)
dist0 = np.linalg.norm(np.array(record0) - np.array(saddle_point), axis=1)
dist1 = np.linalg.norm(np.array(record1) - np.array(saddle_point), axis=1)
dist2 = np.linalg.norm(np.array(record2) - np.array(saddle_point), axis=1)

plt.semilogy(dist, label='$\mathrm{NN{-}HiSD}$', color='brown', marker='s', markevery=40)
plt.semilogy(dist0, label='$\mathrm{NN{-}HiSD}_\mathrm{Hb}$, $\gamma$=0.7', color='red')
plt.semilogy(dist1, label='$\mathrm{NN{-}HiSD}_\mathrm{Hb}$, $\gamma$=0.8', color='gray')
plt.semilogy(dist2, label='$\mathrm{NN{-}HiSD}_\mathrm{NA}$', color='blue', marker='o', markevery=20)
plt.legend(fontsize=15)
plt.xlabel('Iteration', fontsize=fontsize)
plt.ylabel('$||x^{(n)}-x_{NN}^*||_2$', fontsize=20, labelpad=-5)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.grid(True)
# plt.savefig('ADAD.png', dpi=300)
plt.show()