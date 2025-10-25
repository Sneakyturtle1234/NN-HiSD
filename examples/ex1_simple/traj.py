import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from functions import Potential_2d
from functions import Potential_2d_grad

sys.path.append(".../..")
import utils, core

current_dir = os.path.dirname(os.path.abspath(__file__))

fontsize = 13
n = 100
Region = np.array([[-1, 7], [-1, 7]])

model_path = os.path.join(current_dir, 'parameters/simple2d.pth')
model_info = torch.load(model_path)
net = utils.networks.DNN(input_size=model_info.get('input_size'), layer_sizes=model_info.get('layer_sizes'), 
                         output_size=model_info.get('output_size')).double()
net.load_state_dict(model_info['state_dict'])
print(f"Loaded pre-trained model with structure: {model_info.get('model_type')}")
print(f"Model configuration - Input size: {model_info.get('input_size')}, "
        f"Layers: {model_info.get('layer_sizes')}, "
        f"Output size: {model_info.get('output_size')}")

X, Y, grid, true_values = utils.create_regular_dataset(Potential_2d, Region, n)
grid = torch.tensor(grid, dtype=torch.float64)
predicted_values = net(grid).detach().numpy()

plt.figure(figsize=(5, 3))
plt.title('Exact Potential', fontsize=fontsize+5)
plt.contour(X, Y, true_values.reshape(n, n), cmap='viridis', levels=30)
plt.colorbar().ax.tick_params(labelsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)

grad = lambda w: Potential_2d_grad(w)
w0 = np.array([0.7, 0.7])
dt = 5e-2
k = 1
steps = 1000
index = np.arange(0, steps, 10)
start_time = time.time()
saddle_Origin, record_Origin = core.hisd(grad=grad, w0=w0, v0=None, dt=dt, ds=dt,
                                         max_iter=steps, k=k, sub_iter=1,
                                         momentum=0.8, step_interval=100)

end_time = time.time()
print(f"Running Time: {end_time - start_time:.4f} seconds")


start_time = time.time()
saddle_OriginNes, record_OriginNes = core.hisd(grad=grad, w0=w0, v0=None, dt=dt, ds=dt,
                                               nesterov=True, restart=20, max_iter=steps, k=k, sub_iter=1,
                                               momentum=0.0, step_interval=100, style=1)

end_time = time.time()
print(f"Running Time: {end_time - start_time:.4f} seconds")

record_Origin = np.array(record_Origin)
record_OriginNes = np.array(record_OriginNes)
plt.plot(record_Origin[index, 0], record_Origin[index, 1], 'o-', color='green', markersize=3, label='$\mathrm{HiSD}_\mathrm{Hb}$')
plt.plot(record_OriginNes[index, 0], record_OriginNes[index, 1], '*-', markersize=5, color='blue', label='$\mathrm{HiSD}_\mathrm{NA}$')
plt.plot(w0[0], w0[1], 'o', color='brown', markersize=10, label='Initial point')
plt.plot(saddle_OriginNes[0], saddle_OriginNes[1], '*', color='red', markersize=10, label='Saddle point')
plt.legend(fontsize=fontsize)
print(saddle_OriginNes)
# plt.savefig("Simple1.png", dpi=300)

plt.show()

plt.figure(figsize=(5, 3))
plt.title('Surrogate Potential', fontsize=fontsize+5)
plt.contour(X, Y, predicted_values.reshape(n, n), cmap='viridis', levels=30)

start_time = time.time()
saddle_NN, record_NN = core.hisdnn(model=net, w0=w0, v0=None, dt=dt, ds=dt,
                                     max_iter=steps, k=k, sub_iter=1,
                                     momentum=0.8, step_interval=100)

end_time = time.time()
print(f"Running Time: {end_time - start_time:.4f} seconds")

start_time = time.time()

saddle_NN_Nes, record_NN_Nes = core.hisdnn(model=net, w0=w0, v0=None, dt=dt, ds=dt,
                                nesterov=True, restart=20, max_iter=steps, k=k, sub_iter=1,
                                momentum=0.0, step_interval=100)

end_time = time.time()
print(f"Running Time: {end_time - start_time:.4f} seconds")

record_NN = np.array(record_NN)
record_NN_Nes = np.array(record_NN_Nes)
plt.plot(record_NN[index, 0], record_NN[index, 1], 'o-', markersize=3, color='green', label='$\mathrm{NN{-}HiSD}_\mathrm{Hb}$')
plt.plot(record_NN_Nes[index, 0], record_NN_Nes[index, 1], '*-', markersize=5, color='blue', label='$\mathrm{NN{-}HiSD}_\mathrm{NA}$')
plt.plot(w0[0], w0[1], 'o', color='brown', markersize=10, label='Initial point')
plt.plot(saddle_NN_Nes[0], saddle_NN_Nes[1], '*', color='red', markersize=10, label='Saddle point')
plt.legend(fontsize=fontsize)
plt.colorbar().ax.tick_params(labelsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)
print(saddle_NN_Nes)

# plt.savefig("Simple2.png", dpi=300)
plt.show()

plt.figure(figsize=(5, 3))
plt.title('Error Function', fontsize=fontsize+5)
error = true_values.reshape(n, n) - predicted_values.reshape(n, n)
plt.imshow(error, extent=[Region[0,0], Region[0,1], Region[1,0], Region[1,1]],
           origin='lower', cmap='RdBu', vmin=-0.004, vmax=0.004, aspect='auto')
plt.colorbar().ax.tick_params(labelsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)
# plt.savefig("Simple_Error.png", dpi=300)
plt.show()