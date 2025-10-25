import os
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from functions import MB, MB_grad

current_dir = os.path.dirname(os.path.abspath(__file__))

import sys
sys.path.append(".../..")
import utils, core

def create_uniform_dataset(Region, num_points):
    x = np.linspace(Region[0, 0], Region[0, 1], num_points)
    y = np.linspace(Region[1, 0], Region[1, 1], num_points)
    X, Y = np.meshgrid(x, y)
    grid = torch.tensor(np.c_[X.ravel(), Y.ravel()], dtype=torch.float64)
    Z = MB(grid)

    return X, Y, grid, Z


n = 100
Region = np.array([[-1.5, 0.25], [0, 1.95]])

model_path = os.path.join(current_dir, 'parameters/mb.pth')
model_info = torch.load(model_path)
net = utils.networks.DNN(input_size=model_info.get('input_size'), layer_sizes=model_info.get('layer_sizes'), 
                         output_size=model_info.get('output_size')).double()
net.load_state_dict(model_info['state_dict'])
print(f"Loaded pre-trained model with structure: {model_info.get('model_type')}")
print(f"Model configuration - Input size: {model_info.get('input_size')}, "
        f"Layers: {model_info.get('layer_sizes')}, "
        f"Output size: {model_info.get('output_size')}")

X, Y, grid, true_values = create_uniform_dataset(Region, n)
predicted_values = net(grid).detach().numpy()
plt.figure(figsize=(5, 4))
plt.title('True Potential')
plt.contour(X, Y, true_values.reshape(n, n), cmap='viridis', levels=30, alpha=0.5)
plt.colorbar()

grad = lambda w: MB_grad(w, add=False)
w0 = np.array([0.15, 1.5])
dt = 3e-4
k = 1
steps = 500
index = np.arange(0, steps, 1)
start_time = time.time()
saddle_Origin, record_Origin = core.hisd(grad=grad, w0=w0, v0=None, dt=dt, ds=dt,
                                    max_iter=steps, k=k, sub_iter=1,
                                    momentum=0.8, step_interval=100)

end_time = time.time() 
print(f"Running Time: {end_time - start_time:.4f} seconds")
start_time = time.time()

saddle_Origin2, record_Origin2 = core.hisd(grad=grad, w0=w0, v0=None, dt=dt, ds=dt,
                                    max_iter=steps, k=k, sub_iter=1,
                                    momentum=0.0, step_interval=100)

end_time = time.time() 
print(f"Running Time: {end_time - start_time:.4f} seconds")
start_time = time.time()

saddle_OriginNes, record_OriginNes = core.hisd(grad=grad, w0=w0, v0=None, dt=dt, ds=dt,
                                             nesterov=True, max_iter=steps, k=k, sub_iter=1, restart=100,
                                             momentum=0.0, step_interval=100, style=1)

end_time = time.time() 
print(f"Running Time: {end_time - start_time:.4f} seconds")

record_Origin = np.array(record_Origin)
record_Origin2 = np.array(record_Origin2)
record_OriginNes = np.array(record_OriginNes)
plt.plot(record_Origin2[index, 0], record_Origin2[index, 1], 'o-', color='black', label='$\mathrm{HiSD}_\mathrm{Hb}$, $\gamma$=0', markersize=5)
plt.plot(record_Origin[index, 0], record_Origin[index, 1], 'o-', color='green', label='$\mathrm{HiSD}_\mathrm{Hb}$, $\gamma$=0.8', markersize=5)
plt.plot(record_OriginNes[index, 0], record_OriginNes[index, 1], 'o-', color='blue', label='$\mathrm{HiSD}_\mathrm{NA}$', markersize=5)
plt.legend()
print(saddle_OriginNes)
# plt.savefig("MB1.jpg", dpi=300)
plt.show()

plt.figure(figsize=(5, 4))
plt.title('Learned Potential')
plt.contour(X, Y, predicted_values.reshape(n, n), cmap='viridis', levels=30, alpha=0.5)
plt.colorbar()

start_time = time.time()
saddle_NN, record_NN = core.hisdnn(model=net, w0=w0, v0=None, dt=dt, ds=dt,
                              max_iter=steps, k=k, sub_iter=1,
                              momentum=0.8, step_interval=100)

end_time = time.time() 
print(f"Running Time: {end_time - start_time:.4f} seconds")
start_time = time.time()

saddle_NN2, record_NN2 = core.hisdnn(model=net, w0=w0, v0=None, dt=dt, ds=dt,
                              max_iter=steps, k=k, sub_iter=1,
                              momentum=0.0, step_interval=100)

end_time = time.time() 
print(f"Running Time: {end_time - start_time:.4f} seconds")
start_time = time.time()

saddle_NN_Nes, record_NN_Nes = core.hisdnn(model=net, w0=w0, v0=None, dt=dt, ds=dt,
                                       nesterov=True, max_iter=steps, k=k, sub_iter=1, restart=100,
                                       momentum=0.0, step_interval=100)

end_time = time.time() 
print(f"Running Time: {end_time - start_time:.4f} seconds")

record_NN = np.array(record_NN)
record_NN2 = np.array(record_NN2)
record_NN_Nes = np.array(record_NN_Nes)
plt.plot(record_NN2[index, 0], record_NN2[index, 1], 'o-', color='black', label='$\mathrm{NN{-}HiSD}_\mathrm{Hb}$, $\gamma$=0', markersize=5)
plt.plot(record_NN[index, 0], record_NN[index, 1], 'o-', color='green', label='$\mathrm{NN{-}HiSD}_\mathrm{Hb}$, $\gamma$=0.8', markersize=5)
plt.plot(record_NN_Nes[index, 0], record_NN_Nes[index, 1], 'o-', color='blue', label='$\mathrm{NN{-}HiSD}_\mathrm{NA}$', markersize=5)
plt.legend()
print(saddle_NN_Nes)
# plt.savefig("MB2.jpg", dpi=300)
plt.show()


plt.figure(figsize=(5, 4))
plt.title('Potential Function Errors')
error = true_values.reshape(n, n) - predicted_values.reshape(n, n)
plt.imshow(error, extent=[Region[0,0], Region[0,1], Region[1,0], Region[1,1]],
           origin='lower', cmap='RdBu', vmin=-0.1, vmax=0.1, aspect='auto')
plt.colorbar()
# plt.savefig("MB_Error.jpg", dpi=300)
plt.show()