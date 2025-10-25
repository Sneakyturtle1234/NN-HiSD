import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import sys

sys.path.append(".../..")
import utils, core

n = 720
Region = np.array([[-180, 180], [-180, 180]])
x = np.linspace(Region[0, 0], Region[0, 1], n)
y = np.linspace(Region[1, 0], Region[1, 1], n)
X, Y = np.meshgrid(x, y)
grid = torch.tensor(np.c_[X.ravel(), Y.ravel()], dtype=torch.float64)
net = utils.networks.DNN(input_size=2, layer_sizes=[128, 128, 128, 128, 128], output_size=1)
net.double()
model_filename = 'nanma.pth'
dir = os.path.dirname(os.path.abspath(__file__))
net.load_state_dict(torch.load(os.path.join(dir, model_filename)))
Z = net(grid/180).detach().numpy()

set1 = [[11.3, 37.7], [14.0, 160.4], [125.2, 44.5]]
set2 = [[17.6, 53.5], [-1.1, -68.2], [73.8, 106.2], [-103.2, -67.7], [127.4, -118.7], [-98.1, 135.0]]
set3 = [[68.7, -69.7], [-83.2, 74.0], [-146.4, 168.4]]
for i in range(len(set1)):
    plt.plot(set1[i][0], set1[i][1], 'o', color="#66b3e7", markersize=5)
    plt.text(set1[i][0], set1[i][1], f'M{i+1}', fontsize=12, verticalalignment='bottom', horizontalalignment='right')
for i in range(len(set2)):
    plt.plot(set2[i][0], set2[i][1], 'o', color="#fde397", markersize=5)
    plt.text(set2[i][0], set2[i][1], f'S{i+1}', fontsize=12)
for i in range(len(set3)):
    plt.plot(set3[i][0], set3[i][1], 'o', color="#d9958f", markersize=5)
    plt.text(set3[i][0], set3[i][1], f'Z{i+1}', fontsize=12, verticalalignment='top', horizontalalignment='right')

plt.contour(X, Y, Z.reshape(n, n), cmap='viridis', levels=25, alpha=0.6, linestyles='dashed')
plt.xlabel('Angle 1 ($\phi$)', fontsize=15, labelpad=0)
plt.ylabel('Angle 2 ($\psi$)', fontsize=15, labelpad=0)
plt.tick_params(axis='both', which='major', labelsize=12)
# plt.savefig("Potential.png", dpi=300)
plt.show()