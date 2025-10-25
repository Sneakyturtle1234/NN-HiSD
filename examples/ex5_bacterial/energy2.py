import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
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

def make_grid(Region, lines):
    x = np.linspace(Region[0, 0], Region[0, 1], lines)
    y = np.linspace(Region[1, 0], Region[1, 1], lines)
    grid = np.meshgrid(x, y)
    grid = np.array(grid)
    grid = torch.tensor(grid, dtype=torch.float64)

    return x, y, grid

Region = np.array([[-250, 250], [-250, 250]])
[in_dim, out_dim] = [2, 1]
layer = [128, 128, 128]
x, y, grid = make_grid(Region, 80)
model = DNN(in_dim, layer, out_dim)
model_filename = 'bacterial.pth'
dir = os.path.dirname(os.path.abspath(__file__))
model.load_state_dict(torch.load(os.path.join(dir, model_filename)))
model.double()

values = model((grid.view(2, -1).t() - model.mean)/model.std).detach().numpy().reshape(80, 80) + 0.05
contour = plt.contourf(x, y, values, levels=40, cmap='Spectral_r')
cbar = plt.colorbar(contour, ticks=np.arange(3.0, 7.5, 0.5)).ax.tick_params(labelsize=15)

set1 = [[22, -46], [-71, -49], [-82, -132], [32, -149]]
set2 = [[-21, 51], [123, 53], [28, 43], [-154, -50], [-21, -48], [-89, -83], [-23, -165], [69, -105]]
set3 = [[-28, 85], [-10, 14], [50, 53], [-32, -88]]
for i in range(len(set1)):
    plt.plot(set1[i][0], set1[i][1], 'o', color="#93f0f0", markersize=5)
    plt.text(set1[i][0], set1[i][1], f'M{i+1}', fontsize=12, verticalalignment='bottom', horizontalalignment='right')
for i in range(len(set2)):
    plt.plot(set2[i][0], set2[i][1], 'o', color="#92d050", markersize=5)
    plt.text(set2[i][0], set2[i][1], f'S{i+1}', fontsize=12, verticalalignment='bottom', horizontalalignment='right')
for i in range(len(set3)):
    plt.plot(set3[i][0], set3[i][1], 'o', color="#fee599", markersize=5)
    plt.text(set3[i][0], set3[i][1], f'Z{i+1}', fontsize=12, verticalalignment='bottom', horizontalalignment='left')

plt.xlabel('t-SNE 1', fontsize=15)
plt.ylabel('t-SNE 2', fontsize=15, labelpad=-15)
plt.text(1.12, 1.0, '($k_B$T)', transform=plt.gca().transAxes, ha='left', va='bottom', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)
# plt.savefig('saddle.png', dpi=300)
plt.show()
