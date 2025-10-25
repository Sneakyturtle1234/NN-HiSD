import scipy.interpolate
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.interpolate

current_dir = os.path.dirname(os.path.abspath(__file__))
data = np.load(os.path.join(current_dir, 'tsne_s28.npy'))
value = np.loadtxt(os.path.join(current_dir, 'num.txt'))
value = 13.75 - np.log(value)


grid_x, grid_y = np.mgrid[min(data[:,0]):max(data[:,0]):1000j, min(data[:,1]):max(data[:,1]):1000j]
grid_z = scipy.interpolate.griddata(data, value, (grid_x, grid_y), method='linear')
im=plt.imshow(grid_z.T, extent=(min(data[:,0]), max(data[:,0]), min(data[:,1]), max(data[:,1])), origin='lower', cmap='Spectral_r', interpolation='nearest')
plt.scatter(data[:,0], data[:,1], s=5)

plt.xlabel('t-SNE 1', fontsize=15)
plt.ylabel('t-SNE 2', labelpad=-15, fontsize=15)
plt.xlim([-250, 250])
plt.ylim([-250, 250])

points = [
    (-200, -3, 'A'),
    (-27, -94, 'B'),
    (-31, -16, 'C'),
    (52, 48, 'D'),
    (-10, 21, 'E'),
    (41, -155, 'L3'),
    (71, -50, 'L4'),
    (90, 100, 'L5'),
    (-35, 90, 'L6'),
]

for x, y, label in points:
    plt.scatter(x, y, color='brown', s=20, marker='*')
    plt.text(x, y, f' {label}', fontsize=12, ha='left', va='center')

cbar = plt.colorbar(im).ax.tick_params(labelsize=13)
plt.text(1.12, 1.0, '($k_B$T)', transform=plt.gca().transAxes, ha='left', va='bottom',fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)
# plt.savefig('Picture.png', dpi=300)
plt.show()