import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

sys.path.append(".../..")
import utils, core
current_dir = os.path.dirname(os.path.abspath(__file__))

def Potential(x0, add=True):
    x = torch.tensor(x0, dtype=torch.float64)
    s = torch.where(torch.arange(x.size(1)) < 5, torch.tensor(-500), torch.tensor(1))
    x_star = torch.ones_like(x)
    x = torch.tensor(x, dtype=torch.float64, requires_grad=False)
    s = torch.tensor(s, dtype=torch.float64)
    x_star = torch.tensor(x_star, dtype=torch.float64)
    part1 = torch.sum(100.0 * (x[:, 1:] - x[:,:-1] ** 2.0) ** 2.0 + (1 - x[:,:-1]) ** 2.0, dim=1)
    part2 = torch.sum(s * (torch.atan(x - x_star)) ** 2, dim=1) if add else 0
    return (part1 + part2).numpy()

dimension = 7
Region = []
for _ in range(dimension):
    Region.append([0.8, 1.2])
Region = np.array(Region)
train, label = utils.create_random_dataset(Potential, Region=Region, num_points=10000) 
train = np.array(train)
label = np.array(label)

def sample_points_in_sphere(xc, radius, Nsam):
    distances = np.linalg.norm(train - xc, axis=1)
    mask = distances <= radius
    selected_points = train[mask, :]
    selected_values = label[mask]
    num = len(selected_points)

    if len(selected_points) > Nsam:
        indices = np.random.choice(len(selected_points), Nsam, replace=False)
        selected_points = selected_points[indices, :]
        selected_values = selected_values[indices]
        num = Nsam
    
    return selected_points, selected_values, num

def gp_derivative(gp, x):
    if x.ndim == 1:
        x = x.reshape(1, -1)

    X = x[:, np.newaxis, :] - gp.X_train_.reshape(1, -1, gp.X_train_.shape[1])

    c = gp.kernel_.k1.constant_value
    l = gp.kernel_.k2.length_scale
    A = gp.alpha_  # A 的形状为 (n_train,)

    f = np.exp(-np.sum(X**2, axis=-1) / (2 * l**2))
    df = f[..., np.newaxis] * (-X / (l**2))
    
    gradient = c * np.einsum("ijk,j->ik", df, A)
    return gradient.flatten()


kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=0.5, length_scale_bounds=(1e-2, 1e2))
tolN = 100
Nsam = 100
radius = 0.1
Nf = 0
Nm = 1000
Nnew = 100
dt = 1e-4
tolx = 1e-6
toll = 1e-5
tolu = 1e-3
step_interval = 5
xc = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
X, Y, Num = sample_points_in_sphere(xc, radius, Nsam)
while Num < Nsam:
    radius *= 1.2
    X, Y, Num = sample_points_in_sphere(xc, radius, Nsam)

Nf += Nsam
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10).fit(X, Y)

x0 = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]).reshape(1, -1)

grad = lambda x: gp_derivative(gp, x)

xn = xc.copy()
record = [xn]

for j in range(Nm):
    xn_new, record = core.hisd(grad=grad, w0=xn, v0=None, dt=dt, ds=dt,
                            max_iter=1, k=3, sub_iter=1,
                            momentum=0.0, step_interval=10)

    if np.linalg.norm(xn_new - xn, ord=np.inf) <= tolx:
        xn = xn_new
        print("It is converged")
        break

    if np.linalg.norm(xn_new - xc) > radius:
        _, std = gp.predict(xc.reshape(1, -1), return_std=True)
        r = np.max(std)

        if r < toll:
            xc = xn_new
            radius *= 2
        elif r > tolu:
            xn_new = xc
            radius /= 2
        else:
            xc = xn_new

        X_new, Y_new, Num = sample_points_in_sphere(xc, radius, Nnew)

        while Num < Nnew:
            radius *= 1.2
            X_new, Y_new, Num = sample_points_in_sphere(xc, radius, Nnew)

        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10).fit(X_new, Y_new)

    xn = xn_new

    if j % step_interval == 0:
        print('Iteration: ' + str(j) + f'|| Norm of gradient: {np.linalg.norm(grad(xn)):.8f}' + f'  radius: {radius}')
        print(xn)
        if j == 1: record = []
        record.append(xn.reshape(-1, ))

print(xn)