import os
import sys
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import subprocess
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

sys.path.append(".../..")
import utils, core
current_dir = os.path.dirname(os.path.abspath(__file__))

def dataset(w, namd_path):
    x = w * 180
    x = (x / 5.0) % 1 * 5
    with open(os.path.join(current_dir, 'MDparams/gprhisd/vacuum.in'), 'r') as f:
        lines = f.readlines()
    lines[8] = "  " + lines[8].split()[0] + f' {x[0] - 180}\n'
    lines[9] = "  " + lines[9].split()[0] + f' {x[0] + 180}\n'
    lines[32] = "  " + lines[32].split()[0] + f' {x[1] - 180}\n'
    lines[33] = "  " + lines[33].split()[0] + f' {x[1] + 180}\n'
    with open(os.path.join(current_dir, 'MDparams/gprhisd/vacuum.in'), 'w') as f:
        f.writelines(lines)

    cmd = f'{namd_path}/namd2 +p32 ./MDparams/gprhisd/vacuum.conf > ./MDparams/gprhisd/vacuum.log'
    subprocess.run(cmd, shell=True, check=True, cwd=current_dir)

    output_file = os.path.join(current_dir, 'MDparams/gprhisd/output/vacuum.pmf')
    data = np.loadtxt(output_file)
    X = np.array(data[:, 0:2]) / 180
    Y = np.array(data[:, 2])

    return X, Y

dimension = 2
Region = []
xc = np.array([80, -70])
xc = xc / 180
for _ in range(dimension):
    Region.append([-1, 7])
Region = np.array(Region)

if len(sys.argv) != 2:
    print("Usage: python examples/ex4_nanma/gpr.py path/to/NAMD/")
    sys.exit(1)
namd_path = sys.argv[1]
train, label = dataset(xc, namd_path)
train = np.array(train)
label = np.array(label).reshape(-1)

def sample_points_in_sphere(xc, radius, Nsam):
    distances = np.linalg.norm(train - xc, axis=1)
    mask = distances <= radius
    selected_points = train[mask]
    selected_values = label[mask]
    num = len(selected_points)

    if len(selected_points) > Nsam:
        indices = np.random.choice(len(selected_points), Nsam, replace=False)
        selected_points = selected_points[indices]
        selected_values = selected_values[indices]
        num = Nsam
    
    return selected_points, selected_values, num

def gp_derivative(gp, x):
    if x.ndim == 1:
        x = x.reshape(1, -1)

    X = x[:, np.newaxis, :] - gp.X_train_.reshape(1, -1, gp.X_train_.shape[1])

    c = gp.kernel_.k1.constant_value
    l = gp.kernel_.k2.length_scale
    A = gp.alpha_ 

    f = np.exp(-np.sum(X**2, axis=-1) / (2 * l**2))
    df = f[..., np.newaxis] * (-X / (l**2))
    
    gradient = c * np.einsum("ijk,j->ik", df, A)
    return gradient.flatten()


kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=0.5, length_scale_bounds=(1e-2, 1e2))
Nsam = 20
radius = 1
Nf = 0
Nm = 10000
Nnew = 20
dt = 1e-2
tolx = 1e-10
toll = 1e-5
tolu = 1e-3
step_interval = 1
momentum = 0.2
X, Y, Num = sample_points_in_sphere(xc, radius, Nsam)
while Num < Nsam:
    radius *= 1.2
    X, Y, Num = sample_points_in_sphere(xc, radius, Nsam)

Nf += Nsam
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10).fit(X, Y)

x0 = np.array(xc).reshape(1, -1)


grad = lambda x: gp_derivative(gp, x)

xn = xc.copy()
record = [xn]

x_pre = xn
start_time = time.time()

for j in range(Nm):
    xn_new, record = core.hisd(grad=grad, w0=xn, v0=None, dt=dt, ds=dt,
                            max_iter=1, k=1, sub_iter=1,
                            momentum=0.0, step_interval=10, w_pre=x_pre)
    
    x_pre = xn

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
            x_pre = xc
            radius /= 2
        else:
            xc = xn_new

        train, label = dataset(xc, namd_path)
        X_new, Y_new, Num = sample_points_in_sphere(xc, radius, Nnew)

        while Num < Nnew:
            radius *= 1.2
            X_new, Y_new, Num = sample_points_in_sphere(xc, radius, Nnew)

        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10).fit(X_new, Y_new)

    xn = xn_new

    if j % step_interval == 0:
        print('Iteration: ' + str(j) + f'|| Norm of gradient: {np.linalg.norm(grad(xn)):.8f}' + f'|| radius: {radius}')
        print(xn * 180)
        if j == 1: record = []
        record.append(xn.reshape(-1, ))

end_time = time.time()
print(f"Total time: {end_time - start_time:.2f} seconds")

print(xn * 180)