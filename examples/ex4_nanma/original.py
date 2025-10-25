import numpy as np
import torch
import os
import time
import sys
import subprocess

sys.path.append(".../..")
import core

if len(sys.argv) != 2:
    print("Usage: python examples/ex4_nanma/original.py path/to/NAMD/")
    sys.exit(1)

namd_path = sys.argv[1]
current_dir = os.path.dirname(os.path.abspath(__file__))

def grad(w):
    x = w * 180
    with open(os.path.join(current_dir, 'MDparams/hisd/vacuum.in'), 'r') as f:
        lines = f.readlines()
    lines[8] = "  " + lines[8].split()[0] + f' {x[0] - 180}\n'
    lines[9] = "  " + lines[9].split()[0] + f' {x[0] + 180}\n'
    lines[32] = "  " + lines[32].split()[0] + f' {x[1] - 180}\n'
    lines[33] = "  " + lines[33].split()[0] + f' {x[1] + 180}\n'
    with open(os.path.join(current_dir, 'MDparams/hisd/vacuum.in'), 'w') as f:
        f.writelines(lines)

    cmd = f'{namd_path}/namd2 +p32 ./MDparams/hisd/vacuum.conf > ./MDparams/hisd/vacuum.log'
    subprocess.run(cmd, shell=True, check=True, cwd=current_dir)

    output_file = os.path.join(current_dir, 'MDparams/hisd/output/vacuum.pmf')
    data = np.loadtxt(output_file)
    h = 2.5
    f00 = data[(np.abs(data[:,0] - (x[0] - h)) < 1e-2) & (np.abs(data[:,1] - (x[1] - h)) < 1e-2), 2][0]
    f10 = data[(np.abs(data[:,0] - (x[0] + h)) < 1e-2) & (np.abs(data[:,1] - (x[1] - h)) < 1e-2), 2][0]
    f01 = data[(np.abs(data[:,0] - (x[0] - h)) < 1e-2) & (np.abs(data[:,1] - (x[1] + h)) < 1e-2), 2][0]
    f11 = data[(np.abs(data[:,0] - (x[0] + h)) < 1e-2) & (np.abs(data[:,1] - (x[1] + h)) < 1e-2), 2][0]

    h0 = h / 180
    dfdx = ((f10 + f11) - (f00 + f01)) / (4*h0)
    dfdy = ((f01 + f11) - (f00 + f10)) / (4*h0)
    return np.array([dfdx, dfdy])


k = 1
steps = 100
dt = 1e-2
w = np.array([-80, 75])

start_time = time.time()

w = w / 180
w_saddle, _ = core.hisd(grad=grad, w0=w, v0=None, method='sirqit',
                        dt=dt, ds=dt, max_iter=steps, k=k,
                        report=True, initial_hessian='full_hessian',
                        sub_iter=1, momentum=0, step_interval=1)
end_time = time.time()
print(f"Running Time: {end_time - start_time:.4f} s")
print(w_saddle * 180)