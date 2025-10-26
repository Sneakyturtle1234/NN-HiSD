import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from functions import Potential_3d_grad

sys.path.append(".../..")
import utils, core

current_dir = os.path.dirname(os.path.abspath(__file__))

w0 = np.array([6, 1, 6])

k = 0
steps = 10000
dt = 1e-2

grad = lambda w: Potential_3d_grad(w, alpha=6)
for i in range(20):
    addition = np.random.normal(0, 0.05, 3)
    w = w0 + addition
    w_saddle, record = core.hisd(grad=grad, w0=w, v0=None, method='sirqit',
                   dt=dt, ds=dt, max_iter=steps, k=k, report=True, initial_hessian='full_hessian',
                   sub_iter=1, momentum=0.5, step_interval=5000, stopping=True)
    if np.linalg.norm(grad(record[9999])) < 1e-5:
        print(f"[{w_saddle[0]:.2f}, {w_saddle[1]:.2f}, {w_saddle[2]:.2f}]")
