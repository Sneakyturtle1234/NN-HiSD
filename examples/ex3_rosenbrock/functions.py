import numpy as np
import torch

def Potential(x, add=True):
    x = np.array(x, dtype=np.float64)
    s = np.ones(x.shape[1], dtype=np.float64)
    s[:5] = -500
    x_star = np.ones_like(x, dtype=np.float64)
    part1 = np.sum(100.0 * (x[:, 1:] - x[:, :-1] ** 2.0) ** 2.0 + (1 - x[:, :-1]) ** 2.0, axis=1)
    if add:
        part2 = np.sum(s * (np.arctan(x - x_star)) ** 2, axis=1)
    else:
        part2 = 0
    return part1 + part2

def Potential_grad(w=np.array([1,1,1,1,1])):
    w = np.array(w, dtype=np.float64)
    n = len(w)
    output = np.zeros(n, dtype=np.float64)
    output[0] = -400 * w[0] * (w[1] - w[0] ** 2) - 2 * (1 - w[0])
    for i in range(1, n - 1, 1):
        output[i] = 200 * (w[i] - w[i - 1] ** 2) - 400 * w[i] * (w[i + 1] - w[i] ** 2) - 2 * (1 - w[i])
    output[n - 1] = 200 * (w[n - 1] - w[n - 2] ** 2)
    s = np.ones(n, dtype=np.float64)
    s[0:5] = -50000
    addition = 2 * s * np.arctan(w - 1) * 1 / (1 + (w - 1) ** 2)
    output = output + addition
    return output