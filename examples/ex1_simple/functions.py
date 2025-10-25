import numpy as np
import torch

def Potential_2d(w):
    M = np.array([[0.8, -0.2], [-0.2, 0.5]])
    part1 = 0.5 * np.sum(w @ M * w, axis=1)
    part2 = np.sum(np.arctan(w - 5), axis=1)
    return (part1 - 5 * part2)[:, np.newaxis]

def Potential_2d_grad(w):
    x = w[0:2]
    alpha = 5
    M = np.array([[0.8, -0.2], [-0.2, 0.5]])
    part1 = np.matmul(M, x)
    part2 = 1 / (1 + (x - 5 * np.ones(x.size)) ** 2)
    return part1 - alpha * part2

def Potential_3d(w):
    x = w[0:3]
    alpha = w[3]
    A = -0.5
    B = 0.2
    M = np.array([[0.8, -0.2, A], [-0.2, 0.5, B], [A, B, 1]])
    part1 = 0.5 * np.matmul(x.T, np.matmul(M, x))
    part2 = np.sum(np.arctan(x - 5 * np.ones(x.size)))
    return part1 - alpha * part2

def Potential_3d_grad(w, alpha=7, A=-0.5, B=0.2):
    x = w[0:3]
    M = np.array([[0.8, -0.2, A], [-0.2, 0.5, B], [A, B, 1]])
    part1 = np.matmul(M, x)
    part2 = 1 / (1 + (x - 5 * np.ones(x.size)) ** 2)
    return part1 - alpha * part2