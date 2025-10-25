import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(".../..")
import utils
from functions import Potential

epochs = 30000

grad_diff_history = np.load("grad_diff_history.npy").tolist() if os.path.exists("grad_diff_history.npy") else []
hessian_diff_history = np.load("hessian_diff_history.npy").tolist() if os.path.exists("hessian_diff_history.npy") else []

font = 22
plt.figure(figsize=(10, 6))
plt.semilogy(range(0, epochs, 200), grad_diff_history, label="Max Gradient Differences", color="blue", marker="o", markersize=3)
plt.semilogy(range(0, epochs, 200), hessian_diff_history, label="Max Hessian Differences", color="red", marker="x", markersize=3)
plt.xlabel("Epochs", fontsize=font)
plt.ylabel("L2 Norm", fontsize=font)
plt.title("Gradient & Hessian Error vs. Epochs", fontsize=font)
plt.legend(fontsize=font)
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.savefig("grad_hessian_diff_plot.png", dpi=300)
plt.show()