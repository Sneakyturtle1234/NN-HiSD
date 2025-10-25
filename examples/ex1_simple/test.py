import matplotlib.pyplot as plt



font = 22
plt.figure(figsize=(10, 6))
plt.semilogy(range(0, epochs, 200), grad_diff_history, label="Max Gradient Differences", color="blue", marker="o", markersize=3)
plt.semilogy(range(0, epochs, 200), hessian_diff_history, label="Max Hessian Differences", color="red", marker="x", markersize=3)
plt.xlabel("Epochs", fontsize=font)
plt.ylabel("L2 Norm", fontsize=font)
plt.title("Gradient & Hessian Error vs. Epochs", fontsize=font)
plt.legend(fontsize=font)
plt.grid(True)
# plt.savefig("grad_hessian_diff_plot.png", dpi=300)
plt.show()