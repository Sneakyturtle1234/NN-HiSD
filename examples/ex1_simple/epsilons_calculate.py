import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(".../..")
import utils, core

def Potential_2d(w):
    M = torch.tensor([[0.8, -0.2], [-0.2, 0.5]], dtype=w.dtype, device=w.device)
    part1 = 0.5 * torch.sum(w @ M * w, dim=1)
    part2 = torch.sum(torch.atan(w - 5), dim=1)
    return (part1 - 5 * part2).unsqueeze(1)

def calculate_derivatives(net, Potential, num_points=1000):
    center = np.array([1.2842, 3.4484])
    radius = 0.5

    angles = np.random.uniform(0, 2 * np.pi, num_points)
    radii = np.sqrt(np.random.uniform(0, 1, num_points)) * radius
    x_samples = center[0] + radii * np.cos(angles)
    y_samples = center[1] + radii * np.sin(angles)
    points = np.column_stack((x_samples, y_samples))
    points = torch.tensor(points, dtype=torch.float64, requires_grad=True)

    net_output = net(points)
    potential_output = Potential(points)

    grad_net = torch.autograd.grad(net_output, points, grad_outputs=torch.ones_like(net_output), create_graph=True)[0]
    grad_potential = torch.autograd.grad(potential_output, points, grad_outputs=torch.ones_like(potential_output), create_graph=True)[0]
    grad_diff = grad_net - grad_potential
    max_grad_diff_norm = torch.max(torch.norm(grad_diff, p=2, dim=1)).item()

    hessian_net = []
    hessian_potential = []
    for i in range(dimension):
        second_grad_net = torch.autograd.grad(grad_net[:, i], points, grad_outputs=torch.ones_like(grad_net[:, i]), create_graph=True)[0]
        second_grad_potential = torch.autograd.grad(grad_potential[:, i], points, grad_outputs=torch.ones_like(grad_potential[:, i]), create_graph=True)[0]
        hessian_net.append(second_grad_net)
        hessian_potential.append(second_grad_potential)
    hessian_net = torch.stack(hessian_net, dim=1)
    hessian_potential = torch.stack(hessian_potential, dim=1)
    hessian_diff = hessian_net - hessian_potential
    max_hessian_diff_norm = torch.max(torch.norm(hessian_diff, p=2, dim=(1, 2))).item()

    lipschitz_constant = 0
    for i in range(num_points):
        for j in range(i + 1, num_points):
            hessian_diff_norm = torch.norm(hessian_diff[i] - hessian_diff[j], p=2)
            points_diff_norm = torch.norm(points[i] - points[j], p=2)
            if points_diff_norm > 0:
                lipschitz_constant = max(lipschitz_constant, (hessian_diff_norm / points_diff_norm).item())

    return {
        "grad_diff": max_grad_diff_norm,
        "hessian_diff": max_hessian_diff_norm,
        "max_third_grad_norm": lipschitz_constant,
    }

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'parameters/simple2d.pth')
    model_info = torch.load(model_path)
    net = utils.networks.DNN(input_size=model_info.get('input_size'), layer_sizes=model_info.get('layer_sizes'), 
                            output_size=model_info.get('output_size')).double()
    net.load_state_dict(model_info['state_dict'])
    print(f"Loaded pre-trained model with structure: {model_info.get('model_type')}")
    print(f"Model configuration - Input size: {model_info.get('input_size')}, "
        f"Layers: {model_info.get('layer_sizes')}, "
        f"Output size: {model_info.get('output_size')}")

    font = 22
    epochs = 30000
    grad_diff_history = np.load(os.path.join(current_dir, 'parameters/grad_diff_history.npy'))
    hessian_diff_history = np.load( os.path.join(current_dir, 'parameters/hessian_diff_history.npy'))
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

    dimension = 2
    result = calculate_derivatives(net, Potential_2d, num_points=500)

    print("order-1 derivative difference:\n", result["grad_diff"])
    print("order-2 derivative difference:\n", result["hessian_diff"])
    print("order-2 derivative's Lipschitz constant:\n", result["max_third_grad_norm"])

