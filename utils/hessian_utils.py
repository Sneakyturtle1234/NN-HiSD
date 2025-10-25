import numpy as np
import scipy.sparse.linalg as linalg
import copy
import torch

__all__ = ['given_eigenvector', 'sirqit', 'lobpcg', 'atiken', 'calculate_derivatives']

def single_hessian_vector_product(grad, w, v, L=1e-3):
    '''
    Compute the Jacobi-Vector Product by central finite difference method.
    Mathematical theory sketch:
       \nabla F(x)v \approx [F(x+lv) - F(x-lv)]/(2*L)
    Here L is the Dimer length, which helps to approximate the Jacobi-Vector Product.
    One can adjsut L to obtain the desired accuracy.
    '''
    return (grad(w + L * v) - grad(w - L * v)) / (2.0 * L)


def batch_hessian_vector_product(grad, w, v, L=1e-3):
    '''
    Compute the Jacobi-Vector Product in the batch, i.e.
    this function calculate
       \nabla F(x)[v_1,...v_k] = [\nabla F(x)v_1,...,\nabla F(x)v_k]
    '''
    hvp = lambda x: single_hessian_vector_product(grad, w, x, L)
    hvpmat = map(hvp, list(np.transpose(v)))
    result = []
    for item in hvpmat:
        result.append(item)
    return np.transpose(np.stack(result))


def hessian(grad, w, L=1e-3):
    '''
    Compute the Jacobi matrix.
    '''
    dimension = w.shape[0]
    return batch_hessian_vector_product(grad, w, np.eye(dimension), L)


###############################################################################################
def given_eigenvector(grad, w, L, D, k):
    H = hessian(grad, w, L)
    eigenvalue, eigenvector = np.linalg.eig((H + np.transpose(H)) / 2.0)
    order = np.argsort(eigenvalue)
    eigenvector = eigenvector[:, order]
    return eigenvector[:, 0:k]


def sirqit(grad, w, v, maxiter=10, ds=1e-7, L=1e-3):
    for _ in range(maxiter):
        hv = batch_hessian_vector_product(grad, w, v, L)
        coef = np.triu(v.T @ hv)
        v -= (hv - v * np.diag(coef) - 2.0 * v @ np.triu(coef, k=1)) * ds
        v, _ = np.linalg.qr(v)
    return v


def lobpcg(grad, w, v0, max_iter=10, L=1e-3):
    n = v0.shape[0]
    matvec_fun = lambda z: single_hessian_vector_product(grad, w, z, L)
    matmat_fun = lambda z: batch_hessian_vector_product(grad, w, z, L)
    H = linalg.LinearOperator(shape=(n, n),
                              matvec=matvec_fun,
                              matmat=matmat_fun)
    values, vectors = linalg.lobpcg(A=H, X=v0, maxiter=max_iter, largest=False)
    return values, vectors


def atiken(w_list):
    n = len(w_list)
    w_new = []
    for i in range(n - 2):
        delta2_w = w_list[i] - 2.0 * w_list[i + 1] + w_list[i + 2]
        t = w_list[i] - delta2_w / (delta2_w ** 2).sum() * ((w_list[i + 1] - w_list[i]) ** 2).sum()
        w_new.append(t)
    return w_new

def calculate_derivatives(net):
    center = np.array([1.2842, 3.4484])
    radius = 0.2

    np.random.seed(43)
    num_points = 500
    angles = np.random.uniform(0, 2 * np.pi, num_points)
    radii = np.sqrt(np.random.uniform(0, 1, num_points)) * radius
    x_samples = center[0] + radii * np.cos(angles)
    y_samples = center[1] + radii * np.sin(angles)
    points = np.column_stack((x_samples, y_samples))
    points = torch.tensor(points, dtype=torch.float64, requires_grad=True)

    def Potential(w):
        M = torch.tensor([[0.8, -0.2], [-0.2, 0.5]], dtype=w.dtype, device=w.device)
        part1 = 0.5 * torch.sum(w @ M * w, dim=1)
        part2 = torch.sum(torch.atan(w - 5), dim=1)
        return (part1 - 5 * part2).unsqueeze(1)

    net_output = net(points)
    potential_output = Potential(points)
 
    grad_net = torch.autograd.grad(net_output, points, grad_outputs=torch.ones_like(net_output), create_graph=True)[0]
    grad_potential = torch.autograd.grad(potential_output, points, grad_outputs=torch.ones_like(potential_output), create_graph=True)[0]
    grad_diff = grad_net - grad_potential
    max_grad_diff_norm = torch.max(torch.norm(grad_diff, p=2, dim=1)).item()

    hessian_net = []
    hessian_potential = []
    for i in range(2):
        second_grad_net = torch.autograd.grad(grad_net[:, i], points, grad_outputs=torch.ones_like(grad_net[:, i]), create_graph=True)[0]
        second_grad_potential = torch.autograd.grad(grad_potential[:, i], points, grad_outputs=torch.ones_like(grad_potential[:, i]), create_graph=True)[0]
        hessian_net.append(second_grad_net)
        hessian_potential.append(second_grad_potential)
    hessian_net = torch.stack(hessian_net, dim=1)
    hessian_potential = torch.stack(hessian_potential, dim=1)
    hessian_diff = hessian_net - hessian_potential
    max_hessian_diff_norm = torch.max(torch.norm(hessian_diff, p=2, dim=(1, 2))).item()
    return {
        "grad_diff": max_grad_diff_norm,
        "hessian_diff": max_hessian_diff_norm,
    }
