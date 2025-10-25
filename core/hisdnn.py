import numpy as np
import scipy.sparse.linalg as linalg
import copy
import torch

def hisdnn(model, w0, v0=None, dt=1e-7, momentum=0.0, max_iter=10, 
           nesterov=False, restart=50, ADAD=False,
           k=0, step_interval=1, method='sirqit', **kwargs):
    """
    Neural Network High-index Saddle Dynamics (HiSD) optimization algorithm.
    
    This function implements HiSD for finding saddle points in neural network potential energy surfaces.
    It utilizes automatic differentiation to compute gradients and Hessians from PyTorch models.
    
    Parameters
    ----------
    model : torch.nn.Module
        Neural network model representing the energy function.
    w0 : numpy.ndarray
        Initial parameter vector to start the optimization.
    v0 : numpy.ndarray, optional
        Initial eigenvector(s) corresponding to negative eigenvalues of the Hessian.
    dt : float, default=1e-7
        Learning rate or step size.
    momentum : float, default=0.0
        Momentum factor.
    max_iter : int, default=10
        Maximum number of iterations.
    k : int, default=0
        Morse index of the saddle point to find (0 for minima).
    
    Returns
    -------
    tuple
        Tuple containing (final_parameters, parameter_history)
    """
    w = copy.deepcopy(w0)
    w_pre = copy.deepcopy(w0)
    D = w.shape[0]
    tau = 0.5

    if k > 0:
        if v0 is not None:
            v = copy.deepcopy(v0)
        else:
            v = given_eigenvector(model, w, D, k)
        if v.ndim == 1:
            v = v.reshape(-1, 1)

    w_record = []
    for j in range(1, max_iter + 1):
        if nesterov:
            j0 = j % restart
            gamma_j = j0 / (j0+3)
            w_temp = w + gamma_j * (w - w_pre)
        else:
            w_temp = w

        x = torch.tensor(w_temp, dtype=torch.float64, requires_grad=True)
        outputs = model(x)
        grad_outputs = torch.ones_like(outputs)
        g = torch.autograd.grad(outputs, x, grad_outputs=grad_outputs)[0].numpy()

        if k > 0:
            dw = dt * (g - 2.0 * np.matmul(v, np.matmul(v.T, g)))
        else:
            dw = dt * g

        w_temp = w_temp - dw + momentum * (w - w_pre)
        w_pre = w
        w = w_temp

        if k > 0:
            if method == 'sirqit':
                v = sirqit(model=model, w=w, v=v, maxiter=kwargs['sub_iter'], ds=kwargs['ds'], ADAD=ADAD)
            elif method == 'lobpcg':
                _, v = lobpcg(model=model, w=w, v0=v, max_iter=kwargs['sub_iter'])

        if j % step_interval == 0:
            print('Iteration: ' + str(j) + f'|| Norm of gradient: {np.linalg.norm(g):.8f}')
        w_record.append(w.reshape(-1, ))

        #if np.linalg.norm(g) < 1e-8:
        #    break

    return w, w_record

###############################################################################################
def single_hessian_vector_product(model, w, v, L=1e-3, ADAD=True):
    if ADAD:
        w0 = torch.tensor(w, dtype=torch.float64, requires_grad=True)
        output = model(w0)
        grad_output = torch.autograd.grad(output, w0, create_graph=True)[0]
        Hessian = torch.zeros((len(w), len(w)))
        for i in range(len(w)):
            grad_grad = torch.autograd.grad(grad_output[i], w0, create_graph=True)[0]
            Hessian[i] = grad_grad.view(-1)
        Direction = np.matmul(Hessian.detach().numpy(), v)
        return Direction
    else:
        w1 = torch.tensor(w + L * v, requires_grad=True)
        w2 = torch.tensor(w - L * v, requires_grad=True)
        output1 = model(w1)
        output2 = model(w2)
        grad_outputs = torch.ones_like(output1)
        grad1 = torch.autograd.grad(output1, w1, grad_outputs=grad_outputs)[0].numpy()
        grad2 = torch.autograd.grad(output2, w2, grad_outputs=grad_outputs)[0].numpy()
        return (grad1 - grad2) / (2 * L)


def batch_hessian_vector_product(model, w, v, ADAD=True):
    hvp = lambda x: single_hessian_vector_product(model, w, x, ADAD=ADAD)
    hvpmat = map(hvp, list(np.transpose(v)))
    result = []
    for item in hvpmat:
        result.append(item)
    return np.transpose(np.stack(result))


def hessian(model, w, L=1e-3):
    '''
    Compute the Hessian matrix.
    '''
    dimension = w.shape[0]
    return batch_hessian_vector_product(model, w, np.eye(dimension))


###############################################################################################
def given_eigenvector(model, w, D, k):
    H = hessian(model, w)
    eigenvalue, eigenvector = np.linalg.eig((H + np.transpose(H)) / 2.0)
    order = np.argsort(eigenvalue)
    eigenvector = eigenvector[:, order]
    return eigenvector[:, 0:k]


def lobpcg(model, w, v0, max_iter=10, L=1e-3):
    """
    Locally Optimal Block Preconditioned Conjugate Gradient method for finding smallest eigenvalues.
    
    This function finds the smallest eigenvalues and corresponding eigenvectors of the Hessian matrix
    using the LOBPCG method with hessian-vector product calculations.
    
    Parameters
    ----------
    model : torch.nn.Module
        Neural network model representing the energy function.
    w : numpy.ndarray
        Current parameter vector at which to compute the Hessian.
    v0 : numpy.ndarray
        Initial guess for the eigenvectors.
    max_iter : int, default=10
        Maximum number of iterations for the LOBPCG algorithm.
    L : float, default=1e-3
        Regularization parameter for hessian calculation.
    
    Returns
    -------
    tuple
        Tuple containing (eigenvalues, eigenvectors)
    """
    matvec_fun = lambda z: single_hessian_vector_product(model, w, z)
    matmat_fun = lambda z: batch_hessian_vector_product(model, w, z)
    H = linalg.LinearOperator(shape=(len(v0), len(v0)), matvec=matvec_fun, matmat=matmat_fun)
    values, vectors = linalg.lobpcg(A=H, X=v0, maxiter=max_iter, largest=False)
    return values, vectors

def sirqit(model, w, v, maxiter=10, ds=1e-7, L=1e-3, ADAD=True):
    """
    Subspace Iteration with Rayleigh Quotient and Inverse Iteration for eigenvector refinement.
    
    This function refines a set of eigenvectors corresponding to the smallest eigenvalues
    of the Hessian matrix using an iterative subspace method.
    
    Parameters
    ----------
    model : torch.nn.Module
        Neural network model representing the energy function.
    w : numpy.ndarray
        Current parameter vector at which to compute the Hessian.
    v : numpy.ndarray
        Initial eigenvectors to refine.
    maxiter : int, default=10
        Maximum number of refinement iterations.
    ds : float, default=1e-7
        Step size for the iteration.
    L : float, default=1e-3
        Regularization parameter.
    ADAD : bool, default=True
        Whether to use automatic differentiation for Hessian calculations.
    
    Returns
    -------
    numpy.ndarray
        Refined eigenvectors after subspace iteration.
    """
    for _ in range(maxiter):
        hv = batch_hessian_vector_product(model, w, v, ADAD=ADAD)
        coef = np.triu(v.T @ hv)
        v -= (hv - v * np.diag(coef) - 2.0 * v @ np.triu(coef, k=1)) * ds
        v, _ = np.linalg.qr(v)
    return v