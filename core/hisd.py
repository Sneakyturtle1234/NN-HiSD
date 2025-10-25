import numpy as np
import scipy.sparse.linalg as linalg
import copy
import warnings
from utils import sirqit, lobpcg, given_eigenvector

warnings.filterwarnings('ignore')

def hisd(grad, w0, v0=None, method='sirqit', dt=1e-7, momentum=0.0,
         nesterov=False, restart=50, max_iter=10, k=0, L=1e-3, report=True,
         initial_hessian='full_hessian', step_interval=1, **kwargs):
    """
    High-index Saddle Dynamics (HiSD) algorithm implementation.
    
    This function implements the HiSD method for finding saddle points in potential energy surfaces.
    It can find minima (k=0) or saddle points with specific Morse index (k>0).
    
    Parameters
    ----------
    grad : function
        Gradient function that takes a vector w and returns its gradient.
    w0 : numpy.ndarray
        Initial position vector to start the optimization.
    v0 : numpy.ndarray, optional
        Initial eigenvector(s) corresponding to negative eigenvalues of the Hessian.
        Required when k > 0.
    method : str, default='sirqit'
        Method to update the eigenvectors. Options: 'sirqit' or 'lobpcg'.
    dt : float, default=1e-7
        Learning rate or step size for parameter updates.
    momentum : float, default=0.0
        Momentum factor for parameter updates.
    nesterov : bool, default=False
        Whether to use Nesterov momentum.
    restart : int, default=50
        Restart frequency for Nesterov momentum.
    max_iter : int, default=10
        Maximum number of iterations.
    k : int, default=0
        Morse index of the saddle point to find (0 for minima, positive for saddle points).
    L : float, default=1e-3
        Step size of dimer method.
    report : bool, default=True
        Whether to report progress during optimization.
    initial_hessian : str, default='full_hessian'
        Method to compute the initial Hessian.
    step_interval : int, default=1
        Interval at which to report progress.
    **kwargs : dict
        Additional parameters.
    
    Returns
    -------
    numpy.ndarray or tuple
        If report=True, returns a tuple (w, w_record) where:
            - w is the final optimized position
            - w_record is a list of positions during optimization
        If report=False, returns only the final optimized position w.
    """
    w = copy.deepcopy(w0)
    w_pre = copy.deepcopy(w0)
    D = w.shape[0]
    tau = 0.5

    if k > 0:
        if v0 is not None:
            v = copy.deepcopy(v0)
        else:
            if initial_hessian == 'full_hessian': v = given_eigenvector(grad, w, L, D, k)
        if v.ndim == 1:
            v = v.reshape(-1, 1)

    for j in range(1, max_iter + 1):
        if nesterov:
            j0 = j % restart
            gamma_j = j0 / (j0+3)
            w_temp = w + gamma_j * (w - w_pre)
        else:
            w_temp = w 
        
        g = grad(w)

        if k > 0:
            dw = dt * (g - 2.0 * np.matmul(v, np.matmul(v.T, g)))
        else:
            dw = dt * g

        w_temp = w_temp - dw + momentum * (w - w_pre)
        w_pre = w
        w = w_temp

        if k > 0:
            if method == 'sirqit':
                v = sirqit(grad=grad,
                           w=w,
                           v=v,
                           maxiter=kwargs['sub_iter'],
                           ds=kwargs['ds'],
                           L=L)
            elif method == 'lobpcg':
                _, v = lobpcg(grad=grad,
                              w=w,
                              v0=v,
                              max_iter=kwargs['sub_iter'],
                              L=L)

        if report:
            if j % step_interval == 0:
                print('Iteration: ' + str(j) + f'|| Norm of gradient: {np.linalg.norm(g):.8f}')
            if j == 1: w_record = []
            w_record.append(w.reshape(-1, ))

        if 'Stopping' in kwargs and np.linalg.norm(g) < 1e-8:
            break 

    if report:
        return w, w_record
    else:
        return w