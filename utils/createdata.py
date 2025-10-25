import numpy as np
import torch

__all__ = ['create_random_dataset', 'create_random_dataset_grad', 'create_regular_dataset', 'create_random_dataset_Noise']

def create_random_dataset(Potential, Region, num_points, seed=42):
    """Create a random dataset of points within a specified region.
    
    Args:
        Potential: Function to evaluate the potential energy at each point.
        Region: Array defining the boundaries of each dimension.
        num_points: Number of random points to generate.
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple of (grid points, potential values).
    """
    np.random.seed(seed)
    dimension = np.shape(Region)[0]
    grid = np.zeros((num_points, dimension))
    for i in range(dimension):
        grid[:, i] = np.random.uniform(Region[i, 0], Region[i, 1], num_points)
    grid = np.array(grid)
    Z = Potential(grid)

    return grid, Z

def create_random_dataset_grad(Potential, Potential_grad, Region, num_points, proportion=0.15, seed=42):
    """Create a random dataset with potential values and gradients at some points.
    
    Args:
        Potential: Function to evaluate the potential energy.
        Potential_grad: Function to evaluate the gradient of the potential.
        Region: Array defining the boundaries of each dimension.
        num_points: Number of random points to generate.
        ratio: Fraction of points to include gradients for.
        
    Returns:
        Tuple of (grid points, combined data including potential, gradient flags, and gradients).
    """
    np.random.seed(seed)
    dimension = np.shape(Region)[0]
    grid = np.zeros((num_points, dimension))
    for i in range(dimension):
        grid[:, i] = np.random.uniform(Region[i, 0], Region[i, 1], num_points)
    Z = Potential(grid)

    size = int(proportion * num_points)
    indices = np.random.choice(len(Z), size=size, replace=False)
    sign = np.zeros((len(Z), 1), dtype=np.float64)
    grads = np.zeros((len(Z), dimension), dtype=np.float64)
    sign[indices] = 1
    for i in indices:
        grads[i, :] = Potential_grad(grid[i, :])
    Z = np.concatenate((Z, sign, grads), axis=-1)

    return grid, Z


def create_regular_dataset(Potential, Region, num_points):
    """Create a regular grid dataset within a 2D region.
    
    Args:
        Potential: Function to evaluate the potential energy.
        Region: 2D array defining the boundaries for x and y dimensions.
        num_points: Number of points along each dimension.
        
    Returns:
        Tuple of (X meshgrid, Y meshgrid, flattened grid points, potential values).
    """
    x = np.linspace(Region[0, 0], Region[0, 1], num_points)
    y = np.linspace(Region[1, 0], Region[1, 1], num_points)
    X, Y = np.meshgrid(x, y)
    grid = np.c_[X.ravel(), Y.ravel()]
    Z = Potential(grid)

    return X, Y, grid, Z

def create_random_dataset_Noise(Potential, Region, num_points, Ratio=1.0, std_dev=0.1, seed=42):
    """Create a random dataset with additive noise to potential values.
    
    Args:
        Potential: Function to evaluate the potential energy.
        Region: Array defining the boundaries of each dimension.
        num_points: Number of random points to generate.
        Ratio: Fraction of points to add noise to.
        std_dev: Standard deviation of the Gaussian noise.
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple of (grid points, noisy potential values).
    """
    np.random.seed(seed)
    dimension = np.shape(Region)[0]
    grid = np.zeros((num_points, dimension))
    for i in range(dimension):
        grid[:, i] = np.random.uniform(Region[i, 0], Region[i, 1], num_points)
    Z = Potential(grid).squeeze(axis=-1)
    num_error_samples = int(num_points * Ratio)
    indices = np.random.choice(num_points, num_error_samples, replace=False)
    noise = np.random.normal(0, std_dev, num_error_samples)
    Z[indices] += noise
    Z = Z[:, np.newaxis]

    return grid, Z

def create_parametric_dataset(Potential, Region, num_points):
    np.random.seed(42)
    num_alpha = 10
    num_points = num_points // 10
    x = np.random.uniform(Region[0, 0], Region[0, 1], num_points * num_alpha)
    y = np.random.uniform(Region[1, 0], Region[1, 1], num_points * num_alpha)
    w = np.random.uniform(Region[2, 0], Region[2, 1], num_points * num_alpha)
    alpha = np.linspace(Region[3, 0], Region[3, 1], num_alpha)
    alpha = np.repeat(alpha, num_points)
    grid = np.column_stack((x, y, w, alpha))
    Z = torch.ones([num_points * num_alpha, 1], dtype=torch.float64)
    for i in range(num_points * num_alpha):
        Z[i] = Potential(grid[i, :])
    grid = torch.tensor(grid, dtype=torch.float64)

    return grid, Z