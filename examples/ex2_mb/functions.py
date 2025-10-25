import numpy as np
import torch

import numpy as np

def MB(w, add=False):
    A = np.array([-200.0, -100.0, -170.0, 15.0], dtype=np.float64)
    a = np.array([-1.0, -1.0, -6.5, 0.7], dtype=np.float64)
    b = np.array([0.0, 0.0, 11.0, 0.6], dtype=np.float64)
    c = np.array([-10.0, -10.0, -6.5, 0.7], dtype=np.float64)
    x_bar = np.array([1.0, 0.0, -0.5, -1.0], dtype=np.float64)
    y_bar = np.array([0.0, 0.5, 1.5, 1.0], dtype=np.float64)

    w = np.array(w, dtype=np.float64)
    
    dx = w[:, 0, np.newaxis] - x_bar
    dy = w[:, 1, np.newaxis] - y_bar
    values = A * np.exp(a * (dx ** 2) + b * dx * dy + c * (dy ** 2))

    if not add:
        return np.sum(values, axis=1, keepdims=True)
    else:
        A_5, a_5, c_5 = 500.0, -0.1, -0.1
        x_bar_5, y_bar_5 = -0.5582, 1.4417
        additional_term = A_5 * np.sin(w[:, 0] * w[:, 1]) * np.exp(
            a_5 * (w[:, 0] - x_bar_5) ** 2 + c_5 * (w[:, 1] - y_bar_5) ** 2)
        return np.sum(values, axis=1, keepdims=True) + additional_term[:, np.newaxis]

def MB_grad(w, add=False):
    A = np.array([-200.0, -100.0, -170.0, 15.0])
    a = np.array([-1.0, -1.0, -6.5, 0.7])
    b = np.array([0.0, 0.0, 11.0, 0.6])
    c = np.array([-10.0, -10.0, -6.5, 0.7])
    x_bar = np.array([1.0, 0.0, -0.5, -1.0])
    y_bar = np.array([0.0, 0.5, 1.5, 1.0])

    dx = w[0] - x_bar
    dy = w[1] - y_bar
    kernel = np.exp(a * (dx ** 2) + b * dx * dy + c * (dy ** 2))

    g1 = (A * kernel * (2.0 * a * dx + b * dy)).sum()
    g2 = (A * kernel * (2.0 * c * dy + b * dx)).sum()
    if not add:
        return np.array([g1, g2])
    else:
        A_5, a_5, c_5 = 500.0, -0.1, -0.1
        x_bar_5, y_bar_5 = -0.5582, 1.4417

        dx_5, dy_5 = w[0] - x_bar_5, w[1] - y_bar_5
        kernel_5 = a_5 * (dx_5) ** 2 + c_5 * (dy_5) ** 2
        w_product = w[0] * w[1]
        delta_g1 = A_5 * np.cos(w_product) * w[1] * np.exp(kernel_5) + A_5 * np.sin(w_product) * np.exp(
            kernel_5) * 2.0 * a_5 * dx_5
        delta_g2 = A_5 * np.cos(w_product) * w[0] * np.exp(kernel_5) + A_5 * np.sin(w_product) * np.exp(
            kernel_5) * 2.0 * c_5 * dy_5
        return np.array([g1 + delta_g1, g2 + delta_g2])

def MMB(w, add=True):
    A = np.array([-200.0, -100.0, -170.0, 15.0], dtype=np.float64)
    a = np.array([-1.0, -1.0, -6.5, 0.7], dtype=np.float64)
    b = np.array([0.0, 0.0, 11.0, 0.6], dtype=np.float64)
    c = np.array([-10.0, -10.0, -6.5, 0.7], dtype=np.float64)
    x_bar = np.array([1.0, 0.0, -0.5, -1.0], dtype=np.float64)
    y_bar = np.array([0.0, 0.5, 1.5, 1.0], dtype=np.float64)

    w = np.array(w, dtype=np.float64)
    
    dx = w[:, 0, np.newaxis] - x_bar
    dy = w[:, 1, np.newaxis] - y_bar
    values = A * np.exp(a * (dx ** 2) + b * dx * dy + c * (dy ** 2))

    if not add:
        return np.sum(values, axis=1, keepdims=True)
    else:
        A_5, a_5, c_5 = 500.0, -0.1, -0.1
        x_bar_5, y_bar_5 = -0.5582, 1.4417
        additional_term = A_5 * np.sin(w[:, 0] * w[:, 1]) * np.exp(
            a_5 * (w[:, 0] - x_bar_5) ** 2 + c_5 * (w[:, 1] - y_bar_5) ** 2)
        return np.sum(values, axis=1, keepdims=True) + additional_term[:, np.newaxis]

def MMB_grad(w, add=True):
    A = np.array([-200.0, -100.0, -170.0, 15.0])
    a = np.array([-1.0, -1.0, -6.5, 0.7])
    b = np.array([0.0, 0.0, 11.0, 0.6])
    c = np.array([-10.0, -10.0, -6.5, 0.7])
    x_bar = np.array([1.0, 0.0, -0.5, -1.0])
    y_bar = np.array([0.0, 0.5, 1.5, 1.0])

    dx = w[0] - x_bar
    dy = w[1] - y_bar
    kernel = np.exp(a * (dx ** 2) + b * dx * dy + c * (dy ** 2))

    g1 = (A * kernel * (2.0 * a * dx + b * dy)).sum()
    g2 = (A * kernel * (2.0 * c * dy + b * dx)).sum()
    if not add:
        return np.array([g1, g2])
    else:
        A_5, a_5, c_5 = 500.0, -0.1, -0.1
        x_bar_5, y_bar_5 = -0.5582, 1.4417

        dx_5, dy_5 = w[0] - x_bar_5, w[1] - y_bar_5
        kernel_5 = a_5 * (dx_5) ** 2 + c_5 * (dy_5) ** 2
        w_product = w[0] * w[1]
        delta_g1 = A_5 * np.cos(w_product) * w[1] * np.exp(kernel_5) + A_5 * np.sin(w_product) * np.exp(
            kernel_5) * 2.0 * a_5 * dx_5
        delta_g2 = A_5 * np.cos(w_product) * w[0] * np.exp(kernel_5) + A_5 * np.sin(w_product) * np.exp(
            kernel_5) * 2.0 * c_5 * dy_5
        return np.array([g1 + delta_g1, g2 + delta_g2])