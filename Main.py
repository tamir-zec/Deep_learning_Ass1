from typing import Tuple, Dict, List
from collections.abc import Callable
import numpy as np


def initialize_parameters(layer_dims: Tuple) -> Dict:
    layers = {
        'W': list(),
        'b': list()
    }
    for layer in layer_dims:
        weights = np.random.randn(1, layer)
        bias = np.random.random()
        layers['W'].append(weights)
        layers['b'].append(bias)

    return layers


def linear_forward(A: np.ndarray, W: np.ndarray, b: np.ndarray) -> [List, Dict]:
    z = np.dot(W.transpose(), A) + b
    linear_cache = {
        'A': A,
        'W': W,
        'b': b,
    }
    return z, linear_cache


def softmax(Z: np.ndarray) -> [List, Dict]:
    exp = np.exp(Z)
    sum_array = np.sum(exp, axis=0)
    ans = exp / sum_array
    return ans, {'Z': Z}


def relu(Z: np.ndarray) -> [List, Dict]:
    ans = np.maximum(Z, 0)
    return ans, {'Z': Z}


def linear_activation_forward(A_prev: np.ndarray, W: np.ndarray, b: np.ndarray,
                              activation: Callable) -> Tuple[np.ndarray, Dict]:
    z, z_cache = linear_forward(A_prev, W, b)
    new_A, activation_cache = activation(z)
    z_cache.update(activation_cache)
    return new_A, z_cache


def L_model_forward(X: np.ndarray, parameters: np.ndarray, use_batchnorm: bool) -> [np.ndarray, List]:
    pass


def compute_cost(AL: np.ndarray, Y: np.ndarray) -> np.ndarray:
    pass


def apply_batchnorm(A: np.ndarray) -> np.ndarray:
    pass
