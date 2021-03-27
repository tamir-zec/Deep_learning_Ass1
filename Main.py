from typing import Tuple, Dict, List
from collections.abc import Callable
import numpy as np


def initialize_parameters(layer_dims: Tuple) -> Dict:
    layers = {
        'W': list(),
        'b': list()
    }
    for idx in range(1, len(layer_dims)):
        weights = np.random.randn(layer_dims[idx-1], layer_dims[idx])
        bias = np.random.randn(layer_dims[idx], 1)
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


def L_model_forward(X: np.ndarray, parameters: Dict, use_batchnorm: bool) -> [np.ndarray, List]:
    weights = parameters['W']
    biases = parameters['b']
    A_prev = X
    all_caches = list()
    for layer_weight, layer_bias in zip(weights[:-1], biases[:-1]):
        A_prev, cache = linear_activation_forward(A_prev, layer_weight, layer_bias, relu)
        all_caches.append(cache)
        if use_batchnorm:
            A_prev = apply_batchnorm(A_prev)
    AL, cache = linear_activation_forward(A_prev, weights[-1], biases[-1], softmax)
    all_caches.append(cache)
    return AL, all_caches


def compute_cost(AL: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    :param AL:
    :param Y: np.ndarry - one hot vector for each sample. each column is one sample. rows are classes
    :return:
    """
    pred_log = np.log(AL)
    y_pred_actual = np.multiply(pred_log, Y)
    cost_sum = np.sum(y_pred_actual)
    cost = cost_sum/AL.shape[1]
    return cost


def apply_batchnorm(A: np.ndarray) -> np.ndarray:
    mean = np.mean(A, axis=1)
    std = np.std(A, axis=1)
    epsilon = 1e-3
    denominator = np.sqrt((std ** 2)+epsilon)
    mean = np.reshape(mean, (mean.shape[0], 1))
    numerator = A - mean
    diag = np.diag(1/denominator)
    return np.matmul(diag, numerator)
