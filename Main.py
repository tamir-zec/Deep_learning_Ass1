from typing import Tuple, Dict, List
from collections.abc import Callable
import numpy as np


def initialize_parameters(layer_dims: List) -> Dict :
    layers = {
        'W': list(),
        'b': list()
    }
    # lets try smart init
    np.random.seed(3)
    for idx in range(1, len(layer_dims)) :
        weights = np.random.randn(layer_dims[idx - 1], layer_dims[idx]) * np.sqrt(2 / layer_dims[idx - 1])
        bias = np.random.randn(layer_dims[idx], 1)
        layers['W'].append(weights)
        layers['b'].append(bias)

    return layers


def linear_forward(A: np.ndarray, W: np.ndarray, b: np.ndarray) -> [List, Dict] :
    """
    :param A: [curr features X samples]
    :param W: [curr features X next features]
    :param b: [next features X 1]
    :return: Z: [next features X samples] , cache -dict with A, W, b
    """
    z = np.matmul(W.transpose(), A) + b
    linear_cache = {
        'A' : A,
        'W' : W,
        'b' : b,
    }
    return z, linear_cache


def softmax(Z: np.ndarray) -> [List, Dict] :
    exp = np.exp(Z)
    sum_array = np.sum(exp, axis=0)
    ans = exp / sum_array
    return ans, {'Z' : Z}


def relu(Z: np.ndarray) -> [List, Dict] :
    ans = np.maximum(Z, 0)
    return ans, {'Z' : Z}


def linear_activation_forward(A_prev: np.ndarray, W: np.ndarray, b: np.ndarray,
                              activation: Callable) -> Tuple[np.ndarray, Dict] :
    z, z_cache = linear_forward(A_prev, W, b)
    new_A, activation_cache = activation(z)
    z_cache.update(activation_cache)
    return new_A, z_cache


def L_model_forward(X: np.ndarray, parameters: Dict, use_batchnorm: bool) -> [np.ndarray, List] :
    """
    :param X: [flatten input X  samples] - model input
    :param parameters:
    :param use_batchnorm:
    :return: [softmax result, all_caches]
    """
    weights = parameters['W']
    biases = parameters['b']
    A_prev = X
    all_caches = list()
    for layer_weight, layer_bias in zip(weights[:-1], biases[:-1]) :
        A_prev, cache = linear_activation_forward(A_prev, layer_weight, layer_bias, relu)
        all_caches.append(cache)
        if use_batchnorm :
            A_prev = apply_batchnorm(A_prev)
    AL, cache = linear_activation_forward(A_prev, weights[-1], biases[-1], softmax)
    all_caches.append(cache)
    return AL, all_caches


def compute_cost(AL: np.ndarray, Y: np.ndarray) -> float :
    """
    :param AL: [classes X samples] - prediction of softmax layer
    :param Y: [classes X samples] - np.ndarry - one hot vector for each sample. each column is one sample. rows are classes
    :return:
    """
    pred_log = np.log(AL)
    y_pred_actual = np.multiply(pred_log, Y)
    y_pred_actual[np.isnan(y_pred_actual)] = 0
    cost_sum = np.sum(y_pred_actual)
    cost = cost_sum / AL.shape[1]
    return -cost


def apply_batchnorm(A: np.ndarray) -> np.ndarray :
    mean = np.mean(A, axis=1, keepdims=True)
    std = np.std(A, axis=1)
    epsilon = 1e-3
    denominator = np.sqrt((std ** 2) + epsilon)
    # mean = np.reshape(mean, (mean.shape[0], 1))
    numerator = A - mean
    diag = np.diag(1 / denominator)
    return np.matmul(diag, numerator)


def Linear_backward(dZ: np.ndarray, cache: Dict) -> Tuple :
    """
    :param dZ: [next features x samples]
    :param cache: of current layer
                  'A': [curr features X samples]
                  'W': [current features X next features]
                  'b': [next features X 1]
    :return: da_prev: [curr features X samples]
             dw: [current features X next features]
             db: [next features X 1]
    """
    A = cache['A']
    m = A.shape[1]
    dW = np.matmul(A, dZ.transpose()) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m  # we sum on rows because this layer bias is [next features X 1]
    da_prev = np.matmul(cache['W'], dZ)
    return da_prev, dW, db


def linear_activation_backward(dA: np.ndarray, cache: Dict, activation: Callable) -> Tuple :
    grad_z = activation(dA, cache)
    return Linear_backward(grad_z, cache)


def relu_backward(dA: np.ndarray, activation_cache: Dict) -> np.ndarray :
    dA_new = np.array(dA, copy=True)
    dA_new[activation_cache['Z'] <= 0] = 0
    # dA_new[dA_new != 0] = 1
    return dA_new


def softmax_backward(dA: np.ndarray, activation_cache: Dict) -> np.ndarray :
    dz = activation_cache['Z'] - dA
    return dz


def derivative_cross_entropy(AL: np.ndarray, Y: np.ndarray) -> np.ndarray :
    """
    :param AL:  [classes X samples] - result of softmax
    :param Y: [classes X samples] - one hot vector of real classes
    :return: [classes X samples]
    """
    return np.abs(Y - AL)


def L_model_backward(AL: np.ndarray, Y: np.ndarray, caches: List) -> Dict :
    """
    :param AL:  [classes,samples] - softmax probabilites
    :param Y:   [classes,samples] - one hot vector real ans
    :param caches: List[{A,W,b,Z}]
    :return: Dict[gradients]
    """
    grads = {}
    dA = derivative_cross_entropy(AL, Y)
    da_prev, dW, db = linear_activation_backward(dA, caches[-1], softmax_backward)
    layers = len(caches) - 1
    grads["dA" + str(layers)] = da_prev
    grads["dW" + str(layers)] = dW
    grads["db" + str(layers)] = db

    for layer in reversed((range(layers))) :
        da_prev, dW, db = linear_activation_backward(da_prev, caches[layer], relu_backward)
        grads["dA" + str(layer)] = da_prev
        grads["dW" + str(layer)] = dW
        grads["db" + str(layer)] = db

    return grads


def Update_parameters(parameters: Dict, grads: Dict, learning_rate: float) -> Dict :
    """
    :param parameters: {'W': list of weights for each layer, 'b': list of bias for each layer} - look at init
    :param grads: Dict result of L_model_backward
    :param learning_rate:
    :return: Updated parameters
    """
    ans = {
        'W' : list(),
        'b' : list()
    }
    for idx in range(len(parameters['W'])) :
        W = parameters['W'][idx]
        b = parameters['b'][idx]
        dW = grads["dW" + str(idx)]
        db = grads["db" + str(idx)]
        ans['W'].append(W - learning_rate * dW)
        ans['b'].append(b - learning_rate * db)

    return ans


def generator_by_batch(X: np.ndarray, Y: np.ndarray, batch_size: int):
    samples = X.shape[1]
    curr = 0
    while True :
        if curr + batch_size >= samples :
            yield X[:, curr :], Y[:, curr :]
            curr = 0
        yield X[:, curr : curr + batch_size], Y[:, curr : curr + batch_size]
        curr += batch_size


def L_layer_model(X: np.ndarray, Y: np.ndarray, layer_dims: List, learning_rate: float,
                  num_iterations: int, batch_size: int, use_batchnorm: bool = False,
                  validation: Tuple[np.ndarray, np.ndarray] = None,
                  early_stopping: Callable = None) -> [Dict, List[float]]:
    """
    :param X: [height*width , samples] - model input
    :param Y: [classes, samples] - one hot vector of real labels
    :param layer_dims: *without input & without number of classes
    :param learning_rate:
    :param num_iterations:
    :param batch_size:
    :param use_batchnorm:
    :param validation:
    :param early_stopping:
    :return: [parameters, costs]:
              * parameters - dictionary of weights and bias (like init)
              * costs = list of floats

    """
    costs = list()
    full_dims = [X.shape[0]] + layer_dims
    params = initialize_parameters(full_dims)
    input_gen = generator_by_batch(X, Y, batch_size)
    for curr_iter in range(1, num_iterations + 1) :
        curr_inp, curr_labels = next(input_gen)
        AL, caches = L_model_forward(curr_inp, params, use_batchnorm)
        if curr_iter % 100 == 0 :
            cost_AL, _ = L_model_forward(X, params, use_batchnorm)
            cost = compute_cost(cost_AL, Y)
            costs.append(cost)
            # if validation is not None:
            #     val_acc = Predict(validation[0], validation[1], params, use_batchnorm)
            #     train_acc = Predict(X, Y, params, use_batchnorm)
            #     val_ans, _ = L_model_forward(validation[0], params, use_batchnorm)
            #     val_cost = compute_cost(val_ans, validation[1])
            #     print("iter num: {} - train cost: {:.3f} , train acc: {:.3f}  - val acc: {:.3f} , val-cost: {:.3f}".
            #           format(curr_iter, cost, train_acc, val_acc, val_cost))

            if early_stopping is not None:
                early_ans = early_stopping(params, use_batchnorm)
                if early_ans is not None:
                    return early_ans, costs

        grads = L_model_backward(AL, curr_labels, caches)
        params = Update_parameters(params, grads, learning_rate)

    return params, costs


def Predict(X: np.ndarray, Y: np.ndarray, parameters: Dict, use_batchnorm: bool = False) -> float:
    """
       :param X: [height*width , samples] - model input
       :param Y: [classes, samples] - one hot vector of real labels
       :param parameters: Dictionary of Weights and Biases
              :param use_batchnorm:
       :return: accuracy:
   """
    predictions, _ = L_model_forward(X, parameters, use_batchnorm)
    prediction_classes = np.argmax(predictions, axis=0)
    real_y = np.argmax(Y, axis=0)
    difference = (prediction_classes - real_y)
    accuracy = len(difference[difference == 0]) / len(difference)
    return accuracy
