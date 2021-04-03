from typing import Callable


from tensorflow.keras.datasets import mnist
import numpy as np


import Main

PIXEL_MAX_VALUE = 255


def reshape_x_input(inp):
    return inp.reshape(inp.shape[0], inp.shape[1] * inp.shape[2])


def get_early_stopping_callback(train_x, train_y, val_x, val_y, steps: int) -> Callable:
    count = 0
    max_cost = np.inf
    best_params = None

    def callback(params, use_batchnorm, curr_cost):
        nonlocal count
        nonlocal max_cost
        nonlocal best_params
        if curr_cost <= max_cost:
            count = 0
            best_params = params
            max_cost = curr_cost
        else:
            count += 1
        if count > steps:
            print("early stopping")
            return best_params
        else:
            return None

    return callback


def pre_process_input(X, Y, val_size: float) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    :param X:
    :param Y:
    :param val_size: size of validation in percentage
    :return: train_x, train_y, val_x, val_y
    """
    X = reshape_x_input(X).transpose() / PIXEL_MAX_VALUE
    cut_off_idx = int(X.shape[1] * (1 - val_size))
    stacked = np.vstack((X, Y))
    np.random.shuffle(stacked.transpose())
    shuf_x = stacked[:-1]
    shuf_y = stacked[-1]
    train_x = shuf_x[:, :cut_off_idx]
    train_y = shuf_y[:cut_off_idx]
    val_x = shuf_x[:, cut_off_idx:]
    val_y = shuf_y[cut_off_idx:]
    # convert to categorical
    train_y = np.eye(10)[train_y.astype(int)].transpose()
    val_y = np.eye(10)[val_y.astype(int)].transpose()
    return train_x, train_y, val_x, val_y


def old_test_split(x_train, y_train, x_test, y_test):
    from sklearn.model_selection import train_test_split
    from tensorflow.python.keras.utils.np_utils import to_categorical
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, stratify=y_train,
                                                      random_state=42)
    y_train = to_categorical(y_train, 10)
    y_val = to_categorical(y_val, 10)
    y_test = to_categorical(y_test, 10)
    x_train = reshape_x_input(x_train).transpose() / PIXEL_MAX_VALUE
    x_val = reshape_x_input(x_val).transpose() / PIXEL_MAX_VALUE
    x_test = reshape_x_input(x_test).transpose() / PIXEL_MAX_VALUE
    y_train = y_train.transpose()
    y_val = y_val.transpose()
    y_test = y_test.transpose()
    return x_train, y_train, x_val, y_val, x_test, y_test


def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train, x_val, y_val = pre_process_input(x_train, y_train, 0.1)
    # x_train, y_train, x_val, y_val, x_test, y_test = old_test_split(x_train, y_train, x_test, y_test)

    early_stopping = get_early_stopping_callback(x_train, y_train, x_val, y_val, 3)
    learning_rate = 0.009
    Main.L_layer_model(x_train, y_train, [20, 7, 5, 10], learning_rate, 480 * 150, 256, use_batchnorm=False,
                       validation=(x_val, y_val), early_stopping=early_stopping)


if __name__ == "__main__":
    main()
