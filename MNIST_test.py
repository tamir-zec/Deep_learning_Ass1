from typing import Callable

from tensorflow.keras.datasets import mnist
import numpy as np
import Main

PIXEL_MAX_VALUE = 255


def reshape_x_input(inp):
    return inp.reshape(inp.shape[0], inp.shape[1] * inp.shape[2])


def get_early_stopping_callback(train_x, train_y, val_x, val_y, steps: int) -> Callable:
    count = 0
    max_val_acc = -1
    best_params = None

    def callback(params, use_batchnorm):
        nonlocal count
        nonlocal max_val_acc
        nonlocal best_params
        val_acc = Main.Predict(val_x, val_y, params, use_batchnorm)
        if val_acc >= max_val_acc:
            count = 0
            best_params = params
            max_val_acc = val_acc
        else:
            count += 1
        if count > steps:
            print("early stopping")
            return best_params
        else:
            return None

    return callback


def pre_process_input(X, Y, val_size: int) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
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


def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train, x_val, y_val = pre_process_input(x_train, y_train, 0.1)


    early_stopping = get_early_stopping_callback(x_train, y_train, x_val, y_val, 20)
    learning_rate = 0.009
    Main.L_layer_model(x_train, y_train, [20, 7, 5, 10], learning_rate, 480 * 150, 100, use_batchnorm=False,
                       validation=(x_val, y_val), early_stopping=early_stopping)


if __name__ == "__main__":
    main()
