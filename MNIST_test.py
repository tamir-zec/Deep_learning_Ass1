from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import Main

PIXEL_MAX_VALUE = 255


def reshape_x_input(inp):
    return inp.reshape(inp.shape[0], inp.shape[1] * inp.shape[2])


def get_early_stopping_callback(train_x, train_y, val_x, val_y, steps: int):
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
            return params
        else:
            return None

    return callback


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train, random_state=42)
y_train = to_categorical(y_train, 10)
y_val = to_categorical(y_val, 10)
y_test = to_categorical(y_test, 10)

x_train = reshape_x_input(x_train).transpose() / PIXEL_MAX_VALUE
x_val = reshape_x_input(x_val).transpose() / PIXEL_MAX_VALUE
x_test = reshape_x_input(x_test).transpose() / PIXEL_MAX_VALUE
y_train = y_train.transpose()
y_val = y_val.transpose()
y_test = y_test.transpose()
# print(f'x train shape:{x_train.shape} x test shape: {x_test.shape}')
# print(f'y train shape:{y_train.shape} x val shape: {x_val.shape}')
# print("hello")

early_stopping = get_early_stopping_callback(x_train, y_train, x_val, y_val, 20)
learning_rate = 0.009
Main.L_layer_model(x_train, y_train, [20, 7, 5, 10], learning_rate, 480 * 150, 100, use_batchnorm=False,
                   validation=(x_val, y_val), early_stopping=early_stopping)
