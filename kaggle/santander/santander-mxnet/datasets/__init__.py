import os
import pandas as pd
import numpy as np
import mxnet as mx

FEATURES_COUNT = 369


def _get_file_dir():
    return os.path.dirname(os.path.realpath(__file__))


def _load_test_csv():
    csv_path = os.path.join(_get_file_dir(), 'test.csv')
    csv = pd.read_csv(csv_path)
    csv_values = csv.values

    x = csv_values[:, 1:]

    return x


def _load_train_csv():
    csv_path = os.path.join(_get_file_dir(), 'train.csv')
    csv = pd.read_csv(csv_path)
    csv_values = csv.values

    x = csv_values[:, 1:-1]
    y = csv_values[:, -1]

    return x, y


def _normalize(x):
    x_mean = np.mean(x, axis=0)
    x -= x_mean
    return x


def load_train_dataset(batch_size, split=True, validation_ratio=0.2, normalize=True):
    x_train_all, y_train_all = _load_train_csv()

    if normalize:
        x_train_all = _normalize(x_train_all)

    if not split:
        train_iter = mx.io.NDArrayIter(data=x_train_all, label=y_train_all, batch_size=batch_size, shuffle=False)
        return train_iter

    # split into train/validation according to validation_ratio
    num_training = int(x_train_all.shape[0] * (1.0 - validation_ratio))
    x_train = x_train_all[0:num_training]
    x_val = x_train_all[num_training:]
    y_train = y_train_all[0:num_training]
    y_val = y_train_all[num_training:]

    train_iter = mx.io.NDArrayIter(data=x_train, label=y_train, batch_size=batch_size, shuffle=False)
    val_iter = mx.io.NDArrayIter(data=x_val, label=y_val, batch_size=batch_size, shuffle=False)
    return train_iter, val_iter


def load_test_dataset(batch_size, normalize=True):
    x_test = _load_test_csv()

    if normalize:
        x_test = _normalize(x_test)

    test_iter = mx.io.NDArrayIter(data=x_test, batch_size=batch_size, shuffle=False)
    return test_iter
