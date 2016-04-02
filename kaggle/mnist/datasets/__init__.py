import os
import pandas as pd
import numpy as np


def _get_file_dir():
    return os.path.dirname(os.path.realpath(__file__))


def load_mnist_test():
    csv_path = os.path.join(_get_file_dir(), 'test.csv')
    csv = pd.read_csv(csv_path)

    return csv.values


def load_mnist_train(labels_encoding='original'):
    """

    :param labels_encoding: the encoding used for labels:
        - original: original MNIST values (each label is represented by a integer in the interval 0-9)
        - one-hot: each label is represented by a sparse array containing value 1 only for the correct label.
            example: 4 = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    :return: tuple (x, y), where x are the images and y the labels
    """
    csv_path = os.path.join(_get_file_dir(), 'train.csv')
    csv = pd.read_csv(csv_path)
    csv_values = csv.values

    x = csv_values[:, 1:]
    y = csv_values[:, 0]

    if labels_encoding == 'original':
        y_encoded = y
    elif labels_encoding == 'one-hot':
        y_encoded = np.zeros((y.shape[0], 10))
        y_encoded[range(y.shape[0]), y] = 1
    else:
        raise ValueError('Invalid value for labels_encoding: %s' % labels_encoding)

    return x, y_encoded
