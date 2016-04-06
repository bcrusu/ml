import os
import pandas as pd
import numpy as np


def _get_file_dir():
    return os.path.dirname(os.path.realpath(__file__))


def _load_mnist_test():
    csv_path = os.path.join(_get_file_dir(), 'test.csv')
    csv = pd.read_csv(csv_path)

    return csv.values


def _load_mnist_train():
    csv_path = os.path.join(_get_file_dir(), 'train.csv')
    csv = pd.read_csv(csv_path)
    csv_values = csv.values

    x = csv_values[:, 1:]
    y = csv_values[:, 0]

    return x, y


def _write_uci_format(filename, data):
    path = os.path.join(_get_file_dir(), filename)
    np.savetxt(path, data, fmt='%u', delimiter='\t')


if __name__ == "__main__":
    x_train, y_train = _load_mnist_train()
    x_test = _load_mnist_test()

    _write_uci_format('train.uci', np.hstack((y_train[:, np.newaxis], x_train)))
    _write_uci_format('test.uci', x_test)

    print('done.')
