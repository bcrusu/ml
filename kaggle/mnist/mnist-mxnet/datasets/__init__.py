import os
import pandas as pd
import numpy as np
import mxnet as mx
import scipy.misc

NUM_CLASSES = 10  # each MINST image represents a digit

_original_image_size = 28


def _get_file_dir():
    return os.path.dirname(os.path.realpath(__file__))


def _load_test_csv():
    csv_path = os.path.join(_get_file_dir(), 'test.csv')
    csv = pd.read_csv(csv_path)

    return csv.values


def _load_train_csv(labels_encoding='original'):
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
        y_encoded = np.zeros((y.shape[0], 10), dtype=np.float32)
        y_encoded[range(y.shape[0]), y] = 1.0
    else:
        raise ValueError('Invalid value for labels_encoding: %s' % labels_encoding)

    return x, y_encoded


def _resize_images(images, resize_ratio):
    results = []
    for image in images:
        image_2d = image.reshape((_original_image_size, _original_image_size))
        image_sized = scipy.misc.imresize(image_2d, resize_ratio, interp='bilinear')
        results.append(image_sized.reshape(-1))

    return np.vstack(results)


def load_train_dataset(batch_size, flat=True, split=True, validation_ratio=0.2,
                       resize=False, resize_ratio=0.5):
    x_train_all, y_train_all = _load_train_csv()
    image_size = _original_image_size

    if resize:
        x_train_all = _resize_images(x_train_all, resize_ratio)
        image_size = int(image_size * resize_ratio)

    if not flat:
        x_train_all = x_train_all.reshape((x_train_all.shape[0], 1, image_size, image_size))

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


def load_test_dataset(batch_size, flat=True, resize=False, resize_ratio=0.5):
    x_test = _load_test_csv()
    image_size = _original_image_size

    if resize:
        x_test = _resize_images(x_test, resize_ratio)
        image_size = int(image_size * resize_ratio)

    if not flat:
        x_test = x_test.reshape(x_test.shape[0], 1, image_size, image_size)

    test_iter = mx.io.NDArrayIter(data=x_test, batch_size=batch_size, shuffle=False)
    return test_iter
