import os
import pandas as pd
import numpy as np
import mxnet as mx

IMAGE_SIZE = 96
INPUT_SIZE = IMAGE_SIZE * IMAGE_SIZE
FEATURE_NO = 15

_train_dataset = None
_test_dataset = None


def _get_file_dir():
    return os.path.dirname(os.path.realpath(__file__))


def _str_to_int_array(str_array):
    result = []
    for str in str_array:
        result.append(np.fromstring(str, dtype=np.int8, sep=' '))

    return np.vstack(result)


def _load_test_csv():
    global _test_dataset
    if _test_dataset is None:
        csv_path = os.path.join(_get_file_dir(), 'test.csv')
        csv = pd.read_csv(csv_path)

        pixels = _str_to_int_array(csv.values[:, 1])
        _test_dataset = pixels

    return _test_dataset


def _load_train_csv():
    global _train_dataset
    if _train_dataset is None:
        csv_path = os.path.join(_get_file_dir(), 'training.csv')
        csv = pd.read_csv(csv_path)

        points = np.array(csv.values[:, :30], dtype=np.float32)
        pixels = _str_to_int_array(csv.values[:, 30])
        _train_dataset = (points, pixels)

    return _train_dataset


def load_train_dataset(batch_size, feature_no=0, flat=True, split=True, validation_ratio=0.2):
    """
    :type feature_no: int
         0 = left_eye_center
         1 = right_eye_center
         2 = left_eye_inner_corner
         3 = left_eye_outer_corner
         4 = right_eye_inner_corner
         5 = right_eye_outer_corner
         6 = left_eyebrow_inner_end
         7 = left_eyebrow_outer_end
         8 = right_eyebrow_inner_end
         9 = right_eyebrow_outer_end
        10 = nose_tip
        11 = mouth_left_corner
        12 = mouth_right_corner
        13 = mouth_center_top_lip
        14 = mouth_center_bottom_lip
    """
    points, pixels = _load_train_csv()

    # select (x,y) pairs for feature_no
    points = points[:, feature_no * 2: feature_no * 2 + 2]

    # remove samples where feature is missing
    non_nan_mask = np.equal(np.sum(np.isnan(points), axis=1), 0)
    points = points[non_nan_mask]
    pixels = pixels[non_nan_mask]

    if not flat:
        pixels = pixels.reshape(pixels.shape[0], 1, IMAGE_SIZE, IMAGE_SIZE)

    if not split:
        train_iter = mx.io.NDArrayIter(data=pixels, label=points, batch_size=batch_size, shuffle=False)
        return train_iter

    # split into train/validation according to validation_ratio
    num_training = int(pixels.shape[0] * (1.0 - validation_ratio))
    pixels_train = pixels[0:num_training]
    pixels_val = pixels[num_training:]
    points_train = points[0:num_training]
    points_val = points[num_training:]

    train_iter = mx.io.NDArrayIter(data=pixels_train, label=points_train, batch_size=batch_size, shuffle=False)
    val_iter = mx.io.NDArrayIter(data=pixels_val, label=points_val, batch_size=batch_size, shuffle=False)
    return train_iter, val_iter


def load_test_dataset(batch_size, flat=True):
    pixels = _load_test_csv()

    if not flat:
        pixels = pixels.reshape(pixels.shape[0], 1, IMAGE_SIZE, IMAGE_SIZE)

    test_iter = mx.io.NDArrayIter(data=pixels, batch_size=batch_size, shuffle=False)
    return test_iter
