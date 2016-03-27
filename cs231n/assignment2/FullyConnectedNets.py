import time
import math
import itertools
import numpy as np
import matplotlib.pyplot as plt
from cs231n.data_utils import get_CIFAR10_data

# Global settings
plt.interactive(False)


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


# load CIFAR-10 data
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()

