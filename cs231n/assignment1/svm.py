import time
import math
import itertools
import numpy as np
import matplotlib.pyplot as plt
from cs231n.data_utils import load_CIFAR10
from cs231n.classifiers.linear_svm import svm_loss_naive
from cs231n.classifiers.linear_svm import svm_loss_vectorized
from cs231n.classifiers import LinearSVM

# Global settings
plt.interactive(False)

# Load the raw CIFAR-10 data.
X_train, y_train, X_test, y_test = load_CIFAR10('./cs231n/datasets/cifar-10-batches-py')

# Split the data into train, val, and test sets. In addition we will
# create a small development set as a subset of the training data;
# we can use this for development so our code runs faster.
num_training = int(X_train.shape[0] * 0.9)
num_validation = X_train.shape[0] - num_training
num_test = 1000
num_dev = 1000

# Our validation set will be num_validation points from the original
# training set.
mask = range(num_training, num_training + num_validation)
X_val = X_train[mask]
y_val = y_train[mask]

# Our training set will be the first num_train points from the original
# training set.
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

# We will also make a development set, which is a small subset of
# the training set.
mask = np.random.choice(num_training, num_dev, replace=False)
X_dev = X_train[mask]
y_dev = y_train[mask]

# We use the first num_test points of the original test set as our
# test set.
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

# Preprocessing: reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

# Preprocessing: subtract the mean image
# first: compute the image mean based on the training data
mean_image = np.mean(X_train, axis=0)

# second: subtract the mean image from train and test data
X_train -= mean_image
X_val -= mean_image
X_test -= mean_image
X_dev -= mean_image

# third: append the bias dimension of ones (i.e. bias trick) so that our SVM
# only has to worry about optimizing a single weight matrix W.
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])


def test_svm_loss_naive():
    # generate a random SVM weight matrix of small numbers
    W = np.random.randn(3073, 10) * 0.0001

    # Compute the loss and its gradient at W.
    loss, grad = svm_loss_naive(W, X_dev, y_dev, 0.0)

    # Numerically compute the gradient along several randomly chosen dimensions, and
    # compare them with your analytically computed gradient. The numbers should match
    # almost exactly along all dimensions.
    from cs231n.gradient_check import grad_check_sparse
    f = lambda w: svm_loss_naive(w, X_dev, y_dev, 0.0)[0]
    grad_numerical = grad_check_sparse(f, W, grad)

    # do the gradient check once again with regularization turned on
    # you didn't forget the regularization gradient did you?
    loss, grad = svm_loss_naive(W, X_dev, y_dev, 1e2)
    f = lambda w: svm_loss_naive(w, X_dev, y_dev, 1e2)[0]
    grad_numerical = grad_check_sparse(f, W, grad)


def test_svm_loss_vectorized():
    # generate a random SVM weight matrix of small numbers
    W = np.random.randn(3073, 10) * 0.0001

    tic = time.time()
    loss_naive, grad_naive = svm_loss_naive(W, X_dev, y_dev, 0.00001)
    toc = time.time()
    print('Naive: %e computed in %fs' % (loss_naive, toc - tic))

    tic = time.time()
    loss_vectorized, grad_vectorized = svm_loss_vectorized(W, X_dev, y_dev, 0.00001)
    toc = time.time()
    print('Vectorized: %e computed in %fs' % (loss_vectorized, toc - tic))

    # The losses should match but your vectorized implementation should be much faster.
    print('Loss difference: %f' % (loss_naive - loss_vectorized))

    # The loss is a single number, so it is easy to compare the values computed
    # by the two implementations. The gradient on the other hand is a matrix, so
    # we use the Frobenius norm to compare them.
    difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
    print('Gradient difference: %f' % difference)


def test_stochastic_gradient_descent():
    svm = LinearSVM()
    tic = time.time()
    loss_hist = svm.train(X_train, y_train, learning_rate=1e-7, reg=5e4,
                          batch_size=32, num_iters=1000, verbose=True)
    toc = time.time()
    print('Train took %fs' % (toc - tic))

    # plot loss to iterations
    # plt.plot(loss_hist)
    # plt.xlabel('Iteration number')
    # plt.ylabel('Loss value')
    # plt.show()

    y_train_pred = svm.predict(X_train)
    print('training accuracy: %f' % (np.mean(y_train == y_train_pred),))
    y_val_pred = svm.predict(X_val)
    print('validation accuracy: %f' % (np.mean(y_val == y_val_pred),))


def hyperparameters_tuning():
    learning_rates = np.linspace(75e-9, 80e-9, 5)
    regularization_strengths = np.linspace(3e4, 5e4, 3)

    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the fraction
    # of data points that are correctly classified.
    results = {}
    best_val = -1    # The highest validation accuracy that we have seen so far.
    best_svm = None  # The LinearSVM object that achieved the highest validation rate.

    for learning_rate, regularization_strength in itertools.product(learning_rates, regularization_strengths):
        print("learning_rate = %e; regularization_strength = %e" % (learning_rate, regularization_strength))

        svm = LinearSVM()
        svm.train(X_train, y_train, learning_rate=learning_rate,
                  reg=regularization_strength, batch_size=32, num_iters=1000, verbose=True)

        y_train_pred = svm.predict(X_train)
        train_accuracy = np.mean(y_train == y_train_pred)
        y_val_pred = svm.predict(X_val)
        val_accuracy = np.mean(y_val == y_val_pred)

        results[(learning_rate, regularization_strength)] = (train_accuracy, val_accuracy)

        if val_accuracy > best_val:
            best_val = val_accuracy
            best_svm = svm

    # Print out results.
    for lr, reg in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
                    lr, reg, train_accuracy, val_accuracy))

    print('best validation accuracy achieved during cross-validation: %f' % best_val)

    # Visualize the cross-validation results
    x_scatter = [math.log10(x[0]) for x in results]
    y_scatter = [math.log10(x[1]) for x in results]

    # plot training accuracy
    marker_size = 100
    colors = [results[x][0] for x in results]
    plt.subplot(2, 1, 1)
    plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
    plt.colorbar()
    plt.xlabel('log learning rate')
    plt.ylabel('log regularization strength')
    plt.title('CIFAR-10 training accuracy')

    # plot validation accuracy
    colors = [results[x][1] for x in results] # default size of markers is 20
    plt.subplot(2, 1, 2)
    plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
    plt.colorbar()
    plt.xlabel('log learning rate')
    plt.ylabel('log regularization strength')
    plt.title('CIFAR-10 validation accuracy')
    plt.show()

    # Evaluate the best svm on test set
    y_test_pred = best_svm.predict(X_test)
    test_accuracy = np.mean(y_test == y_test_pred)
    print('linear SVM on raw pixels final test set accuracy: %f' % test_accuracy)

hyperparameters_tuning()
