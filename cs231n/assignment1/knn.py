import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt
from cs231n.classifiers import KNearestNeighbor

# Global settings
plt.interactive(False)

# Load the raw CIFAR-10 data.
X_train, y_train, X_test, y_test = load_CIFAR10('./cs231n/datasets/cifar-10-batches-py')

# Subsample the data for more efficient code execution in this exercise
num_training = 500
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 100
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
# print(X_train.shape, X_test.shape)


def visualize_random():
    # Visualize some examples from the dataset.
    # We show a few examples of training images from each class.
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    samples_per_class = 7
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(y_train == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X_train[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.show()


def test():
    classifier = KNearestNeighbor()
    classifier.train(X_train, y_train)

    dists = classifier.compute_distances_two_loops(X_test)
    # print(dists.shape)
    # plt.imshow(dists, interpolation='none')
    # plt.show()

    y_test_pred = classifier.predict_labels(dists, k=1)

    # Compute and print the fraction of correctly predicted examples
    # k=1
    num_correct = np.sum(y_test_pred == y_test)
    accuracy = float(num_correct) / num_test
    print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))

    # k=5
    y_test_pred = classifier.predict_labels(dists, k=5)
    num_correct = np.sum(y_test_pred == y_test)
    accuracy = float(num_correct) / num_test
    print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))


def cross_validation():
    num_folds = 5
    k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

    X_train_folds = np.array(np.array_split(X_train, num_folds))
    y_train_folds = np.array(np.array_split(y_train, num_folds))

    # A dictionary holding the accuracies for different values of k that we find
    # when running cross-validation. After running cross-validation,
    # k_to_accuracies[k] should be a list of length num_folds giving the different
    # accuracy values that we found when using that value of k.
    k_to_accuracies = {}

    for k in k_choices:
        k_to_accuracies[k] = np.zeros(num_folds)
        for fold in range(num_folds):
            train_folds = np.where(np.array(range(num_folds)) != fold)
            X_train_current = np.vstack(X_train_folds[train_folds])
            X_validate_current = X_train_folds[fold]
            y_train_current = np.hstack(y_train_folds[train_folds])
            y_validate_current = y_train_folds[fold]

            classifier = KNearestNeighbor()
            classifier.train(X_train_current, y_train_current)
            y_predicted = classifier.predict(X_validate_current, k)

            num_correct = np.sum(y_predicted == y_validate_current)
            accuracy = float(num_correct) / num_test

            k_to_accuracies[k][fold] = accuracy

    # Print out the computed accuracies
    for k in sorted(k_to_accuracies):
        for accuracy in k_to_accuracies[k]:
            print('k = %d, accuracy = %f' % (k, accuracy))


cross_validation()
