import time
import math
import itertools
import numpy as np
import matplotlib.pyplot as plt
from cs231n.data_utils import load_CIFAR10
from cs231n.classifiers.neural_net import TwoLayerNet
from cs231n.vis_utils import visualize_grid

# Global settings
plt.interactive(False)


def get_CIFAR10_data(num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = './cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    num_training = int(X_train.shape[0] * 0.9)
    num_validation = X_train.shape[0] - num_training  # 10% for validation

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Reshape data to rows
    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    return X_train, y_train, X_val, y_val, X_test, y_test


# load CIFAR-10 data
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()


def show_net_weights(net):
    W1 = net.params['W1']
    W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.gca().axis('off')
    plt.show()


def test_train():
    input_size = 32 * 32 * 3
    hidden_size = 50
    num_classes = 10
    net = TwoLayerNet(input_size, hidden_size, num_classes)

    # Train the network
    stats = net.train(X_train, y_train, X_val, y_val,
                      num_iters=1000, batch_size=200,
                      learning_rate=1e-4, learning_rate_decay=0.95,
                      reg=0.5, verbose=True)

    # Predict on the validation set
    val_acc = (net.predict(X_val) == y_val).mean()
    print('Validation accuracy: ', val_acc)

    # Plot the loss function and train / validation accuracies
    plt.subplot(2, 1, 1)
    plt.plot(stats['loss_history'])
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(stats['train_acc_history'], label='train')
    plt.plot(stats['val_acc_history'], label='val')
    plt.title('Classification accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Clasification accuracy')
    plt.show()

    # Visualize the weights of the network
    show_net_weights(net)


def hyperparameters_tuning():
    input_size = 32 * 32 * 3
    num_classes = 10

    results = {}
    best_val = -1
    best_net = None

    max_runs = 5
    for _ in range(max_runs):
        learning_rate = 10**np.random.uniform(-3.454, -3.453)
        regularization_strength = 10**np.random.uniform(0.332, 0.335)
        hidden_size = 50

        print("hidden_size: %d learning_rate_log10: %e; regularization_strength_log10: %e" %
              (hidden_size, np.log10(learning_rate), np.log10(regularization_strength)))

        net = TwoLayerNet(input_size, hidden_size, num_classes)
        # Train the network
        stats = net.train(X_train, y_train, X_val, y_val, num_iters=2000, batch_size=100,
                          learning_rate=learning_rate, learning_rate_decay=0.95,
                          reg=regularization_strength, verbose=True)

        y_train_pred = net.predict(X_train)
        train_accuracy = np.mean(y_train == y_train_pred)
        y_val_pred = net.predict(X_val)
        val_accuracy = np.mean(y_val == y_val_pred)

        results[(hidden_size, learning_rate, regularization_strength)] = (train_accuracy, val_accuracy)

        if val_accuracy > best_val:
            best_val = val_accuracy
            best_net = net

    # Print out results.
    for hidden_size, lr, reg in sorted(results):
        train_accuracy, val_accuracy = results[(hidden_size, lr, reg)]
        print('hidden_size: %d lr: %e reg: %e train accuracy: %f val accuracy: %f' % (
            hidden_size, lr, reg, train_accuracy, val_accuracy))

    print('best validation accuracy achieved during cross-validation: %f' % best_val)

    # Evaluate the best net on test set
    y_test_pred = best_net.predict(X_test)
    test_accuracy = np.mean(y_test == y_test_pred)
    print('final test set accuracy: %f' % (test_accuracy,))

    # visualize the weights of the best network
    show_net_weights(best_net)

hyperparameters_tuning()
