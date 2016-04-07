import os
import numpy as np
import datasets
import nets

batch_size = 100
checkpoint_dir = 'work'
submission_out_dir = 'work'


def load_mnist_dataset():
    x_test = datasets.load_mnist_test()
    return x_test


def save_predicted_labels(predicted_labels):
    predicted_labels_count = predicted_labels.shape[0]
    out = np.c_[range(1, predicted_labels_count + 1), predicted_labels]

    out_csv_path = os.path.join(submission_out_dir, 'results.csv')
    np.savetxt(out_csv_path, out,
               delimiter=',',
               header='ImageId,Label',
               comments='', fmt='%d')


def run_test():
    test_data = load_mnist_dataset()
    test_data_count = test_data.shape[0]
    predicted_labels = np.zeros(test_data_count)

    # TODO: run model on test data

    save_predicted_labels(predicted_labels)


if __name__ == '__main__':
    run_test()
