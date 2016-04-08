import os
import numpy as np
import mxnet as mx
import datasets

batch_size = 100
load_model_prefix = 'work/model'
load_epoch = 15  # saved epoch train parameters to load
submission_out_dir = 'work'


def load_mnist_dataset():
    x_test = datasets.load_mnist_test()
    test_iter = mx.io.NDArrayIter(data=x_test, batch_size=batch_size, shuffle=False)

    return test_iter


def save_predicted_labels(predicted_labels):
    predicted_labels_count = predicted_labels.shape[0]
    out = np.c_[range(1, predicted_labels_count + 1), predicted_labels]

    out_csv_path = os.path.join(submission_out_dir, 'results.csv')
    np.savetxt(out_csv_path, out,
               delimiter=',',
               header='ImageId,Label',
               comments='', fmt='%d')


def run_test():
    test_iter = load_mnist_dataset()

    model = mx.model.FeedForward.load(load_model_prefix, load_epoch, ctx=mx.cpu())
    y_predicted_onehot = model.predict(test_iter)
    y_predicted = np.argmax(y_predicted_onehot, axis=1)

    save_predicted_labels(y_predicted)


if __name__ == '__main__':
    run_test()
