import os
import numpy as np
import mxnet as mx
import datasets

batch_size = 100
load_model_prefix = 'work/model'
load_epoch = 3  # saved epoch train parameters to load
submission_out_dir = 'work'


def save_predicted_labels(predicted_labels):
    predicted_labels_count = predicted_labels.shape[0]
    out = np.c_[range(1, predicted_labels_count + 1), predicted_labels]

    out_csv_path = os.path.join(submission_out_dir, 'results.csv')
    np.savetxt(out_csv_path, out,
               delimiter=',',
               header='ImageId,Label',
               comments='', fmt='%d')


def save_submission_result():
    test_iter = datasets.load_test_dataset(batch_size, flat=False)

    model = mx.model.FeedForward.load(load_model_prefix, load_epoch, ctx=mx.cpu())
    y_predicted_onehot = model.predict(test_iter)
    y_predicted = np.argmax(y_predicted_onehot, axis=1)

    save_predicted_labels(y_predicted)


def run_validation():
    train_iter = datasets.load_train_dataset(batch_size, flat=False, split=False)
    true_labels = train_iter.label[0][1]

    model = mx.model.FeedForward.load(load_model_prefix, load_epoch, ctx=mx.cpu())
    predicted_labels_onehot = model.predict(train_iter)
    predicted_labels = np.argmax(predicted_labels_onehot, axis=1)

    accuracy = np.mean(predicted_labels == true_labels)
    print('Accuracy on entire train dataset: %f' % accuracy)

if __name__ == '__main__':
    # save_submission_result()
    run_validation()
