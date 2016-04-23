import os
import numpy as np
import mxnet as mx
import datasets

batch_size = 200
load_model_prefix = 'work/model'
load_epoch = 3  # saved epoch train parameters to load
submission_out_dir = 'work'


def save_labels(file_name, labels):
    labels_count = labels.shape[0]
    out = np.c_[range(1, labels_count + 1), labels]

    out_csv_path = os.path.join(submission_out_dir, file_name)
    np.savetxt(out_csv_path, out, delimiter=',', comments='', fmt='%d')


def save_predicted_labels(predicted):
    predicted_labels_count = predicted.shape[0]
    out = np.c_[range(1, predicted_labels_count + 1), predicted]

    out_csv_path = os.path.join(submission_out_dir, 'results.csv')
    np.savetxt(out_csv_path, out,
               delimiter=',',
               header='ID,TARGET',
               comments='', fmt='%d')


def save_submission_result():
    test_iter = datasets.load_test_dataset(batch_size)

    model = mx.model.FeedForward.load(load_model_prefix, load_epoch, ctx=mx.cpu())
    y_predicted = model.predict(test_iter)

    save_predicted_labels(y_predicted)


def run_validation():
    train_iter = datasets.load_train_dataset(batch_size, split=False)
    true_labels = train_iter.label[0][1] == 1

    model = mx.model.FeedForward.load(load_model_prefix, load_epoch, ctx=mx.cpu())
    predicted_labels = model.predict(train_iter).ravel()

    predicted_labels_l = predicted_labels >= .5

    accuracy = np.mean(predicted_labels_l == true_labels)
    print('Accuracy on entire train dataset: %f' % accuracy)

    save_labels('val_true.csv', true_labels)
    save_labels('val_predicted.csv', predicted_labels_l)

if __name__ == '__main__':
    # save_submission_result()
    run_validation()
