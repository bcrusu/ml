import os
import numpy as np
import mxnet as mx
import datasets

batch_size = 100
load_model_prefix = 'work/model_%d'
load_epoch = 5  # saved epoch train parameters to load
submission_out_dir = 'work'
train_accuracies = np.array([0.99871429, 0.99752381, 0.9975, 0.99488095, 0.99719048, 0.99611905,
                             0.99785714, 0.99633333, 0.99519048, 0.99519048])


def save_predicted_labels(predicted_labels):
    predicted_labels_count = predicted_labels.shape[0]
    out = np.c_[range(1, predicted_labels_count + 1), predicted_labels]

    out_csv_path = os.path.join(submission_out_dir, 'results.csv')
    np.savetxt(out_csv_path, out,
               delimiter=',',
               header='ImageId,Label',
               comments='', fmt='%d')


def save_submission_result():
    test_iter = datasets._load_test_csv()

    # take prediction in order of model train accuracies
    y_predicted = np.zeros(test_iter.shape[0])
    labels_by_accuracy = np.argsort(train_accuracies)
    for label in reversed(labels_by_accuracy):
        model = mx.model.FeedForward.load(load_model_prefix % label, load_epoch, ctx=mx.cpu())
        predict_onehot = model.predict(test_iter)
        predict = np.argmax(predict_onehot, axis=1) * (label + 1)

        y_predicted_zero = np.equal(y_predicted, 0)
        y_predicted[y_predicted_zero] = predict[y_predicted_zero]

    y_predicted_gt = np.greater(y_predicted, 0)
    y_predicted[y_predicted_gt] -= 1

    save_predicted_labels(y_predicted)


def run_validation():
    x_train, y_train = datasets._load_train_csv()

    # evaluate model for each label
    predict_results = {}
    for label in range(10):
        model = mx.model.FeedForward.load(load_model_prefix % label, load_epoch, ctx=mx.cpu())
        predict_onehot = model.predict(x_train)
        predict = np.argmax(predict_onehot, axis=1)
        predict_results[label] = predict * (label + 1)

    # take prediction in order of model train accuracies
    y_predicted = np.zeros_like(y_train)
    labels_by_accuracy = np.argsort(train_accuracies)
    for label in reversed(labels_by_accuracy):
        predict_result = predict_results[label]
        y_predicted_zero = np.equal(y_predicted, 0)
        y_predicted[y_predicted_zero] = predict_result[y_predicted_zero]

    y_predicted_gt = np.greater(y_predicted, 0)
    y_predicted[y_predicted_gt] -= 1

    accuracy = np.mean(y_predicted == y_train)
    print('Accuracy on entire train dataset: %f' % accuracy)


if __name__ == '__main__':
    save_submission_result()
    # run_validation()
