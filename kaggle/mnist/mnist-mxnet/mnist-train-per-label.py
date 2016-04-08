import logging
import datasets
import nets
import mxnet as mx
import numpy as np

learning_rate = 1e-3  # initial learning rate
batch_size = 100
num_epochs = 3
save_model_prefix = 'work/model_%d'


def create_iterator_for_label(x_train_all, y_train_all, label):
    label_mask = np.equal(y_train_all, label)
    x_train_label = x_train_all[label_mask]

    nonlabel_mask = np.not_equal(y_train_all, label)
    x_train_nonlabel = x_train_all[nonlabel_mask]

    x_train_all = np.vstack((x_train_label, x_train_nonlabel))
    y_train_all = np.hstack((np.ones(x_train_label.shape[0]), np.full(x_train_nonlabel.shape[0], -1)))

    all_iter = mx.io.NDArrayIter(data=x_train_all, label=y_train_all, batch_size=batch_size, shuffle=True)
    return all_iter


def setup_logging():
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)


def run_training_for_label(train_iterator, label):
    # create net
    network = nets.get_two_layer_net(600, 2)

    epoch_size = train_iterator.data[0][1].shape[0] / batch_size

    lr_factor = 0.9
    lr_factor_epoch = 1
    lr_scheduler = mx.lr_scheduler.FactorScheduler(step=max(int(epoch_size * lr_factor_epoch), 1), factor=lr_factor)
    optimizer = mx.optimizer.Adam(learning_rate=learning_rate, wd=0.5, lr_scheduler=lr_scheduler)

    # He init style
    initializer = mx.init.Xavier(rnd_type="gaussian", factor_type="in", magnitude=2)

    model = mx.model.FeedForward(
        symbol=network.create_symbol(),
        ctx=mx.cpu(),
        num_epoch=num_epochs,
        initializer=initializer,
        optimizer=optimizer)

    batch_end_callback = [mx.callback.Speedometer(batch_size, 50)]
    epoch_end_callback = [mx.callback.do_checkpoint(save_model_prefix % label)]

    print('Model fit running for label %d...' % label)
    model.fit(
        X=train_iterator,
        kvstore=None,  # not required when running on CPU
        batch_end_callback=batch_end_callback,
        epoch_end_callback=epoch_end_callback)

    # predict and calculate train accuracy
    true_labels = train_iterator.label[0][1]
    predicted_labels_onehot = model.predict(train_iterator)
    predicted_labels = np.argmax(predicted_labels_onehot, axis=1)

    accuracy = np.mean(predicted_labels == true_labels)
    print('Accuracy on entire train dataset: %f' % accuracy)

    return accuracy


def run_training():
    x_train_all, y_train_all = datasets._load_train_csv()

    per_label_accuracies = np.zeros(10)
    for label in range(10):
        train_iterator = create_iterator_for_label(x_train_all, y_train_all, label)
        accuracy = run_training_for_label(train_iterator, label)
        per_label_accuracies[label] = accuracy

    print('Per-label accuracies:')
    print(per_label_accuracies)

if __name__ == '__main__':
    setup_logging()
    run_training()
