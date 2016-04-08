import logging
import datasets
import nets
import mxnet as mx

learning_rate = 5e-4  # initial learning rate
batch_size = 100
num_epochs = 15
save_model_prefix = 'work/model'


def load_mnist_dataset():
    x_train_all, y_train_all = datasets.load_mnist_train()

    num_training = int(x_train_all.shape[0] * 0.8)  # 20% for validation

    x_train = x_train_all[0:num_training]
    x_val = x_train_all[num_training:]
    y_train = y_train_all[0:num_training]
    y_val = y_train_all[num_training:]

    train_iter = mx.io.NDArrayIter(data=x_train, label=y_train, batch_size=batch_size, shuffle=False)
    val_iter = mx.io.NDArrayIter(data=x_val, label=y_val, batch_size=batch_size, shuffle=False)

    return train_iter, val_iter, x_train.shape[0]


def setup_logging():
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)


def run_training():
    # load MNIST data as NDArray iterators
    train_iter, val_iter, num_train = load_mnist_dataset()

    # create net
    network = nets.get_two_layer_net(250)

    epoch_size = num_train / batch_size

    lr_factor = 0.9
    lr_factor_epoch = 1
    lr_scheduler = mx.lr_scheduler.FactorScheduler(step=max(int(epoch_size * lr_factor_epoch), 1), factor=lr_factor)
    optimizer = mx.optimizer.Adam(learning_rate=learning_rate, wd=0.00001, lr_scheduler=lr_scheduler)

    # He init style
    initializer = mx.init.Xavier(rnd_type="gaussian", factor_type="in", magnitude=2)

    model = mx.model.FeedForward(
        symbol=network.create_symbol(),
        ctx=mx.cpu(),
        num_epoch=num_epochs,
        initializer=initializer,
        optimizer=optimizer)

    eval_metrics = ['accuracy']
    batch_end_callback = [mx.callback.Speedometer(batch_size, 50)]
    epoch_end_callback = [mx.callback.do_checkpoint(save_model_prefix)]

    print('Model fit running...')
    model.fit(
        X=train_iter,
        eval_data=val_iter,
        eval_metric=eval_metrics,
        kvstore=None,  # not required when running on CPU
        batch_end_callback=batch_end_callback,
        epoch_end_callback=epoch_end_callback)


if __name__ == '__main__':
    setup_logging()
    run_training()
