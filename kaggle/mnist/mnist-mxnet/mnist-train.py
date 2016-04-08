import logging
import datasets
import nets
import mxnet as mx

learning_rate = 5e-4  # initial learning rate
batch_size = 100
num_epochs = 3
save_model_prefix = 'work/model'


def setup_logging():
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)


def run_training():
    # load MNIST data as NDArray iterators
    train_iter, val_iter, num_train = datasets.load_train_dataset(batch_size, flat=False, split=True)

    # create net
    network = nets.get_two_layer_conv_net()

    epoch_size = num_train / batch_size

    lr_factor = 0.85
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
