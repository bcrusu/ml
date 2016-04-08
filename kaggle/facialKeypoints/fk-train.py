import logging
import datasets
import nets
import mxnet as mx

learning_rate = 5e-3  # initial learning rate
batch_size = 100
num_epochs = 5
save_model_prefix = 'work/model-f%0.2d'


def setup_logging():
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)


def run_training_for_feature(feature_no):
    train_iter, val_iter = datasets.load_train_dataset(batch_size, feature_no=feature_no, flat=True, split=True)

    # create net
    network = nets.get_two_layer_fc_net(1024)

    epoch_size = train_iter.data[0][1].shape[0] / batch_size

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

    eval_metrics = ['rmse']
    batch_end_callback = [mx.callback.Speedometer(batch_size, 10)]
    epoch_end_callback = [mx.callback.do_checkpoint(save_model_prefix % feature_no)]

    model.fit(
        X=train_iter,
        eval_data=val_iter,
        eval_metric=eval_metrics,
        kvstore=None,  # not required when running on CPU
        batch_end_callback=batch_end_callback,
        epoch_end_callback=epoch_end_callback)


def run_training():
    for feature_no in range(datasets.FEATURE_NO):
        print('Running training for feature no. %d...' % feature_no)
        run_training_for_feature(feature_no)

        # TODO: remove below
        break


if __name__ == '__main__':
    setup_logging()
    run_training()
