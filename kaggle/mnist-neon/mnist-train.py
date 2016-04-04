from neon.initializers import Gaussian
from neon.optimizers import GradientDescentMomentum, Schedule
from neon.layers import Conv, Pooling, GeneralizedCost, Affine
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti, Misclassification
from neon.models import Model
from neon.data import ArrayIterator
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser
import datasets


def load_mnist_dataset():
    x_train_all, y_train_all = datasets.load_mnist_train()

    num_training = int(x_train_all.shape[0] * 0.8)  # 20% for validation

    x_train = x_train_all[0:num_training]
    x_val = x_train_all[num_training:]
    y_train = y_train_all[0:num_training]
    y_val = y_train_all[num_training:]

    lshape = (1, 28, 28)
    return {
        'train': (x_train, y_train),
        'validation': (x_val, y_val),
        'train_iterator': ArrayIterator(x_train, y_train, nclass=10, lshape=lshape),
        'validation_iterator': ArrayIterator(x_val, y_val, nclass=10, lshape=lshape)
    }


def run_training():
    # init neon backend
    parser = NeonArgparser(__doc__)
    args = parser.parse_args()

    # load data-set
    dataset = load_mnist_dataset()
    train_set = dataset['train_iterator']
    valid_set = dataset['validation_iterator']

    init_norm = Gaussian(loc=0.0, scale=0.01)
    opt_gdm = GradientDescentMomentum(learning_rate=1e-4, momentum_coef=0.9, wdecay=0.95,
                                      schedule=Schedule(step_config=[200, 250, 300], change=0.1))

    conv_params = dict(init=init_norm, batch_norm=False, activation=Rectlin(), padding=1)

    layers = [Conv((5, 5, 32), **conv_params),
              Pooling(2, op='max'),
              Conv((5, 5, 64), **conv_params),
              Pooling(2, op='max'),
              Affine(nout=1024, init=init_norm, activation=Rectlin()),
              Affine(nout=10, init=init_norm, activation=Softmax())]

    cost = GeneralizedCost(costfunc=CrossEntropyMulti())
    net = Model(layers=layers)

    # configure callbacks
    callbacks = Callbacks(net, eval_set=valid_set, **args.callback_args)

    net.fit(train_set, optimizer=opt_gdm, num_epochs=10, cost=cost, callbacks=callbacks)
    print('Misclassification error = %.1f%%' % (net.eval(valid_set, metric=Misclassification()) * 100))


if __name__ == '__main__':
    run_training()
