import mxnet as mx


class TwoLayerConvNet:
    """ Two layer convolutional net. Architecture:
     input - (conv - batch_norm - relu - dropout - 2x2 max pool) x2 - fc1 - batch_norm - relu - dropout - fc2 - softmax
    """

    def __init__(self, input_size, output_size, dropout_pct=0.25):
        self.input_size = input_size
        self.output_size = output_size
        self.dropout_pct = dropout_pct

    def create_symbol(self):
        data = mx.symbol.Variable('data')

        # first convolution layer
        conv1 = mx.symbol.Convolution(data=data, kernel=(5, 5), num_filter=32)
        bn1 = mx.symbol.BatchNorm(data=conv1)
        relu1 = mx.symbol.Activation(data=bn1, act_type="relu")
        dropout1 = mx.symbol.Dropout(data=relu1, p=self.dropout_pct)
        pool1 = mx.symbol.Pooling(data=dropout1, pool_type="max", kernel=(2, 2), stride=(2, 2))

        # second convolution layer
        conv2 = mx.symbol.Convolution(data=pool1, kernel=(5, 5), num_filter=64)
        bn2 = mx.symbol.BatchNorm(data=conv2)
        relu2 = mx.symbol.Activation(data=bn2, act_type="relu")
        dropout2 = mx.symbol.Dropout(data=relu2, p=self.dropout_pct)
        pool2 = mx.symbol.Pooling(data=dropout2, pool_type="max", kernel=(2, 2), stride=(2, 2))

        flatten = mx.symbol.Flatten(data=pool2)

        # first full-connected layer
        fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=256)
        bn3 = mx.symbol.BatchNorm(data=fc1)
        relu3 = mx.symbol.Activation(data=bn3, act_type="relu")
        dropout3 = mx.symbol.Dropout(data=relu3, p=self.dropout_pct)

        # second full-connected layer
        fc2 = mx.symbol.FullyConnected(data=dropout3, num_hidden=128)
        bn4 = mx.symbol.BatchNorm(data=fc2)
        relu4 = mx.symbol.Activation(data=bn4, act_type="relu")
        dropout4 = mx.symbol.Dropout(data=relu4, p=self.dropout_pct)

        # third full-connected layer
        fc2 = mx.symbol.FullyConnected(data=dropout4, num_hidden=self.output_size)

        softmax = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
        return softmax
