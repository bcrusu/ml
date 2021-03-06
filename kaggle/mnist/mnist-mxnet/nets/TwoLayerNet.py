import mxnet as mx


class TwoLayerNet:
    """ Two layer fully-connected net. Architecture:
     input - hidden layer - ReLU - output layer - softmax
    """

    def __init__(self, hidden_size, output_size, dropout_pct=0.5):
        """
        :param hidden_size: the size of hidden layer
        """

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_pct = dropout_pct

    def create_symbol(self):
        data = mx.symbol.Variable('data')

        # hidden layer
        fc1 = mx.symbol.FullyConnected(data=data, num_hidden=self.hidden_size, name='hidden_layer')
        bn = mx.symbol.BatchNorm(data=fc1)
        relu = mx.symbol.Activation(data=bn, act_type="relu")
        dropout = mx.symbol.Dropout(data=relu, p=self.dropout_pct)

        # output layer
        fc2 = mx.symbol.FullyConnected(data=dropout, num_hidden=self.output_size, name='output_layer')

        softmax = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
        return softmax
