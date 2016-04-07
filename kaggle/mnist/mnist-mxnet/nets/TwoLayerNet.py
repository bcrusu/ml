import mxnet as mx


class TwoLayerNet:
    """ Two layer fully-connected net. Architecture:
     input - hidden layer - ReLU - output layer - softmax
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        :param hidden_size: the size of hidden layer
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def create_symbol(self):
        data = mx.symbol.Variable('data')

        # hidden layer
        fc1 = mx.symbol.FullyConnected(data=data, num_hidden=self.hidden_size, name='hidden_layer')
        relu = mx.symbol.Activation(data=fc1, act_type="relu")
        dropout = mx.symbol.Dropout(data=relu, p=0.25)

        # output layer
        fc2 = mx.symbol.FullyConnected(data=dropout, num_hidden=self.output_size, name='output_layer')

        softmax = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
        return softmax
