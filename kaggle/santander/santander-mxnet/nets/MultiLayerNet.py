import mxnet as mx


class MultiLayerNet:
    """ Two layer fully-connected net. Architecture:
     input - hidden layer - bach_norm - ReLU - dropout - output layer - logit
    """

    def __init__(self, layer_sizes, dropout_pct=0.5):
        self.layer_sizes = layer_sizes
        self.dropout_pct = dropout_pct

    def _create_layer(self, size, name, prev_layer):
        fc = mx.symbol.FullyConnected(data=prev_layer, num_hidden=size, name=name)
        bn = mx.symbol.BatchNorm(data=fc)
        relu = mx.symbol.Activation(data=bn, act_type="relu")
        #dropout = mx.symbol.Dropout(data=relu, p=self.dropout_pct)
        return relu

    def create_symbol(self):
        data = mx.symbol.Variable('data')

        prev_layer = data
        for index, layer_size in enumerate(self.layer_sizes):
            name = 'hidden_layer' + str(index + 1)
            layer = self._create_layer(layer_size, name, prev_layer)
            prev_layer = layer

        # output layer
        prev_layer = mx.symbol.FullyConnected(data=prev_layer, num_hidden=1, name='output_layer')

        net = mx.symbol.LogisticRegressionOutput(data=prev_layer, name='softmax')
        return net
