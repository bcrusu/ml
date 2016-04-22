from datasets import NUM_CLASSES


def get_two_layer_net(hidden_size, output_size=NUM_CLASSES):
    from nets.TwoLayerNet import TwoLayerNet
    return TwoLayerNet(hidden_size, output_size)


def get_two_layer_conv_net(output_size=NUM_CLASSES):
    from nets.TwoLayerConvNet import TwoLayerConvNet
    return TwoLayerConvNet(output_size)
