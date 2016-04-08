from datasets import IMAGE_SIZE, INPUT_SIZE, NUM_CLASSES


def get_two_layer_net(hidden_size):
    from nets.TwoLayerNet import TwoLayerNet
    return TwoLayerNet(INPUT_SIZE, hidden_size, NUM_CLASSES)


def get_two_layer_conv_net():
    from nets.TwoLayerConvNet import TwoLayerConvNet
    return TwoLayerConvNet(IMAGE_SIZE, NUM_CLASSES)
