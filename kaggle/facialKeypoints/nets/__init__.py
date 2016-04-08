from datasets import IMAGE_SIZE, INPUT_SIZE


def get_two_layer_fc_net(hidden_size):
    from nets.TwoLayerNet import TwoLayerNet
    return TwoLayerNet(INPUT_SIZE, hidden_size, 2)


def get_two_layer_conv_net(output_size=2):
    pass
    # from nets.TwoLayerConvNet import TwoLayerConvNet
    # return TwoLayerConvNet(IMAGE_SIZE, output_size)
