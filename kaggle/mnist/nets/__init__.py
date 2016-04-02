IMAGE_SIZE = 28  # MNIST images size is 28x28 pixels

INPUT_SIZE = IMAGE_SIZE * IMAGE_SIZE
NUM_CLASSES = 10  # each MINST image represents a digit


def get_two_layer_net(hidden_size):
    from nets.TwoLayerNet import TwoLayerNet
    return TwoLayerNet(INPUT_SIZE, hidden_size, NUM_CLASSES)
