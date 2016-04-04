import math
import tensorflow as tf


class TwoLayerConvNet:
    """ Two layer convolutional net. Architecture:
     input - (conv - relu - 2x2 max pool) x2 - fc1 - relu - fc2 - softmax
    """

    def __init__(self, image_size, output_size):
        self.image_size = image_size
        self.output_size = output_size

    def _get_he_init_std(self, fan_in):
        return math.sqrt(2.0 / fan_in)

    def _create_weight_variable(self, shape, init_std):
        init = tf.truncated_normal(shape, stddev=init_std)
        var = tf.Variable(initial_value=init, name='weights')
        return var

    def _create_conv_tensor(self, prev_tensor, scope_name, shape):
        filter_height, filter_width, in_channels, out_channels = shape

        with tf.name_scope(scope_name):
            w_init_std = self._get_he_init_std(in_channels * filter_height * filter_width)
            w = self._create_weight_variable(shape, w_init_std)

            b_init = tf.zeros([out_channels])
            b = tf.Variable(b_init, name='biases')

            conv2d = tf.nn.conv2d(prev_tensor, w, strides=[1, 1, 1, 1], padding='SAME')
            result = tf.nn.relu(conv2d + b)

            return result

    def _create_fc_tensor(self, prev_tensor, scope_name, shape, add_relu=True):
        in_dim, out_dim = shape

        with tf.name_scope(scope_name):
            w_init_std = self._get_he_init_std(in_dim)
            w = self._create_weight_variable(shape, w_init_std)

            b_init = tf.zeros([out_dim])
            b = tf.Variable(b_init, name='biases')

            result = tf.matmul(prev_tensor, w) + b
            if add_relu:
                result = tf.nn.relu(result)

            return result

    def _create_scores_tensor(self, images):
        images_reshaped = tf.reshape(images, [-1, self.image_size, self.image_size, 1])

        # 1st convolutional layer
        conv1 = self._create_conv_tensor(images_reshaped, 'conv1', (5, 5, 1, 32))
        max_pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # 2nd convolutional layer
        conv2 = self._create_conv_tensor(max_pool1, 'conv2', (5, 5, 32, 64))
        max_pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # 1st fully-connected layer
        fc1_in_dim = 7 * 7 * 64  # 28x28 halved twice
        fc1_out_dim = 1024
        max_pool2_flat = tf.reshape(max_pool2, [-1, fc1_in_dim])
        fc1 = self._create_fc_tensor(max_pool2_flat, 'fc1', (fc1_in_dim, fc1_out_dim), add_relu=True)

        # 2nd fully-connected layer
        fc2_out_dim = 10
        fc2 = self._create_fc_tensor(fc1, 'fc2', (fc1_out_dim, fc2_out_dim), add_relu=False)

        return fc2

    def _create_loss_tensor(self, scores, labels):
        """Calculates the loss from the scores and the labels.
        Args:
          scores: Scores tensor, float32 - [batch_size, NUM_CLASSES].
          labels: Labels tensor, float32 - [batch_size, NUM_CLASSES].
        Returns:
          loss: Loss tensor of type float.
        """
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(scores, labels, name='xentropy')
        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        return loss

    def _create_train_op(self, loss, learning_rate):
        """Sets up the training Ops.

        Creates a summarizer to track the loss over time in TensorBoard.

        Creates an optimizer and applies the gradients to all trainable variables.

        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train.

        Args:
          loss: Loss tensor, from loss().
          learning_rate: The learning rate to use for gradient descent.

        Returns:
          train_op: The Op for training.
        """
        # Add a scalar summary for the snapshot loss.
        tf.scalar_summary(loss.op.name, loss)

        # Create the gradient descent optimizer with the given learning rate.
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        train_op = optimizer.minimize(loss, global_step=global_step)

        return train_op

    def _create_input_placeholders(self, batch_size):
        """Generate placeholder variables to represent the input tensors.
        Args:
          batch_size: The batch size will be baked into both placeholders.
        Returns:
          images_placeholder: Images placeholder.
          labels_placeholder: Labels placeholder - one-hot encoded.
        """
        images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, self.image_size * self.image_size))
        labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size, self.output_size))

        return images_placeholder, labels_placeholder

    def build_graph(self, batch_size, learning_rate=1e-5):
        images_placeholder, labels_placeholder = self._create_input_placeholders(batch_size)

        scores = self._create_scores_tensor(images_placeholder)
        loss = self._create_loss_tensor(scores, labels_placeholder)
        train = self._create_train_op(loss, learning_rate)

        return (images_placeholder, labels_placeholder), scores, loss, train
