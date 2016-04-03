import math
import tensorflow as tf


class TwoLayerNet:
    """ Two layer fully-connected net. Architecture:
     input - hidden layer - ReLU - output layer - softmax
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        :param hidden_size: the size of hidden layer
        :param std: standard deviation used to initialize the weights
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def _create_scores_tensor(self, images):
        """Forward operation
        :param images: input placeholder
        :return: output logits tensor
        """
        with tf.name_scope('hidden_layer'):
            w_init_std = 1.0 / math.sqrt(float(self.input_size))
            w_init = tf.truncated_normal([self.input_size, self.hidden_size], stddev=w_init_std)
            w = tf.Variable(initial_value=w_init, name='weights')

            b_init = tf.zeros([self.hidden_size])
            b = tf.Variable(b_init, name='biases')

            hidden = tf.nn.relu(tf.matmul(images, w) + b)

        with tf.name_scope('output_layer'):
            w_init_std = 1.0 / math.sqrt(float(self.hidden_size))
            w_init = tf.truncated_normal([self.hidden_size, self.output_size], stddev=w_init_std)
            w = tf.Variable(initial_value=w_init, name='weights')

            b_init = tf.zeros([self.output_size])
            b = tf.Variable(b_init, name='biases')

            output = tf.matmul(hidden, w) + b

        return output

    def _create_loss_tensor(self, scores, labels):
        """Calculates the loss from the logits and the labels.
        Args:
          logits: Logits tensor, float32 - [batch_size, NUM_CLASSES].
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
        images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, self.input_size))
        labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size, self.output_size))

        return images_placeholder, labels_placeholder

    def build_graph(self, batch_size, learning_rate):
        images_placeholder, labels_placeholder = self._create_input_placeholders(batch_size)

        scores = self._create_scores_tensor(images_placeholder)
        loss = self._create_loss_tensor(scores, labels_placeholder)
        train = self._create_train_op(loss, learning_rate)

        return (images_placeholder, labels_placeholder), scores, loss, train
