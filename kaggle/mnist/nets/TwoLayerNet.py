"""Builds the MNIST network.

Implements the inference/loss/training pattern for model building.

1. inference() - Builds the model as far as is required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.

This file is used by the various "fully_connected_*.py" files and not meant to
be run.
"""
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

    def inference(self, images):
        """
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

    def loss(self, logits, labels):
        """Calculates the loss from the logits and the labels.
        Args:
          logits: Logits tensor, float - [batch_size, NUM_CLASSES].
          labels: Labels tensor, int32 - [batch_size].
        Returns:
          loss: Loss tensor of type float.
        """
        # Convert from sparse integer labels in the range [0, NUM_CLASSES)
        # to 1-hot dense float vectors (that is we will have batch_size vectors,
        # each with NUM_CLASSES values, all of which are 0.0 except there will
        # be a 1.0 in the entry corresponding to the label).
        batch_size = tf.size(labels)
        labels = tf.expand_dims(labels, 1)
        indices = tf.expand_dims(tf.range(0, batch_size), 1)
        concated = tf.concat(1, [indices, labels])
        onehot_labels = tf.sparse_to_dense(concated, tf.pack([batch_size, self.output_size]), 1.0, 0.0)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, onehot_labels, name='xentropy')
        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        return loss

    def training(self, loss, learning_rate):
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
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        train_op = optimizer.minimize(loss, global_step=global_step)

        return train_op

    def evaluation(self, logits, labels):
        """Evaluate the quality of the logits at predicting the label.

        Args:
          logits: Logits tensor, float - [batch_size, NUM_CLASSES].
          labels: Labels tensor, int32 - [batch_size], with values in the
            range [0, NUM_CLASSES).

        Returns:
          A scalar int32 tensor with the number of examples (out of batch_size)
          that were predicted correctly.
        """
        # For a classifier model, we can use the in_top_k Op.
        # It returns a bool tensor with shape [batch_size] that is true for
        # the examples where the label's is was in the top k (here k=1)
        # of all logits for that example.
        correct = tf.nn.in_top_k(logits, labels, 1)

        # Return the number of true entries.
        return tf.reduce_sum(tf.cast(correct, tf.int32))

    def get_input_placeholders(self, batch_size):
        """Generate placeholder variables to represent the input tensors.
        These placeholders are used as inputs by the rest of the model building
        code and will be fed from the downloaded data in the .run() loop, below.
        Args:
          batch_size: The batch size will be baked into both placeholders.
        Returns:
          images_placeholder: Images placeholder.
          labels_placeholder: Labels placeholder.
        """
        # Note that the shapes of the placeholders match the shapes of the full
        # image and label tensors, except the first dimension is now batch_size
        # rather than the full size of the train or test data sets.
        images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, self.input_size))
        labels_placeholder = tf.placeholder(tf.int32, shape=batch_size)

        return images_placeholder, labels_placeholder
