import time
import tensorflow as tf
import numpy as np
import datasets
import nets


def load_mnist_dataset():
    x_train_all, y_train_all = datasets.load_mnist_train(labels_encoding='one-hot')
    x_test = datasets.load_mnist_test()

    num_training = int(x_train_all.shape[0] * 0.8)  # 20% for validation

    x_train = x_train_all[0:num_training]
    x_val = x_train_all[num_training:]
    y_train = y_train_all[0:num_training]
    y_val = y_train_all[num_training:]

    # Package data into a dictionary
    return {
        'train': (x_train, y_train),
        'validation': (x_val, y_val),
        'test': x_test
    }


# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_string('train_dir', 'tf_work/train', 'Directory to put the training data.')

data = load_mnist_dataset()


def fill_feed_dict(data_set, images_pl, labels_pl):
    """Fills the feed_dict for training the given step.
    A feed_dict takes the form of:
    feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ....
    }
    Args:
      data_set: The set of images and labels, from input_data.read_data_sets()
      images_pl: The images placeholder, from placeholder_inputs().
      labels_pl: The labels placeholder, from placeholder_inputs().
    Returns:
      feed_dict: The feed dictionary mapping from placeholders to values.
    """
    x, y = data_set
    mask = np.random.choice(x.shape[0], FLAGS.batch_size)
    x_batch = x[mask]
    y_batch = y[mask]

    feed_dict = {
        images_pl: x_batch,
        labels_pl: y_batch
    }
    return feed_dict


def do_eval(sess,
            forward_op,
            images_placeholder,
            labels_placeholder,
            data_set):
    """Runs one evaluation against the full epoch of data.
    Args:
      sess: The session in which the model has been trained.
      forward_op: The forward operation.
      images_placeholder: The images placeholder.
      labels_placeholder: The labels placeholder.
      data_set: The set of images and labels to evaluate.
    """
    x, y = data_set

    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = x.shape[0] // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size

    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,
                                   images_placeholder,
                                   labels_placeholder)
        logits = sess.run(forward_op, feed_dict=feed_dict)
        predicted_labels = np.argmax(logits, axis=1)

        true_labels = feed_dict[labels_placeholder]
        true_count += np.sum(true_labels[range(true_labels.shape[0]), predicted_labels])

    precision = true_count / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
          (num_examples, true_count, precision))


def run_training():
    print('run_training...')

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        net = nets.get_two_layer_net(100)

        # Generate placeholders for the images and labels.
        images_placeholder, labels_placeholder = net.get_input_placeholders(FLAGS.batch_size)

        # Build a Graph that computes predictions from the inference model.
        forward_op = net.forward(images_placeholder)

        # Add to the Graph the Ops for loss calculation.
        loss = net.loss(forward_op, labels_placeholder)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = net.training(loss, FLAGS.learning_rate)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        # Run the Op to initialize the variables.
        init = tf.initialize_all_variables()
        sess.run(init)

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph_def=sess.graph_def)

        # And then after everything is built, start the training loop.
        for step in range(FLAGS.max_steps):
            start_time = time.time()

            # Fill a feed dictionary with the actual set of images and labels
            # for this particular training step.
            feed_dict = fill_feed_dict(data['train'],
                                       images_placeholder,
                                       labels_placeholder)

            # Run one step of the model.  The return values are the activations
            # from the `train_op` (which is discarded) and the `loss` Op.  To
            # inspect the values of your Ops or variables, you may include them
            # in the list passed to sess.run() and the value tensors will be
            # returned in the tuple from the call.
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

            duration = time.time() - start_time

            # Write the summaries and print an overview fairly often.
            if step % 100 == 0:
                # Print status to stdout.
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                # Update the events file.
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                saver.save(sess, FLAGS.train_dir, global_step=step)
                # Evaluate against the training set.
                print('Training Data Eval:')
                do_eval(sess,
                        forward_op,
                        images_placeholder,
                        labels_placeholder,
                        data['train'])
                # Evaluate against the validation set.
                print('Validation Data Eval:')
                do_eval(sess,
                        forward_op,
                        images_placeholder,
                        labels_placeholder,
                        data['validation'])


def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
