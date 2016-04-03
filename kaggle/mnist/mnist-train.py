import time
import tensorflow as tf
import numpy as np
import datasets
import nets

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_string('train_dir', 'work/tf/train', 'Directory to put the training data.')


def load_mnist_dataset():
    x_train_all, y_train_all = datasets.load_mnist_train(labels_encoding='one-hot')

    num_training = int(x_train_all.shape[0] * 0.8)  # 20% for validation

    x_train = x_train_all[0:num_training]
    x_val = x_train_all[num_training:]
    y_train = y_train_all[0:num_training]
    y_val = y_train_all[num_training:]

    # Package data into a dictionary
    return {
        'train': (x_train, y_train),
        'validation': (x_val, y_val)
    }


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


def do_eval(sess, forward, images_placeholder, labels_placeholder, data_set):
    """Runs one evaluation against the full epoch of data.
    Args:
      sess: The session in which the model has been trained.
      forward: The forward operation.
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
        feed_dict = fill_feed_dict(data_set, images_placeholder, labels_placeholder)

        scores = sess.run(forward, feed_dict=feed_dict)
        predicted_labels = np.argmax(scores, axis=1)

        true_labels = feed_dict[labels_placeholder]
        true_count += np.sum(true_labels[range(true_labels.shape[0]), predicted_labels])

    precision = true_count / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' % (num_examples, true_count, precision))


def run_training():
    data = load_mnist_dataset()

    with tf.Graph().as_default():
        net = nets.get_two_layer_conv_net()
        placeholders, scores, loss, train = net.build_graph(FLAGS.batch_size, FLAGS.learning_rate)
        images_placeholder, labels_placeholder = placeholders

        summary = tf.merge_all_summaries()
        saver = tf.train.Saver()

        session = tf.Session()
        init = tf.initialize_all_variables()
        session.run(init)

        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph_def=session.graph_def)

        # Start the training loop.
        for step in range(FLAGS.max_steps):
            start_time = time.time()

            # Fill a feed dictionary with an train images and labels batch
            feed_dict = fill_feed_dict(data['train'], images_placeholder, labels_placeholder)

            # Run one forward and backward pass steps of the model.
            _, loss_value = session.run([train, loss], feed_dict=feed_dict)

            duration = time.time() - start_time

            # Write the summaries and print an loss overview.
            if step % 100 == 0:
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))

                summary_str = session.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                saver.save(session, FLAGS.train_dir, global_step=step)

                print('Training Data Eval:')
                do_eval(session, scores, images_placeholder, labels_placeholder, data['train'])

                print('Validation Data Eval:')
                do_eval(session, scores, images_placeholder, labels_placeholder, data['validation'])


def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
