import os
import tensorflow as tf
import numpy as np
import datasets
import nets

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_string('checkpoint_dir', 'work/tf', 'Directory to put the training data.')
flags.DEFINE_string('submission_out_dir', 'work', 'Directory to put the final submission .csv file.')


def load_mnist_dataset():
    x_test = datasets.load_mnist_test()
    return x_test


def evaluate(session, scores, images_placeholder, images):
    """Runs the model for the input images.
    Args:
      session: The tf session.
      scores: The forward operation, returning the class scores.
      images_placeholder: The images placeholder.
      images: The set of images to evaluate.

    Returns:
        the predicted class labels for the input images
    """
    feed_dict = {
        images_placeholder: images
    }

    scores = session.run(scores, feed_dict=feed_dict)
    predicted_labels = np.argmax(scores, axis=1)

    return predicted_labels


def save_predicted_labels(predicted_labels):
    predicted_labels_count = predicted_labels.shape[0]
    out = np.c_[range(1, predicted_labels_count + 1), predicted_labels]

    out_csv_path = os.path.join(FLAGS.submission_out_dir, 'results.csv')
    np.savetxt(out_csv_path, out,
               delimiter=',',
               header='ImageId,Label',
               comments='', fmt='%d')


def run_test():
    test_data = load_mnist_dataset()
    test_data_count = test_data.shape[0]
    predicted_labels = np.zeros(test_data_count)

    with tf.Graph().as_default():
        # build model graph
        net = nets.get_two_layer_net(200)
        placeholders, scores, loss, train = net.build_graph(FLAGS.batch_size)
        images_placeholder, labels_placeholder = placeholders

        # init session
        session = tf.Session()
        init = tf.initialize_all_variables()
        session.run(init)

        # load latest checkpoint
        latest_checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if latest_checkpoint_path is None:
            raise FileNotFoundError('Latest training checkpoint not found.')

        saver = tf.train.Saver()
        saver.restore(session, latest_checkpoint_path)

        # evaluate all test samples
        num_batches = test_data_count // FLAGS.batch_size
        for current_batch in range(num_batches):
            batch_from = current_batch * FLAGS.batch_size
            batch_to = (current_batch + 1) * FLAGS.batch_size
            images_batch = test_data[batch_from:batch_to]

            eval_result = evaluate(session, scores, images_placeholder, images_batch)
            predicted_labels[batch_from:batch_to] = eval_result

    save_predicted_labels(predicted_labels)


def main(_):
    run_test()


if __name__ == '__main__':
    tf.app.run()
