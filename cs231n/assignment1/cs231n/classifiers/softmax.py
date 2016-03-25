import numpy as np


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W)
        scores -= np.max(scores)  # numeric stability trick
        scores_exp = np.exp(scores)

        softmax_scores = scores_exp / np.sum(scores_exp)
        correct_softmax_scores = np.zeros(num_classes)
        correct_softmax_scores[y[i]] = 1

        # calculate loss
        loss -= np.log(softmax_scores[y[i]])

        # calculate gradient
        margins = correct_softmax_scores - softmax_scores
        for j in range(num_classes):
            dW[:, j] -= X[i, :] * margins[j]

    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)

    dW /= num_train
    dW += W * reg

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    num_train = X.shape[0]

    scores = X.dot(W)
    scores -= np.max(scores, axis=1, keepdims=True)  # numeric stability trick
    scores_exp = np.exp(scores)
    scores_exp_sum = np.sum(scores_exp, axis=1, keepdims=True)

    softmax_scores = scores_exp / scores_exp_sum
    correct_softmax_scores = np.zeros_like(scores)
    correct_softmax_scores[range(num_train), y] = 1

    # calculate loss
    loss = - np.sum(np.log(softmax_scores[range(num_train), y]))

    # calculate gradient
    margins = correct_softmax_scores - softmax_scores
    dW = - X.T.dot(margins)

    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)

    dW /= num_train
    dW += W * reg

    return loss, dW
