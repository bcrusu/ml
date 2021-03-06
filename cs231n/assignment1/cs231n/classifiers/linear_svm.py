import numpy as np

OF_SAME_SHAPE_AS_W_ = """
    Structured SVM loss function, naive implementation (with loops).

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


def svm_loss_naive(W, X, y, reg):
    OF_SAME_SHAPE_AS_W_
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin

                for xi in range(dW.shape[0]):
                    dW[xi, j] += X[i][xi]
                    dW[xi, y[i]] -= X[i][xi]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)

    dW /= num_train
    dW += W * reg

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss
    scores = X.dot(W)
    correct_class_scores = scores[range(num_train), y]

    margins = np.maximum(0, scores - correct_class_scores[:, np.newaxis] + 1)
    margins[range(num_train), y] = 0

    loss = np.sum(margins)
    loss /= num_train

    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)

    # compute the gradient
    # TODO: find better solution (no loops)
    for i in range(num_train):
        for j in range(num_classes):
            if margins[i, j] == 0:
                continue

            dW[:, j] += X[i][:]
            dW[:, y[i]] -= X[i][:]

    dW /= num_train
    dW += W * reg

    return loss, dW
