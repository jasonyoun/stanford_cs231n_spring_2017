import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train): # N = 500
    scores = X[i].dot(W) # (1 x D) x (D x C) = (1 x C)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes): # C = 10
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        # Refer to the lecture note
        # 'http://cs231n.github.io/optimization-1/#gradcompute'
        dW[:, j] += (X[i, :]).T
        dW[:, y[i]] += -(X[i, :]).T

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  num_train = X.shape[0]

  # Choose the correct score for each training data.
  scores = X.dot(W) # (N x D) x (D x C) = (N x C)
  correct_scores = scores[(range(num_train),y)]

  # Get the loss after computing the margins.
  margins = np.maximum(0, scores - correct_scores.reshape(num_train, 1) + 1)
  margins[(range(num_train), y)] = 0
  loss = np.sum(margins) / num_train

  # Regularization.
  loss += reg * np.sum(W * W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  margin_fail_count_mask = np.zeros(margins.shape) # (N x C)

  # Count the number of classes that didn't meet the desired margin.
  margin_fail_count_mask[margins > 0] = 1
  margin_fail_count = np.sum(margin_fail_count_mask, axis=1) # (N, )

  # Fill in the mask with 'margin_fail_count' and compute the gradient.
  margin_fail_count_mask[(range(num_train), y)] = - margin_fail_count
  dW = X.T.dot(margin_fail_count_mask) / num_train

  # Regularization.
  dW += 2 * reg * W

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
