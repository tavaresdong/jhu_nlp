#!/usr/bin/env python

import numpy as np
import random

from utils import softmax, sigmoid, sigmoid_grad, gradcheck_naive

def forward_backward_prop(X, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network
    Compute the forward propagation and for the cross entropy cost,
    the backward propagation for the gradients for all parameters.
    Notice the gradients computed here are different from the gradients in
    the assignment sheet: they are w.r.t. weights, not inputs.
    Arguments:
    X -- M x Dx matrix, where each row is a training example x.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    # Forward pass
    z1 = np.matmul(X, W1) + b1
    h = sigmoid(z1)
    z2 = np.matmul(h, W2) + b2
    y_hat = softmax(z2)
    cost = -np.sum(labels * np.log(y_hat))

    # Backward pass
    m = labels.shape[0]
    gradz2 = y_hat - labels

    # h is shape (1, H), gradz2 is shape (1, Dy)
    # so we need to flip the dimensions of h for it to multiply with gradz2
    gradW2 = np.matmul(h.T, gradz2)
    gradb2 = np.matmul(np.ones((1, m)), gradz2)
    gradz1 = np.matmul(gradz2, W2.T) * sigmoid_grad(sigmoid(z1))
    gradW1 = np.matmul(X.T, gradz1)
    gradb1 = np.matmul(np.ones((1, m)), gradz1)

    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad

def __sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print ("Running sanity check...")

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)

if __name__ == "__main__":
    __sanity_check()

