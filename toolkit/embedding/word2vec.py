#!/usr/bin/env python

import numpy as np
import random

from utils import softmax
from utils import gradcheck_naive
from utils import sigmoid, sigmoid_grad

def normalize_rows(x):
    """ Row normalization function
    Implement a function that normalizes each row of a matrix to have
    unit length.
    """
    sqrt_sum = np.sqrt(np.sum(np.power(x, 2), axis=1)).reshape((-1,1))
    x = x / sqrt_sum

    return x


def test_normalize_rows():
    print ("Testing normalizeRows...")
    x = normalize_rows(np.array([[3.0, 4.0], [1, 2]]))
    print (x)
    ans = np.array([[0.6,0.8],[0.4472136,0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print ("")


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models
    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.
    Arguments:
    predicted -- numpy ndarray, predicted word vector (\hat{v} in
                 the written component)
    target -- integer, the index of the target word
    outputVectors -- "output" vectors (as rows) for all tokens
    dataset -- needed for negative sampling, unused here.
    Return:
    cost -- cross entropy cost for the softmax word prediction
    gradPred -- the gradient with respect to the predicted word
           vector
    grad -- the gradient with respect to all the other word
           vectors
    We will not provide starter code for this function, but feel
    free to reference the code you previously wrote for this
    assignment!
    """

    # Here y and y_hat are all in one dimension
    # outputVectors is V * H matrix
    # y is one dimension length V matrix
    # predicted is v_c,
    y = np.zeros(outputVectors.shape[0])
    y[target] = 1
    y_hat = softmax(outputVectors.dot(predicted)).flatten()
    cost = np.sum(y * np.log(y_hat)) * (-1)

    # grad_theta id dCE / dtheta, where theta is vector of [u_w * vc]
    grad_theta = y_hat - y

    # grad_pred is dCE / d_vc, equals U * (y_hat - y)
    grad_pred = outputVectors.T.dot(grad_theta)

    # grad_theta: [d0, ..., d10000], predicted:[p0, ..., p300]
    # then grad is 10000 * 300, which is of the same shape of outputVectors
    grad = np.outer(grad_theta, predicted)

    return cost, grad_pred, grad


def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target
    Draw K "negative samples" from dataset, and these points
    drawn should not be the same with target
    """

    indices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset, K=10):
    """ Negative sampling cost function for word2vec models
    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.
    Note: See test_word2vec below for dataset's initialization.
    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!

    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))

    ### YOUR CODE HERE
    grad = np.zeros(outputVectors.shape)

    # Jneg_sample(o, vc, u) = - log(sigmoid(uo * vc)) - sum_1_k(log(sigmoid(-uk * vc)))
    predictedProb = sigmoid(outputVectors[target].dot(predicted))
    cost = - np.log(predictedProb)
    grad_pred = outputVectors[target] * (predictedProb - 1.0)
    grad[target] = predicted * (predictedProb - 1.0)
    for idx in indices[1:]:
        u = outputVectors[idx]
        product = u.dot(predicted)
        cost -= np.log(sigmoid(-product))
        grad_pred += u * sigmoid(product)
        grad[idx] += predicted * (sigmoid(-product) - 1) * (-1)

    return cost, grad_pred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec
    Implement the skip-gram model in this function.
    Arguments:
    currentWord -- a string of the current center word
    C -- integer, context size
    contextWords -- list of no more than 2*C strings, the context words
    tokens -- a dictionary that maps words to their indices in
              the word vector list
    inputVectors -- "input" word vectors (as rows) for all tokens
    outputVectors -- "output" word vectors (as rows) for all tokens
    word2vecCostAndGradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.
    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    c = tokens[currentWord]
    v_c = inputVectors[c]
    for contextWord in contextWords:
        j = tokens[contextWord]
        cost_j, gradPred_j, grad_j = word2vecCostAndGradient(v_c, j, outputVectors, dataset)
        cost = cost + cost_j
        gradIn[c] += gradPred_j.T
        gradOut += grad_j

    return cost, gradIn, gradOut


def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """CBOW model in word2vec
    Implement the continuous bag-of-words model in this function.
    Arguments/Return specifications: same as the skip-gram model
    Extra credit: Implementing CBOW is optional, but the gradient
    derivations are not. If you decide not to implement CBOW, remove
    the NotImplementedError.
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    target = tokens[currentWord]
    v_hat = np.sum(np.asarray([inputVectors[tokens[w]] for w in contextWords]), axis=0)
    cost, gradPred, grad = word2vecCostAndGradient(v_hat, target, outputVectors, dataset)
    for w in contextWords:
        gradIn[tokens[w]] += gradPred
    gradOut = grad

    return cost, gradIn, gradOut


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N // 2, :]
    outputVectors = wordVectors[N // 2:, :]
    for i in range(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N // 2, :] += gin / batchsize / denom
        grad[N // 2:, :] += gout / batchsize / denom

    return cost, grad


def test_word2vec():
    """ Interface to the dataset for negative sampling """
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in range(2*C)]

    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalize_rows(np.random.randn(10, 3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print("==== Gradient check for skip-gram ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)
    print("\n==== Gradient check for CBOW      ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)

    print("\n=== Results ===")
    print(skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
    print(skipgram("c", 1, ["a", "b"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient))
    print(cbow("a", 2, ["a", "b", "c", "a"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
    print(cbow("a", 2, ["a", "b", "a", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient))


if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()