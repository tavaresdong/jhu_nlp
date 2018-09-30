import numpy as np
import random


def softmax(x):
    """Compute the softmax function for each row of the input x.

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code. You might find numpy
    functions np.exp, np.sum, np.reshape, np.max, and numpy
    broadcasting useful for this task.

    Numpy broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

    You should also make sure that your code works for a single
    D-dimensional vector (treat the vector as a single row) and
    for N x D matrices. This may be useful for testing later. Also,
    make sure that the dimensions of the output match the input.

    Arguments:
    x -- A D dimensional vector or N x D dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        #raise NotImplementedError
        x = x - np.max(x, axis=1, keepdims=True)
        x = np.exp(x)
        sums = np.sum(x, axis=1, keepdims=True)
        x = x / sums
    else:
        x = x - np.max(x)
        x = np.exp(x)
        x = x / np.sum(x)

    assert x.shape == orig_shape
    return x

def sigmoid(x):
    """
    Computer the sigmoid value of a given x
    :param x:  the input variable, a scalar or numpy arrary
    :return: sigmoid(x)
    """
    x = np.exp(-x) + 1
    return 1 / x

def sigmoid_grad(x):
    """
    Calculate the gradient of the softmax function
    :param x: scalar or numpy array
    :return: your computed gradient
    """
    comp = 1 - x
    grad = x * comp
    return grad

def gradcheck_naive(f, x):
    """ Gradient check for a function f.
    Arguments:
    f -- a function that takes a single argument and outputs the
         cost and its gradients
    x -- the point (numpy array) to check the gradient at
    """

    rndstate = random.getstate()
    random.setstate(rndstate)
    fx, grad = f(x) # Evaluate function value at original point
    h = 1e-4        # Do not change this!

    # Iterate over all indexes ix in x to check the gradient.
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        # Must copy here, we
        # calculate f(x1, x2, ... xk + delte, xk+1, ...) so
        # aside from xk, the rest should be unchanged
        x_plus_d = np.copy(x)
        x_minus_d = np.copy(x)

        x_plus_d[ix] += h
        x_minus_d[ix] -= h

        random.setstate(rndstate)
        fx_plus, _ = f(x_plus_d)

        random.setstate(rndstate)
        fx_minus, _ = f(x_minus_d)

        numgrad = (fx_plus - fx_minus) / (2 * h)

        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5:
            print ("Gradient check failed.")
            print ("First gradient error found at index %s" % str(ix))
            print ("Your gradient: %f \t Numerical gradient: %f" % (
                grad[ix], numgrad))
            return False

        it.iternext()

    print ("Gradient check passed!")
    return True

def __sanity_check():
    """
    Some basic sanity checks.
    """
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print("Running sanity checks...")
    gradcheck_naive(quad, np.array(123.456))      # scalar test
    gradcheck_naive(quad, np.random.randn(3,))    # 1-D test
    gradcheck_naive(quad, np.random.randn(4,5))   # 2-D test
    print("Passed gradient check test")

def __test_sigmoid_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print ("Running basic tests...")
    x = np.array([[1, 2], [-1, -2]])
    f = sigmoid(x)
    g = sigmoid_grad(f)
    print (f)
    f_ans = np.array([
        [0.73105858, 0.88079708],
        [0.26894142, 0.11920292]])
    assert np.allclose(f, f_ans, rtol=1e-05, atol=1e-06)
    print (g)
    g_ans = np.array([
        [0.19661193, 0.10499359],
        [0.19661193, 0.10499359]])
    assert np.allclose(g, g_ans, rtol=1e-05, atol=1e-06)
    print ("You should verify these results by hand!\n")


def __test_softmax_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print ("Running basic tests...")
    test1 = softmax(np.array([1,2]))
    print (test1)
    ans1 = np.array([0.26894142,  0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print (test2)
    ans2 = np.array([
        [0.26894142, 0.73105858],
        [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[-1001,-1002]]))
    print (test3)
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)

    print ("You should be able to verify these results by hand!\n")


if __name__ == "__main__":
    __test_softmax_basic()
    __test_sigmoid_basic()
    __sanity_check()
