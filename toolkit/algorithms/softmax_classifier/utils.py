import numpy as np
import tensorflow as tf

def softmax(x):
    '''
    Compute the softmax function with tensorflow
    softmax(x) = softmax(x + c) for any constant c
    we can use this to reduce max(x) to avoid overflow in exponential
    and there is at least one 0, some underflow may happen but is harmless
    :param x: a tensor with shape (n_samples, n_features)
    :return: tf.Tensor with shape(n_samples, n_features)
    '''
    max_by_row = tf.reduce_max(x, axis=1, keepdims=True)
    reduce_max = x - max_by_row
    exp = tf.exp(reduce_max)
    sum_exp = tf.reduce_sum(exp, axis=1, keepdims=True)
    result = exp / sum_exp
    return result

def cross_entropy_loss(y, yhat):
    '''
    Compute cross entropy loss for y and yhat
        CE(y, yhat) = -Sum(y_i * logyhat_i)
    :param y: one-hot vector of shape (n_samples, n_features)
    :param yhat: vector of shape (n_samples, n_features)
    :return: summed loss across all examples
    '''
    log_yhat = tf.log(yhat)
    mult = tf.multiply(tf.to_float(y), log_yhat)
    loss = tf.reduce_sum(mult)
    return -loss


def get_minibatches(data, minibatch_size, shuffle=True):
    """
    Iterates through the provided data one minibatch at at time. You can use this function to
    iterate through data in minibatches as follows:
        for inputs_minibatch in get_minibatches(inputs, minibatch_size):
            ...
    Or with multiple data sources:
        for inputs_minibatch, labels_minibatch in get_minibatches([inputs, labels], minibatch_size):
            ...
    Args:
        data: there are two possible values:
            - a list or numpy array
            - a list where each element is either a list or numpy array
        minibatch_size: the maximum number of items in a minibatch
        shuffle: whether to randomize the order of returned data
    Returns:
        minibatches: the return value depends on data:
            - If data is a list/array it yields the next minibatch of data.
            - If data a list of lists/arrays it returns the next minibatch of each element in the
              list. This can be used to iterate through multiple data sources
              (e.g., features and labels) at the same time.
    """
    list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
    data_size = len(data[0]) if list_data else len(data)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    for minibatch_start in np.arange(0, data_size, minibatch_size):
        minibatch_indices = indices[minibatch_start:minibatch_start + minibatch_size]
        yield [minibatch(d, minibatch_indices) for d in data] if list_data \
            else minibatch(data, minibatch_indices)


def minibatch(data, minibatch_idx):
    return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]

def __test_all_close(name, actual, expected):
    if actual.shape != expected.shape:
        raise ValueError("{:} failed, expected output to have shape {:} but has shape {:}"
                         .format(name, expected.shape, actual.shape))
    if np.max(np.fabs(actual - expected)) > 1e-6:
        raise ValueError("{:} failed, expected {:} but value is {:}".format(name, expected, actual))
    else:
        print(name, "passed!")

def __test_softmax():
    """
    Some simple tests of softmax to get you started.
    Warning: these are not exhaustive.
    """

    test1 = softmax(tf.constant(np.array([[1001, 1002], [3, 4]]), dtype=tf.float32))
    with tf.Session() as sess:
            test1 = sess.run(test1)
    __test_all_close("Softmax test 1", test1, np.array([[0.26894142, 0.73105858],
                                                      [0.26894142, 0.73105858]]))

    test2 = softmax(tf.constant(np.array([[-1001, -1002]]), dtype=tf.float32))
    with tf.Session() as sess:
            test2 = sess.run(test2)
    __test_all_close("Softmax test 2", test2, np.array([[0.73105858, 0.26894142]]))

    print("Basic (non-exhaustive) softmax tests pass\n")

def __test_cross_entropy_loss():
    """
    Some simple tests of cross_entropy_loss to get you started.
    Warning: these are not exhaustive.
    """
    y = np.array([[0, 1], [1, 0], [1, 0]])
    yhat = np.array([[.5, .5], [.5, .5], [.5, .5]])

    test1 = cross_entropy_loss(tf.constant(y, dtype=tf.int32), tf.constant(yhat, dtype=tf.float32))
    with tf.Session() as sess:
        test1 = sess.run(test1)
    expected = -3 * np.log(.5)
    __test_all_close("Cross-entropy test 1", test1, expected)

    print("Basic (non-exhaustive) cross-entropy tests pass")


if __name__ == "__main__":
    __test_softmax()
    __test_cross_entropy_loss()