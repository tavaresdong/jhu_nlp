import argparse
import numpy as np
import sys
sys.path.append("..")

from stanford_sent import StanfordSentiment
from sgd import load_saved_params, sgd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

def getSentenceFeatures(tokens, wordVectors, sentence):
    """
    Obtain the sentence feature for sentiment analysis by averaging its
    word vectors
    """

    # Implement computation for the sentence features given a sentence.

    # Inputs:
    # tokens -- a dictionary that maps words to their indices in
    #           the word vector list
    # wordVectors -- word vectors (each row) for all tokens
    # sentence -- a list of words in the sentence of interest

    # Output:
    # - sentVector: feature vector for the sentence

    sentVector = np.zeros((wordVectors.shape[1],))

    # Take the average of the word vectors
    wordVecs = [wordVectors[tokens[word]] for word in sentence]
    arr = np.asarray(wordVecs)
    sentVector = np.mean(arr, axis=0).reshape((-1,))

    assert sentVector.shape == (wordVectors.shape[1],)
    print(sentVector.shape)
    return sentVector

def getRegularizationValues():
    """Try different regularizations
    Return a sorted list of values to try.
    """
    values = None   # Assign a list of floats in the block below
    values = np.logspace(-4, 2, num=100, base=10)
    return sorted(values)

def chooseBestModel(results):
    """Choose the best model based on dev set performance.
    Arguments:
    results -- A list of python dictionaries of the following format:
        {
            "reg": regularization,
            "clf": classifier,
            "train": trainAccuracy,
            "dev": devAccuracy,
            "test": testAccuracy
        }
    Each dictionary represents the performance of one model.
    Returns:
    Your chosen result dictionary.
    """
    bestResult = None
    bestResult = max(results, key=lambda result: result["dev"])

    return bestResult

def accuracy(y, yhat):
    """ Precision for classifier """
    assert(y.shape == yhat.shape)
    return np.sum(y == yhat) * 100.0 / y.size

def outputPredictions(dataset, features, labels, clf, filename):
    """ Write the predictions to file """
    pred = clf.predict(features)
    with open(filename, "w") as f:
        print("True\tPredicted\tText", f)
        for i in range(len(dataset)):
            print("%d\t%d\t%s" % (
                labels[i], pred[i], " ".join(dataset[i][0])), f)

def __main__():
    dataset = StanfordSentiment(root_dir="../datasets")
    tokens = dataset.tokens()
    numWords = len(tokens)

    _, wordVectors, _ = load_saved_params("../saved_params")

    # Concatenate the input/output word embedding vectors to form the final word representation
    inputVectors, outputVectors = wordVectors[:numWords, :], wordVectors[numWords:, :]
    print(inputVectors.shape)
    print(outputVectors.shape)
    wordVectors = np.concatenate((inputVectors, outputVectors), axis=1)
    dimVectors = wordVectors.shape[1]

    # Load the train set
    trainset = dataset.getTrainSentences()
    nTrain = len(trainset)
    trainFeatures = np.zeros((nTrain, dimVectors))
    trainLabels = np.zeros((nTrain,), dtype=np.int32)
    for i in range(nTrain):
        words, trainLabels[i] = trainset[i]
        trainFeatures[i, :] = getSentenceFeatures(tokens, wordVectors, words)

    # Prepare dev set features
    devset = dataset.getDevSentences()
    nDev = len(devset)
    devFeatures = np.zeros((nDev, dimVectors))
    devLabels = np.zeros((nDev,), dtype=np.int32)
    for i in range(nDev):
        words, devLabels[i] = devset[i]
        devFeatures[i, :] = getSentenceFeatures(tokens, wordVectors, words)

    # Prepare test set features
    testset = dataset.getTestSentences()
    nTest = len(testset)
    testFeatures = np.zeros((nTest, dimVectors))
    testLabels = np.zeros((nTest,), dtype=np.int32)
    for i in range(nTest):
        words, testLabels[i] = testset[i]
        testFeatures[i, :] = getSentenceFeatures(tokens, wordVectors, words)

    # We will save our results from each run
    # and see if the regularization value performs well
    results = []
    regValues = getRegularizationValues()
    for reg in regValues:
        print("Training for reg=%f" % reg)
        # Note: add a very small number to regularization to please the library
        clf = LogisticRegression(C=1.0/(reg + 1e-12))
        clf.fit(trainFeatures, trainLabels)

        # Test on train set
        pred = clf.predict(trainFeatures)
        trainAccuracy = accuracy(trainLabels, pred)
        print("Train accuracy (%%): %f" % trainAccuracy)

        # Test on dev set
        pred = clf.predict(devFeatures)
        devAccuracy = accuracy(devLabels, pred)
        print("Dev accuracy (%%): %f" % devAccuracy)

        # Test on test set
        # Note: always running on test is poor style. Typically, you should
        # do this only after validation.
        pred = clf.predict(testFeatures)
        testAccuracy = accuracy(testLabels, pred)
        print("Test accuracy (%%): %f" % testAccuracy)

        results.append({
            "reg": reg,
            "clf": clf,
            "train": trainAccuracy,
            "dev": devAccuracy,
            "test": testAccuracy})

    # Print the accuracies
    print("")
    print("=== Recap ===")
    print("Reg\t\tTrain\tDev\tTest")
    for result in results:
        print("%.2E\t%.3f\t%.3f\t%.3f" % (
            result["reg"],
            result["train"],
            result["dev"],
            result["test"]))
    print("")

    bestResult = chooseBestModel(results)
    print("Best regularization value: %0.2E" % bestResult["reg"])
    print("Test accuracy (%%): %f" % bestResult["test"])


def __test_getRegularizationValues():
    values = getRegularizationValues()
    print(values)
    print(len(values))

def __test_getSentenceFeatures():
    dataset = StanfordSentiment()
    tokens = dataset.tokens()
    numWords = len(tokens)

    _, wordVectors, _ = load_saved_params("../saved_params")

    # Concatenate the input/output word embedding vectors to form the final word representation
    wordVectors = np.concatenate((wordVectors[:numWords, :], wordVectors[numWords:, :]), axis=1)
    sentence = "make a splash".split()
    sent_feature = getSentenceFeatures(tokens, wordVectors, sentence)
    print(sent_feature)

if __name__ == "__main__":
    #__test_getSentenceFeatures()
    #__test_getRegularizationValues()
    __main__()