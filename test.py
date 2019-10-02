"""
This file contains misc. codes for testing the machine learning algorithms.
"""

import numpy as np
import random
from sys import exit, maxsize
from sklearn import datasets
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix

from NeuralNetworks import *
from SVM import *
from DecisionTree import *

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def dSigmoid(x):
    return np.multiply(sigmoid(x), 1 - sigmoid(x))

def splitData(X, Y, testing = 0.3):
    """Randomly split the data into two groups.
    Parameters
    ----------
    X, Y : List
    testing : float
        If testing is less than 1, then it will be treated as fraction.
        Or its floor value will be treated as the testing group size.
    """
    if len(X) != len(Y):
        print("len(X) = " + str(len(X)) + "  len(Y) = " + str(len(Y)))
        exit()
    if testing > len(X):
        print("Too many to sample: " + str(testing))
        exit()
    XTrain, XTest = [], []
    YTrain, YTest = [], []
    indices = [i for i in range(len(X))]
    random.shuffle(indices)
    amountTest = testing
    if testing < 1:
        amountTest = np.floor(len(X) * testing)
    for i in range(len(X)):
        if i < amountTest:
            XTest.append(X[indices[i]])
            YTest.append(Y[indices[i]])
        else:
            XTrain.append(X[indices[i]])
            YTrain.append(Y[indices[i]])
    return XTrain, XTest, YTrain, YTest
    
def getAccuracy(predicted, groundTruth):
    if len(predicted) != len(groundTruth):
        print("List length unmatched: len(predicted) = " + str(len(predicted)) + "  len(groundTruth) = " + str(len(groundTruth)))
        exit()
    if len(predicted) == 0:
        return 0
    accuracy = 0
    for i in range(len(predicted)):
        if predicted[i] == groundTruth[i]:
            accuracy += 1
    accuracy /= len(predicted)
    return accuracy

def getData(dataset):
    X, Y = [], []
    if dataset == "iris":
        print("Selecting iris data (first 100 rows and first two columns).")
        iris = datasets.load_iris()
        X = iris.data[:100, :2].tolist()
        Y = iris.target[:100].tolist()
    elif dataset == "banknote":
        print("Selecting banknote data.")
        banknotes = open("data_banknote_authentication.txt")
        X = []
        Y = []
        for line in banknotes:
            lineSplitted = line.split(",")
            row = []
            for i in range(len(lineSplitted) - 1):
                row.append(float(lineSplitted[i]))
            X.append(row)
            Y.append(int(lineSplitted[-1]))
        banknotes.close()
    else:
        print("Unrecognized dataset name: " + str(dataset))
        exit()
    print("Data size: (" + str(len(X)) + ", " + str(len(X[0])) + ")")
    print("Classes: " + str(len(set(Y))))
    return X, Y


X, labels = getData("banknote")
XTrain, XTest, YTrain, YTest = splitData(X, labels)
clf = NeuralNetwork([4])
clf.fit(XTrain, YTrain, epochs = 20, learningRate = 0.01, regularC = 0.001, verbose = True)
YPredict = clf.predict(XTest).argmax(axis = 1)
print(YPredict[:20])
print(accuracy_score(YTest, YPredict))
print(confusion_matrix(YTest, YPredict))