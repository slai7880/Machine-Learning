"""
This program implements the good-old-fashioned neural networks.

Limitations:
1. The activation function is fixed for all layers except for the
last one.
2. The loss function is fixed to be cross entropy.
3. Only L2 regularization is implemented.
"""

import numpy as np
import random
from sys import exit, maxsize

class NeuralNetwork:
    def __init__(self, nodes): # nodes do not include the biases
        """
        Initializes the object.
        Parameters
        ----------
        nodes : List[int]
            The number of nodes in each layer.
        Returns
        -------
        None
        """
        if len(nodes) < 1:
            print("Not enough layer, len(nodes) = " + str(nodes))
            exit()
        for i in range(len(nodes)):
            if nodes[i] == 0:
                print("Layer " + str(i) + " is empty.")
        self.randomFactor = 1
        self.nodes = nodes
        self.W = [] # weights
        for i in range(1, len(nodes)):
            weights = np.random.normal(size = (nodes[i], nodes[i - 1] + 1))
            self.W.append(weights)
        # set the default activation function
        self.setActivationFunction(self.sigmoid, self.dSigmoid)
    
    def setActivationFunction(self, activate, dActivate):
        """
        Sets the avtivation function and its corresponding derivative.
        Parameters
        ----------
        activate, dActivate : python function
        Returns
        -------
        None
        """
        self.activate = activate
        self.dActivate = dActivate
    
    #################################################################
    # Default activation function and its derivative.
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
    def dSigmoid(self, x):
        return np.multiply(self.sigmoid(x), 1 - self.sigmoid(x))
    
    #################################################################
    
    def vectorize(self, Y):
        """
        Creates a one hot representation for each element in Y.
        Parameters
        ----------
        Y : array
            The elements inside must be integers.
        Returns
        -------
        results : array
            Each element is a one hot vector representation of the
            corresponding element in Y.
        """
        results = []
        NClass = len(set([y for y in Y]))
        for i in range(len(Y)):
            y = np.array([0] * NClass)
            y[Y[i]] = 1
            results.append(y)
        return results
    
    def softmax(self, L):
        """
        Computes the softmax for L.
        """
        exp = np.exp(L)
        return exp / exp.sum()
    
    def fit(self, X, Y,
            batchSize = 1, epochs = 10, learningRate = 0.001, regularC = 0,
            verbose = False):
        """
        Fits the model.
        Parameters
        ----------
        X : array-like
            Each row is a sample input.
        Y : array-like
            The input labels.
        batchSize : int
        epochs : int
        learningRate : float
        regularC : float
            This is the regularization coefficient.
        verbose : boolean
            True if printout is desired.
        Returns
        -------
        None
        """
        assert len(X) != 0 and len(X) == len(Y) and len(X[0]) != 0
        if isinstance(X, list): # just for convention
            X = np.matrix(X)
        Y = self.vectorize(Y) # essential for network output comparison
        
        # add two weight matrices to the list, given input and output dimensions
        self.W.insert(0, np.random.normal(size = (self.nodes[0], X.shape[1] + 1)))
        self.W.append(np.random.normal(size = (len(Y[0]), self.nodes[-1] + 1)))
        for epoch in range(epochs):
            loss = 0
            # outputsBatch records the outputs per layer for each sample
            YTrue, outputsBatch = [], []
            for i in range(X.shape[0]):
                YTrue.append(Y[i])
                outputs = self.forward(X[i, :])
                outputsBatch.append(outputs)
                if i % batchSize == 0 or i == X.shape[0] - 1: # do one backprog per batch
                    probabilities = [self.softmax(o[-1]) for o in outputsBatch]
                    loss += self.getLoss(YTrue, probabilities, regularC)
                    self.backprog(YTrue, outputsBatch, learningRate, regularC)
                    YTrue, outputsBatch = [], []
            if verbose:
                print("Epoch: " + str(epoch) + "  loss = " + str(loss))
    
    def predict(self, X):
        """
        Predicts on X.
        Parameters
        ----------
        X : array-like
            Sample inputs.
        Returns
        -------
        results : numpy array
            Each element is a probability array from the last layer
            of the network.
        """
        if isinstance(X, list):
            X = np.matrix(X)
        results = []
        for i in range(X.shape[0]):
            outputs = self.forward(X[i, :])
            results.append(np.squeeze(np.array(self.softmax(outputs[-1]))))
        return np.array(results)
    
    def forward(self, x):
        """
        Run a forward propagation.
        Parameters
        ----------
        x : numpy array
            Single sample.
        Returns
        -------
        outputs : List[numpy matrix]
            The outputs per layer before activation.
        """
        outputs = [x.T] # stores the outputs for each layer 
        for i in range(len(self.W)):
            # the extra 1 is the bias term
            inputVector = np.vstack((self.activate(outputs[-1]), np.matrix([[1]])))
            
            A = self.W[i] * inputVector
            outputs.append(A)
        return outputs
        
        
    def backprog(self, YTrue, outputsBatch, learningRate, regularC):
        """
        Run a backward propagation.
        Parameters
        ----------
        YTrue : List[numpy array]
            Each element should be a one hot encoding of a sample.
        outputBatch : List[List[numpy matrix]]
            Each element is a list of outputs per layer from a sample.
        learningRate : float
        regularC : float
            This is the regularization coefficient.
        Returns
        -------
        None
        """
        gradientsBatch = [np.zeros(w.shape) for w in self.W]
        for i in range(len(YTrue)):
            delta = [self.softmax(outputsBatch[i][-1]) - np.matrix(YTrue[i]).T]
            for j in range(len(outputsBatch[i]) - 2, 0, -1):
                delta.insert(0, np.multiply(self.dActivate(outputsBatch[i][j]), self.W[j][:, :-1].T * delta[0]))
            for j in range(len(self.W)):
                gradientsBatch[j] += delta[j] * np.hstack((self.activate(outputsBatch[i][j]).T, np.matrix([[1]])))
        for j in range(len(self.W)):
            self.W[j] = (1 - learningRate * regularC) * self.W[j] - learningRate * gradientsBatch[j]
    
    def getLoss(self, YTrue, probabilities, regularC):
        """
        Computes cross entropy loss with L2 regularization.
        Parameters
        ----------
        YTrue : List[numpy array]
            Each element should be a one hot encoding of a sample.
        probabilities : List[numpy array]
            Each elelemt should be a probability array from the last layer.
        regularC : float
            This is the regularization coefficient.
        Returns
        -------
        loss : float
        """
        loss = 0
        for i in range(len(YTrue)):
            loss -= YTrue[i] * np.log(probabilities[i])
        loss = np.squeeze(np.array(loss))
        
        # regularization
        loss += regularC * np.sum([np.linalg.norm(w, "fro")**2 for w in self.W])
        return loss / len(YTrue)
            
        
#========================= End of Section ==========================#
